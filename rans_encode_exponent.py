import argparse
import os
import json
import re
import safetensors.torch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
from collections import defaultdict
import time
import traceback
import sys
import math
import io # For in-memory bitstream
import contextlib
import pickle # To save the independent streams

# --- FP8 E4M3 Type Handling ---
FP8_E4M3_DTYPE = None
FP8_E4M3_SUPPORTED = False
try:
    FP8_E4M3_DTYPE = torch.float8_e4m3fn
    FP8_E4M3_SUPPORTED = True
    print(f"Targeting FP8 dtype: {FP8_E4M3_DTYPE}")
except AttributeError:
    warnings.warn(
        f"FP8 type torch.float8_e4m3fn not found. Need PyTorch 2.1+."
    )

# --- Constants ---
EXPONENT_SHIFT = 3
EXPONENT_MASK = 0b1111 # 0-15
NUM_EXPONENTS = 16 # Vocabulary size for exponents
EXPONENT_BITS = 4  # Original bits per exponent

# --- Arithmetic Coding Library (Insert full classes here as before) ---
python3 = sys.version_info.major >= 3
# ... (ArithmeticCoderBase, ArithmeticEncoder, ArithmeticDecoder, BitInputStream, BitOutputStream classes) ...
class ArithmeticCoderBase(object):
	def __init__(self, numbits):
		if numbits < 1: raise ValueError("State size out of range")
		self.num_state_bits = numbits
		self.full_range = 1 << self.num_state_bits
		self.half_range = self.full_range >> 1
		self.quarter_range = self.half_range >> 1
		self.minimum_range = self.quarter_range + 2
		self.maximum_total = self.minimum_range
		self.state_mask = self.full_range - 1
		self.low = 0
		self.high = self.state_mask
	def update(self, freqs, symbol):
		low = self.low; high = self.high
		range_ = high - low + 1
		total = int(freqs[-1])
		symlow = int(freqs[symbol-1]) if symbol > 0 else 0
		symhigh = int(freqs[symbol])
		newlow  = low + symlow  * range_ // total
		newhigh = low + symhigh * range_ // total - 1
		self.low = newlow; self.high = newhigh
		while ((self.low ^ self.high) & self.half_range) == 0:
			self.shift(); self.low  = ((self.low  << 1) & self.state_mask)
			self.high = ((self.high << 1) & self.state_mask) | 1
		while (self.low & ~self.high & self.quarter_range) != 0:
			self.underflow(); self.low = (self.low << 1) ^ self.half_range
			self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1
	def shift(self): raise NotImplementedError()
	def underflow(self): raise NotImplementedError()

class ArithmeticEncoder(ArithmeticCoderBase):
	def __init__(self, numbits, bitout):
		super(ArithmeticEncoder, self).__init__(numbits); self.output = bitout; self.num_underflow = 0
	def write(self, freqs, symbol): self.update(freqs, symbol)
	def finish(self): self.output.write(1) # Final bit is 1 for encoding
	def shift(self):
		bit = self.low >> (self.num_state_bits - 1); self.output.write(bit)
		for _ in range(self.num_underflow): self.output.write(bit ^ 1)
		self.num_underflow = 0
	def underflow(self): self.num_underflow += 1

class ArithmeticDecoder(ArithmeticCoderBase): # Keep decoder for completeness
    def __init__(self, numbits, bitin):
        super(ArithmeticDecoder, self).__init__(numbits); self.input = bitin; self.code = 0
        for _ in range(self.num_state_bits): self.code = self.code << 1 | self.read_code_bit()
    def read(self, freqs):
        total = int(freqs[-1]); range_ = self.high - self.low + 1
        offset = self.code - self.low; value = ((offset + 1) * total - 1) // range_
        start = 0; end = len(freqs)
        while end - start > 1:
            middle = (start + end) >> 1; low = int(freqs[middle-1]) if middle > 0 else 0
            if low > value: end = middle
            else: start = middle
        symbol = start; self.update(freqs, symbol); return symbol
    def shift(self): self.code = ((self.code << 1) & self.state_mask) | self.read_code_bit()
    def underflow(self): self.code = (self.code & self.half_range) | ((self.code << 1) & (self.state_mask >> 1)) | self.read_code_bit()
    def read_code_bit(self):
        temp = self.input.read();
        if temp == -1: temp = 0
        return temp

class BitInputStream(object): # Keep for completeness
	def __init__(self, inp): self.input = inp; self.currentbyte = 0; self.numbitsremaining = 0
	def read(self):
		if self.currentbyte == -1: return -1
		if self.numbitsremaining == 0:
			temp = self.input.read(1);
			if len(temp) == 0: self.currentbyte = -1; return -1
			self.currentbyte = temp[0] if python3 else ord(temp); self.numbitsremaining = 8
		assert self.numbitsremaining > 0; self.numbitsremaining -= 1
		return (self.currentbyte >> self.numbitsremaining) & 1
	def read_no_eof(self): result = self.read(); assert result != -1; return result
	def close(self): self.input.close(); self.currentbyte = -1; self.numbitsremaining = 0

class BitOutputStream(object):
	def __init__(self, out): self.output = out; self.currentbyte = 0; self.numbitsfilled = 0
	def write(self, b):
		assert b in (0, 1); self.currentbyte = (self.currentbyte << 1) | b; self.numbitsfilled += 1
		if self.numbitsfilled == 8:
			towrite = bytes((self.currentbyte,)) if python3 else chr(self.currentbyte)
			self.output.write(towrite); self.currentbyte = 0; self.numbitsfilled = 0
	def close(self):
		while self.numbitsfilled != 0: self.write(0)
		# Do not close self.output here, let the caller manage it (like io.BytesIO)

# --- Helper Functions ---

def parse_index(index_path):
    if not os.path.exists(index_path): raise FileNotFoundError(f"Index file not found: {index_path}")
    with open(index_path, 'r') as f: index_data = json.load(f)
    if "weight_map" not in index_data: raise ValueError("Index file missing 'weight_map' key.")
    return index_data["weight_map"]

def find_tensors_for_layer(weight_map, target_layer):
    layer_proj_experts = defaultdict(list)
    expert_pattern = re.compile(rf"layers\.{target_layer}\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    alt_expert_pattern = re.compile(rf"model\.layers\.{target_layer}\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    print(f"Searching for expert tensors in layer {target_layer}...")
    count = 0
    for tensor_name, filename in weight_map.items():
        match = expert_pattern.search(tensor_name) or alt_expert_pattern.search(tensor_name)
        if match:
            expert_num = int(match.group(1)); proj_type = match.group(2)
            layer_proj_experts[proj_type].append({'expert': expert_num,'name': tensor_name,'file': filename})
            count += 1
    print(f"Found {count} expert tensor entries.")
    for proj in layer_proj_experts: layer_proj_experts[proj].sort(key=lambda x: x['expert'])
    return layer_proj_experts

def load_validated_tensor(f_handle, tensor_name, device):
     try: tensor = f_handle.get_tensor(tensor_name)
     except Exception as e: raise IOError(f"Load fail '{tensor_name}': {e}")
     if tensor.ndim != 2: raise ValueError(f"Tensor '{tensor_name}' not 2D")
     if tensor.numel() == 0: raise ValueError(f"Tensor '{tensor_name}' empty")
     # No dtype validation here, focus on structure
     return tensor.to(device)

def extract_all_exponents_np(tensor_2d):
    if tensor_2d.numel() == 0: return np.array([], dtype=np.uint8)
    # Perform extraction on CPU for numpy compatibility
    byte_view = tensor_2d.cpu().view(torch.uint8)
    exponents = (byte_view >> EXPONENT_SHIFT) & EXPONENT_MASK
    return exponents.numpy()

def prepare_layer_exponent_by_row(layer_proj_experts, model_dir, device):
    """Loads tensors, extracts exponents, groups by row index."""
    # row_data[row_index] = list of exponent values for that row across all tensors
    row_data = defaultdict(list)
    last_opened_file = None
    f_handle = None
    tensors_processed = 0
    expected_shape = None
    total_elements = 0

    proj_order = ['gate_proj', 'up_proj', 'down_proj']

    try:
        for proj_type in proj_order:
            if proj_type not in layer_proj_experts: continue
            print(f"  Processing projection: {proj_type}")
            expert_list = layer_proj_experts[proj_type]

            for expert_info in expert_list:
                filepath = os.path.join(model_dir, expert_info['file'])
                tensor_name = expert_info['name']
                try:
                    if filepath != last_opened_file:
                        if f_handle: pass # No close
                        f_handle = safetensors.torch.safe_open(filepath, framework="pt", device="cpu")
                        last_opened_file = filepath

                    tensor = load_validated_tensor(f_handle, tensor_name, device)

                    # Determine expected shape and check consistency
                    if expected_shape is None:
                        expected_shape = tensor.shape
                        print(f"    Detected tensor shape: {expected_shape}")
                    elif tensor.shape != expected_shape:
                         print(f"    Warning: Skipping {tensor_name}, shape {tensor.shape} != {expected_shape}")
                         continue

                    exponents_np = extract_all_exponents_np(tensor) # Get exponents as 2D numpy array
                    tensors_processed += 1
                    total_elements += exponents_np.size

                    # Append row data
                    for r in range(exponents_np.shape[0]):
                        row_data[r].append(exponents_np[r]) # Append the entire row array

                except (IOError, TypeError, ValueError, KeyError) as e:
                    print(f"\n    Warning: Skipping tensor {tensor_name} due to error: {e}")
                except Exception as e:
                    print(f"\n    Warning: Skipping tensor {tensor_name} due to unexpected error: {e}")
                    traceback.print_exc()
    finally:
        pass # No explicit close needed

    if not row_data:
        return None, 0

    print(f"\nConcatenating exponent sequences for each row... Processed {tensors_processed} tensors.")
    # Concatenate the collected row arrays for each row index
    row_sequences = {}
    max_row_index = -1
    for r in sorted(row_data.keys()):
        if row_data[r]: # Only process if data was collected for this row
             row_sequences[r] = np.concatenate(row_data[r])
             max_row_index = r
        else:
             print(f"Warning: No valid exponent data collected for row {r}")

    print(f"Created sequences for rows 0 to {max_row_index}.")
    return row_sequences, total_elements

# --- PyTorch LSTM Model (Same as before) ---
class ExponentPredictorLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super().__init__(); self.hidden_dim = hidden_dim; self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
    def forward(self, x, hidden):
        embedded = self.embedding(x); lstm_out, hidden = self.lstm(embedded, hidden)
        last_out = lstm_out[:, -1, :]; output = self.fc(last_out)
        log_probs = self.log_softmax(output); return log_probs, hidden
    def init_hidden(self, device, batch_size=1): # Added batch_size parameter
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

def detach_hidden(hidden):
    if isinstance(hidden, torch.Tensor): return hidden.detach()
    else: return tuple(detach_hidden(h) for h in hidden)

# --- Modified Compression Function for ONE Sequence ---
def compress_single_sequence(sequence_np, row_index, model, optimizer, criterion, bitout, device, vocab_size, ac_precision):
    """Trains LSTM while encoding *one* sequence (e.g., one row)."""
    # Creates a new coder for each sequence, writing to the SAME bitout stream
    coder = ArithmeticEncoder(ac_precision, bitout)

    model.train() # Ensure model is in training mode for this sequence
    hidden = model.init_hidden(device, batch_size=1) # Batch size is 1 here
    total_loss = 0
    total_bits_theoretical = 0
    sequence_len = len(sequence_np)

    if sequence_len == 0:
        print(f"Warning: Empty sequence provided for row {row_index}. Skipping encoding.")
        # Need to handle how the decoder knows this sequence is empty/missing
        # Or ensure empty sequences are not passed in.
        # For simplicity, we'll just return 0 bits, assuming prepare func filters empty.
        return 0, 0

    prob_scale = 1 << 16

    # --- Encode the very first symbol using a uniform distribution ---
    first_symbol = sequence_np[0]
    uniform_freq = np.cumsum(np.full(vocab_size, prob_scale // vocab_size + 1))
    coder.write(uniform_freq, first_symbol)
    total_bits_theoretical += math.log2(vocab_size)

    # --- Main Loop ---
    current_input_symbol = torch.tensor([[first_symbol]], dtype=torch.long).to(device)

    for t in range(1, sequence_len):
        # Predict, Encode, Train (as in previous script's loop)
        log_probs, hidden = model(current_input_symbol, hidden)
        hidden = detach_hidden(hidden)

        probs_np = torch.exp(log_probs).squeeze().detach().cpu().numpy()
        probs_np = np.maximum(probs_np, 1e-9); probs_np /= np.sum(probs_np)
        freq_table = np.cumsum(probs_np * prob_scale + 1).astype(np.uint64)
        target_symbol = sequence_np[t]
        coder.write(freq_table, target_symbol)

        target_tensor = torch.tensor([target_symbol], dtype=torch.long).to(device)
        loss = criterion(log_probs, target_tensor)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        total_loss += loss.item()
        actual_log_prob = log_probs[0, target_symbol].item()
        total_bits_theoretical -= actual_log_prob / math.log(2)
        current_input_symbol = torch.tensor([[target_symbol]], dtype=torch.long).to(device)

    # --- Finish Encoding for THIS sequence ---
    coder.finish() # Finish the individual stream for this row

    avg_loss = total_loss / (sequence_len - 1) if sequence_len > 1 else 0
    return total_bits_theoretical, avg_loss


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Compress layer exponents row-by-row using LSTM and Arithmetic Coding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model_dir", help="Path to model directory.")
    parser.add_argument("--layer", type=int, required=True, help="Target layer number.")
    parser.add_argument("--output_file", type=str, default="compressed_rows.pkl", help="Output file to save compressed row data.")
    parser.add_argument("--device", default="cpu", help="Device ('cpu', 'cuda:0', etc.).")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="LSTM hidden dimension.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--ac_precision", type=int, default=32, help="Arithmetic coder precision.")

    args = parser.parse_args()

    # --- Input Validation ---
    if not FP8_E4M3_SUPPORTED: print("Error: FP8 type required."); return
    if args.embedding_dim <= 0 or args.hidden_dim <= 0 or args.lr <= 0: print("Error: Dims/LR must be positive."); return
    if args.ac_precision < 16 or args.ac_precision > 60: print("Warning: ac_precision outside typical range.");

    # --- Device Setup ---
    try:
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available(): print(f"Warning: CUDA '{args.device}' N/A. Using CPU."); device = torch.device("cpu")
        print(f"Using device for computation: {device}")
    except Exception as e: print(f"Error setting device: {e}. Using CPU."); device = torch.device("cpu")

    # --- Find Tensors and Prepare Row Sequences ---
    index_path = os.path.join(args.model_dir, "model.safetensors.index.json");
    if not os.path.exists(index_path): index_path = os.path.join(args.model_dir, "index.json")
    try:
        print(f"Parsing index file: {index_path}"); weight_map = parse_index(index_path)
        # Process ALL projections together for row data
        layer_proj_experts = find_tensors_for_layer(weight_map, args.layer)
        if not layer_proj_experts: print(f"Error: No expert tensors found for layer {args.layer}."); return

        print(f"\nPreparing exponent sequences grouped by row for layer {args.layer}...")
        row_sequences, total_elements = prepare_layer_exponent_by_row(
            layer_proj_experts, args.model_dir, device
        )
        if row_sequences is None or total_elements == 0: print("Error: Failed to create row sequences."); return

    except FileNotFoundError: print(f"Error: Index file not found: {index_path}"); return
    except Exception as e: print(f"Error during index/sequence prep: {e}"); traceback.print_exc(); return

    # --- Setup Model (Shared Weights) ---
    print("\nSetting up shared LSTM model...")
    # Use a single model instance - weights will be shared across row compressions
    model = ExponentPredictorLSTM(
        vocab_size=NUM_EXPONENTS, embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim, num_layers=2
    ).to(device)
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()

    # --- Compress Each Row Sequence Independently ---
    print("\nStarting compression process for each row...")
    start_compress_time = time.time()
    compressed_row_data = {} # Store compressed bytes for each row
    total_theoretical_bits = 0
    total_actual_bits = 0
    processed_rows = 0

    sorted_row_indices = sorted(row_sequences.keys())

    for row_index in sorted_row_indices:
        sequence_np = row_sequences[row_index]
        sequence_len = len(sequence_np)
        if sequence_len == 0: continue # Skip empty rows

        print(f"  Compressing Row {row_index} (length {sequence_len:,})...")
        # Create a new buffer and bitstream for each row
        compressed_buffer = io.BytesIO()
        bitout = BitOutputStream(compressed_buffer)

        # Compress this single row's sequence
        # The model state (hidden) is reset inside compress_single_sequence
        # Optimizer state persists and is updated by each sequence
        theoretical_bits, avg_loss = compress_single_sequence(
            sequence_np, row_index, model, optimizer, criterion, bitout,
            device, NUM_EXPONENTS, args.ac_precision
        )

        bitout.close() # Finalize stream for this row
        compressed_bytes = compressed_buffer.getvalue()
        compressed_row_data[row_index] = compressed_bytes # Store compressed bytes
        total_theoretical_bits += theoretical_bits
        total_actual_bits += len(compressed_bytes) * 8
        processed_rows += 1
        print(f"    Row {row_index}: Actual Bits = {len(compressed_bytes)*8:,}, Theoretical Bits = {theoretical_bits:,.1f}, Avg Loss = {avg_loss:.4f}")

    end_compress_time = time.time()
    print(f"\nFinished compressing {processed_rows} rows in {end_compress_time - start_compress_time:.2f} seconds.")

    # --- Save Compressed Data ---
    print(f"\nSaving compressed row data to {args.output_file}...")
    try:
        with open(args.output_file, 'wb') as f_out:
            # Save metadata needed for potential decoding
            metadata = {
                'layer': args.layer,
                'num_rows': max(row_sequences.keys()) + 1 if row_sequences else 0,
                'total_elements': total_elements,
                'ac_precision': args.ac_precision,
                # Store LSTM params if needed for separate decoder
                'model_params': {
                    'embedding_dim': args.embedding_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': 2 # Fixed
                }
                # Could also save final model state_dict if not training decoder online
            }
            pickle.dump(metadata, f_out)
            pickle.dump(compressed_row_data, f_out)
        print("Compressed data saved.")
    except Exception as e:
        print(f"Error saving compressed data: {e}")
        traceback.print_exc()


    # --- Final Report ---
    original_size_bits = total_elements * EXPONENT_BITS

    print("\n--- Overall Compression Results ---")
    print(f"Layer Analyzed:         {args.layer}")
    print(f"Total Exponents:        {total_elements:,}")
    print(f"Original Size:          {original_size_bits:,.0f} bits ({original_size_bits/8.0:,.1f} bytes)")
    print(f"Total Compressed Size:  {total_actual_bits:,.0f} bits ({total_actual_bits/8.0:,.1f} bytes)")
    print(f"Total Theoretical Size: {total_theoretical_bits:,.0f} bits (based on -log2(p))")

    if original_size_bits > 0:
        compression_ratio_actual = (original_size_bits - total_actual_bits) / original_size_bits * 100
        bits_per_exponent_actual = total_actual_bits / total_elements if total_elements else 0
        compression_ratio_theory = (original_size_bits - total_theoretical_bits) / original_size_bits * 100
        bits_per_exponent_theory = total_theoretical_bits / total_elements if total_elements else 0

        print(f"\nCompression Ratio (Actual): {compression_ratio_actual:.2f}%")
        print(f"Bits/Exponent (Actual):   {bits_per_exponent_actual:.4f} (vs {EXPONENT_BITS} original)")
        print(f"Compression Ratio (Theory): {compression_ratio_theory:.2f}%")
        print(f"Bits/Exponent (Theory):   {bits_per_exponent_theory:.4f}")
    else:
        print("\nCannot calculate ratios (original size is 0).")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
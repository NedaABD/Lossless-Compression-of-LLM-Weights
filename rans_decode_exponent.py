import argparse
import os
import json
import re
import safetensors.torch
import torch
import torch.nn as nn
# import torch.optim as optim # No optimizer needed for decoding
import numpy as np
import warnings
from collections import defaultdict
import time
import traceback
import sys
import math
import io # For in-memory bitstream
import contextlib
import pickle # To load the compressed streams

# --- FP8 E4M3 Type Handling (Less critical, but good context) ---
# ... (Keep the FP8 check section for context if desired) ...
FP8_E4M3_DTYPE = None
FP8_E4M3_SUPPORTED = False
try:
    FP8_E4M3_DTYPE = torch.float8_e4m3fn
    FP8_E4M3_SUPPORTED = True
    print(f"Targeting FP8 dtype (context): {FP8_E4M3_DTYPE}")
except AttributeError:
    warnings.warn(f"FP8 type torch.float8_e4m3fn not found.")


# --- Constants ---
EXPONENT_SHIFT = 3
EXPONENT_MASK = 0b1111 # 0-15
NUM_EXPONENTS = 16 # Vocabulary size for exponents

# --- Arithmetic Coding Library (Insert full classes here) ---
python3 = sys.version_info.major >= 3
# ... (ArithmeticCoderBase, ArithmeticEncoder, ArithmeticDecoder, BitInputStream, BitOutputStream classes) ...
# --- Arithmetic coding core classes ---
class ArithmeticCoderBase(object):
	def __init__(self, numbits):
		if numbits < 1: raise ValueError("State size out of range")
		self.num_state_bits = numbits; self.full_range = 1 << self.num_state_bits
		self.half_range = self.full_range >> 1; self.quarter_range = self.half_range >> 1
		self.minimum_range = self.quarter_range + 2; self.maximum_total = self.minimum_range
		self.state_mask = self.full_range - 1; self.low = 0; self.high = self.state_mask
	def update(self, freqs, symbol):
		low = self.low; high = self.high; range_ = high - low + 1
		total = int(freqs[-1]); symlow = int(freqs[symbol-1]) if symbol > 0 else 0
		symhigh = int(freqs[symbol]); newlow  = low + symlow  * range_ // total
		newhigh = low + symhigh * range_ // total - 1; self.low = newlow; self.high = newhigh
		while ((self.low ^ self.high) & self.half_range) == 0:
			self.shift(); self.low  = ((self.low  << 1) & self.state_mask)
			self.high = ((self.high << 1) & self.state_mask) | 1
		while (self.low & ~self.high & self.quarter_range) != 0:
			self.underflow(); self.low = (self.low << 1) ^ self.half_range
			self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1
	def shift(self): raise NotImplementedError()
	def underflow(self): raise NotImplementedError()

class ArithmeticEncoder(ArithmeticCoderBase): # Keep for completeness
	def __init__(self, numbits, bitout): super(ArithmeticEncoder, self).__init__(numbits); self.output = bitout; self.num_underflow = 0
	def write(self, freqs, symbol): self.update(freqs, symbol)
	def finish(self): self.output.write(1)
	def shift(self): bit = self.low >> (self.num_state_bits - 1); self.output.write(bit); [self.output.write(bit ^ 1) for _ in range(self.num_underflow)]; self.num_underflow = 0
	def underflow(self): self.num_underflow += 1

class ArithmeticDecoder(ArithmeticCoderBase):
    def __init__(self, numbits, bitin):
        super(ArithmeticDecoder, self).__init__(numbits); self.input = bitin; self.code = 0
        for _ in range(self.num_state_bits): self.code = self.code << 1 | self.read_code_bit()
    def read(self, freqs):
        total = int(freqs[-1]); range_ = self.high - self.low + 1
        # Check for zero range, which can happen if the state becomes invalid
        if range_ == 0: raise EOFError("Arithmetic decoder state is invalid (zero range).")
        offset = self.code - self.low; value = ((offset + 1) * total - 1) // range_
        # Basic range check for value
        if not (0 <= value < total): raise ValueError(f"Decoder value {value} out of range [0, {total})")
        start = 0; end = len(freqs)
        while end - start > 1:
            middle = (start + end) >> 1; low_bound = int(freqs[middle-1]) if middle > 0 else 0
            if low_bound > value: end = middle
            else: start = middle
        symbol = start; self.update(freqs, symbol); return symbol
    def shift(self): self.code = ((self.code << 1) & self.state_mask) | self.read_code_bit()
    def underflow(self): self.code = (self.code & self.half_range) | ((self.code << 1) & (self.state_mask >> 1)) | self.read_code_bit()
    def read_code_bit(self):
        temp = self.input.read();
        if temp == -1:
            # Need a way to signal EOF *within* the read process if necessary
            # Or just treat as 0s. Treating as 0s matches reference implementation.
            temp = 0
        return temp

class BitInputStream(object):
	def __init__(self, inp): self.input = inp; self.currentbyte = 0; self.numbitsremaining = 0; self.eof_reached = False
	def read(self):
		if self.eof_reached: return -1 # Stay at -1 once EOF is hit
		if self.numbitsremaining == 0:
			temp_bytes = self.input.read(1);
			if len(temp_bytes) == 0: self.currentbyte = -1; self.eof_reached = True; return -1
			self.currentbyte = temp_bytes[0]; self.numbitsremaining = 8
		assert self.numbitsremaining > 0; self.numbitsremaining -= 1
		return (self.currentbyte >> self.numbitsremaining) & 1
	# Removed read_no_eof as ArithmeticDecoder handles EOF as 0 now
	def close(self): self.input.close(); self.currentbyte = -1; self.numbitsremaining = 0

class BitOutputStream(object): # Keep for completeness
	def __init__(self, out): self.output = out; self.currentbyte = 0; self.numbitsfilled = 0
	def write(self, b): assert b in (0, 1); self.currentbyte=(self.currentbyte << 1)|b; self.numbitsfilled+=1; \
		if self.numbitsfilled==8: towrite=bytes((self.currentbyte,)); self.output.write(towrite); self.currentbyte=0; self.numbitsfilled=0
	def close(self): while self.numbitsfilled!=0: self.write(0)

# --- PyTorch LSTM Model (Same as encoder) ---
class ExponentPredictorLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super().__init__(); self.hidden_dim = hidden_dim; self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1) # Keep LogSoftmax for consistency if loading weights
    def forward(self, x, hidden):
        embedded = self.embedding(x); lstm_out, hidden = self.lstm(embedded, hidden)
        last_out = lstm_out[:, -1, :]; output = self.fc(last_out)
        log_probs = self.log_softmax(output); return log_probs, hidden
    def init_hidden(self, device, batch_size=1):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Decode layer exponents row-by-row using batched LSTM prediction (GPU/MPS) "
                    "and sequential Arithmetic Decoding (CPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("compressed_file", help="Path to the input file containing compressed row data (.pkl).")
    parser.add_argument("--output_safetensor", type=str, default="decoded_exponents.safetensors",
                        help="Output file to save the reconstructed exponent tensor.")
    parser.add_argument("--device", default="auto", help="Device for LSTM computation ('cpu', 'cuda:0', 'mps', 'auto').")
    # Add arg to load model weights if encoder saved them separately
    # parser.add_argument("--model_weights", type=str, default=None, help="Path to saved LSTM model state_dict (optional).")

    args = parser.parse_args()

    # --- Device Setup ---
    if args.device == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        try:
            device = torch.device(args.device)
            # Check availability for non-CPU devices
            if device.type == 'cuda' and not torch.cuda.is_available():
                print(f"Warning: CUDA '{args.device}' requested but not available. Using CPU.")
                device = torch.device("cpu")
            elif device.type == 'mps' and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                 print(f"Warning: MPS '{args.device}' requested but not available/built. Using CPU.")
                 device = torch.device("cpu")
        except Exception as e:
            print(f"Error setting device '{args.device}': {e}. Using CPU.")
            device = torch.device("cpu")
    print(f"Using device for LSTM computation: {device}")

    # --- Load Compressed Data ---
    print(f"Loading compressed data from {args.compressed_file}...")
    try:
        with open(args.compressed_file, 'rb') as f_in:
            metadata = pickle.load(f_in)
            compressed_row_data = pickle.load(f_in)
        print("Compressed data loaded.")
        # Extract metadata
        layer = metadata.get('layer', 'Unknown')
        num_rows = metadata.get('num_rows', len(compressed_row_data)) # Infer if needed
        total_elements_original = metadata.get('total_elements', 'Unknown')
        ac_precision = metadata.get('ac_precision', 32)
        model_params = metadata.get('model_params', {})
        embedding_dim = model_params.get('embedding_dim', 64) # Provide defaults
        hidden_dim = model_params.get('hidden_dim', 256)
        num_layers = model_params.get('num_layers', 2)

        print(f"  Metadata: Layer={layer}, Num Rows={num_rows}, AC Precision={ac_precision}")
        print(f"  Model Params: Embed={embedding_dim}, Hidden={hidden_dim}, Layers={num_layers}")
        if not compressed_row_data:
             print("Error: Loaded compressed data dictionary is empty.")
             return
        # Infer sequence length per row if possible (assuming consistency)
        # Note: This assumes all rows had roughly equal length in the original tensor structure
        seq_len_per_row = -1
        if isinstance(total_elements_original, int) and total_elements_original > 0 and num_rows > 0:
             seq_len_per_row = total_elements_original // num_rows
             if total_elements_original % num_rows != 0:
                  print("Warning: Total elements not evenly divisible by num_rows. Inferred sequence length is approximate.")
             print(f"  Inferred sequence length per row: {seq_len_per_row}")
        else:
             print("Warning: Cannot determine exact sequence length per row from metadata.")
             # Need a fallback or user input if lengths vary significantly and aren't stored

    except FileNotFoundError: print(f"Error: Compressed file not found: {args.compressed_file}"); return
    except Exception as e: print(f"Error loading compressed data: {e}"); traceback.print_exc(); return

    # --- Setup Model ---
    print("\nSetting up LSTM model...")
    model = ExponentPredictorLSTM(
        vocab_size=NUM_EXPONENTS,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    model.eval() # Set to evaluation mode
    print(model)
    # Optional: Load weights if they were saved separately by the encoder
    # if args.model_weights: try: model.load_state_dict(torch.load(args.model_weights, map_location=device)); print("Loaded model weights.") except Exception as e: print(f"Error loading weights: {e}")

    # --- Initialize Decoders and States ---
    decoders = {}
    bitstreams = {}
    decoded_rows = {} # Store decoded sequences here
    active_rows = list(sorted(compressed_row_data.keys())) # Rows that have compressed data
    num_active_rows = len(active_rows)

    if num_active_rows == 0: print("Error: No active rows with compressed data found."); return
    if seq_len_per_row <= 0: print("Error: Invalid sequence length per row. Cannot proceed."); return

    print(f"\nInitializing {num_active_rows} decoders...")
    last_symbols = torch.zeros((num_active_rows, 1), dtype=torch.long).to(device)
    prob_scale = 1 << 16 # Must match encoder
    uniform_freq = np.cumsum(np.full(NUM_EXPONENTS, prob_scale // NUM_EXPONENTS + 1)).astype(np.uint64)

    for i, row_index in enumerate(active_rows):
        stream_bytes = compressed_row_data[row_index]
        inp = io.BytesIO(stream_bytes)
        bitin = BitInputStream(inp)
        decoder = ArithmeticDecoder(ac_precision, bitin)
        # Decode the first symbol uniformly
        try:
            first_symbol = decoder.read(uniform_freq)
            decoded_rows[row_index] = [first_symbol] # Start list for this row
            last_symbols[i, 0] = first_symbol # Set initial input for LSTM batch
            decoders[row_index] = decoder
            bitstreams[row_index] = bitin # Keep reference if needed for closing? Unlikely.
        except Exception as e:
            print(f"Error decoding first symbol for row {row_index}: {e}. Skipping row.")
            # Remove row from active processing if first symbol fails
            active_rows.pop(i)
            last_symbols = last_symbols[torch.arange(last_symbols.size(0)) != i] # Remove from batch input
            num_active_rows -= 1
            if num_active_rows == 0: print("Error: All rows failed on first symbol decode."); return


    if num_active_rows == 0: print("No rows successfully initialized."); return
    print(f"Successfully initialized {num_active_rows} rows.")

    # Initialize N independent hidden states (batch dimension is num_active_rows)
    hidden = model.init_hidden(device, batch_size=num_active_rows)


    # --- Main Decoding Loop ---
    print(f"Starting batched prediction / sequential decoding for {seq_len_per_row} steps...")
    start_decode_time = time.time()
    last_print_time = start_decode_time

    with torch.no_grad(): # No gradients needed
        for t in range(1, seq_len_per_row): # Loop for remaining symbols
            # --- 1. Predict in Batch (GPU/MPS) ---
            # Input 'last_symbols' has shape (num_active_rows, 1)
            log_probs_batch, hidden = model(last_symbols, hidden)
            # log_probs_batch shape: (num_active_rows, 16)
            probs_batch_np = torch.exp(log_probs_batch).cpu().numpy() # Move all probs to CPU

            next_symbols_list = []
            rows_to_remove_indices = [] # Indices within the *current* batch

            # --- 2. Decode Sequentially (CPU Bottleneck!) ---
            for i in range(num_active_rows):
                row_index = active_rows[i] # Get the original row index
                decoder = decoders[row_index]

                # Prepare frequency table for row i
                probs_np = np.maximum(probs_batch_np[i], 1e-9)
                probs_np /= np.sum(probs_np) # Re-normalize
                freq_table = np.cumsum(probs_np * prob_scale + 1).astype(np.uint64)

                # Decode symbol for row i using its specific decoder/stream
                try:
                    decoded_symbol = decoder.read(freq_table)
                    decoded_rows[row_index].append(decoded_symbol)
                    next_symbols_list.append(decoded_symbol)
                except (EOFError, ValueError, IndexError) as e:
                    # Handle expected decoding errors (e.g., end of stream, invalid state)
                    print(f"\nWarning: Decoding stopped for row {row_index} at step {t}: {e}")
                    # Mark row for removal from active processing
                    rows_to_remove_indices.append(i)
                    # Append a placeholder to maintain list length temporarily
                    next_symbols_list.append(0) # Value doesn't matter, will be removed
                except Exception as e:
                    # Handle unexpected errors
                    print(f"\nError: Unexpected error decoding row {row_index} at step {t}: {e}")
                    traceback.print_exc()
                    rows_to_remove_indices.append(i)
                    next_symbols_list.append(0)


            # --- 3. Update Batch for Next Iteration (Remove finished/errored rows) ---
            if rows_to_remove_indices:
                print(f"\nRemoving {len(rows_to_remove_indices)} finished/errored rows from batch...")
                # Create mask of rows to KEEP
                keep_mask = torch.ones(num_active_rows, dtype=torch.bool)
                for remove_idx in sorted(rows_to_remove_indices, reverse=True):
                    keep_mask[remove_idx] = False
                    # Remove from active_rows list *carefully* by original index
                    original_row_idx = active_rows.pop(remove_idx)
                    print(f"  Removed original row index: {original_row_idx}")
                    # Clean up decoder/stream if needed (optional)
                    if original_row_idx in decoders: del decoders[original_row_idx]
                    if original_row_idx in bitstreams: bitstreams[original_row_idx].close(); del bitstreams[original_row_idx]


                # Update batch size
                num_active_rows = len(active_rows)
                if num_active_rows == 0:
                    print("All rows finished or errored out.")
                    break # Exit main loop

                # Filter hidden state tuple
                hidden = tuple(h[:, keep_mask, :] for h in hidden)

                # Filter next_symbols_list before converting to tensor
                next_symbols_list = [sym for i, sym in enumerate(next_symbols_list) if keep_mask[i]]

            # Prepare next input batch tensor
            if not next_symbols_list:
                 print("No symbols decoded in this step, ending.")
                 break
            last_symbols = torch.tensor(next_symbols_list, dtype=torch.long).unsqueeze(1).to(device)

            # --- Progress Reporting ---
            current_time = time.time()
            if current_time - last_print_time > 10 or t == seq_len_per_row - 1:
                 percentage = (t + 1) / seq_len_per_row * 100
                 elapsed = current_time - start_decode_time
                 print(f"  Processed Step: {t+1}/{seq_len_per_row} ({percentage:.1f}%) | Active Rows: {num_active_rows} | Time: {elapsed:.1f}s", end='\r')
                 last_print_time = current_time

    end_decode_time = time.time()
    print(f"\nDecoding loop finished in {end_decode_time - start_decode_time:.2f} seconds.")

    # --- Assemble Final Matrix (Optional) ---
    print("\nAssembling decoded rows into final tensor...")
    # Check if all rows have the expected length
    final_rows = []
    max_len_found = 0
    incomplete_rows = 0
    expected_len = seq_len_per_row # Use inferred length
    for r in range(num_rows): # Iterate up to the max original row index
        if r in decoded_rows:
            row_data = decoded_rows[r]
            max_len_found = max(max_len_found, len(row_data))
            if len(row_data) < expected_len:
                incomplete_rows += 1
                # Pad incomplete rows if necessary for stacking
                row_data.extend([0] * (expected_len - len(row_data))) # Pad with 0
            final_rows.append(row_data[:expected_len]) # Truncate just in case
        else:
            # Row was missing entirely or failed at step 0
            print(f"Warning: Row {r} missing from decoded data. Padding with zeros.")
            final_rows.append([0] * expected_len)
            incomplete_rows += 1

    if incomplete_rows > 0:
        print(f"Warning: {incomplete_rows} rows were incomplete or missing, padded with zeros.")
    if max_len_found > expected_len:
         print(f"Warning: Found row with length {max_len_found}, expected {expected_len}. Truncating.")


    try:
        # Stack rows into a single tensor (on CPU)
        decoded_tensor_np = np.array(final_rows, dtype=np.uint8) # Result is uint8 exponents
        decoded_tensor = torch.from_numpy(decoded_tensor_np)
        print(f"Reconstructed exponent tensor shape: {decoded_tensor.shape}")

        # --- Save Reconstructed Tensor ---
        if args.output_safetensor:
             print(f"Saving reconstructed exponent tensor to {args.output_safetensor}...")
             # Create a dictionary for safetensors saving
             # Use a placeholder key name
             tensor_dict = {"reconstructed_exponents": decoded_tensor}
             try:
                  safetensors.torch.save_file(tensor_dict, args.output_safetensor)
                  print("Output saved successfully.")
             except Exception as e:
                  print(f"Error saving output safetensor: {e}")
                  traceback.print_exc()

    except Exception as e:
        print(f"Error assembling or saving final tensor: {e}")
        traceback.print_exc()


    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
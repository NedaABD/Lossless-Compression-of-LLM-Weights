# lossless-compression-of-llm-weights-using-rans-and-lstm

## Overview
Large Language Models (LLMs) have grown to hundreds of billions of parameters, making storage, memory bandwidth, and deployment increasingly expensive. This repository implements a **lossless compression framework for LLM weights**, focusing on the **FP8 (E4M3) exponent bits**, which exhibit a highly skewed distribution and are therefore significantly more compressible than mantissa or sign bits.

The approach combines **online predictive modeling using a two-layer LSTM** with **arithmetic entropy coding via Range Asymmetric Numeral Systems (rANS)**, enabling substantial compression while guaranteeing exact reconstruction of the original values.

---

## Features
- Lossless compression of FP8 (E4M3) exponent bits  
- Online LSTM-based probability prediction  
- rANS arithmetic encoding with exact decoding  
- Row-wise independent compression streams  
- Shared adaptive LSTM per layer or layer-pair  
- Hybrid decoding (GPU/MPS for prediction, CPU for arithmetic decoding)  
- Verified bit-exact reconstruction  

---

## Requirements
- Python 3.9+
- PyTorch (with FP8 E4M3 support)
- NumPy
- safetensors
- pickle

Optional:
- CUDA or Apple Silicon (MPS) for acceleration

---

## Dataset
This project does **not** use a traditional dataset.

- Input consists of **model weight tensors**
- Target architecture: **DeepSeek-R1**
- Weight format: **FP8 (E4M3)**
- Focus on Mixture-of-Experts (MoE) layers
- Approximately **93% of parameters** reside in expert layers

Weights are loaded directly from `.safetensors` files.

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/lossless-compression-of-llm-weights-using-rans-and-lstm.git
cd lossless-compression-of-llm-weights-using-rans-and-lstm
pip install torch numpy safetensors
```

Ensure your PyTorch installation supports FP8 (`torch.float8_e4m3fn`).

---

## Usage

### Encoding (Compression)
```bash
python rans_encode_exponent.py <model_dir> --layer <layer_id> --output_file compressed_layer.pkl
```

This step:
- Parses `model.safetensors.index.json`
- Locates expert tensors (gate, up, down projections)
- Loads FP8 tensors using `safetensors.torch`
- Extracts 4-bit exponent values
- Builds row-wise exponent sequences
- Trains a two-layer LSTM online
- Compresses sequences using rANS
- Writes compressed data and metadata to a `.pkl` file

---

### Decoding (Reconstruction)
```bash
python rans_decode_exponent.py compressed_layer.pkl --output_safetensor decoded_exponents.safetensors
```

This step:
- Loads compressed exponent streams and metadata
- Reinitializes the LSTM with the same architecture
- Decodes exponent sequences exactly using rANS
- Reconstructs exponent tensors
- Saves reconstructed data to `.safetensors`

---

## Code Walkthrough

### Data Loading and Preprocessing
- Parses `model.safetensors.index.json`
- Locates expert tensors:
  - Gate projection
  - Up projection
  - Down projection
- Loads tensors using `safetensors.torch`
- Extracts FP8 exponent bits via bitwise masking and shifting
- Constructs row-wise exponent sequences across experts

---

### LSTM Model Integration
- Two-layer LSTM implemented in PyTorch
- Embedding layer followed by stacked LSTM layers
- LogSoftmax output for probability prediction
- Online training using NLLLoss and Adam optimizer
- BF16 precision to reduce memory overhead
- One shared LSTM per layer or per layer-pair

---

### User Input Processing
- User specifies target layer(s)
- Relevant expert tensors are resolved automatically
- Sequence lengths inferred from tensor shapes
- Metadata stored for decoding consistency

---

### Similarity Calculation
Not applicable.  
Compression is driven by **entropy modeling**, not semantic similarity.

---

### Recommendation
Not applicable.  
This repository implements a **systems-level compression framework**, not a recommendation system.

---

## Example Input and Output

### Input
- FP8 expert weight tensors  
- Example shapes:
  - Up projection: `(2048 × 7160)`
  - Down projection: `(7160 × 2048)`
- Total FP8 expert weight size: ~220 GB
- Exponent component size: ~110 GB

---

### Output
- Compressed exponent data: ~42 GB
- Compression ratio for exponents: ~61.8%
- Average bits per exponent: ~1.53 bits
- Output files:
  - Compressed `.pkl` streams
  - Reconstructed `.safetensors` exponent tensors

---

## Customization
- Modify LSTM hidden size or embedding dimension
- Adjust arithmetic coder precision
- Switch between CPU, CUDA, or MPS backends
- Extend compression to additional layers or layer groups

---

## Limitations
- Mantissa and sign bits are not compressed (near-uniform distribution)
- Python-based rANS implementation introduces runtime overhead
- End-to-end inference integration is not included

---

## Future Enhancements
- Lossy or hybrid compression for mantissa bits
- C++ / CUDA acceleration for rANS
- Transformer-based predictive models
- End-to-end integration with LLM inference pipelines
- Cross-layer statistical modeling

---

## Acknowledgments
This project was developed at **Syracuse University**.

**Authors:**
- Neda Abdolrahimi  
- Naveen Ashok  

**Advisor:**
- Prof. C. K. Mohan

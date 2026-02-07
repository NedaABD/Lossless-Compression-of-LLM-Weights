# Lossless Compression of LLM Weights Using LSTM and rANS

## Overview
Large Language Models (LLMs) have grown to hundreds of billions of parameters, making storage, memory bandwidth, and deployment increasingly expensive. This repository implements a **lossless compression pipeline for LLM weights**, focusing on the **FP8 (E4M3) exponent bits**, which exhibit a highly skewed distribution and are therefore significantly more compressible than mantissa or sign bits.

The core approach combines:
- **Online predictive modeling** using a two-layer LSTM
- **Arithmetic entropy coding** using Range Asymmetric Numeral Systems (rANS)

The method enables substantial storage reduction while guaranteeing **exact reconstruction** of the original weights.

---

## Key Results
- FP8 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
- Target model: DeepSeek-R1 (671B parameters)
- Exponent-only compression:
  - ~110 GB original exponent data
  - ~42 GB compressed
  - ~61.8% reduction
  - ~1.53 bits per exponent (down from 4 bits)
- Compression is fully lossless for exponent bits

---

## Repository Structure
- `rans_encode_exponent.py`  
  Encodes FP8 exponent sequences using an online LSTM and rANS arithmetic coding.

- `rans_decode_exponent.py`  
  Decodes compressed exponent streams and reconstructs exponent tensors exactly.

- `mac_nanogpt.py`  
  Auxiliary NanoGPT training script adapted for Apple Silicon (not required for compression).

- `ANN_Final_Report.pdf`  
  Full technical report describing methodology, experiments, and results.

- `Neda&Naveen_FinalProjectSlides.pdf`  
  Project presentation slides.

---

## Requirements
- Python 3.9+
- PyTorch (with FP8 E4M3 support)
- NumPy
- safetensors

---

## Usage

### Encoding (Compression)
```bash
python rans_encode_exponent.py <model_dir> --layer <layer_id> --output_file compressed_layer.pkl
```

This step:
- Parses `model.safetensors.index.json`
- Locates MoE expert tensors (gate, up, down projections)
- Loads FP8 tensors using `safetensors.torch`
- Extracts 4-bit exponent values
- Builds row-wise exponent sequences across experts
- Trains a two-layer LSTM online to predict exponent probabilities
- Compresses sequences using rANS arithmetic coding
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
- Saves reconstructed exponents to `.safetensors`

---

## Code Walkthrough

### Data Loading and Preprocessing
- Parses `model.safetensors.index.json`
- Resolves expert tensors:
  - Gate projection
  - Up projection
  - Down projection
- Loads tensors using `safetensors.torch`
- Extracts FP8 exponent bits via bitwise masking and shifting
- Constructs row-wise exponent sequences across all experts

---

### LSTM Model Integration
- Two-layer LSTM implemented in PyTorch
- Embedding layer followed by stacked LSTM layers
- LogSoftmax output for probability prediction
- Online training using NLLLoss and Adam optimizer
- BF16 precision used to reduce memory overhead
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
Compression is driven by entropy modeling, not semantic similarity.

---

### Recommendation
Not applicable.  
This repository implements a systems-level compression framework, not a recommendation system.

---

## Example Input and Output

### Input
- FP8 expert weight tensors
- Example shapes:
  - Up projection: (2048 × 7160)
  - Down projection: (7160 × 2048)
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
This project was developed at Syracuse University.

Authors:
- Neda Abdolrahimi
- Naveen Ashok

Advisor:
- Prof. C. K. Mohan


# lossless-compression-of-llm-weights-using-rans-and-lstm

## Overview
Large Language Models (LLMs) have grown to hundreds of billions of parameters, making storage, memory bandwidth, and deployment increasingly expensive.  
This repository contains an implementation of a **lossless compression pipeline for LLM weights**, focusing on the **FP8 (E4M3) exponent bits** of Mixture-of-Experts (MoE) layers.

The project combines **online predictive modeling using a two-layer LSTM** with **Range Asymmetric Numeral Systems (rANS)** arithmetic encoding to significantly reduce storage requirements **without any loss in model fidelity**.

This work was evaluated on the **DeepSeek-R1 (671B parameters)** architecture and demonstrates substantial compression gains for expert-layer weights.

---

## Features
- Lossless compression of FP8 (E4M3) exponent bits
- Online LSTM-based probability prediction
- rANS arithmetic encoding with exact reconstruction
- Row-wise independent compression streams
- Shared adaptive LSTM per layer or layer-pair
- Hybrid decoding: GPU/MPS for prediction, CPU for arithmetic decoding
- Verified bit-exact recovery of original exponents

---

## Requirements
- Python 3.9+
- PyTorch
- NumPy
- safetensors
- pickle
- Apple Silicon (MPS) or CUDA (optional, for acceleration)

---

## Dataset
This project does **not** use a conventional dataset.

- Input data consists of **model weight tensors**
- Target model: **DeepSeek-R1**
- Weight format: **FP8 (E4M3)**
- Focus: Mixture-of-Experts (MoE) layers
- Approximately **93% of parameters** reside in expert layers

The weights are loaded directly from `.safetensors` files.

---

## Installation
```bash
git clone https://github.com/your-username/lossless-compression-of-llm-weights-using-rans-and-lstm.git
cd lossless-compression-of-llm-weights-using-rans-and-lstm
pip install -r requirements.txt

---

## Usage

### Encoding (Compression)
```bash
python rans_encode_exponent.py --layer <layer_id> --model-path <path_to_model>
```

This step:
- Loads FP8 expert weights
- Extracts exponent bits
- Trains an LSTM online
- Compresses exponent sequences using rANS
- Outputs a compressed `.pkl` file containing metadata and encoded streams

---

### Decoding (Reconstruction)
```bash
python rans_decode_exponent.py --input compressed_layer.pkl
```

This step:
- Reconstructs exponent sequences exactly
- Reassembles exponent tensors
- Writes reconstructed data to `.safetensors`

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
- Constructs row-wise exponent sequences across all experts

---

### LSTM Model Integration
- Two-layer LSTM implemented in PyTorch
- Embedding layer + stacked LSTM layers
- LogSoftmax output for probability prediction
- Online training using `NLLLoss` and Adam optimizer
- BF16 precision used to reduce memory overhead
- Shared LSTM per layer or per layer-pair

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



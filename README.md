# Lossless-Compression-of-LLM-Weights
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


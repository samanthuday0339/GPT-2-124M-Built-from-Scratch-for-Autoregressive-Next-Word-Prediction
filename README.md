# GPT-2-124M-Built-from-Scratch-for-Autoregressive-Next-Word-Prediction

A complete **GPT-2 (124M) language model implementation built from first principles using PyTorch**.  
This project focuses on **understanding and implementing the internals of GPT-style Transformers**, including architecture design, training workflow, text generation strategies, and loading official pretrained weights.

---

## 📌 Overview

This repository implements a **decoder-only Transformer (GPT-2 style)** without relying on high-level abstractions.  
Every core component—attention, normalization, residuals, and training loop—is written explicitly for clarity and learning.

**What this project demonstrates:**
- How GPT-2 works internally
- How causal self-attention is implemented
- How large language models are trained
- How pretrained GPT-2 weights can be loaded into a custom model

---

## 🧠 Model Architecture

The model follows the original GPT-2 design:

- Token Embeddings
- Positional Embeddings
- Stacked Transformer Blocks
- Causal Multi-Head Self-Attention
- Feed-Forward Networks
- Layer Normalization
- Residual Connections
- Linear Output Head for next-token prediction

Causal masking ensures the model **never attends to future tokens**, making it autoregressive.

---

## 🧩 Core Components

| Component | Description |
|--------|------------|
Token Embedding | Converts token IDs into dense vectors  
Positional Embedding | Adds sequence order information  
Multi-Head Attention | Parallel attention over multiple heads  
Causal Mask | Prevents information leakage from future tokens  
Feed-Forward Network | Non-linear transformation block  
Layer Normalization | Stabilizes training  
Residual Connections | Improves gradient flow  

---


## 🚀 Features

- ✅ GPT-2 architecture built **from scratch**
- ✅ Custom implementation of LayerNorm and GELU
- ✅ Causal Multi-Head Self-Attention
- ✅ Full training loop with validation
- ✅ Greedy, temperature, and top-k text generation
- ✅ Load official GPT-2 (124M) pretrained weights
- ✅ Modular and educational codebase

---

## 🏋️ Training Pipeline

The model is trained using **next-token prediction**:

- Input: sequence of tokens  
- Target: same sequence shifted by one position  
- Loss: Cross-Entropy over vocabulary  
- Optimizer: AdamW  

Training includes:
- Dataset chunking with sliding windows
- Periodic evaluation on validation data
- Sample text generation during training

---

## ✍️ Text Generation

The project supports multiple decoding strategies:

- **Greedy Decoding** – deterministic next-token selection  
- **Temperature Sampling** – controls randomness  
- **Top-K Sampling** – limits token choices to top probabilities  

These methods significantly affect creativity and coherence.

---

## 📈 Performance Notes

- Training loss typically decreases from ~10 → ~0.5 on small datasets
- Validation loss remains higher due to limited data
- Perplexity is computed as `exp(loss)`
- Pretrained weights dramatically improve generation quality

---

## 🐞 Troubleshooting

| Issue | Solution |
|-----|---------|
Out of memory | Reduce batch size or context length  
Loss diverges | Lower learning rate  
Poor text quality | Train longer or use pretrained weights  
NaN loss | Verify attention mask and normalization  

---

## 🎓 Who Is This For?

- Students learning **Transformers & LLMs**
- ML engineers preparing for **GenAI / NLP interviews**
- Anyone who wants to understand **GPT internals**
- Researchers experimenting with language models

---

## 📚 References

- *Attention Is All You Need* – Vaswani et al.
- *Language Models are Unsupervised Multitask Learners* (GPT-2 Paper)
- *The Illustrated Transformer* – Jay Alammar

---


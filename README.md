# Transformer-Language-Models

**Implementation of a Decoder-Only Transformer (from scratch, in PyTorch)**  

---

# Overview
This repo contains a complete implementation of a Transformer decoder LM:
- Multi‑head causal attention  
- Transformer decoder stack  
- Character-level LM  
- Sampling & analysis too

---

# Installation

```bash
pip install torch numpy
```

Optional for visualization:

```bash
pip install matplotlib seaborn
```

---

# Part 1 — Attention Mechanisms

Implemented in `attention.py`, covering:
- KQV parameter creation  
- Attention score computation  
- Causal mask  
- Multi‑head attention  

---

# Part 2 — Transformer Decoder

In `transformer.py`:
- Token + position embeddings  
- Residual connections  
- LayerNorm  
- Feed‑forward MLP  
- Final LM head  

---

# Part 3 — Language Model

In `lm.py` + `main.py`:
- Create labeled sequences  
- Compute LM loss  
- Full training loop  
- Print loss & sample text during training  

Run training:

```bash
python main.py
```

Switch to Hebrew corpus:

```bash
python main.py --data_dir heb-data/
```

---

# Part 4 — Better Sampling

Top‑k sampling + temperature control implemented in  
`TransformerLM.better_sample_continuation`.

Example:

```python
model.better_sample_continuation(idx, top_k=5, temperature=0.8)
```

---

# Part 5 — Attention Analysis

Explore attention matrices for each layer/head and identify interpretable patterns.

You may visualize attention using:

```python
sns.heatmap(att_matrix)
``` 

---

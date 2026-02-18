# LLM Ablation Tutorial: Building a Language Model from Scratch

A hands-on series of 6 Jupyter notebooks that build a GPT-style language model **from absolute scratch**, one component at a time. Each notebook adds a single architectural element, trains it on Shakespeare, and measures the improvement — creating a **reverse ablation study** that shows exactly what each piece contributes.

## The Notebooks

| # | Notebook | What We Build | Key Concepts |
|---|----------|--------------|-------------|
| **01** | [BPE Tokenizer](notebooks/01_bpe_tokenizer.ipynb) | Byte Pair Encoding from scratch | Pre-tokenization, merge rules, encode/decode, compression ratio |
| **02** | [Bigram Baseline](notebooks/02_bigram_baseline.ipynb) | Context-free language model | Embedding lookup, cross-entropy loss, why context matters |
| **03** | [Self-Attention](notebooks/03_self_attention.ipynb) | Single-head and multi-head attention | Q/K/V projections, causal masking, attention patterns, multi-head parallelism |
| **04** | [Transformer Block](notebooks/04_transformer_block.ipynb) | Complete transformer block | Feed-forward networks, LayerNorm, residual connections, stacking depth |
| **05** | [GPT Architecture](notebooks/05_gpt_model.ipynb) | Full GPT model | Weight tying, scaled residual init, gradient health, parameter breakdown |
| **06** | [Training & Scaling](notebooks/06_training_and_scaling.ipynb) | Training loop + scaling | LR warmup + cosine decay, AdamW, temperature/top-k sampling, scaling laws |

## The Ablation Story

Each notebook trains its model on Tiny Shakespeare and records validation loss. By the end, we can compare every variant side-by-side:

```
Model                          Params     Val Loss
─────────────────────────────────────────────────────
Bigram (no context)                 33K      4.00
+ Single-Head Attention             53K      3.59
+ Multi-Head Attention (4 heads)    86K      3.38
+ Feed-Forward Network             120K      3.29
+ LayerNorm                        120K      3.27
+ Residual Connections             120K      3.18
+ Depth (4 blocks)                 269K      3.06
+ Full GPT Training (tiny)         873K      3.00
+ Scale (GPT-Small)              6,507K      2.70
```

Self-attention is the single biggest jump (context is everything), but every component plays a measurable role.

## Visualizations

Every notebook is packed with matplotlib visualizations:

- **Attention heatmaps** showing which tokens attend to which
- **Embedding space PCA** tracking how token representations evolve layer-by-layer
- **Training curves** with smoothed loss, train/val comparison, and LR schedules
- **Architecture diagrams** built programmatically
- **Gradient health checks** verifying training stability
- **Temperature and top-k** probability distribution comparisons
- **Scaling law plots** showing the parameters-vs-loss relationship

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.x
- matplotlib, numpy

### Install

```bash
conda create -n llm-tutorial python=3.10
conda activate llm-tutorial
pip install torch numpy matplotlib
```

### Run

Open the notebooks in order:

```bash
jupyter notebook notebooks/01_bpe_tokenizer.ipynb
```

Each notebook builds on the previous one. The data (Tiny Shakespeare) and tokenizer are included in `data/`.

GPU is auto-detected (CUDA or Apple MPS) and used when available. All notebooks run fine on CPU — the tiny and small models train in minutes on a modern laptop.

## Project Structure

```
├── data/
│   ├── tinyshakespeare.txt          # Training corpus (~1.1M characters)
│   ├── bpe_tokenizer.json           # Pre-trained BPE tokenizer (512 vocab)
│   └── ablation_metrics.json        # Validation loss for all model variants
├── notebooks/
│   ├── 01_bpe_tokenizer.ipynb
│   ├── 02_bigram_baseline.ipynb
│   ├── 03_self_attention.ipynb
│   ├── 04_transformer_block.ipynb
│   ├── 05_gpt_model.ipynb
│   └── 06_training_and_scaling.ipynb
└── src/llm/                         # Shared module (optional)
```

## Key Design Decisions

- **Self-contained notebooks**: Each notebook includes all code it needs (tokenizer, model classes) so they can be read independently
- **CPU-friendly**: Models are small enough to train on CPU in minutes. GPU accelerates but isn't required
- **Print-heavy**: Every cell prints explanatory output — the notebooks are designed to be read top-to-bottom as a tutorial, not just executed
- **Reverse ablation**: Instead of starting with a full model and removing parts, we build up incrementally so each addition's impact is clear

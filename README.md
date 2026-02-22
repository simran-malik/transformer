# Transformer Architecture

Uses speech data for classification and language modeling tasks.

## Directory Structure

```
code/
├── main.py           # Main entry point for Parts 1, 2, and 3
├── main_sparse.py    # Standalone script for sparse attention (Part 3)
├── transformer.py    # Transformer components: Encoder, Decoder, Classifier
├── sparse_attention.py # Sparse/local attention encoder
├── tokenizer.py      # Word-based tokenizer
├── dataset.py        # Classification and language modeling datasets
├── utilities.py      # Attention visualization and sanity checks
└── speechesdataset/  # Data directory
    ├── train_CLS.tsv       # Classification training (label, text)
    ├── test_CLS.tsv        # Classification test
    ├── train_LM.txt        # Language modeling training
    ├── test_LM_obama.txt
    ├── test_LM_wbush.txt
    └── test_LM_hbush.txt
```

## Dependencies

- Python 3
- PyTorch
- NLTK (for `word_tokenize`)
- matplotlib

Install NLTK data if needed:

```bash
python -c "import nltk; nltk.download('punkt')"
```

## Usage

Run from the `code/` directory. Data is expected in `speechesdataset/`.

### Part 1: Classification

Train an encoder + classifier to predict speaker (3 classes) from speech text.

```bash
python main.py --task part1
```

Produces attention sanity-check plots in `results/encoder/`.

### Part 2: Language Modeling

Train a decoder for language modeling. Reports perplexity on train and on Obama, W. Bush, and H. Bush test sets.

```bash
python main.py --task part2
```

Produces attention sanity-check plots in `results/decoder/`.

### Part 3: Sparse Attention

Train a classification model with local window attention instead of full attention.

```bash
python main.py --task part3
```

Or run the sparse script directly:

```bash
python main_sparse.py
```

Produces attention plots in `results/sparse_encoder/`.

## Module Overview

| File | Purpose |
|------|---------|
| `transformer.py` | `Encoder`, `Decoder`, `Classifier`, full and causal multi-head attention |
| `sparse_attention.py` | `Encoder_Sparse` with local window attention |
| `tokenizer.py` | `SimpleTokenizer` (word-level, NLTK-based) |
| `dataset.py` | `SpeechesClassificationDataset`, `LanguageModelingDataset` |
| `utilities.py` | `Utilities` for attention visualization and sanity checks |

## Hyperparameters (from `main.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 16 | Batch size |
| `block_size` | 32 | Max context length |
| `learning_rate` | 1e-3 | Adam learning rate |
| `n_embd` | 64 | Embedding dimension |
| `n_head` | 2 | Number of attention heads |
| `n_layer` | 4 | Number of transformer layers |
| `epochs_CLS` | 15 | Epochs for classification |
| `max_iters` | 500 | Max LM training iterations |

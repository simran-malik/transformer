# Transformer Architecture

Uses presidential speech data for classification (3-way speaker prediction) and language modeling tasks.

## Directory Structure

```
├── main.py             # Main entry point: classify, generate, sparse
├── transformer.py     # Encoder, Decoder, Classifier; full and causal multi-head attention
├── sparse_attention.py # SparseEncoder with local window attention
├── tokenizer.py       # Word-based tokenizer (NLTK)
├── dataset.py         # SpeechesClassificationDataset, LanguageModelingDataset
├── utilities.py       # Attention visualization and sanity checks
└── speechesdataset/   # Data directory
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

Data is expected in `speechesdataset/`. Results are expected in `results/` .

### Classification (`classify`)

Train an encoder + classifier to predict speaker (3 classes) from speech text.

```bash
python main.py --task classify
```

Produces attention sanity-check plots in `results/encoder/`.

### Language Modeling (`generate`)

Train a decoder for language modeling. Reports perplexity on Obama, W. Bush, and H. Bush test sets.

```bash
python main.py --task generate
```

Produces attention sanity-check plots in `results/decoder/`.

### Sparse Attention (`sparse`)

Train a classification model with `SparseEncoder` (local window attention instead of full attention).

```bash
python main.py --task sparse
```

Produces attention plots in `results/encoder/` (same layout as classify; encoder is sparse).

## Module Overview


| File                  | Purpose                                                                  |
| --------------------- | ------------------------------------------------------------------------ |
| `transformer.py`      | `Encoder`, `Decoder`, `Classifier`; full and causal multi-head attention |
| `sparse_attention.py` | `SparseEncoder`, `SparseMultiHeadAttention` with local window attention  |
| `tokenizer.py`        | `SimpleTokenizer` (word-level, NLTK-based)                               |
| `dataset.py`          | `SpeechesClassificationDataset`, `LanguageModelingDataset`               |
| `utilities.py`        | `Utilities` for attention visualization and sanity checks                |


## Hyperparameters (from `main.py`)


| Parameter       | Value | Description                            |
| --------------- | ----- | -------------------------------------- |
| `batch_size`    | 16    | Batch size                             |
| `block_size`    | 32    | Max context length                     |
| `learning_rate` | 1e-3  | Adam learning rate                     |
| `n_embd`        | 64    | Embedding dimension                    |
| `n_heads`       | 2     | Number of attention heads              |
| `n_layers`      | 4     | Number of transformer layers           |
| `epochs`        | 15    | Epochs for classification              |
| `max_epochs`    | 500   | Max LM training iterations             |
| `eval_epochs`   | 100   | Evaluate perplexity every N iterations |



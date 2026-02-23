import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import os
import torch.optim as optim
import torch.nn as nn
import argparse

from sparse_attention import SparseEncoder
from transformer import Classifier, Decoder, Encoder
from utilities import Utilities
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

seed = 42

def train_classifier(EncoderClass, device, **encoder_kwargs):
    """
    Train an encoder + classifier for speech-to-president classification (3 classes).
    Supports both standard Encoder and SparseEncoder via the EncoderClass parameter.

    Args:
        EncoderClass: The encoder class to instantiate (Encoder or SparseEncoder).
        device: torch device (cuda or cpu) for training.
        **encoder_kwargs: Optional keyword args passed to the encoder constructor (e.g. sparse_type, window_size).
    """
    # Hyperparameters for encoder initialization
    batch_size = 16  # Number of independent sequences we will process in parallel
    block_size = 32  # Maximum context length for predictions
    n_embd = 64  # Embedding dimension
    n_heads = 2  # Number of attention heads
    n_layers = 4  # Number of transformer layers

    # Hyperparameters for classifier initialization
    n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
    n_hidden = 100  # Hidden size for the classifier
    n_output = 3  # Output size for the classifier, we have 3 classes

    # Hyperparameters for encoder-classifier training
    learning_rate = 1e-3  # Learning rate for the optimizer
    epochs = 15 # epochs for classifier training

    tokenizer = load_tokenizer("speechesdataset")

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=partial(collate_batch, block_size=block_size), shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=partial(collate_batch, block_size=block_size), shuffle=True)
    
    encoder = EncoderClass(tokenizer.vocab_size, block_size, n_embd, n_layers, n_heads,**encoder_kwargs)
    classifier = Classifier(n_input, n_hidden, n_output)
    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in classifier.parameters())
    print(f"Total parameters of encoder + classifier: {total_params}")
    
    run_sanity_check(block_size, tokenizer, encoder, "results/encoder", "encoder_pre_training")

    encoder, classifier = run_encoder_training(
        encoder=encoder, 
        classifier=classifier, 
        optimizer=optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()), 
            lr=learning_rate
        ), 
        loss_function=nn.CrossEntropyLoss(), 
        data_loader=train_CLS_loader,
        device=device,
        epochs=epochs)

    print(f"Final Test Accuracy: {compute_classifier_accuracy(encoder, classifier, test_CLS_loader, device):.2f}%")

    run_sanity_check(block_size, tokenizer, encoder, "results/encoder", "encoder_post_training")

def train_decoder(device):
    """
    Train a decoder for language modeling on presidential speeches.
    Evaluates perplexity on Obama, W. Bush, and H. Bush test sets.

    Args:
        device: torch device (cuda or cpu) for training.
    """
    # Hyperparameters for decoder initialization
    batch_size = 16  # Number of independent sequences we will process in parallel
    block_size = 32  # Maximum context length for predictions
    n_embd = 64  # Embedding dimension
    n_heads = 2  # Number of attention heads
    n_layers = 4  # Number of transformer layers

    # Hyperparameters for decoder training
    learning_rate = 1e-3  # Learning rate for the optimizer
    eval_epochs = 100  # How often to evaluate train and test perplexity during training
    max_epochs = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.

    train_text, obama_test_text, wbush_test_text, hbush_test_text = load_individual_texts(
        "speechesdataset/train_LM.txt", 
        "speechesdataset/test_LM_obama.txt", 
        "speechesdataset/test_LM_wbush.txt", 
        "speechesdataset/test_LM_hbush.txt"
    )

    tokenizer = load_tokenizer("speechesdataset")
    train_dataset = LanguageModelingDataset(tokenizer, train_text,  block_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_obama_dataset = LanguageModelingDataset(tokenizer, obama_test_text,  block_size)
    test_obama_loader = DataLoader(test_obama_dataset, batch_size=batch_size, shuffle=True)
    test_wbush_dataset = LanguageModelingDataset(tokenizer, wbush_test_text,  block_size)
    test_wbush_loader = DataLoader(test_wbush_dataset, batch_size=batch_size, shuffle=True)
    test_hbush_dataset = LanguageModelingDataset(tokenizer, hbush_test_text,  block_size)
    test_hbush_loader = DataLoader(test_hbush_dataset, batch_size=batch_size, shuffle=True)

    decoder = Decoder(tokenizer.vocab_size, block_size, n_embd, n_layers, n_heads)
    print(f"Total parameters of Decoder: {sum(p.numel() for p in decoder.parameters())}")

    run_sanity_check(block_size, tokenizer, decoder, "results/decoder", "decoder_pre_training")

    decoder = run_decoder_training(
        decoder=decoder, 
        optimizer=optim.Adam(decoder.parameters(), lr=learning_rate), 
        train_data_loader=train_loader, 
        device=device,
        max_epochs=max_epochs,
        eval_epochs=eval_epochs)

    obama_perplexity = compute_perplexity(decoder, test_obama_loader, device, eval_epochs)
    wbush_perplexity = compute_perplexity(decoder, test_wbush_loader, device, eval_epochs)
    hbush_perplexity = compute_perplexity(decoder, test_hbush_loader, device, eval_epochs)

    print(f"Obama: Test Perplexity: {obama_perplexity:.2f}")
    print(f"W Bush: Test Perplexity: {wbush_perplexity:.2f}")
    print(f"H Bush: Test Perplexity: {hbush_perplexity:.2f}")

    run_sanity_check(block_size, tokenizer, decoder, "results/decoder", "decoder_post_training")

def run_encoder_training(encoder, classifier, optimizer, loss_function, data_loader, device, epochs=15):
    """
    Run training loop for encoder + classifier. Updates both models in place via backpropagation.

    Args:
        encoder: Encoder model (transformer that outputs sequence embeddings).
        classifier: Classification head (predicts 3-way politician label from encoder output).
        optimizer: Optimizer (e.g. Adam) for encoder and classifier parameters.
        loss_function: Loss fn (e.g. CrossEntropyLoss) for classification.
        data_loader: DataLoader yielding (input_ids, labels) batches.
        device: torch device for tensors.
        epochs: Number of training epochs (default 15).

    Returns:
        Tuple of (encoder, classifier) after training.
    """
    encoder.to(device)
    classifier.to(device)

    for epoch in range(epochs):
        encoder.train()
        classifier.train()
        total_loss = 0

        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)

            # FORWARD PASS
            # Get token embeddings from encoder
            encoder_output, _ = encoder(xb) # Shape: (batch_size, block_size, n_embd)
            # Average across the sequence dimension
            classifier_input = encoder_output.mean(dim=1) # Shape: (batch_size, n_embd)
            # Predict politician
            classifier_output = classifier(classifier_input) # Shape: (batch_size, n_output=3)

            # LOSS CALCULATION
            loss = loss_function(classifier_output, yb)

            # BACKPROPAGATION
            optimizer.zero_grad() # Clear previous gradients
            loss.backward() # Compute gradients via chain rule
            optimizer.step() # Update all weights in encoder & classifier

            # LOGGING METRICS
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        accuracy = compute_classifier_accuracy(encoder, classifier, data_loader, device)

        print(f"Epoch {epoch+1:02d} | Epoch Loss: {avg_loss:.4f} | Epoch Acc: {accuracy:.2f}%")

    return encoder, classifier

def run_decoder_training(decoder, optimizer, train_data_loader, device, max_epochs=500, eval_epochs=100):
    """
    Run training loop for the language-modeling decoder. Uses cross-entropy loss internally.

    Args:
        decoder: Decoder model (causal transformer for next-token prediction).
        optimizer: Optimizer (e.g. Adam) for decoder parameters.
        train_data_loader: DataLoader yielding (input_ids, target_ids) batches.
        device: torch device for tensors.
        max_epochs: Max number of training iterations/batches (default 500).
        eval_epochs: Evaluate and log perplexity every N iterations (default 100).

    Returns:
        Trained decoder model.
    """
    decoder.to(device)
    decoder.train()
    for i, (xb, yb) in enumerate(train_data_loader):
        if i >= max_epochs:
            break

        xb, yb = xb.to(device), yb.to(device)

        # FORWARD PASS and LOSS CALCULATION
        _, loss = decoder(xb, yb)

        # BACKPROPAGATION
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Periodically evaluate perplexity
        if i == max_epochs - 1 or (i + 1) % eval_epochs == 0:
            train_perplexity = compute_perplexity(decoder, train_data_loader, device, eval_epochs)
            print(f"Iteration {i + 1:03d} | Train Loss: {loss.item():.4f} | Perplexity: {train_perplexity:.2f}")

    return decoder

def load_tokenizer(data_directory):
    """
    Build a SimpleTokenizer from all non-test text files in the given directory.
    Uses word_tokenize (NLTK) to build vocabulary from training data only.

    Args:
        data_directory: Path to directory containing text files (test files are excluded).

    Returns:
        SimpleTokenizer instance with vocab built from the data.
    """
    print("\nLoading data and creating tokenizer")
    texts = load_texts(data_directory)
    tokenizer = SimpleTokenizer(' '.join(texts))
    print("Vocabulary size is", tokenizer.vocab_size)

    return tokenizer

def load_individual_texts(*files):
    """
    Load raw text from multiple files. Returns one string per file in the same order as input paths.

    Args:
        *files: Variable number of file paths to text files.

    Returns:
        Tuple of strings, each the content of the corresponding file.
    """
    texts = []
    for path in files:
        with open(path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return *texts,

def load_texts(directory):
    """
    Load all text files from a directory, excluding any filename containing "test".
    Used to build tokenizer vocabulary from training data only.

    Args:
        directory: Path to directory containing text files.

    Returns:
        List of strings, one per non-test file.
    """
    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def collate_batch(batch, block_size):
    """
    Collate a batch of (sequences, labels) into padded tensors for the DataLoader.
    Truncates or pads sequences to block_size, stacks labels.

    Args:
        batch: List of (input_ids_tensor, label_tensor) from the dataset.
        block_size: Target sequence length (padding/truncation).

    Returns:
        Tuple of (padded_sequences, labels) as torch tensors.
    """
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def run_sanity_check(block_size, tokenizer, model, results_directory, plot_name_suffix, test_sentence="That's how progress happens -- in societies and in our own lives."):
    """
    Run attention sanity check: pass a fixed sentence through the model and save attention heatmaps.
    Verifies attention weights normalize correctly and produces PNG plots.

    Args:
        block_size: Max sequence length for padding the input.
        tokenizer: Tokenizer to encode the test sentence.
        model: Model (encoder or decoder) that returns attention maps.
        results_directory: Directory to save attention map plots.
        plot_name_suffix: Suffix for plot filenames (e.g. "encoder_pre_training").
        test_sentence: Sentence to run through the model for visualization.
    """
    logging_title = " ".join([word.upper() for word in plot_name_suffix.split('_')])
    print(f"\n{logging_title}: Running Attention Sanity Check")
    utils = Utilities(tokenizer, model, f"{results_directory}", f"{plot_name_suffix}")
    utils.sanity_check(test_sentence, block_size)

def compute_perplexity(decoderLMmodel, data_loader, device, eval_epochs=100):
    """
    Compute perplexity of a decoder language model on a data loader.
    Perplexity = exp(mean cross-entropy loss). Lower is better.

    Args:
        decoderLMmodel: Decoder that returns (logits, loss) given (X, Y).
        data_loader: DataLoader yielding (input_ids, target_ids).
        device: torch device for tensors.
        eval_epochs: Max number of batches to evaluate over (default 100).

    Returns:
        Perplexity (float).
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        _, loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_epochs: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def compute_classifier_accuracy(encoder, classifier, data_loader, device):
    """
    Compute classification accuracy (percentage correct) of encoder + classifier on a data loader.
    Uses mean-pooled encoder output as classifier input. Sets models to eval mode during computation.

    Args:
        encoder: Encoder that outputs (embeddings, attention_maps).
        classifier: Classifier that predicts class from encoder embeddings.
        data_loader: DataLoader yielding (input_ids, labels).
        device: torch device for tensors.

    Returns:
        Accuracy as percentage (0-100).
    """
    encoder.eval()
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            encoder_output, _ = encoder(X)
            classifier_input = encoder_output.mean(dim=1)
            outputs = classifier(classifier_input)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        encoder.train()
        classifier.train()
        return accuracy

def main():
    """
    Entry point: parse command-line task and run the appropriate pipeline.
    Tasks: classify (standard encoder), generate (decoder LM), sparse (sparse encoder).
    """
    parser = argparse.ArgumentParser(description='Transformer Assignment')
    parser.add_argument('--task', type=str, default='classify', help='Task to run: classify, generate, or sparse')
    args = parser.parse_args()

    # Use CUDA if available; otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == 'classify':
        print("\nRunning Classification Task: Classify given speech as belonging to one of 3 presidents")
        train_classifier(Encoder, device)

    elif args.task == 'generate':
        print("\nRunning Language Modeling")
        train_decoder(device)

    elif args.task == "sparse":
        print("\nRunning Classification Task using Sparse Encoder")
        train_classifier(SparseEncoder, device)

if __name__ == "__main__":
    main()

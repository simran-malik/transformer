import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import torch.optim as optim
import torch.nn as nn
import argparse

import main_sparse
from transformer import Classifier, Decoder, Encoder
from utilities import Utilities
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 100 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(encoder, classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
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


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        _, loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():
    parser = argparse.ArgumentParser(description='Transformer Assignment')
    parser.add_argument('--task', type=str, default='cls', help='Task to run: cls, lm, or p3')
    args = parser.parse_args()

    print("\nLoading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    if args.task == 'part1':
        print("Running Part 1: Classification")

        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
        
        encoder = Encoder(tokenizer.vocab_size, block_size, n_embd, n_layer, n_head)
        classifier = Classifier(n_input, n_hidden, n_output)
        total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in classifier.parameters())
        print(f"Total parameters of encoder + classifier: {total_params}")

        print("\nPRE TRAINING: Running Attention Sanity Check")
        utils = Utilities(tokenizer, encoder)
        test_sentence = "That's how progress happens -- in societies and in our own lives."
        utils.sanity_check(test_sentence, block_size)
    
        # Training encoder + classifier for predicting the politician:
        encoder.to(device)
        classifier.to(device)
        # Loss function for classification task
        loss_function = nn.CrossEntropyLoss()
        # Optimizer with all parameters to be trained together
        optimizer = optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()), 
            lr=learning_rate
        )

        for epoch in range(epochs_CLS):
            encoder.train()
            classifier.train()
            total_loss = 0

            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                # FORWARD PASS
                # Get sequence embeddings from encoder
                # Shape: (batch_size, block_size, n_embd)
                encoder_output, _ = encoder(xb)
                # Average across the sequence dimension
                # Shape: (batch_size, n_embd)
                classifier_input = encoder_output.mean(dim=1) # Shape: (B, C)
                # 3. Predict politician
                # Shape: (batch_size, n_output=3)
                classifier_output = classifier(classifier_input)

                # LOSS CALCULATION
                loss = loss_function(classifier_output, yb)

                # BACKPROPAGATION
                # Clear previous gradients
                optimizer.zero_grad() 
                # Compute gradients via chain rule
                loss.backward() 
                # Update all weights in encoder & classifier
                optimizer.step() 

                # LOGGING METRICS
                total_loss += loss.item()

            avg_loss = total_loss / len(train_CLS_loader)
            accuracy = compute_classifier_accuracy(encoder, classifier, train_CLS_loader)

            print(f"Epoch {epoch+1:02d} | Epoch Loss: {avg_loss:.4f} | Epoch Acc: {accuracy:.2f}%")

        print(f"Final Test Accuracy: {compute_classifier_accuracy(encoder, classifier, test_CLS_loader):.2f}%")

        print("\nPOST TRAINING: Running Attention Sanity Check")
        utils = Utilities(tokenizer, encoder)
        test_sentence = "That's how progress happens -- in societies and in our own lives."
        utils.sanity_check(test_sentence, block_size)

    elif args.task == 'part2':
        print("Running Part 2: Language Modeling")

        inputfile = "speechesdataset/train_LM.txt"
        obamatestfile = "speechesdataset/test_LM_obama.txt"
        wbushtestfile = "speechesdataset/test_LM_wbush.txt"
        hbushtestfile = "speechesdataset/test_LM_hbush.txt"

        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        with open(obamatestfile, 'r', encoding='utf-8') as f:
            obamatestdata = f.read()
        with open(wbushtestfile, 'r', encoding='utf-8') as f:
            wbushtestdata = f.read()
        with open(hbushtestfile, 'r', encoding='utf-8') as f:
            hbushtestdata = f.read()

        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
        test_obama_dataset = LanguageModelingDataset(tokenizer, obamatestdata,  block_size)
        test_obama_loader = DataLoader(test_obama_dataset, batch_size=batch_size, shuffle=True)
        test_wbush_dataset = LanguageModelingDataset(tokenizer, wbushtestdata,  block_size)
        test_wbush_loader = DataLoader(test_wbush_dataset, batch_size=batch_size, shuffle=True)
        test_hbush_dataset = LanguageModelingDataset(tokenizer, hbushtestdata,  block_size)
        test_hbush_loader = DataLoader(test_hbush_dataset, batch_size=batch_size, shuffle=True)

        decoder = Decoder(tokenizer.vocab_size, block_size, n_embd, n_layer, n_head)
        decoder.to(device)

        optimizer_LM = optim.Adam(decoder.parameters(), lr=learning_rate)
        print(f"Total parameters of Decoder: {sum(p.numel() for p in decoder.parameters())}")

        print("\nPRE TRAINING: Running Attention Sanity Check")
        utils = Utilities(tokenizer, decoder)
        test_sentence = "That's how progress happens -- in societies and in our own lives."
        utils.sanity_check(test_sentence, block_size)

        # Training Decoder
        decoder.train()
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)

            # Forward pass
            logits, loss = decoder(xb, yb)

            # Backpropagation
            optimizer_LM.zero_grad()
            loss.backward()
            optimizer_LM.step()

            # Periodically evaluate perplexity
            if i == max_iters - 1 or (i + 1) % eval_interval == 0:
                train_perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters)
                print(f"Iteration {i + 1:03d} | Train Loss: {loss.item():.4f} | Perplexity: {train_perplexity:.2f}")

        train_perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters)
        obama_perplexity = compute_perplexity(decoder, test_obama_loader, eval_iters)
        wbush_perplexity = compute_perplexity(decoder, test_wbush_loader, eval_iters)
        hbush_perplexity = compute_perplexity(decoder, test_hbush_loader, eval_iters)

        print(f"Obama: Test Perplexity: {obama_perplexity:.2f}")
        print(f"W Bush: Test Perplexity: {wbush_perplexity:.2f}")
        print(f"H Bush: Test Perplexity: {hbush_perplexity:.2f}")

        print("\nPOST TRAINING: Running Attention Sanity Check")
        utils = Utilities(tokenizer, decoder)
        test_sentence = "That's how progress happens -- in societies and in our own lives."
        utils.sanity_check(test_sentence, block_size)

    elif args.task == "part3":
        main_sparse.main()

if __name__ == "__main__":
    main()

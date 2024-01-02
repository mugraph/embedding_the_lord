import random
import re
import numpy as np
import string
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle

import nltk
from nltk.tokenize import RegexpTokenizer

from tqdm import tqdm

import argparse

torch.manual_seed(1)

def read_text_file(path, readstart, readstop):
    try:
        with open(path, 'r') as file:
            lines = file.readlines()

            # If both readstart and readstop are 0, return the entire content
            if readstart == 0 and readstop == 0:
                return ''.join(lines)

            # Ensure readstart and readstop are within valid range
            readstart = max(0, min(readstart, len(lines)))
            readstop = max(readstart, max(readstop, len(lines)))

            # Extract lines between readstart and readstop
            lines = lines[readstart:readstop]
            result = ''.join(lines)

            return result

    except FileNotFoundError:
        return f"File not found: {path}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

   # nn.Module is the base class for neural network modules
class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def preprocess_text(text, context_size):

    regex = r'\d*?(\w+).\d+:\d+'
    text = re.sub(regex, '', text)

    nltk.download('punkt')
    # Remove punctuation
    translate_table = dict((ord(char), None) for char in string.punctuation)   
    text = text.translate(translate_table)

    # Use word_tokenize on lowercase text to preserve contractions
    tokens = nltk.word_tokenize(text.lower())

    print(f"Number of tokens: {len(tokens)}")

    # Build a list of tuples of size length of tokens minus context size
    # Each tuple is ([ word_i-context_size, ..., word_i-1 ], target word)
    ngrams = [
        (
            # Iteratate over range of context_size
            [tokens[i - j - 1] for j in range(context_size)],
            tokens[i]
        )
        # Iterate over range from context size to length of tokens
        for i in range(context_size, len(tokens))
    ]

    # Print the first 3, just so you can see what they look like.
    print(f"Number of ngrams: {len(ngrams)}")

    return tokens, ngrams

def train_model(file_path,
                name,
                readstart,
                readstop,
                context_size,
                epochs,
                min_dim):
    
    print(f"Context Size: {context_size}")
    print(f"Epochs: {epochs}")

    # Prepare vocabulary
    text = read_text_file(file_path, readstart, readstop)
    tokens, ngrams = preprocess_text(text, context_size)
    vocab = set(tokens)

    # Save the vocab_set to a file using pickle
    Path("./vocab").mkdir(parents=True, exist_ok=True)
    with open(f'./vocab/{name}_vocab.pkl', 'wb') as file:
        pickle.dump(vocab, file)

    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    embedding_dim = min(min_dim, len(vocab) // 2)
    print(f"Number of dims: {embedding_dim}")

    # Print the length of vocabs
    print(f"Number of vocabs: {len(vocab)}")

    losses = {}
    loss_function = nn.NLLLoss()

    # Instantiate NGramLanguageModeler 
    model = NGramLanguageModeler(len(vocab), embedding_dim, context_size)

    # Pass iterator of module params to optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    Path("./models").mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        num_batches = len(ngrams)

        for batch_idx, (context, target) in tqdm(enumerate(ngrams), total=num_batches, desc=f'Epoch {epoch + 1}', leave=False):
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            context_idxs = torch.tensor([word_to_idx[w] for w in context], dtype=torch.long)
        
            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(context_idxs)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_function(log_probs, torch.tensor([word_to_idx[target]], dtype=torch.long))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()

        # Print epoch summary
        avg_loss = total_loss / num_batches
        print(f"\rEpoch {epoch + 1}, Loss: {avg_loss:.4f}")

        # Save the model state after each epoch
        torch.save(model.state_dict(), f"models/{name}_model_epoch_{epoch + 1}_loss_{avg_loss:.4f}.pth")
        print(f"Model saved to .models/{name}_model_epoch_{epoch + 1}_loss_{avg_loss:.4f}.pth")

    # Save the trained model
    torch.save(model.state_dict(), f"./models/{name}_model_final.pth")
    print(f"Model saved to ./models/{name}_model_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NGram Language Model Training Script")

    parser.add_argument("--file", type=str, required=True, help="Path to the text file containing training data")
    parser.add_argument("--name", type=str, required=True, help="Name of the the trained model.")
    parser.add_argument("--readstart", type=int, default=0, help="Number of lines to skip at the start of the file")
    parser.add_argument("--readstop", type=int, default=0, help="Number of lines to skip at the end of the file")
    parser.add_argument("--context_size", type=int, default=2, help="Size of the context window")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--min_dim", type=int, default=50, help="Minimum dimension for embedding vectors")

    args = parser.parse_args()

    train_model(args.file, args.name, args.readstart, args.readstop, args.context_size, args.epochs, args.min_dim)

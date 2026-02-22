import torch.nn as nn
import torch
import math

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        input = self.input_layer(input)
        input = self.activation(input)
        input = self.output_layer(input)

        return input

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Linear transformation to queries, keys and values
        self.query_transform = nn.Linear(model_dim, model_dim)
        self.key_transform = nn.Linear(model_dim, model_dim)
        self.value_transform = nn.Linear(model_dim, model_dim)

        # Linear transformation of concatenated output from all heads
        self.output_linear = nn.Linear(model_dim, model_dim)

        # Softmax normalization applied per token
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, block_size, model_dim = x.shape

        q = self.query_transform(x)
        k = self.key_transform(x)
        v = self.value_transform(x)

        q = q.view(batch_size, block_size, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, block_size, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, block_size, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = self.softmax(scores)
        output = weights @ v 

        output = output.transpose(1, 2).contiguous().view(batch_size, block_size, model_dim)
        output = self.output_linear(output)

        return output, weights

class EncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        # Layer normalization after self-attention
        self.norm1 = nn.LayerNorm(model_dim)
        # Multi-head self attention block
        self.multi_head_self_attention = MultiHeadAttention(model_dim, num_heads)
        # Layer normalization after feedforward layer
        self.norm2 = nn.LayerNorm(model_dim)

        # 2-layer Feedforward Layer
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.ReLU(),
            nn.Linear(4 * model_dim, model_dim)
        )

    def forward(self, x):
        # Layer normalization
        normalized_x = self.norm1(x)

        # Perform multi-headed attention
        attention_output, attention_weights = self.multi_head_self_attention(normalized_x)

        # Residual stabilization
        attention_output = attention_output + x

        # Layer normalization
        normalized_attention_output = self.norm2(attention_output)

        # Add non-linearity using feedforward layer
        feedforward_output = self.feedforward(normalized_attention_output)

        # Residual stabilization
        output = feedforward_output + normalized_attention_output

        return output, [attention_weights]

class Encoder(nn.Module):
    def __init__(self, vocab_size, block_size, model_dim, num_layers, num_heads):
        super().__init__()
        self.model_dim = model_dim

        # Token embedding to map token indices to vectors
        self.token_embeddings = nn.Embedding(vocab_size, model_dim)

        # Positional embedding to map 0 to block_size - 1 positions to vectors
        self.position_embeddings = nn.Embedding(block_size, model_dim)

        # Stack num_layers encoder blocks
        self.blocks = nn.ModuleList([EncoderBlock(model_dim, num_heads) for _ in range(num_layers)])

    def forward(self, input_indices):
        batch_size, block_size = input_indices.shape

        token_embeddings = self.token_embeddings(input_indices)
        positions = torch.arange(0, block_size, dtype=torch.long, device=input_indices.device)
        positional_embeddings = self.position_embeddings(positions)

        # Generate input embeddings with position encoding
        x = token_embeddings + positional_embeddings

        # Pass x through all encoder blocks
        attention_maps = []
        for block in self.blocks:
            x, weights = block(x)
            attention_maps.extend(weights)
            
        return x, attention_maps

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, block_size):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Linear transformation to queries, keys and values
        self.query_transform = nn.Linear(model_dim, model_dim)
        self.key_transform = nn.Linear(model_dim, model_dim)
        self.value_transform = nn.Linear(model_dim, model_dim)

       # Linear transformation of concatenated output from all heads
        self.output_linear = nn.Linear(model_dim, model_dim)

        # Causal mask: Lower triangular matrix of ones
        # Using register_buffer so it's not a trainable parameter but stays with the model
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, block_size, model_dim = x.shape

        q = self.query_transform(x).view(batch_size, block_size, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_transform(x).view(batch_size, block_size, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_transform(x).view(batch_size, block_size, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply Causal Mask
        scores = scores.masked_fill(self.mask[:, :, :block_size, :block_size] == 0, float('-inf'))

        weights = self.softmax(scores)
        output = weights @ v 

        output = output.transpose(1, 2).contiguous().view(batch_size, block_size, model_dim)
        output = self.output_linear(output)

        return output, weights

class DecoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads, block_size):
        super().__init__()
        self.masked_attention = CausalMultiHeadAttention(model_dim, num_heads, block_size)
        
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.ReLU(),
            nn.Linear(4 * model_dim, model_dim),
        )
        
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        # Layer normalization
        normalized_x = self.norm1(x)

        # Perform masked multi-headed attention
        attention_output, attention_weights = self.masked_attention(normalized_x)

        # Residual stabilization
        attention_output = attention_output + x

        # Layer normalization
        normalized_attention_output = self.norm2(attention_output)

        # Add non-linearity using feedforward layer
        feedforward_output = self.feedforward(normalized_attention_output)

        # Residual stabilization
        output = feedforward_output + attention_output

        return output, [attention_weights]

class Decoder(nn.Module):
    def __init__(self, vocab_size, block_size, model_dim, num_layers, num_heads):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, model_dim)
        self.position_embeddings = nn.Embedding(block_size, model_dim)
        
        
        self.blocks = nn.Sequential(*[
            DecoderBlock(model_dim, num_heads, block_size) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size)

    def forward(self, input_indices, targets=None):
        batch_size, block_size = input_indices.shape
        x = self.token_embeddings(input_indices) + self.position_embeddings(torch.arange(block_size, device=input_indices.device))
        
        attention_maps = []
        for block in self.blocks:
            x, weights = block(x)
            attention_maps.extend(weights)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            batch_size, block_size, vocab_size = logits.shape
            loss = nn.functional.cross_entropy(logits.view(batch_size*block_size, vocab_size), targets.view(batch_size*block_size))
            return logits, loss
        
        return logits, attention_maps
import math
import torch
import torch.nn as nn

def get_sparse_attention_mask(block_size, sparse_type="local", window_size=4, block_size_attn=4):
    mask = torch.zeros(block_size, block_size, dtype=torch.bool)
    
    if sparse_type == "local":
        for i in range(block_size):
            for j in range(block_size):
                # Local window: token i can attend to token j if the distance is within window_size
                if abs(i - j) <= window_size:
                    mask[i, j] = True
    else:
        raise ValueError(f"Unknown sparse_type: {sparse_type}. Use 'local'")
        
    return mask

class SparseMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with a sparse mask. Restricts each token to attend only within a local window
    (sparse_type="local") or other patterns, reducing complexity vs full attention.
    """
    def __init__(self, model_dim, num_heads, block_size, sparse_type="local", window_size=4, block_size_attn=4):
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
        
        # Register the sparse mask as a buffer so it moves with the model device
        mask = get_sparse_attention_mask(block_size, sparse_type, window_size, block_size_attn)
        self.register_buffer("sparse_mask", mask)

    def forward(self, x):
        batch_size, seq_len, model_dim = x.shape

        q = self.query_transform(x)
        k = self.key_transform(x)
        v = self.value_transform(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Pre-softmax attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply sparse mask (broadcast to batch and heads)
        # Assuming the sequence length is equal to or less than the defined block_size.
        # If seq_len differs from initialization block_size, slice the mask.
        current_mask = self.sparse_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0) 
        
        # Set scores to -inf where mask is False so they become 0 after softmax
        scores = scores.masked_fill(~current_mask, float('-inf'))

        weights = self.softmax(scores)
        output = weights @ v 

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
        output = self.output_linear(output)

        return output, weights

class SparseEncoderBlock(nn.Module):
    """
    Single encoder layer with sparse attention: LayerNorm -> SparseMultiHeadAttention (residual) ->
    LayerNorm -> FeedForward (residual). Same structure as EncoderBlock but uses sparse attention.
    """
    def __init__(self, model_dim, num_heads, block_size, sparse_type="local", window_size=4, block_size_attn=4):
        super().__init__()
        # Layer normalization after self-attention
        self.norm1 = nn.LayerNorm(model_dim)

        # Multi-head self attention block with Sparse Attention
        self.multi_head_self_attention = SparseMultiHeadAttention(
            model_dim, num_heads, block_size, sparse_type, window_size, block_size_attn
        )

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

class SparseEncoder(nn.Module):
    """
    Transformer encoder with sparse (local-window) attention instead of full attention.
    Same interface as Encoder: token + positional embeddings, stacked SparseEncoderBlocks.
    """
    def __init__(self, vocab_size, block_size, model_dim, num_layers, num_heads, sparse_type="local", window_size=4, block_size_attn=4):
        super().__init__()
        self.model_dim = model_dim

        # Token embedding to map token indices to vectors
        self.token_embeddings = nn.Embedding(vocab_size, model_dim)

        # Positional embedding to map 0 to block_size - 1 positions to vectors
        self.position_embeddings = nn.Embedding(block_size, model_dim)

        # Stack num_layers encoder blocks
        self.blocks = nn.ModuleList([
            SparseEncoderBlock(model_dim, num_heads, block_size, sparse_type, window_size, block_size_attn) 
            for _ in range(num_layers)
        ])

    def forward(self, input_indices):
        _, seq_len = input_indices.shape

        token_embeddings = self.token_embeddings(input_indices)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_indices.device)
        positional_embeddings = self.position_embeddings(positions)

        # Generate input embeddings with position encoding
        x = token_embeddings + positional_embeddings

        # Pass x through all encoder blocks
        attention_maps = []
        for block in self.blocks:
            x, weights = block(x)
            attention_maps.extend(weights)
            
        return x, attention_maps

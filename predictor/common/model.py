# common/model.py
import torch
import torch.nn as nn
import math

class DeckTransformer(nn.Module):
    """The main Transformer model for predicting the next card."""
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dropout, padding_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.card_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        ffn_dim = embedding_dim * 4
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src_embed = self.card_embedding(src) * math.sqrt(self.embedding_dim)
        src_padding_mask = (src == self.padding_idx)
        encoder_output = self.transformer_encoder(src_embed, src_key_padding_mask=src_padding_mask)
        # Aggregate token outputs for a single sequence prediction
        encoder_output = encoder_output.mean(dim=1)
        output = self.output_layer(encoder_output)
        return output
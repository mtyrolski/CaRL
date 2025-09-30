"""Embedding-based subgoal generators for hierarchical latent space search.

This module implements subgoal generators that operate in the learned
embedding space rather than discrete state space, enabling real-time RL
and support for non-discrete environments.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class EmbeddingGenerator(nn.Module):
    """Generator that produces subgoal embeddings from current state embeddings.
    
    Takes the current state embedding and outputs the embedding of a predicted 
    subgoal k steps away, operating entirely in the learned latent space.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: Optional[list[int]] = None,
        k_steps: int = 4,
        dropout: float = 0.1,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.k_steps = k_steps
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Input embedding with k-step conditioning
        self.k_embedding = nn.Embedding(32, embedding_dim // 4)  # Support up to 32 steps
        
        # Main generator network
        input_dim = embedding_dim + embedding_dim // 4  # Current embedding + k embedding
        layers = []
        
        in_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            ])
            in_dim = hidden_dim
            
        # Output layer to produce subgoal embedding
        layers.append(nn.Linear(in_dim, embedding_dim))
        
        self.generator = nn.Sequential(*layers)
        
        # Optional attention mechanism for better long-range dependencies
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(embedding_dim)
            
    def forward(
        self, 
        current_embedding: Tensor, 
        k_steps: Optional[Tensor] = None
    ) -> Tensor:
        """Generate subgoal embedding k steps away from current embedding.
        
        Args:
            current_embedding: Current state embedding [batch_size, embedding_dim]
            k_steps: Number of steps to subgoal [batch_size] or None for default
            
        Returns:
            Predicted subgoal embedding [batch_size, embedding_dim]
        """
        batch_size = current_embedding.size(0)
        
        if k_steps is None:
            k_steps = torch.full((batch_size,), self.k_steps, 
                               device=current_embedding.device, dtype=torch.long)
        
        # Get k-step embeddings
        k_emb = self.k_embedding(k_steps)
        
        # Concatenate current embedding with k-step embedding
        generator_input = torch.cat([current_embedding, k_emb], dim=-1)
        
        # Generate subgoal embedding
        subgoal_embedding = self.generator(generator_input)
        
        # Apply attention if enabled
        if self.use_attention:
            # Use current embedding as query, subgoal as key/value
            attended, _ = self.attention(
                current_embedding.unsqueeze(1),
                subgoal_embedding.unsqueeze(1), 
                subgoal_embedding.unsqueeze(1)
            )
            subgoal_embedding = self.attention_norm(
                subgoal_embedding + attended.squeeze(1)
            )
        
        return subgoal_embedding


class TransformerEmbeddingGenerator(nn.Module):
    """Transformer-based generator for subgoal embeddings.
    
    Uses transformer architecture to model sequence dependencies
    in embedding space for better subgoal generation.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        max_k_steps: int = 32,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_k_steps = max_k_steps
        
        # Position encoding for k-steps
        self.k_embedding = nn.Embedding(max_k_steps, embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        current_embedding: Tensor, 
        k_steps: Optional[Tensor] = None
    ) -> Tensor:
        """Generate subgoal embedding using transformer."""
        batch_size = current_embedding.size(0)
        
        if k_steps is None:
            k_steps = torch.full((batch_size,), 4, 
                               device=current_embedding.device, dtype=torch.long)
        
        # Create sequence: [current_embedding, k_embedding]
        k_emb = self.k_embedding(k_steps)
        sequence = torch.stack([current_embedding, k_emb], dim=1)  # [batch, 2, embed_dim]
        
        # Apply transformer
        transformed = self.transformer(sequence)
        
        # Use the last token (k_embedding position) as subgoal
        subgoal_embedding = transformed[:, -1, :]
        subgoal_embedding = self.output_proj(self.dropout(subgoal_embedding))
        
        return subgoal_embedding


class AutoregressiveEmbeddingGenerator(nn.Module):
    """Autoregressive generator for multi-step subgoal planning in embedding space.
    
    Generates a sequence of intermediate subgoal embeddings leading to the 
    final k-step subgoal, enabling more structured hierarchical planning.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_sequence_length: int = 16,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        
        # LSTM for autoregressive generation
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        current_embedding: Tensor, 
        k_steps: Optional[Tensor] = None,
        return_sequence: bool = False,
    ) -> Tensor:
        """Generate subgoal embedding(s) autoregressively.
        
        Args:
            current_embedding: Current state embedding
            k_steps: Number of steps to generate
            return_sequence: If True, return all intermediate embeddings
            
        Returns:
            Final subgoal embedding or sequence of embeddings
        """
        batch_size = current_embedding.size(0)
        
        if k_steps is None:
            k_steps = torch.full((batch_size,), 4, 
                               device=current_embedding.device, dtype=torch.long)
        
        max_k = min(k_steps.max().item(), self.max_sequence_length)
        
        # Initialize with current embedding
        sequence = [current_embedding]
        hidden = None
        
        # Generate sequence
        input_embedding = current_embedding.unsqueeze(1)  # [batch, 1, embed_dim]
        
        for step in range(max_k):
            lstm_out, hidden = self.lstm(input_embedding, hidden)
            next_embedding = self.output_proj(self.dropout(lstm_out.squeeze(1)))
            sequence.append(next_embedding)
            input_embedding = next_embedding.unsqueeze(1)
        
        if return_sequence:
            return torch.stack(sequence[1:], dim=1)  # Exclude initial embedding
        else:
            # Return embedding at k-step for each sample
            final_embeddings = []
            for i, k in enumerate(k_steps):
                k_idx = min(k.item(), len(sequence) - 1)
                final_embeddings.append(sequence[k_idx][i])
            return torch.stack(final_embeddings)


# Loss functions for embedding generators
def embedding_generator_loss(
    predicted_embedding: Tensor, 
    target_embedding: Tensor,
    embedding_regularization: float = 0.01,
) -> Tensor:
    """Loss function for embedding generator training.
    
    Combines MSE loss for embedding prediction with regularization
    to prevent embedding collapse.
    """
    # Primary loss: MSE between predicted and target embeddings
    mse_loss = nn.functional.mse_loss(predicted_embedding, target_embedding)
    
    # Regularization: encourage diverse embeddings
    embedding_norm = torch.norm(predicted_embedding, dim=-1).mean()
    reg_loss = embedding_regularization * torch.abs(embedding_norm - 1.0)
    
    return mse_loss + reg_loss


def contrastive_embedding_loss(
    predicted_embedding: Tensor,
    positive_embedding: Tensor, 
    negative_embeddings: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    """Contrastive loss for embedding generator training.
    
    Encourages predicted embeddings to be close to positive targets
    and far from negative samples.
    """
    # Normalize embeddings
    pred_norm = nn.functional.normalize(predicted_embedding, dim=-1)
    pos_norm = nn.functional.normalize(positive_embedding, dim=-1)
    neg_norm = nn.functional.normalize(negative_embeddings, dim=-1)
    
    # Compute similarities
    pos_sim = torch.sum(pred_norm * pos_norm, dim=-1) / temperature
    neg_sim = torch.matmul(pred_norm, neg_norm.transpose(-2, -1)) / temperature
    
    # Contrastive loss
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    return nn.functional.cross_entropy(logits, targets)
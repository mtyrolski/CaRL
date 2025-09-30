"""Embedding-conditioned Conditional Low-Level Policy for hierarchical navigation.

This module implements CLLPs that operate on state embeddings rather than
explicit states, enabling hierarchical latent space search and navigation
to subgoals specified in embedding space.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EmbeddingConditionedCLLP(nn.Module):
    """Conditional Low-Level Policy conditioned on state and subgoal embeddings.
    
    Takes current state embedding and target subgoal embedding to produce
    action probabilities for navigating to the subgoal in embedding space.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_actions: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_actions = num_actions
        self.hidden_dims = hidden_dims or [512, 256]
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Embedding processing layers
        self.state_proj = nn.Linear(embedding_dim, embedding_dim)
        self.subgoal_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Cross-attention between state and subgoal embeddings
        if use_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(embedding_dim)
        
        # Policy network
        input_dim = embedding_dim * 2  # Concatenated state and subgoal embeddings
        if use_attention:
            input_dim += embedding_dim  # Add attended features
            
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
            
        # Output layer for action probabilities
        layers.append(nn.Linear(in_dim, num_actions))
        
        self.policy_network = nn.Sequential(*layers)
        
    def forward(
        self, 
        state_embedding: Tensor, 
        subgoal_embedding: Tensor
    ) -> Tensor:
        """Predict action probabilities given state and subgoal embeddings.
        
        Args:
            state_embedding: Current state embedding [batch_size, embedding_dim]
            subgoal_embedding: Target subgoal embedding [batch_size, embedding_dim]
            
        Returns:
            Action logits [batch_size, num_actions]
        """
        # Process embeddings
        state_proj = self.state_proj(state_embedding)
        subgoal_proj = self.subgoal_proj(subgoal_embedding)
        
        # Apply cross-attention if enabled
        if self.use_attention:
            attended, _ = self.cross_attention(
                state_proj.unsqueeze(1),
                subgoal_proj.unsqueeze(1),
                subgoal_proj.unsqueeze(1),
            )
            attended_features = attended.squeeze(1)
            attended_features = self.attention_norm(attended_features)
            
            # Concatenate all features
            combined_features = torch.cat([
                state_proj, subgoal_proj, attended_features
            ], dim=-1)
        else:
            combined_features = torch.cat([state_proj, subgoal_proj], dim=-1)
        
        # Generate action logits
        action_logits = self.policy_network(combined_features)
        
        return action_logits
    
    def get_action_probs(
        self, 
        state_embedding: Tensor, 
        subgoal_embedding: Tensor
    ) -> Tensor:
        """Get action probabilities (softmax of logits)."""
        logits = self.forward(state_embedding, subgoal_embedding)
        return F.softmax(logits, dim=-1)
    
    def sample_action(
        self, 
        state_embedding: Tensor, 
        subgoal_embedding: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """Sample action from policy distribution."""
        logits = self.forward(state_embedding, subgoal_embedding) / temperature
        action_probs = F.softmax(logits, dim=-1)
        
        # Sample from categorical distribution
        action = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
        return action


class HierarchicalEmbeddingCLLP(nn.Module):
    """Hierarchical CLLP that handles multiple levels of subgoals in embedding space.
    
    Can condition on both immediate and long-term subgoals for better
    hierarchical navigation in complex environments.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_actions: int,
        num_hierarchy_levels: int = 2,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_actions = num_actions
        self.num_hierarchy_levels = num_hierarchy_levels
        self.hidden_dims = hidden_dims or [512, 256]
        self.dropout = dropout
        
        # Separate projections for each hierarchy level
        self.state_proj = nn.Linear(embedding_dim, embedding_dim)
        self.subgoal_projections = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) 
            for _ in range(num_hierarchy_levels)
        ])
        
        # Hierarchical attention
        self.hierarchy_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            ) for _ in range(num_hierarchy_levels)
        ])
        
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) 
            for _ in range(num_hierarchy_levels)
        ])
        
        # Hierarchy fusion
        fusion_input_dim = embedding_dim * (1 + num_hierarchy_levels * 2)  # state + (subgoal + attended) per level
        self.hierarchy_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Policy network
        layers = []
        in_dim = embedding_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            ])
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, num_actions))
        self.policy_network = nn.Sequential(*layers)
        
    def forward(
        self, 
        state_embedding: Tensor, 
        subgoal_embeddings: list[Tensor]
    ) -> Tensor:
        """Predict actions given state and hierarchical subgoal embeddings.
        
        Args:
            state_embedding: Current state embedding [batch_size, embedding_dim]
            subgoal_embeddings: List of subgoal embeddings for each hierarchy level
            
        Returns:
            Action logits [batch_size, num_actions]
        """
        assert len(subgoal_embeddings) == self.num_hierarchy_levels
        
        state_proj = self.state_proj(state_embedding)
        hierarchy_features = [state_proj]
        
        # Process each hierarchy level
        for level, subgoal_embedding in enumerate(subgoal_embeddings):
            # Project subgoal embedding
            subgoal_proj = self.subgoal_projections[level](subgoal_embedding)
            
            # Apply attention between state and subgoal
            attended, _ = self.hierarchy_attention[level](
                state_proj.unsqueeze(1),
                subgoal_proj.unsqueeze(1),
                subgoal_proj.unsqueeze(1),
            )
            attended_features = self.attention_norms[level](attended.squeeze(1))
            
            hierarchy_features.extend([subgoal_proj, attended_features])
        
        # Fuse hierarchical features
        combined_features = torch.cat(hierarchy_features, dim=-1)
        fused_features = self.hierarchy_fusion(combined_features)
        
        # Generate action logits
        action_logits = self.policy_network(fused_features)
        
        return action_logits


class ProgressAwareCLLP(nn.Module):
    """CLLP that tracks progress towards subgoals in embedding space.
    
    Maintains an estimate of progress and adjusts actions accordingly,
    useful for long-horizon navigation tasks.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_actions: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_actions = num_actions
        self.hidden_dims = hidden_dims or [512, 256]
        self.dropout = dropout
        
        # Embedding processing
        self.state_proj = nn.Linear(embedding_dim, embedding_dim)
        self.subgoal_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Progress estimation
        self.progress_estimator = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Progress-aware policy
        input_dim = embedding_dim * 2 + 1  # state + subgoal + progress
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
            
        layers.append(nn.Linear(in_dim, num_actions))
        self.policy_network = nn.Sequential(*layers)
        
    def forward(
        self, 
        state_embedding: Tensor, 
        subgoal_embedding: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Predict actions and progress towards subgoal.
        
        Returns:
            Tuple of (action_logits, progress_estimate)
        """
        state_proj = self.state_proj(state_embedding)
        subgoal_proj = self.subgoal_proj(subgoal_embedding)
        
        # Estimate progress
        combined_for_progress = torch.cat([state_proj, subgoal_proj], dim=-1)
        progress = self.progress_estimator(combined_for_progress)
        
        # Generate actions with progress awareness
        combined_for_policy = torch.cat([state_proj, subgoal_proj, progress], dim=-1)
        action_logits = self.policy_network(combined_for_policy)
        
        return action_logits, progress.squeeze(-1)


class ResidualEmbeddingCLLP(nn.Module):
    """CLLP with residual connections for better gradient flow.
    
    Particularly useful for deep networks and complex navigation tasks.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_actions: int,
        num_residual_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_actions = num_actions
        self.num_residual_blocks = num_residual_blocks
        
        # Input projections
        self.state_proj = nn.Linear(embedding_dim, embedding_dim)
        self.subgoal_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Initial fusion
        self.input_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(embedding_dim, dropout)
            for _ in range(num_residual_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, num_actions)
        
    def _make_residual_block(self, dim: int, dropout: float) -> nn.Module:
        """Create a residual block."""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        
    def forward(
        self, 
        state_embedding: Tensor, 
        subgoal_embedding: Tensor
    ) -> Tensor:
        """Forward pass with residual connections."""
        state_proj = self.state_proj(state_embedding)
        subgoal_proj = self.subgoal_proj(subgoal_embedding)
        
        # Initial fusion
        x = self.input_fusion(torch.cat([state_proj, subgoal_proj], dim=-1))
        
        # Apply residual blocks
        for block in self.residual_blocks:
            residual = x
            x = block(x) + residual
            x = F.relu(x)
        
        # Output
        action_logits = self.output_layer(x)
        
        return action_logits


# Loss functions for embedding CLLP training
def cllp_loss(action_logits: Tensor, target_actions: Tensor) -> Tensor:
    """Standard cross-entropy loss for CLLP training."""
    return F.cross_entropy(action_logits, target_actions)


def progress_aware_cllp_loss(
    action_logits: Tensor, 
    progress_estimate: Tensor,
    target_actions: Tensor,
    target_progress: Tensor,
    progress_weight: float = 0.1,
) -> Tensor:
    """Combined loss for progress-aware CLLP."""
    action_loss = F.cross_entropy(action_logits, target_actions)
    progress_loss = F.mse_loss(progress_estimate, target_progress)
    
    return action_loss + progress_weight * progress_loss
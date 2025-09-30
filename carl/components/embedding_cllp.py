"""Embedding-conditioned Conditional Low-Level Policy using HuggingFace transformers.

This module implements CLLPs that operate on state embeddings using BERT
as the backbone architecture, following CaRL's established patterns.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    PreTrainedModel,
    BertForSequenceClassification,
    BertConfig,
    BertModel,
    AutoConfig,
)


class HFEmbeddingConditionedCLLP(BertForSequenceClassification):
    """HuggingFace BERT-based CLLP conditioned on state and subgoal embeddings.
    
    Uses BERT architecture to process combined state and subgoal embeddings
    and predict actions, following the same pattern as existing CLLPs in CaRL.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Additional layers for embedding processing
        self.state_projection = nn.Linear(config.embedding_dim, config.hidden_size)
        self.subgoal_projection = nn.Linear(config.embedding_dim, config.hidden_size)
        
        # Cross-attention between state and subgoal embeddings
        if getattr(config, 'use_attention', True):
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(config.hidden_size)
        
        # Initialize weights
        self.init_weights()
        
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        state_embedding: Optional[Tensor] = None,
        subgoal_embedding: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass for action prediction given state and subgoal embeddings."""
        
        # If we have embeddings directly, process them
        if state_embedding is not None and subgoal_embedding is not None:
            return self._forward_embeddings(
                state_embedding, subgoal_embedding, labels, return_dict
            )
        
        # Otherwise, use standard BERT forward pass
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
    def _forward_embeddings(
        self,
        state_embedding: Tensor,
        subgoal_embedding: Tensor, 
        labels: Optional[Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Tensor]:
        """Process state and subgoal embeddings to predict actions."""
        
        # Project embeddings to BERT hidden size
        state_proj = self.state_projection(state_embedding)
        subgoal_proj = self.subgoal_projection(subgoal_embedding)
        
        # Apply cross-attention if configured
        if hasattr(self, 'cross_attention'):
            attended, _ = self.cross_attention(
                state_proj.unsqueeze(1),
                subgoal_proj.unsqueeze(1),
                subgoal_proj.unsqueeze(1),
            )
            attended_features = self.attention_norm(attended.squeeze(1))
            
            # Concatenate features
            combined_features = torch.cat([
                state_proj, subgoal_proj, attended_features
            ], dim=-1)
            
            # Project back to hidden size
            combined_features = nn.Linear(
                combined_features.size(-1), 
                self.config.hidden_size,
                device=combined_features.device
            )(combined_features)
        else:
            # Simple concatenation and projection
            combined_features = torch.cat([state_proj, subgoal_proj], dim=-1)
            combined_features = nn.Linear(
                combined_features.size(-1),
                self.config.hidden_size,
                device=combined_features.device
            )(combined_features)
        
        # Pass through BERT classifier
        # Create fake sequence with combined features as CLS token
        inputs_embeds = combined_features.unsqueeze(1)  # [batch, 1, hidden_size]
        attention_mask = torch.ones(
            inputs_embeds.size(0), 1, 
            device=inputs_embeds.device
        )
        
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        action_logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(action_logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': action_logits,
        }
    
    def get_action_probs(
        self, 
        state_embedding: Tensor, 
        subgoal_embedding: Tensor
    ) -> Tensor:
        """Get action probabilities (softmax of logits)."""
        outputs = self._forward_embeddings(state_embedding, subgoal_embedding)
        return F.softmax(outputs['logits'], dim=-1)
    
    def sample_action(
        self, 
        state_embedding: Tensor, 
        subgoal_embedding: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """Sample action from policy distribution."""
        outputs = self._forward_embeddings(state_embedding, subgoal_embedding)
        logits = outputs['logits'] / temperature
        action_probs = F.softmax(logits, dim=-1)
        
        # Sample from categorical distribution
        action = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
        return action


class HFHierarchicalEmbeddingCLLP(PreTrainedModel):
    """Hierarchical CLLP using BERT that handles multiple levels of subgoals."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # BERT backbone
        self.bert = BertModel(config, add_pooling_layer=True)
        
        # Embedding projections for different hierarchy levels
        self.state_projection = nn.Linear(config.embedding_dim, config.hidden_size)
        self.subgoal_projections = nn.ModuleList([
            nn.Linear(config.embedding_dim, config.hidden_size) 
            for _ in range(config.num_hierarchy_levels)
        ])
        
        # Hierarchical attention
        self.hierarchy_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads // 2,  # Fewer heads per level
                dropout=config.attention_probs_dropout_prob,
                batch_first=True,
            ) for _ in range(config.num_hierarchy_levels)
        ])
        
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) 
            for _ in range(config.num_hierarchy_levels)
        ])
        
        # Fusion and classifier
        fusion_input_dim = config.hidden_size * (1 + config.num_hierarchy_levels * 2)
        self.hierarchy_fusion = nn.Linear(fusion_input_dim, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self.init_weights()
        
    def forward(
        self,
        state_embedding: Tensor,
        subgoal_embeddings: list[Tensor],
        labels: Optional[Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Tensor]:
        """Predict actions given state and hierarchical subgoal embeddings."""
        
        assert len(subgoal_embeddings) == self.config.num_hierarchy_levels
        
        state_proj = self.state_projection(state_embedding)
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
        pooled_output = self.dropout(fused_features)
        action_logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(action_logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': action_logits,
        }


class HFProgressAwareCLLP(PreTrainedModel):
    """BERT-based CLLP that tracks progress towards subgoals."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # BERT backbone
        self.bert = BertModel(config, add_pooling_layer=True)
        
        # Embedding processing
        self.state_projection = nn.Linear(config.embedding_dim, config.hidden_size)
        self.subgoal_projection = nn.Linear(config.embedding_dim, config.hidden_size)
        
        # Progress estimation
        self.progress_estimator = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.intermediate_size // 4),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size // 4, 1),
            nn.Sigmoid(),
        )
        
        # Progress-aware policy
        self.classifier = nn.Linear(config.hidden_size * 2 + 1, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self.init_weights()
        
    def forward(
        self,
        state_embedding: Tensor,
        subgoal_embedding: Tensor,
        labels: Optional[Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Tensor]:
        """Predict actions and progress towards subgoal."""
        
        state_proj = self.state_projection(state_embedding)
        subgoal_proj = self.subgoal_projection(subgoal_embedding)
        
        # Estimate progress
        combined_for_progress = torch.cat([state_proj, subgoal_proj], dim=-1)
        progress = self.progress_estimator(combined_for_progress).squeeze(-1)
        
        # Generate actions with progress awareness
        combined_for_policy = torch.cat([state_proj, subgoal_proj, progress.unsqueeze(-1)], dim=-1)
        combined_for_policy = self.dropout(combined_for_policy)
        action_logits = self.classifier(combined_for_policy)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(action_logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': action_logits,
            'progress': progress,
        }


# Configuration class for embedding CLLPs
class EmbeddingCLLPConfig(BertConfig):
    """Configuration for embedding-conditioned CLLPs."""
    
    def __init__(
        self,
        vocab_size=144,  # State vocabulary size
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        embedding_dim=64,  # Embedding space dimension
        num_labels=4,  # Number of actions (e.g., 4 for Sokoban)
        use_attention=True,  # Whether to use cross-attention
        num_hierarchy_levels=2,  # For hierarchical CLLP
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            num_labels=num_labels,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            **kwargs
        )
        self.embedding_dim = embedding_dim
        self.use_attention = use_attention
        self.num_hierarchy_levels = num_hierarchy_levels


# Register with transformers
AutoConfig.register("embedding_cllp", EmbeddingCLLPConfig)


# Convenience functions
def create_embedding_cllp_config(**kwargs) -> EmbeddingCLLPConfig:
    """Create configuration for embedding CLLP."""
    return EmbeddingCLLPConfig(**kwargs)


# Loss functions for embedding CLLP training
def cllp_loss(outputs: Dict[str, Tensor], labels: Tensor) -> Tensor:
    """Standard cross-entropy loss for CLLP training."""
    if 'loss' in outputs and outputs['loss'] is not None:
        return outputs['loss']
    
    logits = outputs['logits']
    return F.cross_entropy(logits, labels)


def progress_aware_cllp_loss(
    outputs: Dict[str, Tensor],
    labels: Tensor,
    target_progress: Optional[Tensor] = None,
    progress_weight: float = 0.1,
) -> Tensor:
    """Combined loss for progress-aware CLLP."""
    action_loss = cllp_loss(outputs, labels)
    
    if 'progress' in outputs and target_progress is not None:
        progress_loss = F.mse_loss(outputs['progress'], target_progress)
        return action_loss + progress_weight * progress_loss
    
    return action_loss
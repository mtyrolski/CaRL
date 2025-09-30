"""Embedding-based subgoal generators using HuggingFace transformers.

This module implements subgoal generators that operate in learned embedding
space using BART as the backbone architecture, following CaRL's established
patterns with HuggingFace models.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    PreTrainedModel,
    BartForConditionalGeneration,
    BartConfig,
    BartModel,
    AutoConfig,
    AutoModel,
)


class HFEmbeddingGenerator(BartForConditionalGeneration):
    """HuggingFace BART-based generator for subgoal embeddings.
    
    Uses BART encoder-decoder architecture to generate subgoal embeddings
    from current state embeddings, following the same pattern as existing
    generators in CaRL.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Additional layers for embedding processing
        self.embedding_projection = nn.Linear(config.d_model, config.embedding_dim)
        self.k_embedding = nn.Embedding(config.max_k_steps, config.embedding_dim // 4)
        
        # Initialize weights
        self.init_weights()
        
    def generate_embedding(
        self,
        current_embedding: Tensor,
        k_steps: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate subgoal embedding from current embedding."""
        batch_size = current_embedding.size(0)
        
        if k_steps is None:
            k_steps = torch.full((batch_size,), 4, 
                               device=current_embedding.device, dtype=torch.long)
        
        # Combine current embedding with k-step information
        k_emb = self.k_embedding(k_steps)
        
        # Use BART encoder to process the combined embedding
        # We'll treat the embedding as a sequence of tokens
        encoder_outputs = self.model.encoder(
            inputs_embeds=current_embedding.unsqueeze(1),  # Add sequence dimension
            attention_mask=attention_mask,
        )
        
        # Generate subgoal embedding using decoder
        decoder_outputs = self.model.decoder(
            inputs_embeds=k_emb.unsqueeze(1),  # k embedding as decoder input
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
        )
        
        # Project to embedding space
        subgoal_embedding = self.embedding_projection(decoder_outputs.last_hidden_state.squeeze(1))
        
        return subgoal_embedding
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        k_steps: Optional[Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """Forward pass for embedding generation."""
        
        # If we're given embeddings directly, use the embedding generation method
        if inputs_embeds is not None and k_steps is not None:
            subgoal_embedding = self.generate_embedding(inputs_embeds, k_steps, attention_mask)
            
            loss = None
            if labels is not None:
                # MSE loss between predicted and target embeddings
                loss = F.mse_loss(subgoal_embedding, labels)
            
            return {
                'loss': loss,
                'embedding': subgoal_embedding,
                'logits': subgoal_embedding,  # For compatibility
            }
        
        # Otherwise, use standard BART forward pass
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            return_dict=return_dict,
        )


class HFTransformerEmbeddingGenerator(PreTrainedModel):
    """Transformer-based embedding generator using BART backbone.
    
    Specialized version that focuses on embedding-to-embedding generation
    with better support for different k-step values.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Use BART model as backbone
        self.bart = BartModel(config)
        
        # Embedding processing layers
        self.input_projection = nn.Linear(config.embedding_dim, config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.embedding_dim)
        
        # K-step conditioning
        self.k_embedding = nn.Embedding(config.max_k_steps, config.d_model)
        
        # Initialize weights
        self.init_weights()
        
    def forward(
        self,
        current_embedding: Tensor,
        k_steps: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Tensor]:
        """Generate subgoal embeddings using transformer."""
        batch_size = current_embedding.size(0)
        
        if k_steps is None:
            k_steps = torch.full((batch_size,), 4, 
                               device=current_embedding.device, dtype=torch.long)
        
        # Project current embedding to model dimension
        current_proj = self.input_projection(current_embedding).unsqueeze(1)
        
        # Get k-step embeddings
        k_emb = self.k_embedding(k_steps).unsqueeze(1)
        
        # Create encoder sequence: [current_embedding, k_embedding]
        encoder_inputs = torch.cat([current_proj, k_emb], dim=1)
        
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, 2, device=current_embedding.device)
        
        # Use BART encoder-decoder
        encoder_outputs = self.bart.encoder(
            inputs_embeds=encoder_inputs,
            attention_mask=attention_mask,
        )
        
        decoder_outputs = self.bart.decoder(
            inputs_embeds=k_emb,  # Start decoding from k embedding
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
        )
        
        # Project back to embedding space
        subgoal_embedding = self.output_projection(decoder_outputs.last_hidden_state.squeeze(1))
        
        loss = None
        if labels is not None:
            # MSE loss for embedding prediction
            loss = F.mse_loss(subgoal_embedding, labels)
        
        return {
            'loss': loss,
            'embedding': subgoal_embedding,
            'logits': subgoal_embedding,
        }


class HFAutoregressiveEmbeddingGenerator(PreTrainedModel):
    """Autoregressive embedding generator using BART for multi-step planning."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # BART backbone
        self.bart = BartModel(config)
        
        # Embedding processing
        self.embedding_projection = nn.Linear(config.embedding_dim, config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.embedding_dim)
        
        # Initialize weights
        self.init_weights()
        
    def forward(
        self,
        current_embedding: Tensor,
        k_steps: Optional[Tensor] = None,
        return_sequence: bool = False,
        labels: Optional[Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Tensor]:
        """Generate subgoal embeddings autoregressively."""
        batch_size = current_embedding.size(0)
        
        if k_steps is None:
            k_steps = torch.full((batch_size,), 4, 
                               device=current_embedding.device, dtype=torch.long)
        
        max_k = min(k_steps.max().item(), getattr(self.config, 'max_sequence_length', 16))
        
        # Project input embedding
        encoder_input = self.embedding_projection(current_embedding).unsqueeze(1)
        
        # Encode
        encoder_outputs = self.bart.encoder(inputs_embeds=encoder_input)
        
        # Generate sequence autoregressively
        generated_embeddings = []
        decoder_input = encoder_input  # Start with encoded current state
        
        for step in range(max_k):
            decoder_outputs = self.bart.decoder(
                inputs_embeds=decoder_input,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
            )
            
            # Project to embedding space
            step_embedding = self.output_projection(decoder_outputs.last_hidden_state[:, -1:, :])
            generated_embeddings.append(step_embedding.squeeze(1))
            
            # Use generated embedding as next input
            decoder_input = torch.cat([decoder_input, step_embedding], dim=1)
        
        if return_sequence:
            final_output = torch.stack(generated_embeddings, dim=1)
        else:
            # Return embedding at k-step for each sample
            final_embeddings = []
            for i, k in enumerate(k_steps):
                k_idx = min(k.item() - 1, len(generated_embeddings) - 1)
                final_embeddings.append(generated_embeddings[k_idx][i])
            final_output = torch.stack(final_embeddings)
        
        loss = None
        if labels is not None:
            if return_sequence:
                loss = F.mse_loss(final_output, labels)
            else:
                loss = F.mse_loss(final_output, labels)
        
        return {
            'loss': loss,
            'embedding': final_output,
            'logits': final_output,
        }


# Configuration class for embedding generators
class EmbeddingGeneratorConfig(BartConfig):
    """Configuration for embedding generators."""
    
    def __init__(
        self,
        vocab_size=144,  # State vocabulary size
        d_model=512,
        encoder_layers=4,
        decoder_layers=4,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        embedding_dim=64,  # Embedding space dimension
        max_k_steps=32,  # Maximum k steps supported
        max_sequence_length=16,  # For autoregressive generation
        dropout=0.1,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            decoder_attention_heads=decoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            decoder_ffn_dim=decoder_ffn_dim,
            dropout=dropout,
            **kwargs
        )
        self.embedding_dim = embedding_dim
        self.max_k_steps = max_k_steps
        self.max_sequence_length = max_sequence_length


# Register with transformers
AutoConfig.register("embedding_generator", EmbeddingGeneratorConfig)


# Convenience functions
def create_embedding_generator_config(**kwargs) -> EmbeddingGeneratorConfig:
    """Create configuration for embedding generator."""
    return EmbeddingGeneratorConfig(**kwargs)


# Loss functions for embedding generators
def embedding_generator_loss(
    outputs: Dict[str, Tensor], 
    labels: Tensor,
    embedding_regularization: float = 0.01,
) -> Tensor:
    """Loss function for embedding generator training."""
    if 'loss' in outputs and outputs['loss'] is not None:
        return outputs['loss']
        
    predicted_embedding = outputs['embedding']
    
    # Primary loss: MSE between predicted and target embeddings
    mse_loss = F.mse_loss(predicted_embedding, labels)
    
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
    """Contrastive loss for embedding generator training."""
    # Normalize embeddings
    pred_norm = F.normalize(predicted_embedding, dim=-1)
    pos_norm = F.normalize(positive_embedding, dim=-1)
    neg_norm = F.normalize(negative_embeddings, dim=-1)
    
    # Compute similarities
    pos_sim = torch.sum(pred_norm * pos_norm, dim=-1) / temperature
    neg_sim = torch.matmul(pred_norm, neg_norm.transpose(-2, -1)) / temperature
    
    # Contrastive loss
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    return F.cross_entropy(logits, targets)
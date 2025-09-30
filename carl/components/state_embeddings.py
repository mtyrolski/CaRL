"""State embedding models using HuggingFace transformers for hierarchical latent space search.

This module implements Autoencoder (AE) and Variational Autoencoder (VAE)
models using HuggingFace transformer backbones for learning compressed latent 
embeddings of game states, particularly for Sokoban domain.
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    PreTrainedModel, 
    BertModel, 
    BertConfig,
    AutoModel,
    AutoConfig
)


class HFStateAutoencoder(PreTrainedModel):
    """HuggingFace-based autoencoder for state embeddings.
    
    Uses BERT encoder as the backbone and adds a decoder head for reconstruction.
    This follows the HuggingFace patterns used throughout CaRL.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Use BERT encoder as backbone
        self.encoder = BertModel(config, add_pooling_layer=False)
        
        # Decoder layers to reconstruct from hidden states
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.vocab_size),
        )
        
        # Initialize weights
        self.init_weights()
        
    def encode(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Encode input to embedding using BERT encoder."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token embedding as state embedding
        return outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
    def decode(self, embeddings: Tensor) -> Tensor:
        """Decode embeddings back to token space."""
        return self.decoder(embeddings)
        
    def forward(
        self, 
        input_ids: Tensor, 
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass returning reconstruction and embedding."""
        # Encode to embedding
        embedding = self.encode(input_ids, attention_mask)
        
        # Decode back to token space
        reconstruction_logits = self.decode(embedding)
        
        loss = None
        if labels is not None:
            # Reconstruction loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reconstruction_logits.view(-1, self.config.vocab_size), 
                          labels.view(-1))
        
        return {
            'loss': loss,
            'logits': reconstruction_logits,
            'embedding': embedding,
            'reconstruction': reconstruction_logits,
        }


class HFStateVAE(PreTrainedModel):
    """HuggingFace-based Variational Autoencoder for state embeddings.
    
    Uses BERT encoder as backbone with additional VAE components.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Use BERT encoder as backbone
        self.encoder = BertModel(config, add_pooling_layer=False)
        
        # VAE components
        self.mu_head = nn.Linear(config.hidden_size, config.latent_dim)
        self.logvar_head = nn.Linear(config.hidden_size, config.latent_dim)
        
        # Decoder from latent space back to token space
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.vocab_size),
        )
        
        # KL weight
        self.beta = getattr(config, 'beta', 1.0)
        
        # Initialize weights
        self.init_weights()
        
    def encode(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Encode input to mean and log variance."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        mu = self.mu_head(hidden_state)
        logvar = self.logvar_head(hidden_state)
        
        return mu, logvar
        
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z: Tensor) -> Tensor:
        """Decode latent embedding back to token space."""
        return self.decoder(z)
        
    def forward(
        self, 
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass returning reconstruction, embedding, and KL divergence."""
        # Encode to mean and log variance
        mu, logvar = self.encode(input_ids, attention_mask)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode back to token space
        reconstruction_logits = self.decode(z)
        
        # Compute KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        loss = None
        if labels is not None:
            # Reconstruction loss
            reconstruction_loss = F.cross_entropy(
                reconstruction_logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                reduction='none'
            ).view(labels.shape[0], -1).mean(dim=1)
            
            # Total VAE loss
            loss = reconstruction_loss + self.beta * kl_div
            loss = loss.mean()
        
        return {
            'loss': loss,
            'logits': reconstruction_logits,
            'embedding': z,
            'reconstruction': reconstruction_logits,
            'mu': mu,
            'logvar': logvar,
            'kl_div': kl_div,
        }


# Configuration classes for the custom models
class StateEmbeddingConfig(BertConfig):
    """Configuration for state embedding models."""
    
    def __init__(
        self,
        vocab_size=144,  # Flattened state size (12x12 for Sokoban)
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_dropout_prob=0.1,
        latent_dim=64,  # For VAE
        beta=1.0,  # KL weight for VAE
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            **kwargs
        )
        self.latent_dim = latent_dim
        self.beta = beta


# Register the models with transformers
from transformers import AutoConfig, AutoModel

AutoConfig.register("state_embedding", StateEmbeddingConfig)
AutoModel.register(StateEmbeddingConfig, HFStateAutoencoder)


# Convenience functions for creating models
def create_state_autoencoder_config(**kwargs) -> StateEmbeddingConfig:
    """Create configuration for state autoencoder."""
    return StateEmbeddingConfig(**kwargs)


def create_state_vae_config(**kwargs) -> StateEmbeddingConfig:
    """Create configuration for state VAE."""
    return StateEmbeddingConfig(**kwargs)


# Loss functions for training
def autoencoder_loss(outputs: Dict[str, Tensor], labels: Tensor) -> Tensor:
    """Standard reconstruction loss for autoencoder."""
    if 'loss' in outputs and outputs['loss'] is not None:
        return outputs['loss']
    
    logits = outputs['logits']
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))


def vae_loss(outputs: Dict[str, Tensor], labels: Tensor, beta: float = 1.0) -> Tensor:
    """VAE loss combining reconstruction and KL divergence."""
    if 'loss' in outputs and outputs['loss'] is not None:
        return outputs['loss']
    
    logits = outputs['logits']
    kl_div = outputs['kl_div']
    
    reconstruction_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        labels.view(-1),
        reduction='none'
    ).view(labels.shape[0], -1).mean(dim=1)
    
    total_loss = reconstruction_loss + beta * kl_div
    return total_loss.mean()
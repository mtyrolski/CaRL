"""State embedding models for hierarchical latent space search.

This module implements Autoencoder (AE) and Variational Autoencoder (VAE)
models for learning compressed latent embeddings of game states, particularly
for Sokoban domain. These embeddings enable hierarchical planning in continuous
latent space rather than discrete state space.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaseStateEmbedding(nn.Module, ABC):
    """Base class for state embedding models."""
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [512, 256]
        
    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """Encode state to embedding."""
        raise NotImplementedError
        
    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """Decode embedding back to state."""
        raise NotImplementedError
        
    @abstractmethod
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass returning reconstruction and other outputs."""
        raise NotImplementedError


class StateAutoencoder(BaseStateEmbedding):
    """Standard autoencoder for state embeddings.
    
    Learns a deterministic mapping from states to latent embeddings
    that can be used for hierarchical planning.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(input_dim, embedding_dim, hidden_dims)
        self.dropout = dropout
        
        # Build encoder
        encoder_layers = []
        in_dim = input_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        encoder_layers.append(nn.Linear(in_dim, embedding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder  
        decoder_layers = []
        in_dim = embedding_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: Tensor) -> Tensor:
        """Encode state to embedding."""
        return self.encoder(x)
        
    def decode(self, z: Tensor) -> Tensor:
        """Decode embedding back to state."""
        return self.decoder(z)
        
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass returning reconstruction."""
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return {
            'reconstruction': reconstruction,
            'embedding': embedding,
        }


class StateVAE(BaseStateEmbedding):
    """Variational autoencoder for state embeddings.
    
    Learns a probabilistic mapping from states to latent embeddings
    with regularization via KL divergence to prior distribution.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
        beta: float = 1.0,
    ) -> None:
        super().__init__(input_dim, embedding_dim, hidden_dims)
        self.dropout = dropout
        self.beta = beta  # KL divergence weight
        
        # Build encoder
        encoder_layers = []
        in_dim = input_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.encoder_backbone = nn.Sequential(*encoder_layers)
        
        # Mean and log variance heads
        self.mu_head = nn.Linear(in_dim, embedding_dim)
        self.logvar_head = nn.Linear(in_dim, embedding_dim)
        
        # Build decoder
        decoder_layers = []
        in_dim = embedding_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode state to mean and log variance."""
        features = self.encoder_backbone(x)
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        return mu, logvar
        
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z: Tensor) -> Tensor:
        """Decode embedding back to state."""
        return self.decoder(z)
        
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass returning reconstruction and KL divergence."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        return {
            'reconstruction': reconstruction,
            'embedding': z,
            'mu': mu,
            'logvar': logvar,
            'kl_div': kl_div,
        }


class Conv2DStateAutoencoder(BaseStateEmbedding):
    """Convolutional autoencoder for 2D state representations like Sokoban boards.
    
    Specifically designed for spatial game states that can benefit from
    convolutional processing rather than fully connected layers.
    """
    
    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        embedding_dim: int,
        conv_channels: Optional[list[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        # Calculate flattened input dimension for compatibility
        input_dim = input_channels * input_height * input_width
        super().__init__(input_dim, embedding_dim)
        
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.conv_channels = conv_channels or [32, 64, 128]
        self.dropout = dropout
        
        # Build convolutional encoder
        encoder_layers = []
        in_channels = input_channels
        for out_channels in self.conv_channels:
            encoder_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout),
            ])
            in_channels = out_channels
            
        self.conv_encoder = nn.Sequential(*encoder_layers)
        
        # Calculate feature size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            conv_output = self.conv_encoder(dummy_input)
            self.feature_size = conv_output.numel()
            self.conv_output_shape = conv_output.shape[1:]
            
        # Fully connected encoder
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )
        
        # Fully connected decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, self.feature_size),
            nn.ReLU(inplace=True),
        )
        
        # Build convolutional decoder
        decoder_layers = []
        in_channels = self.conv_channels[-1]
        for out_channels in reversed(self.conv_channels[:-1]):
            decoder_layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
            ])
            in_channels = out_channels
            
        # Final layer to get back to input channels
        decoder_layers.append(
            nn.ConvTranspose2d(
                in_channels, input_channels, 
                kernel_size=3, stride=2, padding=1, output_padding=1
            )
        )
        self.conv_decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: Tensor) -> Tensor:
        """Encode 2D state to embedding."""
        # Reshape if input is flattened
        if x.dim() == 2:
            x = x.view(-1, self.input_channels, self.input_height, self.input_width)
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
            
        conv_features = self.conv_encoder(x)
        flattened = conv_features.view(conv_features.size(0), -1)
        embedding = self.fc_encoder(flattened)
        return embedding
        
    def decode(self, z: Tensor) -> Tensor:
        """Decode embedding back to 2D state."""
        fc_output = self.fc_decoder(z)
        conv_input = fc_output.view(-1, *self.conv_output_shape)
        reconstruction_2d = self.conv_decoder(conv_input)
        
        # Crop or pad to match input size
        if reconstruction_2d.shape[-2:] != (self.input_height, self.input_width):
            reconstruction_2d = F.interpolate(
                reconstruction_2d, 
                size=(self.input_height, self.input_width), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Return flattened for compatibility with existing pipeline
        return reconstruction_2d.view(reconstruction_2d.size(0), -1) 
        
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass returning reconstruction."""
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return {
            'reconstruction': reconstruction,
            'embedding': embedding,
        }


# Loss functions for training embeddings
def autoencoder_loss(outputs: dict[str, Tensor], targets: Tensor) -> Tensor:
    """Standard reconstruction loss for autoencoder."""
    reconstruction = outputs['reconstruction']
    return F.mse_loss(reconstruction, targets)


def vae_loss(outputs: dict[str, Tensor], targets: Tensor, beta: float = 1.0) -> Tensor:
    """VAE loss combining reconstruction and KL divergence."""
    reconstruction = outputs['reconstruction']
    kl_div = outputs['kl_div']
    
    reconstruction_loss = F.mse_loss(reconstruction, targets)
    kl_loss = torch.mean(kl_div)
    
    return reconstruction_loss + beta * kl_loss
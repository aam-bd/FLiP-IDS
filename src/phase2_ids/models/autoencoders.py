"""
Cosine-Targeted Autoencoder (CT-AE) for privacy-preserving feature encoding.

Implements the CT-AE architecture from the SOH-FL paper with a loss function
combining reconstruction error and cosine similarity terms. This enables
similarity-based aggregation while preserving privacy by sharing only
low-dimensional latent representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from ...common.logging import get_logger

logger = get_logger(__name__)


class CosineTargetedAutoencoder(nn.Module):
    """
    Cosine-Targeted Autoencoder for privacy-preserving feature encoding.
    
    The CT-AE loss combines reconstruction error with cosine similarity:
    loss = w_rec * MSE(recon, x) + w_cos * (1 - cosine(z, z_target))
    
    This ensures that encoded vectors retain directional information
    crucial for similarity-based matching in federated learning.
    """
    
    def __init__(self,
                 input_dim: int = 35,
                 latent_dim: int = 32,
                 hidden_dims: List[int] = [128, 64],
                 activation: str = 'relu',
                 dropout_rate: float = 0.2):
        """
        Initialize CT-AE.
        
        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Encoder layers
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        self.encoder_layers = nn.ModuleList()
        
        for i in range(len(encoder_dims) - 1):
            self.encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            if i < len(encoder_dims) - 2:  # No dropout on latent layer
                self.encoder_layers.append(nn.Dropout(dropout_rate))
        
        # Decoder layers (symmetric to encoder)
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        self.decoder_layers = nn.ModuleList()
        
        for i in range(len(decoder_dims) - 1):
            self.decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i < len(decoder_dims) - 2:  # No dropout on output layer
                self.decoder_layers.append(nn.Dropout(dropout_rate))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        h = x
        
        for i, layer in enumerate(self.encoder_layers):
            if isinstance(layer, nn.Linear):
                h = layer(h)
                if i < len(self.encoder_layers) - 1:  # No activation on final layer
                    h = self.activation(h)
            else:  # Dropout layer
                h = layer(h)
        
        return h
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent space to input space.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed input of shape (batch_size, input_dim)
        """
        h = z
        
        for i, layer in enumerate(self.decoder_layers):
            if isinstance(layer, nn.Linear):
                h = layer(h)
                if i < len(self.decoder_layers) - 1:  # No activation on output layer
                    h = self.activation(h)
            else:  # Dropout layer
                h = layer(h)
        
        return h
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstructed_input, latent_representation)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict:
        """Get model architecture information."""
        return {
            'model_type': 'CosineTargetedAutoencoder',
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'total_parameters': self.count_parameters()
        }


class CTAELoss(nn.Module):
    """
    Cosine-Targeted Autoencoder loss function.
    
    Combines reconstruction loss with cosine similarity loss:
    loss = w_rec * reconstruction_loss + w_cos * cosine_similarity_loss
    """
    
    def __init__(self, w_rec: float = 0.7, w_cos: float = 0.3):
        """
        Initialize CT-AE loss.
        
        Args:
            w_rec: Weight for reconstruction loss
            w_cos: Weight for cosine similarity loss
        """
        super().__init__()
        self.w_rec = w_rec
        self.w_cos = w_cos
        
        self.mse_loss = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
    
    def forward(self, 
                x_input: torch.Tensor,
                x_recon: torch.Tensor,
                z_latent: torch.Tensor,
                z_target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CT-AE loss.
        
        Args:
            x_input: Original input
            x_recon: Reconstructed input
            z_latent: Latent representation
            z_target: Target latent representation for cosine similarity
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Reconstruction loss
        recon_loss = self.mse_loss(x_recon, x_input)
        
        # Cosine similarity loss
        if z_target is not None:
            # Cosine similarity between current and target latent vectors
            cos_sim = self.cosine_similarity(z_latent, z_target)
            cos_loss = torch.mean(1 - cos_sim)  # Minimize (1 - cosine_similarity)
        else:
            # If no target, encourage unit norm (alternative approach)
            z_norm = F.normalize(z_latent, p=2, dim=1)
            cos_loss = torch.mean(torch.sum((z_latent - z_norm) ** 2, dim=1))
        
        # Combined loss
        total_loss = self.w_rec * recon_loss + self.w_cos * cos_loss
        
        loss_components = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'cosine_loss': cos_loss.item()
        }
        
        return total_loss, loss_components


class CTAETrainer:
    """Training utilities for Cosine-Targeted Autoencoder."""
    
    def __init__(self, 
                 model: CosineTargetedAutoencoder,
                 device: torch.device,
                 w_rec: float = 0.7,
                 w_cos: float = 0.3):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.criterion = CTAELoss(w_rec=w_rec, w_cos=w_cos)
        self.logger = logger
    
    def train_epoch(self,
                   train_loader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   use_cosine_target: bool = True) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            use_cosine_target: Whether to use cosine targeting
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_cos_loss = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(train_loader):
            if isinstance(data, (list, tuple)):
                data = data[0]  # Take only input data, ignore labels
            
            data = data.to(self.device)
            batch_size = data.size(0)
            
            optimizer.zero_grad()
            
            # Forward pass
            x_recon, z_latent = self.model(data)
            
            # Create target for cosine similarity (optional)
            z_target = None
            if use_cosine_target and batch_size > 1:
                # Use a shifted version of the batch as target
                # This encourages similar patterns to have similar encodings
                indices = torch.randperm(batch_size)
                z_target = z_latent[indices]
            
            # Compute loss
            loss, loss_components = self.criterion(data, x_recon, z_latent, z_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss_components['total_loss']
            total_recon_loss += loss_components['reconstruction_loss']
            total_cos_loss += loss_components['cosine_loss']
            num_batches += 1
        
        metrics = {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': total_recon_loss / num_batches,
            'cosine_loss': total_cos_loss / num_batches
        }
        
        return metrics
    
    def evaluate(self,
                test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_cos_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data in test_loader:
                if isinstance(data, (list, tuple)):
                    data = data[0]
                
                data = data.to(self.device)
                
                # Forward pass
                x_recon, z_latent = self.model(data)
                
                # Compute loss (without cosine target for evaluation)
                loss, loss_components = self.criterion(data, x_recon, z_latent, None)
                
                total_loss += loss_components['total_loss']
                total_recon_loss += loss_components['reconstruction_loss']
                total_cos_loss += loss_components['cosine_loss']
                num_batches += 1
        
        metrics = {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': total_recon_loss / num_batches,
            'cosine_loss': total_cos_loss / num_batches
        }
        
        return metrics
    
    def encode_data(self, data_loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Encode data to latent space.
        
        Args:
            data_loader: Data loader
            
        Returns:
            Encoded representations
        """
        self.model.eval()
        encodings = []
        
        with torch.no_grad():
            for data in data_loader:
                if isinstance(data, (list, tuple)):
                    data = data[0]
                
                data = data.to(self.device)
                z = self.model.encode(data)
                encodings.append(z.cpu().numpy())
        
        return np.vstack(encodings)
    
    def compute_similarity_matrix(self, 
                                encodings1: np.ndarray,
                                encodings2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute cosine similarity matrix between encodings.
        
        Args:
            encodings1: First set of encodings
            encodings2: Second set of encodings (optional)
            
        Returns:
            Similarity matrix
        """
        if encodings2 is None:
            encodings2 = encodings1
        
        # Normalize encodings
        encodings1_norm = encodings1 / (np.linalg.norm(encodings1, axis=1, keepdims=True) + 1e-8)
        encodings2_norm = encodings2 / (np.linalg.norm(encodings2, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(encodings1_norm, encodings2_norm.T)
        
        return similarity_matrix


def create_ctae_model(config: Dict) -> CosineTargetedAutoencoder:
    """
    Factory function to create CT-AE model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        CT-AE instance
    """
    return CosineTargetedAutoencoder(
        input_dim=config.get('input_dim', 35),
        latent_dim=config.get('latent_dim', 32),
        hidden_dims=config.get('hidden_dims', [128, 64]),
        activation=config.get('activation', 'relu'),
        dropout_rate=config.get('dropout_rate', 0.2)
    )


# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    model = CosineTargetedAutoencoder(input_dim=35, latent_dim=16)
    
    print(f"Model info: {model.get_model_info()}")
    
    # Test forward pass
    batch_size = 8
    input_dim = 35
    x = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        x_recon, z = model(x)
        z_only = model.encode(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_recon.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction error: {F.mse_loss(x, x_recon).item():.6f}")
    
    # Test loss function
    criterion = CTAELoss(w_rec=0.7, w_cos=0.3)
    loss, components = criterion(x, x_recon, z, z[torch.randperm(batch_size)])
    print(f"Loss components: {components}")

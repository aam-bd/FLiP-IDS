"""
1D CNN classifier for intrusion detection.

Implements a simple but effective 1D CNN architecture suitable for tabular
sequential data and per-flow feature vectors as used in the SOH-FL experiments.
The model is designed to be lightweight yet effective for federated learning scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from ...common.logging import get_logger

logger = get_logger(__name__)


class CNN1DBlock(nn.Module):
    """Basic 1D CNN block with convolution, batch norm, and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class CNN1DClassifier(nn.Module):
    """
    1D CNN for intrusion detection classification.
    
    Architecture designed for feature vectors from Phase 1 profiling,
    suitable for federated learning with MAML meta-learning.
    """
    
    def __init__(self, 
                 input_dim: int = 35,
                 hidden_channels: List[int] = [64, 128, 64],
                 kernel_size: int = 3,
                 dropout: float = 0.3,
                 num_classes: int = 10):
        """
        Initialize 1D CNN classifier.
        
        Args:
            input_dim: Input feature dimension (e.g., 35 selected features)
            hidden_channels: List of channel sizes for conv layers
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            num_classes: Number of output classes (attack types + benign)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        
        # Input projection to create sequence-like data
        # Transform (batch, features) to (batch, channels, sequence_length)
        self.input_projection = nn.Linear(input_dim, hidden_channels[0] * 8)
        self.sequence_length = 8
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        
        for i in range(len(hidden_channels) - 1):
            self.conv_layers.append(
                CNN1DBlock(
                    hidden_channels[i], 
                    hidden_channels[i + 1],
                    kernel_size=kernel_size,
                    dropout=dropout
                )
            )
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1] // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Input validation
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.size(1)}")
        
        # Project to sequence-like representation
        x = self.input_projection(x)  # (batch, hidden_channels[0] * sequence_length)
        x = x.view(batch_size, self.hidden_channels[0], self.sequence_length)
        
        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_feature_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature embeddings before classification layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature embeddings
        """
        batch_size = x.size(0)
        
        # Forward through CNN layers
        x = self.input_projection(x)
        x = x.view(batch_size, self.hidden_channels[0], self.sequence_length)
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict:
        """Get model architecture information."""
        return {
            'model_type': 'CNN1DClassifier',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'hidden_channels': self.hidden_channels,
            'sequence_length': self.sequence_length,
            'total_parameters': self.count_parameters()
        }


class CNN1DTrainer:
    """Training utilities for CNN1D classifier."""
    
    def __init__(self, model: CNN1DClassifier, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.logger = logger
    
    def train_epoch(self, 
                   train_loader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': 100.0 * correct / total
        }
        
        return metrics
    
    def evaluate(self,
                test_loader: torch.utils.data.DataLoader,
                criterion: nn.Module) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            test_loader: Test data loader
            criterion: Loss function
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        metrics = {
            'loss': total_loss / len(test_loader),
            'accuracy': 100.0 * correct / total
        }
        
        return metrics
    
    def predict(self, data_loader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on data.
        
        Args:
            data_loader: Data loader
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                
                outputs = self.model(data)
                probs = F.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)


def create_cnn1d_model(config: Dict) -> CNN1DClassifier:
    """
    Factory function to create CNN1D model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        CNN1D classifier instance
    """
    return CNN1DClassifier(
        input_dim=config.get('input_dim', 35),
        hidden_channels=config.get('hidden_channels', [64, 128, 64]),
        kernel_size=config.get('kernel_size', 3),
        dropout=config.get('dropout', 0.3),
        num_classes=config.get('num_classes', 10)
    )


# Example usage and testing
if __name__ == "__main__":
    # Test model creation and forward pass
    model = CNN1DClassifier(input_dim=35, num_classes=5)
    
    print(f"Model info: {model.get_model_info()}")
    
    # Test forward pass
    batch_size = 16
    input_dim = 35
    x = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        outputs = model(x)
        embeddings = model.get_feature_embeddings(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")

"""
Phase 2: Self-Labeled Personalized Federated Learning IDS

This module implements the second phase of the IoT security framework,
focusing on collaborative intrusion detection using the SOH-FL approach
with meta-learning, cosine-targeted autoencoders, and similarity-based aggregation.

Key components:
- CNN-1D models for intrusion detection
- Cosine-Targeted Autoencoder (CT-AE) for privacy-preserving feature encoding
- MAML utilities for meta-learning and personalization
- Federated learning server and client implementations
- Self-labeling workflow with similarity-based aggregation (BS-Agg)
"""

from .models.cnn_1d import CNN1DClassifier
from .models.autoencoders import CosineTargetedAutoencoder
from .models.maml import MAMLTrainer, MAMLOptimizer
from .federation.server import FederatedServer
from .federation.client import FederatedClient
from .federation.data_pipe import DataPipeline

__all__ = [
    "CNN1DClassifier",
    "CosineTargetedAutoencoder", 
    "MAMLTrainer",
    "MAMLOptimizer",
    "FederatedServer",
    "FederatedClient",
    "DataPipeline"
]

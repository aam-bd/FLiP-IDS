"""
Federated learning client implementing the SOH-FL gateway simulation.

Each client represents an IoT gateway managing multiple IoT devices. The client:
- Holds Phase 1 labeled historical data as local training set
- Splits data into support/query sets for meta-learning
- Encodes data using CT-AE for privacy-preserving similarity computation
- Receives custom annotation models from server via BS-Agg
- Performs local adaptation and intrusion detection
"""

import copy
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict
from dataclasses import dataclass

from ..models.cnn_1d import CNN1DClassifier, CNN1DTrainer
from ..models.autoencoders import CosineTargetedAutoencoder, CTAETrainer
from ..models.maml import MAMLTrainer, create_support_query_split
from .data_pipe import IoTFlowDataset
from ...common.logging import get_logger, MetricsLogger
from ...common.metrics import calculate_metrics
from ...common.utils import Timer, set_seed

logger = get_logger(__name__)


@dataclass
class ClientConfig:
    """Configuration for federated client."""
    client_id: str
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.005
    
    # MAML parameters
    maml_inner_lr: float = 0.001
    maml_inner_steps: int = 1
    support_ratio: float = 0.6
    
    # Adaptation parameters
    adaptation_steps: int = 5
    adaptation_lr: float = 0.01
    
    # CT-AE parameters
    ct_ae_epochs: int = 5
    ct_ae_lr: float = 0.001


class FederatedClient:
    """
    Federated learning client for SOH-FL implementation.
    
    Simulates an IoT gateway that:
    1. Manages local IoT device data from Phase 1
    2. Participates in federated learning with MAML
    3. Uses CT-AE for privacy-preserving encoding
    4. Performs self-labeling with server-provided annotation models
    5. Adapts locally for personalized intrusion detection
    """
    
    def __init__(self,
                 client_id: str,
                 model_config: Dict,
                 client_config: ClientConfig,
                 device: torch.device):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            model_config: CNN1D model configuration
            client_config: Client configuration
            device: Computing device
        """
        self.client_id = client_id
        self.config = client_config
        self.device = device
        
        # Initialize local model
        self.local_model = CNN1DClassifier(**model_config)
        self.local_model.to(device)
        self.model_trainer = CNN1DTrainer(self.local_model, device)
        
        # Initialize CT-AE
        self.ct_ae = CosineTargetedAutoencoder(
            input_dim=model_config.get('input_dim', 35),
            latent_dim=32  # Fixed latent dimension
        )
        self.ct_ae.to(device)
        self.ct_ae_trainer = CTAETrainer(self.ct_ae, device)
        
        # Initialize MAML trainer
        self.maml_trainer = MAMLTrainer(
            self.local_model, device,
            inner_lr=client_config.maml_inner_lr,
            inner_steps=client_config.maml_inner_steps
        )
        
        # Data storage
        self.local_data: Dict[str, Any] = {}
        self.data_loaders: Dict[str, DataLoader] = {}
        
        # Training history
        self.training_history: List[Dict] = []
        self.round_number = 0
        
        # Encodings for similarity computation
        self.historical_encoding: Optional[np.ndarray] = None
        self.current_encoding: Optional[np.ndarray] = None
        
        # Metrics tracking
        self.logger = logger
        self.metrics_logger = MetricsLogger(logger, f"Client_{client_id}")
        
        set_seed(42)
    
    def load_local_data(self, data_dict: Dict[str, Any]):
        """
        Load local data from Phase 1 profiling results.
        
        Args:
            data_dict: Dictionary containing training data splits
        """
        self.local_data = data_dict
        
        # Create datasets and data loaders
        self.data_loaders = {}
        
        for split_name in ['support', 'query', 'test']:
            if f'{split_name}_features' in data_dict and f'{split_name}_labels' in data_dict:
                features = data_dict[f'{split_name}_features']
                labels = data_dict[f'{split_name}_labels']
                
                if isinstance(features, pd.DataFrame):
                    features = features.values
                if isinstance(labels, pd.Series):
                    labels = labels.values
                
                dataset = IoTFlowDataset(features, labels)
                loader = DataLoader(
                    dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    drop_last=False
                )
                self.data_loaders[split_name] = loader
        
        # Log data statistics
        total_samples = sum(len(loader.dataset) for loader in self.data_loaders.values())
        self.logger.info(f"Client {self.client_id} loaded {total_samples} samples")
        for split, loader in self.data_loaders.items():
            self.logger.info(f"  {split}: {len(loader.dataset)} samples")
    
    def update_global_model(self, global_model_state: OrderedDict):
        """
        Update local model with global model parameters.
        
        Args:
            global_model_state: Global model state dict
        """
        self.local_model.load_state_dict(global_model_state)
        self.logger.info(f"Client {self.client_id} updated with global model")
    
    def train_local_epochs(self, num_epochs: Optional[int] = None) -> Dict[str, float]:
        """
        Train local model for specified epochs.
        
        Args:
            num_epochs: Number of local epochs (default: config value)
            
        Returns:
            Training metrics
        """
        if num_epochs is None:
            num_epochs = self.config.local_epochs
        
        if 'support' not in self.data_loaders:
            raise ValueError("No support data available for training")
        
        optimizer = optim.Adam(self.local_model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        epoch_metrics = []
        
        with Timer(f"Local training ({num_epochs} epochs)", self.logger):
            for epoch in range(num_epochs):
                # Train one epoch
                train_metrics = self.model_trainer.train_epoch(
                    self.data_loaders['support'], optimizer, criterion
                )
                
                # Evaluate on query set if available
                if 'query' in self.data_loaders:
                    val_metrics = self.model_trainer.evaluate(
                        self.data_loaders['query'], criterion
                    )
                    train_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
                
                epoch_metrics.append(train_metrics)
                
                self.metrics_logger.log_training_summary(
                    epoch, train_metrics['loss'],
                    train_metrics.get('val_loss'),
                    {k: v for k, v in train_metrics.items() if k.startswith('val_')}
                )
        
        # Return average metrics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
        
        return avg_metrics
    
    def train_meta_learning(self) -> Dict[str, float]:
        """
        Train using MAML meta-learning approach.
        
        Returns:
            Meta-learning metrics
        """
        if 'support' not in self.data_loaders or 'query' not in self.data_loaders:
            raise ValueError("Both support and query data required for meta-learning")
        
        # Create task batches for meta-learning
        support_loaders = [self.data_loaders['support']]
        query_loaders = [self.data_loaders['query']]
        
        criterion = nn.CrossEntropyLoss()
        
        with Timer("Meta-learning training", self.logger):
            metrics = self.maml_trainer.train_episode(
                support_loaders, query_loaders, criterion
            )
        
        self.metrics_logger.log_metrics(metrics, step=self.round_number)
        return metrics
    
    def train_ct_ae(self) -> Dict[str, float]:
        """
        Train CT-AE on local data.
        
        Returns:
            CT-AE training metrics
        """
        if 'support' not in self.data_loaders:
            raise ValueError("No support data available for CT-AE training")
        
        optimizer = optim.Adam(self.ct_ae.parameters(), lr=self.config.ct_ae_lr)
        
        epoch_metrics = []
        
        with Timer("CT-AE training", self.logger):
            for epoch in range(self.config.ct_ae_epochs):
                train_metrics = self.ct_ae_trainer.train_epoch(
                    self.data_loaders['support'], optimizer, use_cosine_target=True
                )
                epoch_metrics.append(train_metrics)
        
        # Return average metrics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
        
        return avg_metrics
    
    def encode_data(self, data_type: str = 'support') -> np.ndarray:
        """
        Encode local data using CT-AE.
        
        Args:
            data_type: Type of data to encode ('support', 'query', 'test')
            
        Returns:
            Encoded representations
        """
        if data_type not in self.data_loaders:
            raise ValueError(f"No {data_type} data available")
        
        encodings = self.ct_ae_trainer.encode_data(self.data_loaders[data_type])
        
        # Store encodings for similarity computation
        if data_type == 'support':
            self.historical_encoding = np.mean(encodings, axis=0)  # Average encoding
        elif data_type == 'test':
            self.current_encoding = np.mean(encodings, axis=0)
        
        self.logger.info(f"Encoded {data_type} data: {encodings.shape}")
        return encodings
    
    def get_encodings_for_server(self) -> Dict[str, np.ndarray]:
        """
        Get encodings to send to server for similarity computation.
        
        Returns:
            Dictionary with historical and current encodings
        """
        encodings = {}
        
        if self.historical_encoding is not None:
            encodings['historical'] = self.historical_encoding
        
        if self.current_encoding is not None:
            encodings['current'] = self.current_encoding
        
        return encodings
    
    def apply_annotation_model(self, annotation_model_state: OrderedDict) -> Dict[str, Any]:
        """
        Apply server-provided annotation model to prelabel unlabeled data.
        
        Args:
            annotation_model_state: Annotation model state dict from server
            
        Returns:
            Prelabeling results and metrics
        """
        # Create temporary model with annotation weights
        annotation_model = CNN1DClassifier(**{
            'input_dim': self.local_model.input_dim,
            'num_classes': self.local_model.num_classes
        })
        annotation_model.load_state_dict(annotation_model_state)
        annotation_model.to(self.device)
        annotation_model.eval()
        
        # Prelabel test data (representing unlabeled support set as_i)
        if 'test' not in self.data_loaders:
            raise ValueError("No test data available for prelabeling")
        
        predictions = []
        probabilities = []
        confidence_scores = []
        
        with torch.no_grad():
            for data, _ in self.data_loaders['test']:
                data = data.to(self.device)
                
                logits = annotation_model(data)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Calculate confidence (max probability)
                max_probs, _ = torch.max(probs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
                confidence_scores.extend(max_probs.cpu().numpy())
        
        prelabel_results = {
            'predictions': np.array(predictions),
            'probabilities': np.array(probabilities),
            'confidence_scores': np.array(confidence_scores),
            'num_prelabeled': len(predictions),
            'avg_confidence': np.mean(confidence_scores)
        }
        
        self.logger.info(f"Prelabeled {len(predictions)} samples with avg confidence {prelabel_results['avg_confidence']:.3f}")
        
        return prelabel_results
    
    def adapt_with_prelabels(self, prelabel_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Adapt local model using prelabeled data.
        
        Args:
            prelabel_results: Results from prelabeling step
            
        Returns:
            Adaptation metrics
        """
        # Filter high-confidence prelabels
        confidence_threshold = 0.7
        high_conf_mask = prelabel_results['confidence_scores'] >= confidence_threshold
        
        if not np.any(high_conf_mask):
            self.logger.warning("No high-confidence prelabels available")
            return {'adaptation_loss': float('inf'), 'num_adapted_samples': 0}
        
        # Create adaptation dataset from high-confidence prelabels
        test_features = []
        test_labels = []
        
        for data, _ in self.data_loaders['test']:
            test_features.append(data.numpy())
        
        test_features = np.vstack(test_features)
        
        # Filter by confidence
        adapted_features = test_features[high_conf_mask]
        adapted_labels = prelabel_results['predictions'][high_conf_mask]
        
        # Create adaptation dataset
        adaptation_dataset = IoTFlowDataset(adapted_features, adapted_labels)
        adaptation_loader = DataLoader(
            adaptation_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Perform local adaptation
        optimizer = optim.SGD(
            self.local_model.parameters(),
            lr=self.config.adaptation_lr,
            momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()
        
        adaptation_losses = []
        
        with Timer("Local adaptation", self.logger):
            for step in range(self.config.adaptation_steps):
                epoch_loss = 0.0
                num_batches = 0
                
                for data, labels in adaptation_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.local_model(data)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                adaptation_losses.append(avg_loss)
        
        adaptation_metrics = {
            'adaptation_loss': np.mean(adaptation_losses),
            'num_adapted_samples': len(adapted_features),
            'confidence_threshold': confidence_threshold,
            'adaptation_steps': self.config.adaptation_steps
        }
        
        self.logger.info(f"Adapted model with {len(adapted_features)} samples")
        return adaptation_metrics
    
    def evaluate_on_query(self) -> Dict[str, float]:
        """
        Evaluate adapted model on query set (test query aq_i).
        
        Returns:
            Evaluation metrics
        """
        if 'query' not in self.data_loaders:
            self.logger.warning("No query data available for evaluation")
            return {}
        
        criterion = nn.CrossEntropyLoss()
        eval_metrics = self.model_trainer.evaluate(self.data_loaders['query'], criterion)
        
        # Get detailed predictions for metrics calculation
        predictions, probabilities = self.model_trainer.predict(self.data_loaders['query'])
        
        # Get true labels
        true_labels = []
        for _, labels in self.data_loaders['query']:
            true_labels.extend(labels.numpy())
        true_labels = np.array(true_labels)
        
        # Calculate comprehensive metrics
        detailed_metrics = calculate_metrics(
            true_labels, predictions, probabilities, average='macro'
        )
        
        eval_metrics.update(detailed_metrics)
        
        self.logger.info(f"Query evaluation - Accuracy: {eval_metrics.get('accuracy', 0):.4f}, "
                        f"F1: {eval_metrics.get('f1_score', 0):.4f}")
        
        return eval_metrics
    
    def federated_round(self, global_model_state: OrderedDict) -> Dict[str, Any]:
        """
        Execute one round of federated learning.
        
        Args:
            global_model_state: Global model from server
            
        Returns:
            Client update including model and metrics
        """
        self.round_number += 1
        
        with Timer(f"Federated round {self.round_number}", self.logger):
            # Update with global model
            self.update_global_model(global_model_state)
            
            # Train locally (standard FL or meta-learning)
            if hasattr(self.config, 'use_maml') and self.config.use_maml:
                training_metrics = self.train_meta_learning()
            else:
                training_metrics = self.train_local_epochs()
            
            # Evaluate on local data
            eval_metrics = self.evaluate_on_query()
            
            # Combine metrics
            round_metrics = {**training_metrics, **eval_metrics}
            round_metrics['round'] = self.round_number
            round_metrics['client_id'] = self.client_id
            
            # Prepare client update
            client_update = {
                'client_id': self.client_id,
                'round': self.round_number,
                'model_state': self.local_model.state_dict(),
                'metrics': round_metrics,
                'num_samples': len(self.data_loaders.get('support', {}).dataset) if 'support' in self.data_loaders else 0
            }
            
            # Store in history
            self.training_history.append(round_metrics)
            
            return client_update
    
    def self_labeling_workflow(self, annotation_model_state: OrderedDict) -> Dict[str, Any]:
        """
        Execute complete self-labeling workflow.
        
        Args:
            annotation_model_state: Annotation model from server
            
        Returns:
            Self-labeling results and metrics
        """
        workflow_results = {}
        
        with Timer("Self-labeling workflow", self.logger):
            # Step 1: Encode current data
            self.encode_data('test')
            workflow_results['encoding_completed'] = True
            
            # Step 2: Apply annotation model for prelabeling
            prelabel_results = self.apply_annotation_model(annotation_model_state)
            workflow_results['prelabel_results'] = prelabel_results
            
            # Step 3: Adapt with prelabeled data
            adaptation_metrics = self.adapt_with_prelabels(prelabel_results)
            workflow_results['adaptation_metrics'] = adaptation_metrics
            
            # Step 4: Evaluate on query set
            final_metrics = self.evaluate_on_query()
            workflow_results['final_metrics'] = final_metrics
        
        self.logger.info(f"Self-labeling workflow completed - Final F1: {final_metrics.get('f1_score', 0):.4f}")
        
        return workflow_results
    
    def get_client_state(self) -> Dict[str, Any]:
        """Get current client state."""
        return {
            'client_id': self.client_id,
            'round_number': self.round_number,
            'num_samples': sum(len(loader.dataset) for loader in self.data_loaders.values()),
            'data_splits': list(self.data_loaders.keys()),
            'has_historical_encoding': self.historical_encoding is not None,
            'has_current_encoding': self.current_encoding is not None,
            'training_rounds': len(self.training_history)
        }


# Utility functions
def create_client_from_config(client_id: str, config: Dict, device: torch.device) -> FederatedClient:
    """Create federated client from configuration."""
    client_config = ClientConfig(client_id=client_id, **config.get('client', {}))
    model_config = config.get('cnn', {})
    
    return FederatedClient(client_id, model_config, client_config, device)


if __name__ == "__main__":
    # Test client creation and basic functionality
    device = torch.device('cpu')
    
    model_config = {'input_dim': 35, 'num_classes': 5}
    client_config = ClientConfig(client_id='test_client', local_epochs=2)
    
    client = FederatedClient('test_client', model_config, client_config, device)
    
    # Create synthetic local data
    n_samples = 200
    features = np.random.randn(n_samples, 35)
    labels = np.random.randint(0, 5, n_samples)
    
    # Split into support/query/test
    support_size = int(0.5 * n_samples)
    query_size = int(0.3 * n_samples)
    
    local_data = {
        'support_features': features[:support_size],
        'support_labels': labels[:support_size],
        'query_features': features[support_size:support_size+query_size],
        'query_labels': labels[support_size:support_size+query_size],
        'test_features': features[support_size+query_size:],
        'test_labels': labels[support_size+query_size:]
    }
    
    client.load_local_data(local_data)
    
    print(f"Client state: {client.get_client_state()}")
    
    # Test local training
    metrics = client.train_local_epochs(num_epochs=2)
    print(f"Training metrics: {metrics}")
    
    # Test CT-AE encoding
    encodings = client.encode_data('support')
    print(f"Encodings shape: {encodings.shape}")
    
    print("Client test completed successfully!")

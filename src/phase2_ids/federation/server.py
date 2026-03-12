"""
Federated learning server implementing FedAvg and BS-Agg (Similarity-Based Aggregation).

The server orchestrates the SOH-FL federated learning process, including:
- Standard FedAvg for global model training
- BS-Agg for similarity-based helper selection and custom annotation models
- CT-AE encoding coordination for privacy-preserving similarity computation
- Meta-learning coordination across multiple clients
"""

import copy
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, asdict
import json

from ..models.cnn_1d import CNN1DClassifier
from ..models.autoencoders import CosineTargetedAutoencoder, CTAETrainer
from ..models.maml import MAMLTrainer
from ...common.logging import get_logger, MetricsLogger
from ...common.schemas import ClientState, ModelWeights
from ...common.utils import Timer
from ...common.metrics import federated_metrics_summary

logger = get_logger(__name__)


@dataclass
class ServerConfig:
    """Configuration for federated server."""
    num_clients: int = 10
    rounds: int = 50
    participation_rate: float = 0.6
    local_epochs: int = 3
    learning_rate: float = 0.005
    
    # MAML parameters
    maml_inner_lr: float = 0.001
    maml_outer_lr: float = 0.005
    maml_inner_steps: int = 1
    
    # BS-Agg parameters
    gamma_top_helpers: int = 3
    similarity_threshold: float = 0.1
    
    # CT-AE parameters
    latent_dim: int = 32
    ct_ae_epochs: int = 10
    reconstruction_weight: float = 0.7
    cosine_weight: float = 0.3


class FederatedServer:
    """
    Federated learning server for SOH-FL implementation.
    
    Manages global model aggregation, similarity-based helper selection,
    and coordination of the self-labeling workflow.
    """
    
    def __init__(self, 
                 model_config: Dict,
                 server_config: ServerConfig,
                 device: torch.device):
        """
        Initialize federated server.
        
        Args:
            model_config: Configuration for CNN1D model
            server_config: Server configuration
            device: Computing device
        """
        self.config = server_config
        self.device = device
        
        # Initialize global model
        self.global_model = CNN1DClassifier(**model_config)
        self.global_model.to(device)
        
        # Initialize CT-AE for encoding
        self.ct_ae = CosineTargetedAutoencoder(
            input_dim=model_config.get('input_dim', 35),
            latent_dim=server_config.latent_dim
        )
        self.ct_ae.to(device)
        self.ct_ae_trainer = CTAETrainer(
            self.ct_ae, device,
            w_rec=server_config.reconstruction_weight,
            w_cos=server_config.cosine_weight
        )
        
        # Client management
        self.clients: Dict[str, ClientState] = {}
        self.client_models: Dict[str, OrderedDict] = {}
        self.client_encodings: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Training history
        self.round_metrics: List[Dict] = []
        self.current_round = 0
        
        # Similarity cache for BS-Agg
        self.similarity_cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        self.logger = logger
        self.metrics_logger = MetricsLogger(logger, "FederatedServer")
    
    def register_client(self, client_id: str, client_info: Dict) -> bool:
        """
        Register a new client.
        
        Args:
            client_id: Unique client identifier
            client_info: Client information
            
        Returns:
            True if registration successful
        """
        if client_id in self.clients:
            self.logger.warning(f"Client {client_id} already registered")
            return False
        
        client_state = ClientState(
            client_id=client_id,
            round_number=0,
            num_samples=client_info.get('num_samples', 0),
            device_types=client_info.get('device_types', [])
        )
        
        self.clients[client_id] = client_state
        self.logger.info(f"Registered client {client_id} with {client_state.num_samples} samples")
        
        return True
    
    def get_global_model(self) -> OrderedDict:
        """Get current global model state dict."""
        return self.global_model.state_dict()
    
    def select_clients(self, round_num: int) -> List[str]:
        """
        Select clients for participation in current round.
        
        Args:
            round_num: Current round number
            
        Returns:
            List of selected client IDs
        """
        available_clients = list(self.clients.keys())
        num_selected = max(1, int(len(available_clients) * self.config.participation_rate))
        
        # Random selection (can be extended with more sophisticated strategies)
        np.random.seed(round_num)  # For reproducibility
        selected = np.random.choice(available_clients, num_selected, replace=False)
        
        self.logger.info(f"Round {round_num}: Selected {len(selected)}/{len(available_clients)} clients")
        return selected.tolist()
    
    def aggregate_models(self, client_updates: Dict[str, OrderedDict]) -> OrderedDict:
        """
        Aggregate client model updates using FedAvg.
        
        Args:
            client_updates: Dictionary mapping client_id to model state dict
            
        Returns:
            Aggregated global model state dict
        """
        if not client_updates:
            return self.global_model.state_dict()
        
        # Calculate weights based on number of samples
        total_samples = sum(self.clients[cid].num_samples for cid in client_updates.keys())
        weights = {cid: self.clients[cid].num_samples / total_samples 
                  for cid in client_updates.keys()}
        
        # Initialize aggregated parameters
        aggregated_params = OrderedDict()
        
        # Aggregate each parameter
        for param_name in client_updates[list(client_updates.keys())[0]].keys():
            aggregated_params[param_name] = torch.zeros_like(
                client_updates[list(client_updates.keys())[0]][param_name]
            )
            
            for client_id, client_model in client_updates.items():
                aggregated_params[param_name] += weights[client_id] * client_model[param_name]
        
        self.logger.info(f"Aggregated models from {len(client_updates)} clients")
        return aggregated_params
    
    def update_client_encodings(self, client_id: str, encodings: Dict[str, np.ndarray]):
        """
        Update client encodings for similarity computation.
        
        Args:
            client_id: Client identifier
            encodings: Dictionary with 'historical' and 'current' encodings
        """
        self.client_encodings[client_id] = encodings
        
        # Update client state
        if client_id in self.clients:
            self.clients[client_id].historical_encoding = encodings.get('historical', []).tolist()
            self.clients[client_id].current_encoding = encodings.get('current', []).tolist()
        
        self.logger.info(f"Updated encodings for client {client_id}")
    
    def compute_similarity_matrix(self, target_client: str) -> Dict[str, float]:
        """
        Compute similarity between target client and all other clients.
        
        Args:
            target_client: Target client ID
            
        Returns:
            Dictionary mapping client_id to similarity score
        """
        if target_client not in self.client_encodings:
            self.logger.error(f"No encodings found for target client {target_client}")
            return {}
        
        target_encoding = self.client_encodings[target_client].get('current')
        if target_encoding is None:
            self.logger.error(f"No current encoding for target client {target_client}")
            return {}
        
        similarities = {}
        
        for client_id, encodings in self.client_encodings.items():
            if client_id == target_client:
                continue
            
            historical_encoding = encodings.get('historical')
            if historical_encoding is None:
                continue
            
            # Compute cosine similarity
            target_norm = np.linalg.norm(target_encoding)
            hist_norm = np.linalg.norm(historical_encoding)
            
            if target_norm > 0 and hist_norm > 0:
                similarity = np.dot(target_encoding, historical_encoding) / (target_norm * hist_norm)
                similarities[client_id] = float(similarity)
        
        # Cache similarities
        self.similarity_cache[target_client] = similarities
        
        return similarities
    
    def select_helpers(self, target_client: str, gamma: Optional[int] = None) -> List[str]:
        """
        Select top-gamma helper clients using BS-Agg.
        
        Args:
            target_client: Target client ID
            gamma: Number of helpers to select (default: config value)
            
        Returns:
            List of helper client IDs
        """
        if gamma is None:
            gamma = self.config.gamma_top_helpers
        
        # Compute or retrieve similarities
        if target_client not in self.similarity_cache:
            similarities = self.compute_similarity_matrix(target_client)
        else:
            similarities = self.similarity_cache[target_client]
        
        if not similarities:
            self.logger.warning(f"No similarities found for {target_client}")
            return []
        
        # Filter by threshold
        filtered_similarities = {
            cid: sim for cid, sim in similarities.items() 
            if sim >= self.config.similarity_threshold
        }
        
        if not filtered_similarities:
            self.logger.warning(f"No clients above similarity threshold for {target_client}")
            # Fall back to top similarities regardless of threshold
            filtered_similarities = similarities
        
        # Select top-gamma helpers
        sorted_helpers = sorted(filtered_similarities.items(), key=lambda x: x[1], reverse=True)
        top_helpers = [cid for cid, _ in sorted_helpers[:gamma]]
        
        # Update client state
        if target_client in self.clients:
            self.clients[target_client].helper_similarities = filtered_similarities
            self.clients[target_client].selected_helpers = top_helpers
        
        self.logger.info(f"Selected {len(top_helpers)} helpers for {target_client}: {top_helpers}")
        return top_helpers
    
    def create_annotation_model(self, target_client: str, helpers: List[str]) -> OrderedDict:
        """
        Create custom annotation model by aggregating helper models (BS-Agg).
        
        Args:
            target_client: Target client ID
            helpers: List of helper client IDs
            
        Returns:
            Aggregated annotation model state dict
        """
        if not helpers:
            self.logger.warning(f"No helpers available for {target_client}, using global model")
            return self.global_model.state_dict()
        
        # Get helper models
        helper_models = {}
        helper_similarities = self.similarity_cache.get(target_client, {})
        
        for helper_id in helpers:
            if helper_id in self.client_models:
                helper_models[helper_id] = self.client_models[helper_id]
        
        if not helper_models:
            self.logger.warning(f"No helper models available for {target_client}")
            return self.global_model.state_dict()
        
        # Weighted aggregation based on similarities
        total_similarity = sum(helper_similarities.get(hid, 1.0) for hid in helper_models.keys())
        weights = {hid: helper_similarities.get(hid, 1.0) / total_similarity 
                  for hid in helper_models.keys()}
        
        # Aggregate helper models
        aggregated_params = OrderedDict()
        
        for param_name in helper_models[list(helper_models.keys())[0]].keys():
            aggregated_params[param_name] = torch.zeros_like(
                helper_models[list(helper_models.keys())[0]][param_name]
            )
            
            for helper_id, helper_model in helper_models.items():
                aggregated_params[param_name] += weights[helper_id] * helper_model[param_name]
        
        self.logger.info(f"Created annotation model for {target_client} from {len(helpers)} helpers")
        return aggregated_params
    
    def federated_round(self, client_updates: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Execute one round of federated learning.
        
        Args:
            client_updates: Dictionary of client updates
            
        Returns:
            Round results and metrics
        """
        self.current_round += 1
        round_start_time = Timer(f"Federated round {self.current_round}", self.logger)
        
        with round_start_time:
            # Extract model updates
            model_updates = {}
            round_metrics = {}
            
            for client_id, update in client_updates.items():
                if 'model_state' in update:
                    model_updates[client_id] = update['model_state']
                    self.client_models[client_id] = update['model_state']
                
                if 'metrics' in update:
                    round_metrics[client_id] = update['metrics']
                
                # Update client state
                if client_id in self.clients:
                    client_state = self.clients[client_id]
                    client_state.round_number = self.current_round
                    client_state.local_loss = update.get('metrics', {}).get('loss')
                    client_state.local_accuracy = update.get('metrics', {}).get('accuracy')
            
            # Aggregate models
            if model_updates:
                aggregated_model = self.aggregate_models(model_updates)
                self.global_model.load_state_dict(aggregated_model)
            
            # Compute global metrics
            global_metrics = self._compute_global_metrics(round_metrics)
            
            # Log round results
            self.metrics_logger.log_metrics(global_metrics, step=self.current_round)
            
            round_result = {
                'round': self.current_round,
                'participating_clients': list(client_updates.keys()),
                'global_metrics': global_metrics,
                'client_metrics': round_metrics,
                'global_model_state': self.global_model.state_dict()
            }
            
            self.round_metrics.append(round_result)
            
            return round_result
    
    def _compute_global_metrics(self, client_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """Compute global metrics from client metrics."""
        if not client_metrics:
            return {}
        
        # Weight metrics by number of samples
        total_samples = sum(self.clients[cid].num_samples for cid in client_metrics.keys())
        
        global_metrics = {}
        
        # Aggregate common metrics
        for metric_name in ['loss', 'accuracy', 'f1_score']:
            weighted_sum = 0.0
            for client_id, metrics in client_metrics.items():
                if metric_name in metrics:
                    weight = self.clients[client_id].num_samples / total_samples
                    weighted_sum += weight * metrics[metric_name]
            
            if weighted_sum > 0:
                global_metrics[f'global_{metric_name}'] = weighted_sum
        
        return global_metrics
    
    def get_server_state(self) -> Dict[str, Any]:
        """Get current server state."""
        return {
            'current_round': self.current_round,
            'num_clients': len(self.clients),
            'registered_clients': list(self.clients.keys()),
            'global_model_params': sum(p.numel() for p in self.global_model.parameters()),
            'ct_ae_params': sum(p.numel() for p in self.ct_ae.parameters()),
            'round_history': len(self.round_metrics)
        }
    
    def get_federated_summary(self) -> Dict[str, Any]:
        """Get comprehensive federated learning summary."""
        if not self.round_metrics:
            return {}
        
        # Extract client metrics across rounds
        all_client_metrics = {}
        for round_data in self.round_metrics:
            for client_id, metrics in round_data.get('client_metrics', {}).items():
                if client_id not in all_client_metrics:
                    all_client_metrics[client_id] = {}
                
                for metric_name, value in metrics.items():
                    if metric_name not in all_client_metrics[client_id]:
                        all_client_metrics[client_id][metric_name] = []
                    all_client_metrics[client_id][metric_name].append(value)
        
        # Get final metrics (last round)
        final_client_metrics = {}
        if self.round_metrics:
            final_client_metrics = self.round_metrics[-1].get('client_metrics', {})
        
        # Generate summary
        summary = federated_metrics_summary(final_client_metrics)
        summary.update({
            'total_rounds': len(self.round_metrics),
            'server_state': self.get_server_state(),
            'similarity_cache_size': len(self.similarity_cache)
        })
        
        return summary
    
    def save_checkpoint(self, filepath: str):
        """Save server checkpoint."""
        checkpoint = {
            'global_model_state': self.global_model.state_dict(),
            'ct_ae_state': self.ct_ae.state_dict(),
            'server_config': asdict(self.config),
            'current_round': self.current_round,
            'clients': {cid: asdict(client) for cid, client in self.clients.items()},
            'round_metrics': self.round_metrics,
            'similarity_cache': dict(self.similarity_cache)
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Server checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load server checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.global_model.load_state_dict(checkpoint['global_model_state'])
        self.ct_ae.load_state_dict(checkpoint['ct_ae_state'])
        self.current_round = checkpoint['current_round']
        self.round_metrics = checkpoint['round_metrics']
        self.similarity_cache = defaultdict(dict, checkpoint.get('similarity_cache', {}))
        
        # Reconstruct client states
        self.clients = {}
        for cid, client_data in checkpoint.get('clients', {}).items():
            self.clients[cid] = ClientState(**client_data)
        
        self.logger.info(f"Server checkpoint loaded from {filepath}")


# Utility functions for server management
def create_server_from_config(config_path: str, device: torch.device) -> FederatedServer:
    """Create federated server from configuration file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    server_config = ServerConfig(**config.get('federation', {}))
    model_config = config.get('cnn', {})
    
    return FederatedServer(model_config, server_config, device)


if __name__ == "__main__":
    # Test server creation and basic functionality
    device = torch.device('cpu')
    
    model_config = {'input_dim': 35, 'num_classes': 10}
    server_config = ServerConfig(num_clients=5, rounds=10)
    
    server = FederatedServer(model_config, server_config, device)
    
    # Register test clients
    for i in range(3):
        client_info = {'num_samples': 100 + i * 50, 'device_types': ['camera', 'sensor']}
        server.register_client(f'client_{i:02d}', client_info)
    
    print(f"Server state: {server.get_server_state()}")
    
    # Test client selection
    selected = server.select_clients(1)
    print(f"Selected clients: {selected}")
    
    # Test similarity computation (with dummy encodings)
    server.client_encodings = {
        'client_00': {'historical': np.random.randn(32), 'current': np.random.randn(32)},
        'client_01': {'historical': np.random.randn(32), 'current': np.random.randn(32)},
        'client_02': {'historical': np.random.randn(32), 'current': np.random.randn(32)}
    }
    
    similarities = server.compute_similarity_matrix('client_00')
    helpers = server.select_helpers('client_00', gamma=2)
    
    print(f"Similarities: {similarities}")
    print(f"Selected helpers: {helpers}")

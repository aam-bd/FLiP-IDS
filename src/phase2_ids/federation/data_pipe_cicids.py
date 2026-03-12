"""
CIC-IDS2017 specific data pipeline for Phase 2 federated learning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from ..models.autoencoders import CosineTargetedAutoencoder
from ...common.schemas import ModelConfig, DataConfig
from ...common.logging import get_logger

logger = get_logger(__name__)

class CICIDSDataPipeline:
    """Data pipeline specifically designed for CIC-IDS2017 dataset"""
    
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        self.feature_columns = None
        self.label_mapping = None
        self.scaler = None
        
    def load_raw_data(self, data_path: str) -> pd.DataFrame:
        """Load raw CIC-IDS2017 data"""
        logger.info(f"Loading CIC-IDS2017 data from {data_path}")
        
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Loaded {len(data)} samples with {len(data.columns)} columns")
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess CIC-IDS2017 data for federated learning"""
        logger.info("Preprocessing CIC-IDS2017 data...")
        
        # Clean data
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        logger.info(f"After cleaning: {len(data)} samples")
        
        # Create labels
        unique_labels = data['Label'].unique()
        self.label_mapping = {label: i for i, label in enumerate(unique_labels)}
        data['label'] = data['Label'].map(self.label_mapping)
        
        # Select feature columns (exclude Label, day, src_ip)
        self.feature_columns = [col for col in data.columns 
                               if col not in ['Label', 'day', 'src_ip', 'label']]
        
        # Ensure numeric features
        for col in self.feature_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove any remaining NaN
        data = data.dropna()
        
        # Extract features and labels
        X = data[self.feature_columns].values
        y = data['label'].values
        
        logger.info(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features, {len(unique_labels)} classes")
        
        return X, y, self.feature_columns
    
    def create_federated_splits(self, X: np.ndarray, y: np.ndarray, 
                               num_clients: int = 10) -> Dict[int, Dict[str, np.ndarray]]:
        """Create federated data splits for CIC-IDS2017"""
        logger.info(f"Creating federated splits for {num_clients} clients")
        
        # Simple random split for now (can be enhanced with more sophisticated partitioning)
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        client_data = {}
        samples_per_client = n_samples // num_clients
        
        for client_id in range(num_clients):
            start_idx = client_id * samples_per_client
            if client_id == num_clients - 1:  # Last client gets remaining samples
                end_idx = n_samples
            else:
                end_idx = (client_id + 1) * samples_per_client
            
            client_indices = indices[start_idx:end_idx]
            
            client_data[client_id] = {
                'x': X[client_indices],
                'y': y[client_indices]
            }
            
            logger.info(f"Client {client_id}: {len(client_indices)} samples")
        
        return client_data
    
    def prepare_for_phase2(self, data_path: str, num_clients: int = 10) -> Dict:
        """Complete data preparation pipeline for Phase 2"""
        logger.info("Starting CIC-IDS2017 data preparation for Phase 2...")
        
        try:
            # Load and preprocess data
            raw_data = self.load_raw_data(data_path)
            X, y, feature_columns = self.preprocess_data(raw_data)
            
            # Create federated splits
            client_data = self.create_federated_splits(X, y, num_clients)
            
            # Prepare metadata
            metadata = {
                'num_clients': num_clients,
                'num_features': len(feature_columns),
                'num_classes': len(self.label_mapping),
                'feature_columns': feature_columns,
                'label_mapping': self.label_mapping,
                'total_samples': len(X)
            }
            
            logger.info("CIC-IDS2017 data preparation completed successfully")
            logger.info(f"Total samples: {metadata['total_samples']}")
            logger.info(f"Features: {metadata['num_features']}")
            logger.info(f"Classes: {metadata['num_classes']}")
            logger.info(f"Clients: {metadata['num_clients']}")
            
            return {
                'client_data': client_data,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
    
    def simulate_attack_scenario(self, client_data: Dict, attack_ratio: float = 0.3) -> Dict:
        """Simulate attack scenario for evaluation"""
        logger.info(f"Simulating attack scenario with {attack_ratio} attack ratio")
        
        modified_data = {}
        
        for client_id, data in client_data.items():
            X, y = data['x'], data['y']
            
            # Identify benign samples (assuming label 0 is BENIGN)
            benign_mask = (y == 0)
            attack_mask = ~benign_mask
            
            benign_X, benign_y = X[benign_mask], y[benign_mask]
            attack_X, attack_y = X[attack_mask], y[attack_mask]
            
            # Calculate target sizes
            total_samples = len(X)
            target_attack_samples = int(total_samples * attack_ratio)
            target_benign_samples = total_samples - target_attack_samples
            
            # Sample to achieve target ratio
            if len(attack_X) > target_attack_samples:
                attack_indices = np.random.choice(len(attack_X), target_attack_samples, replace=False)
                attack_X = attack_X[attack_indices]
                attack_y = attack_y[attack_indices]
            
            if len(benign_X) > target_benign_samples:
                benign_indices = np.random.choice(len(benign_X), target_benign_samples, replace=False)
                benign_X = benign_X[benign_indices]
                benign_y = benign_y[benign_indices]
            
            # Combine
            combined_X = np.vstack([benign_X, attack_X])
            combined_y = np.hstack([benign_y, attack_y])
            
            # Shuffle
            shuffle_indices = np.random.permutation(len(combined_X))
            combined_X = combined_X[shuffle_indices]
            combined_y = combined_y[shuffle_indices]
            
            modified_data[client_id] = {
                'x': combined_X,
                'y': combined_y
            }
            
            logger.info(f"Client {client_id}: {len(combined_X)} samples "
                       f"({np.sum(combined_y == 0)} benign, {np.sum(combined_y != 0)} attacks)")
        
        return modified_data


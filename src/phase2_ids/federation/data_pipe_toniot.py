"""
Data pipeline for converting TON-IoT dataset to Phase 2 federated learning datasets.

Transforms TON-IoT device data into local datasets for each federated
client, simulating statistical heterogeneity and creating support/query splits
for meta-learning and self-labeling workflows.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ...common.logging import get_logger
from ...common.io import save_data, load_data
from ...common.utils import set_seed

logger = get_logger(__name__)


class IoTFlowDataset(Dataset):
    """Dataset for IoT flow records with features and labels."""
    
    def __init__(self, 
                 features: np.ndarray,
                 labels: np.ndarray,
                 transform: Optional[callable] = None):
        """
        Initialize dataset.
        
        Args:
            features: Feature matrix
            labels: Label array
            transform: Optional data transform
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, label


class TONIoTDataPipeline:
    """Data pipeline for TON-IoT federated learning preparation."""
    
    def __init__(self, 
                 output_dir: Path,
                 num_clients: int = 6,
                 attack_ratio: float = 0.3,
                 support_ratio: float = 0.1,
                 random_state: int = 42):
        """
        Initialize data pipeline.
        
        Args:
            output_dir: Output directory for processed data
            num_clients: Number of federated clients
            attack_ratio: Ratio of attack samples to generate
            support_ratio: Ratio of support samples for meta-learning
            random_state: Random seed
        """
        self.output_dir = Path(output_dir)
        self.num_clients = num_clients
        self.attack_ratio = attack_ratio
        self.support_ratio = support_ratio
        self.random_state = random_state
        
        self.logger = get_logger(__name__)
        set_seed(random_state)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Attack types for simulation
        self.attack_types = ['dos', 'ddos', 'reconnaissance', 'data_exfiltration', 'malware']
        
    def prepare_federated_data(self, profiles_path: str) -> Dict[str, Path]:
        """
        Prepare federated datasets from TON-IoT profiles.
        
        Args:
            profiles_path: Path to TON-IoT CSV file
            
        Returns:
            Dictionary mapping client IDs to their data paths
        """
        self.logger.info(f"Loading TON-IoT profiles from {profiles_path}")
        
        # Load TON-IoT data
        df = pd.read_csv(profiles_path)
        self.logger.info(f"Loaded {len(df)} profiles with {df.shape[1]} features")
        
        # Prepare features and labels
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        features = df[feature_cols].values
        device_labels = df['label'].values
        client_ids = df['src_ip'].values
        
        # Create binary labels: 0 = normal, 1 = attack (simulated)
        normal_labels = np.zeros(len(df))
        
        # Simulate attacks
        attack_features, attack_labels = self._simulate_attacks(features, normal_labels)
        
        # Combine normal and attack data
        all_features = np.vstack([features, attack_features])
        all_labels = np.concatenate([normal_labels, attack_labels])
        all_client_ids = np.concatenate([client_ids, client_ids[:len(attack_features)]])
        
        # Create client datasets
        client_data_paths = {}
        unique_clients = np.unique(all_client_ids)
        
        for i, client_id in enumerate(unique_clients[:self.num_clients]):
            client_mask = all_client_ids == client_id
            client_features = all_features[client_mask]
            client_labels = all_labels[client_mask]
            
            if len(client_features) == 0:
                continue
                
            # Split into train/test/support
            train_features, test_features, train_labels, test_labels = train_test_split(
                client_features, client_labels, test_size=0.3, random_state=self.random_state
            )
            
            # Create support set for meta-learning
            support_size = int(len(train_features) * self.support_ratio)
            if support_size > 0:
                support_indices = np.random.choice(len(train_features), support_size, replace=False)
                support_features = train_features[support_indices]
                support_labels = train_labels[support_indices]
                
                # Remove support samples from training
                train_mask = np.ones(len(train_features), dtype=bool)
                train_mask[support_indices] = False
                train_features = train_features[train_mask]
                train_labels = train_labels[train_mask]
            else:
                support_features = train_features[:10]  # Minimum support set
                support_labels = train_labels[:10]
            
            # Save client data
            client_dir = self.output_dir / f"client_{i}"
            client_dir.mkdir(exist_ok=True)
            
            # Save as numpy arrays
            np.save(client_dir / "train_features.npy", train_features)
            np.save(client_dir / "train_labels.npy", train_labels)
            np.save(client_dir / "test_features.npy", test_features)
            np.save(client_dir / "test_labels.npy", test_labels)
            np.save(client_dir / "support_features.npy", support_features)
            np.save(client_dir / "support_labels.npy", support_labels)
            
            # Save metadata
            metadata = {
                'client_id': i,
                'original_client_id': client_id,
                'num_train': len(train_features),
                'num_test': len(test_features),
                'num_support': len(support_features),
                'num_features': train_features.shape[1],
                'num_classes': len(np.unique(all_labels))
            }
            
            save_data(metadata, client_dir / "metadata.json")
            client_data_paths[f"client_{i}"] = client_dir
            
            self.logger.info(f"Client {i}: {len(train_features)} train, {len(test_features)} test, {len(support_features)} support")
        
        # Save global metadata
        global_metadata = {
            'num_clients': len(client_data_paths),
            'num_features': features.shape[1],
            'num_classes': len(np.unique(all_labels)),
            'attack_ratio': self.attack_ratio,
            'support_ratio': self.support_ratio,
            'total_samples': len(all_features)
        }
        
        save_data(global_metadata, self.output_dir / "global_metadata.json")
        
        self.logger.info(f"Prepared federated data for {len(client_data_paths)} clients")
        return client_data_paths
    
    def _simulate_attacks(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate attack patterns by modifying normal features."""
        num_attacks = int(len(features) * self.attack_ratio)
        attack_indices = np.random.choice(len(features), num_attacks, replace=False)
        
        attack_features = features[attack_indices].copy()
        attack_labels = np.ones(num_attacks)  # All attacks are label 1
        
        # Apply attack patterns
        for i, idx in enumerate(attack_indices):
            attack_type = np.random.choice(self.attack_types)
            attack_features[i] = self._apply_attack_pattern(attack_features[i], attack_type)
        
        return attack_features, attack_labels
    
    def _apply_attack_pattern(self, features: np.ndarray, attack_type: str) -> np.ndarray:
        """Apply attack-specific feature modifications."""
        modified_features = features.copy()
        
        if attack_type == 'dos':
            # DoS: Increase some features significantly
            modified_features[0] *= np.random.uniform(5, 20)  # feature_0
            modified_features[2] *= np.random.uniform(0.1, 0.5)  # feature_2
            modified_features[8] *= np.random.uniform(10, 100)  # feature_8
            
        elif attack_type == 'ddos':
            # DDoS: Very high values
            modified_features[0] *= np.random.uniform(50, 200)  # feature_0
            modified_features[8] *= np.random.uniform(100, 1000)  # feature_8
            modified_features[1] *= np.random.uniform(0.1, 0.3)  # feature_1
            
        elif attack_type == 'reconnaissance':
            # Reconnaissance: Many small connections
            modified_features[1] *= np.random.uniform(0.05, 0.2)  # feature_1
            modified_features[8] *= np.random.uniform(0.1, 0.5)  # feature_8
            modified_features[10] *= np.random.uniform(2, 10)  # feature_10
            
        elif attack_type == 'data_exfiltration':
            # Data exfiltration: Large data transfers
            modified_features[2] *= np.random.uniform(5, 50)  # feature_2
            modified_features[12] *= np.random.uniform(2, 20)  # feature_12
            modified_features[1] *= np.random.uniform(2, 10)  # feature_1
            
        elif attack_type == 'malware':
            # Malware: Unusual patterns
            modified_features[0] *= np.random.uniform(0.1, 0.5)  # feature_0
            modified_features[2] *= np.random.uniform(2, 10)  # feature_2
            modified_features[8] *= np.random.uniform(0.5, 2)  # feature_8
        
        return modified_features
    
    def load_client_data(self, client_path: Path) -> Dict[str, DataLoader]:
        """Load data for a specific client."""
        client_dir = Path(client_path)
        
        # Load data
        train_features = np.load(client_dir / "train_features.npy")
        train_labels = np.load(client_dir / "train_labels.npy")
        test_features = np.load(client_dir / "test_features.npy")
        test_labels = np.load(client_dir / "test_labels.npy")
        support_features = np.load(client_dir / "support_features.npy")
        support_labels = np.load(client_dir / "support_labels.npy")
        
        # Create datasets
        train_dataset = IoTFlowDataset(train_features, train_labels)
        test_dataset = IoTFlowDataset(test_features, test_labels)
        support_dataset = IoTFlowDataset(support_features, support_labels)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        support_loader = DataLoader(support_dataset, batch_size=32, shuffle=False)
        
        return {
            'train': train_loader,
            'test': test_loader,
            'support': support_loader
        }


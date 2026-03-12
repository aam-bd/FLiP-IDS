"""
Data pipeline for converting Phase 1 outputs to Phase 2 federated learning datasets.

Transforms device profiles and flow records into local datasets for each federated
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


class DataPipeline:
    """
    Data pipeline for Phase 2 federated learning setup.
    
    Converts Phase 1 device profiles into federated learning datasets
    with statistical heterogeneity and attack simulation.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.logger = logger
        
        # Attack types for intrusion detection
        self.attack_types = [
            'benign', 'dos', 'ddos', 'reconnaissance', 'theft',
            'man_in_middle', 'malware_cnc', 'data_exfiltration',
            'privilege_escalation', 'zero_day'
        ]
        
        set_seed(random_state)
    
    def load_phase1_data(self, profiles_path: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load Phase 1 device profiles.
        
        Args:
            profiles_path: Path to Phase 1 profiles
            
        Returns:
            Tuple of (features_df, metadata_df)
        """
        profiles_path = Path(profiles_path)
        
        if not profiles_path.exists():
            raise FileNotFoundError(f"Profiles file not found: {profiles_path}")
        
        self.logger.info(f"Loading Phase 1 profiles from {profiles_path}")
        
        # Load profiles
        profiles_df = load_data(profiles_path)
        
        # Separate features from metadata
        feature_columns = [col for col in profiles_df.columns 
                          if col.startswith(('packet_', 'flow_', 'tcp_', 'syn_', 'ack_', 
                                           'fin_', 'rst_', 'psh_', 'urg_', 'protocol_',
                                           'has_', 'src_', 'dst_', 'service_', 'dns_',
                                           'bytes_', 'total_', 'down_', 'avg_', 'variance_',
                                           'subflow_', 'min_', 'unique_'))]
        
        metadata_columns = [col for col in profiles_df.columns if col not in feature_columns]
        
        features_df = profiles_df[feature_columns]
        metadata_df = profiles_df[metadata_columns]
        
        self.logger.info(f"Loaded {len(profiles_df)} profiles with {len(feature_columns)} features")
        
        return features_df, metadata_df
    
    def simulate_attacks(self, 
                        features_df: pd.DataFrame,
                        metadata_df: pd.DataFrame,
                        attack_ratio: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulate network attacks on the dataset.
        
        Args:
            features_df: Feature DataFrame
            metadata_df: Metadata DataFrame
            attack_ratio: Fraction of samples to convert to attacks
            
        Returns:
            Tuple of (augmented_features_df, augmented_metadata_df)
        """
        self.logger.info(f"Simulating attacks with ratio {attack_ratio}")
        
        n_samples = len(features_df)
        n_attacks = int(n_samples * attack_ratio)
        
        # Select samples to convert to attacks
        attack_indices = np.random.choice(n_samples, n_attacks, replace=False)
        
        # Create copies for attack simulation
        attack_features = features_df.iloc[attack_indices].copy()
        attack_metadata = metadata_df.iloc[attack_indices].copy()
        
        # Simulate different attack patterns
        for i, idx in enumerate(attack_indices):
            attack_type = np.random.choice(self.attack_types[1:])  # Exclude 'benign'
            
            # Modify features based on attack type
            attack_features.iloc[i] = self._apply_attack_pattern(
                attack_features.iloc[i], attack_type
            )
            
            # Update metadata
            attack_metadata.loc[attack_metadata.index[i], 'attack_type'] = attack_type
            attack_metadata.loc[attack_metadata.index[i], 'is_attack'] = 1
        
        # Add benign labels to original data
        metadata_df = metadata_df.copy()
        metadata_df['attack_type'] = 'benign'
        metadata_df['is_attack'] = 0
        
        # Combine original and attack data
        combined_features = pd.concat([features_df, attack_features], ignore_index=True)
        combined_metadata = pd.concat([metadata_df, attack_metadata], ignore_index=True)
        
        # Shuffle the combined data
        indices = np.random.permutation(len(combined_features))
        combined_features = combined_features.iloc[indices].reset_index(drop=True)
        combined_metadata = combined_metadata.iloc[indices].reset_index(drop=True)
        
        self.logger.info(f"Generated {len(combined_features)} samples with attacks")
        attack_counts = combined_metadata['attack_type'].value_counts()
        self.logger.info(f"Attack distribution: {attack_counts.to_dict()}")
        
        return combined_features, combined_metadata
    
    def _apply_attack_pattern(self, features: pd.Series, attack_type: str) -> pd.Series:
        """Apply attack-specific feature modifications."""
        modified_features = features.copy()
        
        if attack_type == 'dos':
            # DoS: High packet rate, small packets
            modified_features['packets_per_second'] *= np.random.uniform(5, 20)
            modified_features['avg_packet_size'] *= np.random.uniform(0.1, 0.5)
            modified_features['packet_count_total'] *= np.random.uniform(10, 100)
            
        elif attack_type == 'ddos':
            # DDoS: Very high packet rate, distributed sources
            modified_features['packets_per_second'] *= np.random.uniform(50, 200)
            modified_features['packet_count_total'] *= np.random.uniform(100, 1000)
            modified_features['flow_duration'] *= np.random.uniform(0.1, 0.3)
            
        elif attack_type == 'reconnaissance':
            # Reconnaissance: Many small connections, port scanning
            modified_features['flow_duration'] *= np.random.uniform(0.05, 0.2)
            modified_features['packet_count_total'] *= np.random.uniform(0.1, 0.5)
            modified_features['syn_flag_count'] *= np.random.uniform(2, 10)
            
        elif attack_type == 'data_exfiltration':
            # Data exfiltration: Large outbound traffic
            modified_features['total_bytes_forward'] *= np.random.uniform(10, 100)
            modified_features['bytes_per_second'] *= np.random.uniform(5, 50)
            modified_features['flow_duration'] *= np.random.uniform(2, 10)
            
        elif attack_type == 'malware_cnc':
            # Malware C&C: Regular periodic communication
            modified_features['flow_iat_mean'] = np.random.uniform(10, 60)  # Regular intervals
            modified_features['flow_iat_std'] *= np.random.uniform(0.1, 0.3)  # Low variance
            modified_features['dns_query_count'] *= np.random.uniform(2, 10)
            
        # Add some random noise to make attacks less predictable
        noise_factor = np.random.uniform(0.9, 1.1, len(modified_features))
        modified_features = modified_features * noise_factor
        
        # Ensure non-negative values
        modified_features = modified_features.abs()
        
        return modified_features
    
    def create_federated_splits(self,
                               features_df: pd.DataFrame,
                               metadata_df: pd.DataFrame,
                               num_clients: int = 10,
                               heterogeneity: float = 0.7) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create federated data splits with statistical heterogeneity.
        
        Args:
            features_df: Feature DataFrame
            metadata_df: Metadata DataFrame
            num_clients: Number of federated clients
            heterogeneity: Statistical heterogeneity level (0-1)
            
        Returns:
            Dictionary mapping client_id to data splits
        """
        self.logger.info(f"Creating federated splits for {num_clients} clients")
        
        # Encode labels
        labels = self.label_encoder.fit_transform(metadata_df['attack_type'])
        num_classes = len(self.label_encoder.classes_)
        
        # Create Dirichlet distribution for heterogeneity
        if heterogeneity > 0:
            # Use Dirichlet distribution to create heterogeneous class distributions
            alpha = heterogeneity * num_classes
            class_distributions = np.random.dirichlet([alpha] * num_classes, num_clients)
        else:
            # Uniform distribution (homogeneous)
            class_distributions = np.ones((num_clients, num_classes)) / num_classes
        
        # Assign samples to clients based on class distributions
        client_indices = [[] for _ in range(num_clients)]
        
        for class_idx in range(num_classes):
            class_samples = np.where(labels == class_idx)[0]
            np.random.shuffle(class_samples)
            
            # Calculate number of samples per client for this class
            class_distribution = class_distributions[:, class_idx]
            class_distribution = class_distribution / class_distribution.sum()
            
            samples_per_client = (len(class_samples) * class_distribution).astype(int)
            
            # Distribute samples
            start_idx = 0
            for client_idx in range(num_clients):
                end_idx = start_idx + samples_per_client[client_idx]
                if client_idx == num_clients - 1:  # Last client gets remaining samples
                    end_idx = len(class_samples)
                
                client_indices[client_idx].extend(class_samples[start_idx:end_idx])
                start_idx = end_idx
        
        # Create client datasets
        client_data = {}
        
        for client_idx in range(num_clients):
            client_id = f"client_{client_idx:02d}"
            indices = client_indices[client_idx]
            
            if len(indices) == 0:
                self.logger.warning(f"Client {client_id} has no data")
                continue
            
            client_features = features_df.iloc[indices].reset_index(drop=True)
            client_metadata = metadata_df.iloc[indices].reset_index(drop=True)
            client_labels = labels[indices]
            
            # Create train/test split for each client
            train_features, test_features, train_labels, test_labels = train_test_split(
                client_features, client_labels, test_size=0.3, 
                random_state=self.random_state + client_idx, stratify=client_labels
            )
            
            client_data[client_id] = {
                'train_features': train_features,
                'test_features': test_features,
                'train_labels': train_labels,
                'test_labels': test_labels,
                'metadata': client_metadata,
                'num_samples': len(indices),
                'class_distribution': np.bincount(client_labels, minlength=num_classes)
            }
            
            self.logger.info(f"{client_id}: {len(indices)} samples, "
                           f"classes: {np.bincount(client_labels)}")
        
        return client_data
    
    def create_meta_learning_splits(self, 
                                   client_data: Dict[str, Dict],
                                   support_ratio: float = 0.6) -> Dict[str, Dict]:
        """
        Create support/query splits for meta-learning.
        
        Args:
            client_data: Client data dictionary
            support_ratio: Fraction of training data for support set
            
        Returns:
            Updated client data with support/query splits
        """
        self.logger.info("Creating meta-learning support/query splits")
        
        for client_id, data in client_data.items():
            train_features = data['train_features']
            train_labels = data['train_labels']
            
            # Split training data into support and query sets
            support_features, query_features, support_labels, query_labels = train_test_split(
                train_features, train_labels, 
                train_size=support_ratio,
                random_state=self.random_state,
                stratify=train_labels if len(np.unique(train_labels)) > 1 else None
            )
            
            # Update client data
            data.update({
                'support_features': support_features,
                'support_labels': support_labels,
                'query_features': query_features,
                'query_labels': query_labels
            })
            
            self.logger.info(f"{client_id}: Support={len(support_features)}, "
                           f"Query={len(query_features)}, Test={len(data['test_features'])}")
        
        return client_data
    
    def create_dataloaders(self,
                          client_data: Dict[str, Dict],
                          batch_size: int = 32,
                          shuffle: bool = True) -> Dict[str, Dict[str, DataLoader]]:
        """
        Create PyTorch DataLoaders for each client.
        
        Args:
            client_data: Client data dictionary
            batch_size: Batch size for DataLoaders
            shuffle: Whether to shuffle data
            
        Returns:
            Dictionary mapping client_id to DataLoaders
        """
        client_loaders = {}
        
        for client_id, data in client_data.items():
            loaders = {}
            
            # Create datasets and loaders for each split
            for split_name in ['support', 'query', 'test']:
                if f'{split_name}_features' in data:
                    features = data[f'{split_name}_features'].values
                    labels = data[f'{split_name}_labels']
                    
                    dataset = IoTFlowDataset(features, labels)
                    loader = DataLoader(
                        dataset, 
                        batch_size=batch_size,
                        shuffle=shuffle,
                        drop_last=False
                    )
                    loaders[split_name] = loader
            
            client_loaders[client_id] = loaders
        
        return client_loaders
    
    def save_client_data(self, 
                        client_data: Dict[str, Dict],
                        output_dir: Union[str, Path]):
        """
        Save client data to disk.
        
        Args:
            client_data: Client data dictionary
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving client data to {output_dir}")
        
        for client_id, data in client_data.items():
            client_dir = output_dir / client_id
            client_dir.mkdir(exist_ok=True)
            
            # Save each data split
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    save_data(value, client_dir / f"{key}.csv", format='csv')
                elif isinstance(value, np.ndarray):
                    np.save(client_dir / f"{key}.npy", value)
                elif isinstance(value, (int, float, list, dict)):
                    save_data(value, client_dir / f"{key}.json", format='json')
        
        # Save label encoder
        import joblib
        joblib.dump(self.label_encoder, output_dir / "label_encoder.joblib")
        
        self.logger.info(f"Saved data for {len(client_data)} clients")
    
    def load_client_data(self, input_dir: Union[str, Path]) -> Dict[str, Dict]:
        """
        Load client data from disk.
        
        Args:
            input_dir: Input directory
            
        Returns:
            Client data dictionary
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        self.logger.info(f"Loading client data from {input_dir}")
        
        # Load label encoder
        import joblib
        encoder_path = input_dir / "label_encoder.joblib"
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
        
        client_data = {}
        
        for client_dir in input_dir.iterdir():
            if client_dir.is_dir() and client_dir.name.startswith('client_'):
                client_id = client_dir.name
                data = {}
                
                # Load all data files
                for file_path in client_dir.iterdir():
                    key = file_path.stem
                    
                    if file_path.suffix == '.csv':
                        data[key] = load_data(file_path)
                    elif file_path.suffix == '.npy':
                        data[key] = np.load(file_path)
                    elif file_path.suffix == '.json':
                        data[key] = load_data(file_path)
                
                client_data[client_id] = data
        
        self.logger.info(f"Loaded data for {len(client_data)} clients")
        return client_data
    
    def get_global_statistics(self, client_data: Dict[str, Dict]) -> Dict:
        """Get global statistics across all clients."""
        total_samples = sum(data['num_samples'] for data in client_data.values())
        
        # Aggregate class distributions
        global_class_dist = np.zeros(len(self.label_encoder.classes_))
        for data in client_data.values():
            global_class_dist += data['class_distribution']
        
        stats = {
            'num_clients': len(client_data),
            'total_samples': total_samples,
            'avg_samples_per_client': total_samples / len(client_data),
            'global_class_distribution': global_class_dist.tolist(),
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Test data pipeline
    pipeline = DataPipeline(random_state=42)
    
    # Create synthetic data for testing
    n_samples = 1000
    n_features = 35
    
    # Generate synthetic features
    features_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Generate synthetic metadata
    device_types = ['camera', 'sensor', 'switch', 'hub', 'thermostat']
    metadata_df = pd.DataFrame({
        'device_type': np.random.choice(device_types, n_samples),
        'is_iot': np.ones(n_samples, dtype=int)
    })
    
    print(f"Generated {n_samples} synthetic samples")
    
    # Simulate attacks
    features_attack, metadata_attack = pipeline.simulate_attacks(
        features_df, metadata_df, attack_ratio=0.3
    )
    
    print(f"Attack simulation: {len(features_attack)} total samples")
    
    # Create federated splits
    client_data = pipeline.create_federated_splits(
        features_attack, metadata_attack, num_clients=5, heterogeneity=0.7
    )
    
    # Create meta-learning splits
    client_data = pipeline.create_meta_learning_splits(client_data)
    
    # Get statistics
    stats = pipeline.get_global_statistics(client_data)
    print(f"Global statistics: {stats}")
    
    # Create data loaders
    client_loaders = pipeline.create_dataloaders(client_data, batch_size=16)
    
    print(f"Created DataLoaders for {len(client_loaders)} clients")
    for client_id, loaders in client_loaders.items():
        print(f"  {client_id}: {list(loaders.keys())}")
        break  # Just show first client

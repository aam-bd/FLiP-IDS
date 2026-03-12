"""
Dataset loaders for Phase 1 profiling.

Provides utilities to load and preprocess various IoT datasets including
IoT Sentinel, UNSW-NB15, and other public datasets for training and
evaluation of the device profiling models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import zipfile
import gzip
from urllib.parse import urlparse

from ..common.logging import get_logger
from ..common.utils import ProgressTracker
from ..common.io import save_data, load_data
from ..common.utils import Timer

logger = get_logger(__name__)


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
    
    @abstractmethod
    def load(self, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load dataset.
        
        Returns:
            Tuple of (features_df, labels_df)
        """
        pass
    
    @abstractmethod
    def get_device_types(self) -> List[str]:
        """Get list of device types in the dataset."""
        pass
    
    def download_if_needed(self, url: str, filename: str, 
                          extract: bool = False) -> Path:
        """Download dataset file if it doesn't exist."""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            self.logger.info(f"Dataset file already exists: {filepath}")
            return filepath
        
        self.logger.info(f"Downloading {filename} from {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                if total_size > 0:
                    progress = ProgressTracker(
                        total_size, f"Downloading {filename}", 
                        update_interval=1024*1024  # Update every MB
                    )
                    
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(len(chunk))
                    
                    progress.complete()
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            self.logger.info(f"Downloaded {filename} successfully")
            
            # Extract if needed
            if extract:
                self._extract_file(filepath)
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to download {filename}: {e}")
            if filepath.exists():
                filepath.unlink()
            raise
    
    def _extract_file(self, filepath: Path) -> Path:
        """Extract compressed file."""
        extract_dir = filepath.parent / filepath.stem
        
        if filepath.suffix == '.zip':
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                self.logger.info(f"Extracted {filepath} to {extract_dir}")
        
        elif filepath.suffix == '.gz':
            output_file = filepath.with_suffix('')
            with gzip.open(filepath, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    f_out.write(f_in.read())
            self.logger.info(f"Extracted {filepath} to {output_file}")
            return output_file
        
        return extract_dir


class IoTSentinelLoader(DatasetLoader):
    """
    Loader for IoT Sentinel dataset.
    
    IoT Sentinel is a comprehensive dataset containing network traffic
    from various IoT devices for device identification and behavioral analysis.
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        super().__init__(data_dir)
        
        # IoT Sentinel device types
        self.device_types = [
            'amazon_echo', 'belkin_wemo_switch', 'dropcam', 'google_home',
            'hp_printer', 'insteon_camera', 'lightify_bulb', 'magichome_strip',
            'nest_thermostat', 'netatmo_camera', 'philips_hue', 'ring_doorbell',
            'samsung_smartthings', 'sengled_bulb', 'smart_baby_monitor',
            'tp_link_camera', 'triby_speaker', 'withings_scale'
        ]
    
    def load(self, sample_ratio: float = 1.0, 
             balance_classes: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load IoT Sentinel dataset.
        
        Args:
            sample_ratio: Fraction of data to load (for memory efficiency)
            balance_classes: Whether to balance device type classes
            
        Returns:
            Tuple of (features_df, labels_df)
        """
        # Check if processed data exists
        processed_features = self.data_dir / 'iot_sentinel_features.parquet'
        processed_labels = self.data_dir / 'iot_sentinel_labels.parquet'
        
        if processed_features.exists() and processed_labels.exists():
            self.logger.info("Loading preprocessed IoT Sentinel data...")
            features_df = load_data(processed_features)
            labels_df = load_data(processed_labels)
        else:
            # Generate synthetic IoT Sentinel-like data for demonstration
            self.logger.info("Generating synthetic IoT Sentinel-like data...")
            features_df, labels_df = self._generate_synthetic_data()
            
            # Save processed data
            save_data(features_df, processed_features)
            save_data(labels_df, processed_labels)
        
        # Apply sampling if requested
        if sample_ratio < 1.0:
            n_samples = int(len(features_df) * sample_ratio)
            indices = np.random.choice(len(features_df), n_samples, replace=False)
            features_df = features_df.iloc[indices].reset_index(drop=True)
            labels_df = labels_df.iloc[indices].reset_index(drop=True)
        
        # Balance classes if requested
        if balance_classes:
            features_df, labels_df = self._balance_classes(features_df, labels_df)
        
        self.logger.info(f"Loaded IoT Sentinel data: {len(features_df)} samples, {len(features_df.columns)} features")
        self.logger.info(f"Device type distribution: {labels_df['device_type'].value_counts().to_dict()}")
        
        return features_df, labels_df
    
    def get_device_types(self) -> List[str]:
        """Get list of IoT device types."""
        return self.device_types.copy()
    
    def _generate_synthetic_data(self, n_samples: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic IoT Sentinel-like data for demonstration."""
        np.random.seed(42)
        
        # Feature names (simplified version of the 58 hybrid features)
        feature_names = [
            'packet_length_mean', 'packet_length_std', 'total_bytes_forward',
            'total_bytes_backward', 'flow_duration', 'flow_iat_mean',
            'packets_per_second', 'bytes_per_second', 'tcp_flag_count',
            'syn_flag_count', 'ack_flag_count', 'src_port', 'dst_port',
            'service_http', 'service_https', 'service_dns', 'packet_count_total',
            'down_up_ratio', 'avg_packet_size', 'dns_query_count'
        ]
        
        # Generate features with device-specific patterns
        features_list = []
        labels_list = []
        
        samples_per_device = n_samples // len(self.device_types)
        
        for device_type in self.device_types:
            # Generate device-specific feature patterns
            device_features = self._generate_device_features(
                device_type, samples_per_device, feature_names
            )
            
            features_list.append(device_features)
            
            # Create labels
            device_labels = pd.DataFrame({
                'device_type': [device_type] * samples_per_device,
                'is_iot': [1] * samples_per_device  # All are IoT devices
            })
            labels_list.append(device_labels)
        
        # Add some Non-IoT samples
        non_iot_samples = n_samples // 10
        non_iot_features = self._generate_non_iot_features(non_iot_samples, feature_names)
        features_list.append(non_iot_features)
        
        non_iot_labels = pd.DataFrame({
            'device_type': ['non_iot'] * non_iot_samples,
            'is_iot': [0] * non_iot_samples
        })
        labels_list.append(non_iot_labels)
        
        # Combine all data
        features_df = pd.concat(features_list, ignore_index=True)
        labels_df = pd.concat(labels_list, ignore_index=True)
        
        # Shuffle data
        indices = np.random.permutation(len(features_df))
        features_df = features_df.iloc[indices].reset_index(drop=True)
        labels_df = labels_df.iloc[indices].reset_index(drop=True)
        
        return features_df, labels_df
    
    def _generate_device_features(self, device_type: str, n_samples: int, 
                                feature_names: List[str]) -> pd.DataFrame:
        """Generate synthetic features for a specific device type."""
        # Device-specific parameter patterns
        device_patterns = {
            'amazon_echo': {
                'packet_length_mean': (200, 50), 'flow_duration': (30, 10),
                'service_https': 0.8, 'bytes_per_second': (1000, 300)
            },
            'dropcam': {
                'packet_length_mean': (800, 200), 'flow_duration': (120, 30),
                'service_https': 0.9, 'bytes_per_second': (5000, 1000)
            },
            'philips_hue': {
                'packet_length_mean': (100, 30), 'flow_duration': (10, 5),
                'service_http': 0.7, 'bytes_per_second': (200, 50)
            },
            'nest_thermostat': {
                'packet_length_mean': (150, 40), 'flow_duration': (60, 20),
                'service_https': 0.6, 'bytes_per_second': (500, 100)
            }
        }
        
        # Default pattern for devices not specifically defined
        default_pattern = {
            'packet_length_mean': (300, 100), 'flow_duration': (45, 15),
            'service_http': 0.5, 'bytes_per_second': (800, 200)
        }
        
        pattern = device_patterns.get(device_type, default_pattern)
        
        # Generate features
        features = {}
        
        for feature_name in feature_names:
            if feature_name in pattern:
                if isinstance(pattern[feature_name], tuple):
                    # Gaussian distribution
                    mean, std = pattern[feature_name]
                    features[feature_name] = np.random.normal(mean, std, n_samples)
                else:
                    # Binary feature with probability
                    prob = pattern[feature_name]
                    features[feature_name] = np.random.binomial(1, prob, n_samples)
            else:
                # Default random values
                if 'count' in feature_name or 'port' in feature_name:
                    features[feature_name] = np.random.poisson(5, n_samples)
                elif 'ratio' in feature_name or 'percent' in feature_name:
                    features[feature_name] = np.random.uniform(0, 1, n_samples)
                else:
                    features[feature_name] = np.random.normal(100, 30, n_samples)
        
        # Ensure non-negative values where appropriate
        for feature_name in feature_names:
            if any(keyword in feature_name for keyword in ['length', 'bytes', 'duration', 'count']):
                features[feature_name] = np.maximum(features[feature_name], 0)
        
        return pd.DataFrame(features)
    
    def _generate_non_iot_features(self, n_samples: int, 
                                 feature_names: List[str]) -> pd.DataFrame:
        """Generate synthetic features for Non-IoT devices."""
        features = {}
        
        # Non-IoT devices typically have different traffic patterns
        for feature_name in feature_names:
            if 'packet_length' in feature_name:
                features[feature_name] = np.random.normal(1200, 400, n_samples)
            elif 'bytes' in feature_name:
                features[feature_name] = np.random.normal(10000, 3000, n_samples)
            elif 'duration' in feature_name:
                features[feature_name] = np.random.normal(200, 60, n_samples)
            elif 'service_http' in feature_name:
                features[feature_name] = np.random.binomial(1, 0.3, n_samples)
            elif 'service_https' in feature_name:
                features[feature_name] = np.random.binomial(1, 0.4, n_samples)
            elif 'port' in feature_name:
                features[feature_name] = np.random.choice([80, 443, 22, 21, 25], n_samples)
            else:
                features[feature_name] = np.random.normal(500, 150, n_samples)
        
        # Ensure non-negative values
        for feature_name in feature_names:
            if any(keyword in feature_name for keyword in ['length', 'bytes', 'duration', 'count']):
                features[feature_name] = np.maximum(features[feature_name], 0)
        
        return pd.DataFrame(features)
    
    def _balance_classes(self, features_df: pd.DataFrame, 
                        labels_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Balance device type classes by undersampling majority classes."""
        # Find minimum class size
        class_counts = labels_df['device_type'].value_counts()
        min_count = class_counts.min()
        
        balanced_indices = []
        
        for device_type in class_counts.index:
            device_indices = labels_df[labels_df['device_type'] == device_type].index
            sampled_indices = np.random.choice(
                device_indices, min_count, replace=False
            )
            balanced_indices.extend(sampled_indices)
        
        # Shuffle indices
        balanced_indices = np.random.permutation(balanced_indices)
        
        return (features_df.iloc[balanced_indices].reset_index(drop=True),
                labels_df.iloc[balanced_indices].reset_index(drop=True))


class UNSWLoader(DatasetLoader):
    """
    Loader for UNSW-NB15 dataset adapted for IoT device classification.
    
    The UNSW-NB15 dataset contains network traffic with various attack types
    and can be adapted for IoT security research.
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        super().__init__(data_dir)
        
        self.dataset_urls = {
            'training': 'https://cloudstor.aarnet.edu.au/plus/s/LTDZ7EYzGzJD4Yz/download',
            'testing': 'https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download'
        }
        
        self.device_types = [
            'web_server', 'mail_server', 'workstation', 'mobile_device',
            'iot_sensor', 'iot_camera', 'iot_controller'
        ]
    
    def load(self, subset: str = 'training') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load UNSW-NB15 dataset.
        
        Args:
            subset: Dataset subset ('training' or 'testing')
            
        Returns:
            Tuple of (features_df, labels_df)
        """
        # For demonstration, generate synthetic UNSW-like data
        self.logger.info(f"Generating synthetic UNSW-like data ({subset})...")
        
        n_samples = 15000 if subset == 'training' else 5000
        return self._generate_unsw_synthetic_data(n_samples)
    
    def get_device_types(self) -> List[str]:
        """Get list of device types."""
        return self.device_types.copy()
    
    def _generate_unsw_synthetic_data(self, n_samples: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic UNSW-like data."""
        np.random.seed(42)
        
        # UNSW-NB15 inspired features
        feature_names = [
            'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
            'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit',
            'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack',
            'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len',
            'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
            'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
            'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
        ]
        
        # Generate features
        features = {}
        for feature_name in feature_names:
            if 'ct_' in feature_name or 'pkt' in feature_name:
                features[feature_name] = np.random.poisson(3, n_samples)
            elif 'bytes' in feature_name or 'load' in feature_name:
                features[feature_name] = np.random.lognormal(5, 2, n_samples)
            elif 'is_' in feature_name:
                features[feature_name] = np.random.binomial(1, 0.1, n_samples)
            else:
                features[feature_name] = np.random.normal(0, 1, n_samples)
        
        features_df = pd.DataFrame(features)
        
        # Generate labels with attack types
        attack_types = ['normal', 'dos', 'probe', 'r2l', 'u2r']
        device_types = np.random.choice(self.device_types, n_samples)
        attack_labels = np.random.choice(attack_types, n_samples, 
                                       p=[0.6, 0.2, 0.1, 0.05, 0.05])
        
        # IoT devices are subset of all devices
        is_iot = np.array(['iot_' in dt for dt in device_types]).astype(int)
        
        labels_df = pd.DataFrame({
            'device_type': device_types,
            'attack_type': attack_labels,
            'is_iot': is_iot,
            'is_attack': (attack_labels != 'normal').astype(int)
        })
        
        return features_df, labels_df


def create_dataset_loader(dataset_name: str, data_dir: Union[str, Path]) -> DatasetLoader:
    """
    Factory function to create dataset loaders.
    
    Args:
        dataset_name: Name of the dataset ('iot_sentinel', 'unsw')
        data_dir: Directory to store dataset files
        
    Returns:
        Dataset loader instance
    """
    if dataset_name.lower() == 'iot_sentinel':
        return IoTSentinelLoader(data_dir)
    elif dataset_name.lower() == 'unsw':
        return UNSWLoader(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_multiple_datasets(dataset_names: List[str], 
                          data_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and combine multiple datasets.
    
    Args:
        dataset_names: List of dataset names to load
        data_dir: Directory containing dataset files
        
    Returns:
        Combined (features_df, labels_df)
    """
    all_features = []
    all_labels = []
    
    for dataset_name in dataset_names:
        logger.info(f"Loading dataset: {dataset_name}")
        loader = create_dataset_loader(dataset_name, data_dir)
        features, labels = loader.load()
        
        # Add dataset source column
        labels['dataset_source'] = dataset_name
        
        all_features.append(features)
        all_labels.append(labels)
    
    # Combine datasets
    combined_features = pd.concat(all_features, ignore_index=True, sort=False)
    combined_labels = pd.concat(all_labels, ignore_index=True, sort=False)
    
    # Fill missing features with zeros (different datasets may have different features)
    combined_features = combined_features.fillna(0)
    
    logger.info(f"Combined datasets: {len(combined_features)} samples, {len(combined_features.columns)} features")
    
    return combined_features, combined_labels

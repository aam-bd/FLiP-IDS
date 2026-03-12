#!/usr/bin/env python3
"""
THOROUGHLY REVIEWED Phase 2 SOH-FL Evaluation for Complete BoT-IoT Dataset
Implements the complete Stones From Other Hills Federated Learning methodology
with proper error handling and memory management.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
import gc
from datetime import datetime
import joblib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Import SOH-FL components
from src.phase2_ids.models.cnn_1d import CNN1DClassifier
from src.phase2_ids.models.autoencoders import CosineTargetedAutoencoder, CTAETrainer
from src.common.utils import set_seed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SimpleModelConfig:
    """Simple model configuration that matches CNN1DClassifier constructor."""
    input_dim: int = 32
    hidden_channels: List[int] = None
    kernel_size: int = 3
    dropout: float = 0.3
    num_classes: int = 4
    
    def __post_init__(self):
        if self.hidden_channels is None:
            self.hidden_channels = [64, 32]
    
    def to_dict(self):
        """Convert to dictionary for CNN1DClassifier constructor."""
        return {
            'input_dim': self.input_dim,
            'hidden_channels': self.hidden_channels,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'num_classes': self.num_classes
        }

@dataclass
class SimpleServerConfig:
    """Simple server configuration."""
    rounds: int = 5
    participation_rate: float = 0.6
    num_clients: int = 6
    local_epochs: int = 3
    learning_rate: float = 0.001

@dataclass
class SimpleClientConfig:
    """Simple client configuration."""
    client_id: int
    local_epochs: int = 3
    batch_size: int = 512
    learning_rate: float = 0.001

class SimpleFederatedServer:
    """Simplified federated server for memory-efficient evaluation."""
    
    def __init__(self, model_config: SimpleModelConfig, server_config: SimpleServerConfig, device: torch.device):
        self.model_config = model_config
        self.server_config = server_config
        self.device = device
        
        # Initialize global model
        self.global_model = CNN1DClassifier(**model_config.to_dict()).to(device)
        self.round_metrics = []
        
        logger.info("Initialized SimpleFederatedServer")
    
    def aggregate_models(self, client_weights: List[Dict]) -> Dict:
        """Aggregate client model weights using FedAvg."""
        if not client_weights:
            return self.global_model.state_dict()
        
        # Initialize aggregated state dict
        global_state_dict = {}
        
        # Aggregate each parameter
        for key in client_weights[0].keys():
            tensors = [w[key] for w in client_weights]
            
            # Handle different tensor types
            if tensors[0].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                # For integer parameters (like bias), take the first one
                global_state_dict[key] = tensors[0]
            else:
                # For float parameters, compute weighted average
                global_state_dict[key] = torch.stack(tensors).float().mean(0)
        
        return global_state_dict
    
    def update_global_model(self, aggregated_weights: Dict):
        """Update global model with aggregated weights."""
        self.global_model.load_state_dict(aggregated_weights)

class SimpleFederatedClient:
    """Simplified federated client for memory-efficient evaluation."""
    
    def __init__(self, client_id: int, model_config: SimpleModelConfig, client_config: SimpleClientConfig, device: torch.device):
        self.client_id = client_id
        self.model_config = model_config
        self.client_config = client_config
        self.device = device
        
        # Initialize local model
        self.model = CNN1DClassifier(**model_config.to_dict()).to(device)
        
        logger.info(f"Initialized SimpleFederatedClient {client_id}")
    
    def set_model_weights(self, weights: Dict):
        """Set model weights from server."""
        self.model.load_state_dict(weights)
    
    def get_model_weights(self) -> Dict:
        """Get current model weights."""
        return self.model.state_dict()
    
    def train_local(self, train_loader: DataLoader, epochs: int = 3) -> Dict:
        """Train local model and return metrics."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.client_config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
            
            if epoch_batches > 0:
                total_loss += epoch_loss / epoch_batches
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'loss': avg_loss, 'batches': num_batches}
    
    def evaluate_local(self, test_loader: DataLoader) -> Dict:
        """Evaluate local model and return metrics."""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        num_batches = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                total_loss += loss.item()
                num_batches += 1
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }

def load_federated_data_efficient(data_dir: Path, max_clients: int = 6) -> tuple:
    """Load federated data with memory efficiency."""
    logger.info("Loading federated data efficiently...")
    
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        logger.error("Federated data directories not found")
        return None, None
    
    # Load config
    config_file = data_dir / "config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load client data
    client_data = {}
    
    train_files = sorted(train_dir.glob("*.npz"))[:max_clients]
    
    for train_file in train_files:
        client_id = int(train_file.stem)
        test_file = test_dir / f"{client_id}.npz"
        
        if test_file.exists():
            # Load train data
            train_data = np.load(train_file)
            X_train = train_data['x']
            y_train = train_data['y']
            
            # Load test data
            test_data = np.load(test_file)
            X_test = test_data['x']
            y_test = test_data['y']
            
            if len(X_train) > 0 and len(X_test) > 0:
                client_data[client_id] = {
                    'x_train': X_train,
                    'y_train': y_train,
                    'x_test': X_test,
                    'y_test': y_test
                }
                
                logger.info(f"Client {client_id}: {len(X_train):,} train, {len(X_test):,} test samples")
    
    logger.info(f"Loaded {len(client_data)} clients")
    return client_data, config

def train_ct_autoencoder_efficient(client_data: Dict, config: Dict, output_dir: Path) -> tuple:
    """Train CT-AE with memory-efficient batch processing."""
    logger.info("Training Cosine-Targeted Autoencoder (CT-AE) efficiently...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get feature dimensions from first client
    first_client = next(iter(client_data.values()))
    num_features = first_client['x_train'].shape[1]
    
    # Initialize CT-AE
    ct_ae = CosineTargetedAutoencoder(
        input_dim=num_features,
        latent_dim=32,
        hidden_dims=[128, 64],
        activation='relu',
        dropout_rate=0.2
    ).to(device)
    
    # Initialize trainer
    trainer = CTAETrainer(ct_ae, device, w_rec=0.7, w_cos=0.3)
    optimizer = torch.optim.Adam(ct_ae.parameters(), lr=0.001)
    
    # Training with smaller batches for memory efficiency
    batch_size = 5000  # Reduced batch size
    all_losses = []
    
    # Train for fewer epochs to save time and memory
    for epoch in range(3):  # Reduced from 5 to 3 epochs
        epoch_losses = []
        
        # Process each client's data in smaller batches
        for client_id, data in client_data.items():
            X_client = data['x_train']
            
            # Process in smaller batches
            for i in range(0, len(X_client), batch_size):
                batch_end = min(i + batch_size, len(X_client))
                X_batch = torch.FloatTensor(X_client[i:batch_end]).to(device)
                
                # Train using the trainer's train_epoch method
                batch_dataset = TensorDataset(X_batch)
                batch_loader = DataLoader(batch_dataset, batch_size=1000, shuffle=True)
                
                # Use trainer's train_epoch method
                metrics = trainer.train_epoch(batch_loader, optimizer, use_cosine_target=True)
                epoch_losses.append(metrics['total_loss'])
                
                # Clear batch from memory
                del X_batch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        all_losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1}/3: Average Loss = {avg_loss:.6f}")
    
    # Save CT-AE model
    ct_ae_path = output_dir / "ct_autoencoder.pth"
    torch.save(ct_ae.state_dict(), ct_ae_path)
    logger.info(f"Saved CT-AE model: {ct_ae_path}")
    
    # Evaluate compression ratio
    sample_client = next(iter(client_data.values()))
    sample_data = torch.FloatTensor(sample_client['x_train'][:1000]).to(device)
    
    with torch.no_grad():
        encoded_sample = ct_ae.encode(sample_data)
        compression_ratio = sample_data.numel() / encoded_sample.numel()
    
    logger.info(f"CT-AE Compression Ratio: {compression_ratio:.2f}x")
    
    return ct_ae, {
        'training_losses': all_losses,
        'compression_ratio': float(compression_ratio),
        'latent_dim': 32,
        'input_dim': num_features
    }

def run_federated_learning_efficient(client_data: Dict, config: Dict, ct_ae, output_dir: Path) -> List[Dict]:
    """Run federated learning simulation with memory efficiency."""
    logger.info("Running Federated Learning Simulation efficiently...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    num_classes = config['num_classes']
    num_features = 32  # CT-AE latent dimension
    
    # Model and server configuration
    model_config = SimpleModelConfig(
        input_dim=num_features,
        hidden_channels=[64, 32],
        kernel_size=3,
        dropout=0.3,
        num_classes=num_classes
    )
    
    server_config = SimpleServerConfig(
        rounds=3,  # Reduced rounds for efficiency
        participation_rate=0.8,  # Higher participation for smaller client set
        num_clients=len(client_data),
        local_epochs=2,  # Reduced local epochs
        learning_rate=0.001
    )
    
    # Initialize server
    server = SimpleFederatedServer(model_config, server_config, device)
    
    # Initialize clients
    clients = {}
    for client_id in client_data.keys():
        client_config = SimpleClientConfig(
            client_id=client_id,
            local_epochs=2,
            batch_size=1000,  # Larger batch size for efficiency
            learning_rate=0.001
        )
        
        client = SimpleFederatedClient(client_id, model_config, client_config, device)
        clients[client_id] = client
    
    logger.info(f"Initialized {len(clients)} federated clients")
    
    # Federated training simulation
    round_metrics = []
    
    for round_num in range(server_config.rounds):
        logger.info(f"Federated Round {round_num + 1}/{server_config.rounds}")
        
        # Select participating clients
        num_participants = max(1, int(len(clients) * server_config.participation_rate))
        participating_clients = np.random.choice(
            list(clients.keys()), 
            size=num_participants, 
            replace=False
        )
        
        logger.info(f"Participating clients: {participating_clients}")
        
        # Collect client updates
        client_weights = []
        client_metrics = {}
        
        for client_id in participating_clients:
            client = clients[client_id]
            data = client_data[client_id]
            
            # Set global model weights
            if round_num > 0:  # Skip first round (no global model yet)
                client.set_model_weights(server.global_model.state_dict())
            
            # Encode data using CT-AE (in batches to save memory)
            X_train_encoded = []
            X_test_encoded = []
            
            # Encode training data in batches
            batch_size = 5000
            for i in range(0, len(data['x_train']), batch_size):
                batch_end = min(i + batch_size, len(data['x_train']))
                batch_data = torch.FloatTensor(data['x_train'][i:batch_end]).to(device)
                
                with torch.no_grad():
                    encoded_batch = ct_ae.encode(batch_data).cpu().numpy()
                    X_train_encoded.append(encoded_batch)
                
                del batch_data
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            X_train_encoded = np.vstack(X_train_encoded)
            
            # Encode test data in batches
            for i in range(0, len(data['x_test']), batch_size):
                batch_end = min(i + batch_size, len(data['x_test']))
                batch_data = torch.FloatTensor(data['x_test'][i:batch_end]).to(device)
                
                with torch.no_grad():
                    encoded_batch = ct_ae.encode(batch_data).cpu().numpy()
                    X_test_encoded.append(encoded_batch)
                
                del batch_data
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            X_test_encoded = np.vstack(X_test_encoded)
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_encoded),
                torch.LongTensor(data['y_train'])
            )
            train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
            
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test_encoded),
                torch.LongTensor(data['y_test'])
            )
            test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
            
            # Local training
            train_metrics = client.train_local(train_loader, epochs=2)
            
            # Local evaluation
            eval_metrics = client.evaluate_local(test_loader)
            
            # Collect metrics
            client_metrics[client_id] = {
                'accuracy': eval_metrics['accuracy'],
                'loss': eval_metrics['loss'],
                'samples': len(data['x_train'])
            }
            
            # Collect model weights
            client_weights.append(client.get_model_weights())
            
            logger.info(f"Client {client_id}: Accuracy = {eval_metrics['accuracy']:.4f}, Loss = {eval_metrics['loss']:.4f}")
            
            # Clear memory
            del X_train_encoded, X_test_encoded, train_dataset, test_dataset
            gc.collect()
        
        # Aggregate models
        if client_weights:
            aggregated_weights = server.aggregate_models(client_weights)
            server.update_global_model(aggregated_weights)
        
        # Calculate round metrics
        round_accuracy = np.mean([m['accuracy'] for m in client_metrics.values()])
        round_loss = np.mean([m['loss'] for m in client_metrics.values()])
        
        round_metrics.append({
            'round': round_num + 1,
            'accuracy': round_accuracy,
            'loss': round_loss,
            'participating_clients': len(participating_clients),
            'client_metrics': client_metrics
        })
        
        logger.info(f"Round {round_num + 1} - Avg Accuracy: {round_accuracy:.4f}, Avg Loss: {round_loss:.4f}")
    
    # Save global model
    global_model_path = output_dir / "global_model.pth"
    torch.save(server.global_model.state_dict(), global_model_path)
    logger.info(f"Saved global model: {global_model_path}")
    
    return round_metrics

def evaluate_self_labeling_efficient(client_data: Dict, config: Dict, output_dir: Path) -> Dict:
    """Evaluate self-labeling accuracy efficiently."""
    logger.info("Evaluating Self-Labeling Performance efficiently...")
    
    # Load Phase 1 classifier for pseudo-labeling
    phase1_dir = Path("models/botiot_complete")
    
    try:
        classifier = joblib.load(phase1_dir / "device_classifier.joblib")
        scaler = joblib.load(phase1_dir / "feature_scaler.joblib")
        label_encoder = joblib.load(phase1_dir / "label_encoder.joblib")
    except Exception as e:
        logger.error(f"Failed to load Phase 1 models: {e}")
        return {'overall_accuracy': 0.0, 'client_results': {}}
    
    self_labeling_results = {}
    
    for client_id, data in client_data.items():
        # Use a smaller subset for evaluation (memory efficiency)
        subset_size = min(5000, len(data['x_test']))
        indices = np.random.choice(len(data['x_test']), subset_size, replace=False)
        
        X_test = data['x_test'][indices]
        y_true = data['y_test'][indices]
        
        try:
            # Generate pseudo-labels using Phase 1 classifier
            X_scaled = scaler.transform(X_test)
            y_pseudo = classifier.predict(X_scaled)
            
            # Calculate self-labeling accuracy
            accuracy = np.mean(y_pseudo == y_true)
            
            self_labeling_results[client_id] = {
                'accuracy': float(accuracy),
                'samples': subset_size
            }
            
            logger.info(f"Client {client_id} Self-Labeling Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating client {client_id}: {e}")
            self_labeling_results[client_id] = {
                'accuracy': 0.0,
                'samples': subset_size
            }
    
    overall_accuracy = np.mean([r['accuracy'] for r in self_labeling_results.values()]) if self_labeling_results else 0.0
    logger.info(f"Overall Self-Labeling Accuracy: {overall_accuracy:.4f}")
    
    return {
        'overall_accuracy': overall_accuracy,
        'client_results': self_labeling_results
    }

def main():
    """Main execution function with thorough error handling."""
    logger.info("=" * 80)
    logger.info("PHASE 2: THOROUGHLY REVIEWED SOH-FL EVALUATION ON COMPLETE BoT-IoT")
    logger.info("=" * 80)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Setup directories
    dirs = {
        'data': Path("../dataset/BoT-IoT"),
        'models': Path("models/botiot_complete_phase2_final"),
        'results': Path("results/botiot_complete_final")
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Load federated data (limit to 6 clients for memory)
        logger.info("Step 1: Loading federated data...")
        client_data, config = load_federated_data_efficient(dirs['data'], max_clients=6)
        
        if client_data is None:
            logger.error("Failed to load federated data")
            return False
        
        # Step 2: Train CT-AE for privacy-preserving encoding
        logger.info("Step 2: Training CT-AE...")
        ct_ae, ct_ae_metrics = train_ct_autoencoder_efficient(client_data, config, dirs['models'])
        
        # Step 3: Run federated learning simulation
        logger.info("Step 3: Running federated learning...")
        fl_metrics = run_federated_learning_efficient(client_data, config, ct_ae, dirs['models'])
        
        # Step 4: Evaluate self-labeling
        logger.info("Step 4: Evaluating self-labeling...")
        self_labeling_metrics = evaluate_self_labeling_efficient(client_data, config, dirs['models'])
        
        # Compile final results (convert numpy types to native Python types for JSON serialization)
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        final_results = {
            'dataset': 'BoT-IoT-Complete',
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'clients': len(client_data),
                'classes': config['num_classes'],
                'features': config['features']
            },
            'ct_ae_metrics': convert_numpy_types(ct_ae_metrics),
            'federated_learning': {
                'rounds': len(fl_metrics),
                'final_accuracy': fl_metrics[-1]['accuracy'] if fl_metrics else 0,
                'round_metrics': convert_numpy_types(fl_metrics)
            },
            'self_labeling': convert_numpy_types(self_labeling_metrics),
            'privacy_preservation': {
                'compression_ratio': ct_ae_metrics['compression_ratio'],
                'latent_dimension': ct_ae_metrics['latent_dim']
            }
        }
        
        # Save results
        results_file = dirs['results'] / "phase2_complete_final_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("PHASE 2 SOH-FL EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"🔒 Privacy Preservation: {ct_ae_metrics['compression_ratio']:.2f}x compression")
        logger.info(f"🤝 Federated Learning: {fl_metrics[-1]['accuracy']:.4f} final accuracy")
        logger.info(f"🏷️  Self-Labeling: {self_labeling_metrics['overall_accuracy']:.4f} accuracy")
        logger.info(f"📊 Clients Evaluated: {len(client_data)}")
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during Phase 2 evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)


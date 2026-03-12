"""
Federated training orchestrator for SOH-FL implementation.

Coordinates the complete federated learning process including:
- Server and client initialization
- Multi-round federated training with MAML
- CT-AE encoding and similarity-based aggregation
- Self-labeling workflow execution
- Comprehensive evaluation and reporting
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
import json

from .federation.server import FederatedServer, ServerConfig
from .federation.client import FederatedClient, ClientConfig
from .federation.data_pipe import DataPipeline
from ..common.logging import get_logger, MetricsLogger
from ..common.io import load_config, save_data
from ..common.utils import set_seed, get_device, Timer
from ..common.metrics import federated_metrics_summary

logger = get_logger(__name__)


class FederatedTrainer:
    """
    Orchestrates the complete SOH-FL federated learning process.
    
    Manages server, clients, and coordinates the training workflow
    including standard federated learning and self-labeling phases.
    """
    
    def __init__(self,
                 config: Dict,
                 data_dir: str,
                 device: torch.device,
                 random_state: int = 42):
        """
        Initialize federated trainer.
        
        Args:
            config: Configuration dictionary
            data_dir: Directory containing client data
            device: Computing device
            random_state: Random seed
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.device = device
        self.random_state = random_state
        
        # Initialize components
        self.server: Optional[FederatedServer] = None
        self.clients: Dict[str, FederatedClient] = {}
        self.pipeline = DataPipeline(random_state=random_state)
        
        # Training state
        self.training_history: List[Dict] = []
        self.current_round = 0
        
        # Metrics tracking
        self.logger = logger
        self.metrics_logger = MetricsLogger(logger, "FederatedTrainer")
        
        set_seed(random_state)
    
    def setup_server(self) -> FederatedServer:
        """Initialize federated server."""
        logger.info("Setting up federated server")
        
        # Server configuration
        server_config = ServerConfig(**self.config.get('federation', {}))
        model_config = self.config.get('cnn', {})
        
        # Create server
        self.server = FederatedServer(model_config, server_config, self.device)
        
        logger.info(f"Server initialized with {server_config.num_clients} max clients")
        return self.server
    
    def setup_clients(self) -> Dict[str, FederatedClient]:
        """Initialize federated clients with local data."""
        logger.info(f"Setting up federated clients from {self.data_dir}")
        
        # Load client data
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        client_data = self.pipeline.load_client_data(self.data_dir)
        
        # Create clients
        self.clients = {}
        model_config = self.config.get('cnn', {})
        
        for client_id, data in client_data.items():
            # Client configuration
            client_config = ClientConfig(
                client_id=client_id,
                **self.config.get('client', {})
            )
            
            # Create client
            client = FederatedClient(client_id, model_config, client_config, self.device)
            client.load_local_data(data)
            self.clients[client_id] = client
            
            # Register with server
            if self.server is not None:
                client_info = {
                    'num_samples': data.get('num_samples', 0),
                    'device_types': data.get('device_types', [])
                }
                self.server.register_client(client_id, client_info)
        
        logger.info(f"Initialized {len(self.clients)} clients")
        return self.clients
    
    def run_federated_training(self,
                              num_rounds: int = 50,
                              save_checkpoints: bool = True,
                              checkpoint_dir: str = "checkpoints/") -> Dict[str, Any]:
        """
        Run complete federated training process.
        
        Args:
            num_rounds: Number of federation rounds
            save_checkpoints: Whether to save checkpoints
            checkpoint_dir: Directory for checkpoints
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Starting federated training for {num_rounds} rounds")
        
        # Setup components
        if self.server is None:
            self.setup_server()
        
        if not self.clients:
            self.setup_clients()
        
        # Checkpoint directory
        if save_checkpoints:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        round_results = []
        
        with Timer(f"Federated training ({num_rounds} rounds)", logger):
            for round_num in range(1, num_rounds + 1):
                self.current_round = round_num
                
                logger.info(f"Starting round {round_num}/{num_rounds}")
                
                # Run federated round
                round_result = self._run_single_round()
                round_results.append(round_result)
                
                # Log round metrics
                self.metrics_logger.log_metrics(
                    round_result.get('global_metrics', {}),
                    step=round_num
                )
                
                # Save checkpoint
                if save_checkpoints and round_num % 10 == 0:
                    checkpoint_file = checkpoint_path / f"round_{round_num:03d}.pt"
                    self.server.save_checkpoint(str(checkpoint_file))
        
        # Generate final results
        final_results = self._generate_final_results(round_results)
        
        logger.info("Federated training completed")
        logger.info(f"Final global accuracy: {final_results.get('final_accuracy', 0):.4f}")
        
        return final_results
    
    def _run_single_round(self) -> Dict[str, Any]:
        """Execute a single round of federated learning."""
        # Select participating clients
        selected_clients = self.server.select_clients(self.current_round)
        
        # Get current global model
        global_model_state = self.server.get_global_model()
        
        # Collect client updates
        client_updates = {}
        
        for client_id in selected_clients:
            if client_id in self.clients:
                client = self.clients[client_id]
                
                # Client performs local training
                update = client.federated_round(global_model_state)
                client_updates[client_id] = update
        
        # Server aggregation
        round_result = self.server.federated_round(client_updates)
        
        return round_result
    
    def run_self_labeling_phase(self,
                               target_clients: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run self-labeling phase with CT-AE and BS-Agg.
        
        Args:
            target_clients: Specific clients to run self-labeling for
            
        Returns:
            Self-labeling results
        """
        logger.info("Starting self-labeling phase")
        
        if target_clients is None:
            target_clients = list(self.clients.keys())
        
        self_labeling_results = {}
        
        with Timer("Self-labeling phase", logger):
            # Step 1: Train CT-AE for all clients and encode data
            logger.info("Training CT-AE and encoding data")
            
            for client_id, client in self.clients.items():
                # Train CT-AE
                ct_ae_metrics = client.train_ct_ae()
                
                # Encode historical and current data
                client.encode_data('support')  # Historical
                client.encode_data('test')     # Current
                
                # Update server with encodings
                encodings = client.get_encodings_for_server()
                self.server.update_client_encodings(client_id, encodings)
            
            # Step 2: Run self-labeling workflow for target clients
            logger.info(f"Running self-labeling for {len(target_clients)} clients")
            
            for client_id in target_clients:
                if client_id not in self.clients:
                    continue
                
                client = self.clients[client_id]
                
                # Select helpers using BS-Agg
                helpers = self.server.select_helpers(client_id)
                
                if not helpers:
                    logger.warning(f"No helpers found for {client_id}")
                    continue
                
                # Create annotation model
                annotation_model = self.server.create_annotation_model(client_id, helpers)
                
                # Run self-labeling workflow
                workflow_results = client.self_labeling_workflow(annotation_model)
                
                self_labeling_results[client_id] = {
                    'helpers': helpers,
                    'workflow_results': workflow_results
                }
        
        logger.info(f"Self-labeling completed for {len(self_labeling_results)} clients")
        return self_labeling_results
    
    def evaluate_federation(self) -> Dict[str, Any]:
        """
        Comprehensive evaluation of federated learning results.
        
        Returns:
            Evaluation metrics and analysis
        """
        logger.info("Evaluating federated learning results")
        
        evaluation_results = {}
        
        # Collect final metrics from all clients
        client_metrics = {}
        for client_id, client in self.clients.items():
            final_metrics = client.evaluate_on_query()
            client_metrics[client_id] = final_metrics
        
        # Generate federated summary
        federated_summary = federated_metrics_summary(client_metrics)
        evaluation_results['federated_summary'] = federated_summary
        
        # Server state
        evaluation_results['server_state'] = self.server.get_server_state()
        
        # Training history analysis
        if self.training_history:
            evaluation_results['training_analysis'] = self._analyze_training_history()
        
        # Client state analysis
        client_states = {}
        for client_id, client in self.clients.items():
            client_states[client_id] = client.get_client_state()
        evaluation_results['client_states'] = client_states
        
        return evaluation_results
    
    def _analyze_training_history(self) -> Dict[str, Any]:
        """Analyze training history for trends and insights."""
        if not self.training_history:
            return {}
        
        analysis = {}
        
        # Extract global metrics over rounds
        global_metrics_history = defaultdict(list)
        for round_data in self.training_history:
            for metric_name, value in round_data.get('global_metrics', {}).items():
                global_metrics_history[metric_name].append(value)
        
        # Calculate trends
        for metric_name, values in global_metrics_history.items():
            if len(values) >= 2:
                # Linear trend (simple slope calculation)
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                
                analysis[f'{metric_name}_trend'] = {
                    'slope': float(slope),
                    'initial': float(values[0]),
                    'final': float(values[-1]),
                    'improvement': float(values[-1] - values[0])
                }
        
        # Convergence analysis
        if 'global_accuracy' in global_metrics_history:
            accuracy_values = global_metrics_history['global_accuracy']
            if len(accuracy_values) >= 10:
                # Check if converged (small changes in last 10 rounds)
                recent_std = np.std(accuracy_values[-10:])
                analysis['convergence'] = {
                    'converged': recent_std < 0.01,
                    'recent_std': float(recent_std),
                    'final_accuracy': float(accuracy_values[-1])
                }
        
        return analysis
    
    def _generate_final_results(self, round_results: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive final results."""
        final_results = {
            'total_rounds': len(round_results),
            'num_clients': len(self.clients),
            'round_results': round_results
        }
        
        # Extract final metrics
        if round_results:
            final_round = round_results[-1]
            final_results['final_round'] = final_round
            final_results['final_accuracy'] = final_round.get('global_metrics', {}).get('global_accuracy', 0)
            final_results['final_loss'] = final_round.get('global_metrics', {}).get('global_loss', float('inf'))
        
        # Client metrics summary
        final_client_metrics = {}
        for client_id, client in self.clients.items():
            if client.training_history:
                final_client_metrics[client_id] = client.training_history[-1]
        
        if final_client_metrics:
            final_results['client_metrics'] = final_client_metrics
            final_results['federated_summary'] = federated_metrics_summary(final_client_metrics)
        
        # Server summary
        final_results['server_summary'] = self.server.get_federated_summary()
        
        return final_results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save training results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data(results, output_path, format='json')
        logger.info(f"Results saved to {output_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        if self.server is not None:
            self.server.load_checkpoint(checkpoint_path)
            logger.info(f"Checkpoint loaded from {checkpoint_path}")


# Utility functions for running federated training
def run_federated_experiment(config_path: str,
                            data_dir: str,
                            output_dir: str,
                            num_rounds: int = 50,
                            run_self_labeling: bool = True,
                            device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Run complete federated learning experiment.
    
    Args:
        config_path: Path to configuration file
        data_dir: Directory containing client data
        output_dir: Output directory for results
        num_rounds: Number of federation rounds
        run_self_labeling: Whether to run self-labeling phase
        device: Computing device
        
    Returns:
        Experiment results
    """
    if device is None:
        device = get_device()
    
    # Load configuration
    config = load_config(config_path)
    
    # Initialize trainer
    trainer = FederatedTrainer(config, data_dir, device)
    
    # Run federated training
    training_results = trainer.run_federated_training(
        num_rounds=num_rounds,
        save_checkpoints=True,
        checkpoint_dir=f"{output_dir}/checkpoints"
    )
    
    # Run self-labeling phase if requested
    if run_self_labeling:
        self_labeling_results = trainer.run_self_labeling_phase()
        training_results['self_labeling_results'] = self_labeling_results
    
    # Comprehensive evaluation
    evaluation_results = trainer.evaluate_federation()
    training_results['evaluation'] = evaluation_results
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    trainer.save_results(training_results, output_path / "experiment_results.json")
    
    return training_results


if __name__ == "__main__":
    # Example usage
    config_path = "config/phase2_federation.yaml"
    data_dir = "data/processed/phase2_local"
    output_dir = "results/federated_experiment"
    
    # Run experiment
    results = run_federated_experiment(
        config_path=config_path,
        data_dir=data_dir,
        output_dir=output_dir,
        num_rounds=10,  # Reduced for testing
        run_self_labeling=True
    )
    
    print(f"Experiment completed. Final accuracy: {results.get('final_accuracy', 0):.4f}")
    print(f"Results saved to {output_dir}")

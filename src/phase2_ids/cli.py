"""
Command-line interface for Phase 2 federated learning operations.

Provides CLI commands for:
- Data preparation from Phase 1 profiles
- Federated learning coordination and execution
- CT-AE encoding and similarity computation
- Self-labeling workflow management
- Model evaluation and analysis
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict
import yaml
import torch
import numpy as np
import pandas as pd

from .federation.server import FederatedServer, ServerConfig, create_server_from_config
from .federation.client import FederatedClient, ClientConfig, create_client_from_config
from .federation.data_pipe import DataPipeline
from .federation.data_pipe_toniot import TONIoTDataPipeline
from .train_federated import FederatedTrainer
from ..common.logging import setup_logging, get_logger
from ..common.io import load_config, save_data, load_data
from ..common.utils import set_seed, get_device, Timer
from ..common.metrics import plot_federated_metrics

logger = get_logger(__name__)


def setup_cli_logging(verbose: bool = False):
    """Setup logging for CLI operations."""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level, include_timestamp=True)


def prepare_local_data_command(args):
    """Prepare local datasets from Phase 1 profiles."""
    setup_cli_logging(args.verbose)
    
    profiles_path = Path(args.profiles)
    if not profiles_path.exists():
        logger.error(f"Profiles file not found: {profiles_path}")
        return 1
    
    output_dir = Path(args.output)
    
    logger.info(f"Preparing federated data from {profiles_path}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        with Timer("Data preparation", logger):
            # Initialize pipeline
            pipeline = DataPipeline(random_state=args.random_state)
            
            # Load Phase 1 profiles
            features_df, metadata_df = pipeline.load_phase1_data(profiles_path)
            
            # Simulate attacks
            logger.info(f"Simulating attacks with ratio {args.attack_ratio}")
            features_attack, metadata_attack = pipeline.simulate_attacks(
                features_df, metadata_df, attack_ratio=args.attack_ratio
            )
            
            # Create federated splits
            logger.info(f"Creating federated splits for {args.num_clients} clients")
            client_data = pipeline.create_federated_splits(
                features_attack, metadata_attack,
                num_clients=args.num_clients,
                heterogeneity=args.heterogeneity
            )
            
            # Create meta-learning splits
            client_data = pipeline.create_meta_learning_splits(
                client_data, support_ratio=args.support_ratio
            )
            
            # Save client data
            pipeline.save_client_data(client_data, output_dir)
            
            # Generate statistics
            stats = pipeline.get_global_statistics(client_data)
            stats_path = output_dir / "global_statistics.json"
            save_data(stats, stats_path, format='json')
            
            logger.info(f"Data preparation completed:")
            logger.info(f"  {stats['num_clients']} clients created")
            logger.info(f"  {stats['total_samples']} total samples")
            logger.info(f"  Attack types: {stats['class_names']}")
            logger.info(f"Results saved to {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        return 1


def run_federation_command(args):
    """Run federated learning training."""
    setup_cli_logging(args.verbose)
    set_seed(args.random_state)
    
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    logger.info(f"Running federated learning with config: {config_path}")
    
    try:
        with Timer("Federated learning", logger):
            # Load configuration
            config = load_config(config_path)
            
            # Initialize trainer
            trainer = FederatedTrainer(
                config=config,
                data_dir=args.data_dir,
                device=get_device(args.use_cuda),
                random_state=args.random_state
            )
            
            # Run training
            results = trainer.run_federated_training(
                num_rounds=args.rounds,
                save_checkpoints=True,
                checkpoint_dir=args.checkpoint_dir
            )
            
            # Save results
            output_dir = Path(args.output) if args.output else Path("results/federation")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_path = output_dir / "federation_results.json"
            save_data(results, results_path, format='json')
            
            # Generate plots
            if args.plot_metrics:
                metrics_plot = plot_federated_metrics(
                    results.get('client_metrics', {}),
                    save_path=output_dir / "federation_metrics.png"
                )
            
            logger.info(f"Federated learning completed:")
            logger.info(f"  {results.get('total_rounds', 0)} rounds executed")
            logger.info(f"  {results.get('num_clients', 0)} clients participated")
            logger.info(f"  Final global accuracy: {results.get('final_accuracy', 0):.4f}")
            logger.info(f"Results saved to {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Federated learning failed: {e}")
        return 1


def encode_and_aggregate_command(args):
    """Run CT-AE encoding and similarity-based aggregation."""
    setup_cli_logging(args.verbose)
    
    logger.info("Running CT-AE encoding and similarity-based aggregation")
    
    try:
        with Timer("Encoding and aggregation", logger):
            # Load configuration
            config = load_config(args.config)
            device = get_device(args.use_cuda)
            
            # Initialize server
            server = create_server_from_config(args.config, device)
            
            # Load client data
            data_dir = Path(args.data_dir)
            pipeline = DataPipeline()
            client_data = pipeline.load_client_data(data_dir)
            
            # Initialize clients
            clients = {}
            for client_id, data in client_data.items():
                client = create_client_from_config(client_id, config, device)
                client.load_local_data(data)
                clients[client_id] = client
                
                # Register with server
                server.register_client(client_id, {
                    'num_samples': data.get('num_samples', 0),
                    'device_types': data.get('device_types', [])
                })
            
            # Train CT-AE for each client and encode data
            encoding_results = {}
            for client_id, client in clients.items():
                logger.info(f"Training CT-AE for {client_id}")
                
                # Train CT-AE
                ct_ae_metrics = client.train_ct_ae()
                
                # Encode historical and current data
                historical_encoding = client.encode_data('support')
                current_encoding = client.encode_data('test')
                
                # Update server with encodings
                server_encodings = client.get_encodings_for_server()
                server.update_client_encodings(client_id, server_encodings)
                
                encoding_results[client_id] = {
                    'ct_ae_metrics': ct_ae_metrics,
                    'historical_shape': historical_encoding.shape,
                    'current_shape': current_encoding.shape
                }
            
            # Perform similarity-based aggregation for each client
            aggregation_results = {}
            for client_id in clients.keys():
                logger.info(f"Running BS-Agg for {client_id}")
                
                # Compute similarities
                similarities = server.compute_similarity_matrix(client_id)
                
                # Select helpers
                helpers = server.select_helpers(client_id, gamma=args.gamma)
                
                # Create annotation model
                if helpers:
                    annotation_model = server.create_annotation_model(client_id, helpers)
                    
                    # Run self-labeling workflow
                    workflow_results = clients[client_id].self_labeling_workflow(annotation_model)
                    
                    aggregation_results[client_id] = {
                        'similarities': similarities,
                        'helpers': helpers,
                        'workflow_results': workflow_results
                    }
                else:
                    logger.warning(f"No helpers found for {client_id}")
                    aggregation_results[client_id] = {
                        'similarities': similarities,
                        'helpers': [],
                        'workflow_results': {}
                    }
            
            # Save results
            output_dir = Path(args.output) if args.output else Path("results/encoding_aggregation")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save encoding results
            encoding_path = output_dir / "encoding_results.json"
            save_data(encoding_results, encoding_path, format='json')
            
            # Save aggregation results
            aggregation_path = output_dir / "aggregation_results.json"
            save_data(aggregation_results, aggregation_path, format='json')
            
            # Summary statistics
            total_clients = len(clients)
            successful_aggregations = sum(1 for r in aggregation_results.values() if r['helpers'])
            avg_helpers = np.mean([len(r['helpers']) for r in aggregation_results.values()])
            
            logger.info(f"Encoding and aggregation completed:")
            logger.info(f"  {total_clients} clients processed")
            logger.info(f"  {successful_aggregations} successful aggregations")
            logger.info(f"  Average helpers per client: {avg_helpers:.1f}")
            logger.info(f"Results saved to {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Encoding and aggregation failed: {e}")
        return 1


def evaluate_federation_command(args):
    """Evaluate federated learning results."""
    setup_cli_logging(args.verbose)
    
    results_path = Path(args.results_path)
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return 1
    
    logger.info(f"Evaluating federation results from {results_path}")
    
    try:
        # Load results
        results = load_data(results_path)
        
        # Extract metrics
        client_metrics = results.get('client_metrics', {})
        global_metrics = results.get('global_metrics', {})
        
        if not client_metrics:
            logger.error("No client metrics found in results")
            return 1
        
        # Calculate summary statistics
        from ..common.metrics import federated_metrics_summary
        summary = federated_metrics_summary(client_metrics, global_metrics)
        
        # Log summary
        logger.info("Federation Evaluation Summary:")
        logger.info(f"  Number of clients: {summary.get('num_clients', 0)}")
        
        for metric_name, stats in summary.get('per_metric_stats', {}).items():
            logger.info(f"  {metric_name}:")
            logger.info(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            logger.info(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # Generate plots if requested
        if args.plot_results:
            output_dir = Path(args.output) if args.output else Path("results/evaluation")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot metrics
            for metric_name in ['accuracy', 'f1_score', 'precision', 'recall']:
                if any(metric_name in metrics for metrics in client_metrics.values()):
                    plot_path = output_dir / f"{metric_name}_plot.png"
                    plot_federated_metrics(
                        client_metrics, metric_name=metric_name,
                        save_path=plot_path
                    )
        
        # Save evaluation summary
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            summary_path = output_dir / "evaluation_summary.json"
            save_data(summary, summary_path, format='json')
            
            logger.info(f"Evaluation summary saved to {summary_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


def simulate_zero_day_command(args):
    """Simulate zero-day attack detection."""
    setup_cli_logging(args.verbose)
    
    logger.info("Simulating zero-day attack detection")
    
    try:
        # This would implement a simulation of zero-day attacks
        # and test the self-labeling capability
        
        logger.info("Zero-day simulation not yet implemented")
        return 0
        
    except Exception as e:
        logger.error(f"Zero-day simulation failed: {e}")
        return 1


def create_parser():
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Phase 2 Federated Learning IDS CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--use-cuda', action='store_true',
                       help='Use CUDA if available')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Prepare local data command
    prepare_parser = subparsers.add_parser('prepare-local', help='Prepare federated datasets')
    prepare_parser.add_argument('--profiles', required=True,
                               help='Path to Phase 1 profiles file')
    prepare_parser.add_argument('--output', default='data/processed/phase2_local/',
                               help='Output directory for client data')
    prepare_parser.add_argument('--num-clients', type=int, default=10,
                               help='Number of federated clients')
    prepare_parser.add_argument('--heterogeneity', type=float, default=0.7,
                               help='Statistical heterogeneity level (0-1)')
    prepare_parser.add_argument('--attack-ratio', type=float, default=0.3,
                               help='Fraction of samples to convert to attacks')
    prepare_parser.add_argument('--support-ratio', type=float, default=0.6,
                               help='Fraction of training data for support set')
    prepare_parser.set_defaults(func=prepare_local_data_command)
    
    # Run federation command
    federation_parser = subparsers.add_parser('run-federation', help='Run federated learning')
    federation_parser.add_argument('--config', default='config/phase2_federation.yaml',
                                  help='Configuration file path')
    federation_parser.add_argument('--data-dir', default='data/processed/phase2_local/',
                                  help='Directory containing client data')
    federation_parser.add_argument('--rounds', type=int, default=50,
                                  help='Number of federation rounds')
    federation_parser.add_argument('--output', help='Output directory for results')
    federation_parser.add_argument('--checkpoint-dir', default='checkpoints/',
                                  help='Directory for saving checkpoints')
    federation_parser.add_argument('--plot-metrics', action='store_true',
                                  help='Generate metric plots')
    federation_parser.set_defaults(func=run_federation_command)
    
    # Encode and aggregate command
    encode_parser = subparsers.add_parser('encode-and-aggregate', 
                                         help='Run CT-AE encoding and BS-Agg')
    encode_parser.add_argument('--config', default='config/phase2_federation.yaml',
                              help='Configuration file path')
    encode_parser.add_argument('--data-dir', default='data/processed/phase2_local/',
                              help='Directory containing client data')
    encode_parser.add_argument('--gamma', type=int, default=3,
                              help='Number of top helpers for BS-Agg')
    encode_parser.add_argument('--output', help='Output directory for results')
    encode_parser.set_defaults(func=encode_and_aggregate_command)
    
    # Evaluate federation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate federation results')
    eval_parser.add_argument('results_path', help='Path to federation results file')
    eval_parser.add_argument('--output', help='Output directory for evaluation')
    eval_parser.add_argument('--plot-results', action='store_true',
                            help='Generate evaluation plots')
    eval_parser.set_defaults(func=evaluate_federation_command)
    
    # Simulate zero-day command
    zeroday_parser = subparsers.add_parser('simulate-zero-day', 
                                          help='Simulate zero-day attack detection')
    zeroday_parser.add_argument('--config', default='config/phase2_federation.yaml',
                               help='Configuration file path')
    zeroday_parser.add_argument('--data-dir', default='data/processed/phase2_local/',
                               help='Directory containing client data')
    zeroday_parser.add_argument('--attack-types', nargs='+',
                               default=['zero_day', 'advanced_persistent_threat'],
                               help='Types of zero-day attacks to simulate')
    zeroday_parser.add_argument('--output', help='Output directory for results')
    zeroday_parser.set_defaults(func=simulate_zero_day_command)
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

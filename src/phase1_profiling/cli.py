"""
Command-line interface for Phase 1 profiling operations.

Provides CLI commands for:
- Feature extraction from PCAP files
- Training IoT and device type classifiers
- Device identification and profiling
- Model evaluation and analysis
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np

from .pcap_reader import FlowExtractor
from .feature_extractor import HybridFeatureExtractor
from .selectors import RandomForestSelector, compare_selectors
from .train_identifiers import IoTClassifier, DeviceTypeClassifier, TwoStageClassifier
from .datasets import create_dataset_loader, load_multiple_datasets
from ..common.logging import setup_logging, get_logger
from ..common.io import save_data, load_data, save_model, load_model, load_config
from ..common.metrics import calculate_metrics, confusion_matrix_plot
from ..common.utils import set_seed, Timer

logger = get_logger(__name__)


def setup_cli_logging(verbose: bool = False):
    """Setup logging for CLI operations."""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level, include_timestamp=True)


def extract_features_command(args):
    """Extract features from PCAP file."""
    setup_cli_logging(args.verbose)
    
    pcap_path = Path(args.pcap)
    if not pcap_path.exists():
        logger.error(f"PCAP file not found: {pcap_path}")
        return 1
    
    output_path = Path(args.output) if args.output else pcap_path.with_suffix('.csv')
    
    logger.info(f"Extracting features from {pcap_path}")
    logger.info(f"Output will be saved to {output_path}")
    
    try:
        with Timer("Feature extraction", logger):
            # Extract flows
            flow_extractor = FlowExtractor(
                flow_timeout=args.flow_timeout,
                window_size=args.window_size
            )
            flows = flow_extractor.extract_flows(pcap_path)
            
            if not flows:
                logger.warning("No flows extracted from PCAP file")
                return 1
            
            # Extract features
            feature_extractor = HybridFeatureExtractor()
            features_df = feature_extractor.extract_features_dataframe(flows)
            
            # Add flow metadata
            flow_df = flow_extractor.flows_to_dataframe(flows)
            
            # Combine flow info and features
            combined_df = pd.concat([flow_df, features_df], axis=1)
            
            # Save results
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_data(combined_df, output_path, format='csv')
            
            logger.info(f"Extracted {len(combined_df)} flows with {len(features_df.columns)} features")
            logger.info(f"Results saved to {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return 1


def train_iot_classifier_command(args):
    """Train IoT vs Non-IoT classifier."""
    setup_cli_logging(args.verbose)
    set_seed(args.random_state)
    
    logger.info("Training IoT vs Non-IoT classifier")
    
    try:
        with Timer("IoT classifier training", logger):
            # Load dataset
            if args.dataset_files:
                # Load from specific files
                features_df = pd.concat([load_data(f) for f in args.dataset_files], ignore_index=True)
                
                # Assume labels are in the last column or specified
                if args.label_column:
                    y = features_df[args.label_column].values
                    X = features_df.drop(columns=[args.label_column]).values
                    feature_names = features_df.drop(columns=[args.label_column]).columns.tolist()
                else:
                    logger.error("Label column not specified for custom dataset files")
                    return 1
            else:
                # Load from dataset loader
                loader = create_dataset_loader(args.dataset, args.data_dir)
                features_df, labels_df = loader.load()
                
                # Extract features and labels
                feature_extractor = HybridFeatureExtractor()
                expected_features = feature_extractor.feature_names
                available_features = [col for col in expected_features if col in features_df.columns]
                
                X = features_df[available_features].values
                y = labels_df['is_iot'].values
                feature_names = available_features
            
            logger.info(f"Dataset loaded: {len(X)} samples, {len(feature_names)} features")
            logger.info(f"Class distribution: IoT={np.sum(y)}, Non-IoT={len(y) - np.sum(y)}")
            
            # Feature selection
            if args.feature_selection:
                logger.info(f"Performing feature selection (top {args.n_features})")
                selector = RandomForestSelector(
                    n_features=args.n_features,
                    threshold_alpha=args.selector_alpha
                )
                selector.fit(X, y, feature_names)
                X_selected = selector.transform(X)
                
                # Save feature selector
                selector_path = Path(args.model_output).with_suffix('.selector.joblib')
                save_model(selector, selector_path)
                logger.info(f"Feature selector saved to {selector_path}")
                
                # Evaluate feature selection
                eval_results = selector.evaluate_selection(X, y)
                logger.info(f"Feature selection evaluation: {eval_results}")
                
                selected_features = selector.get_selected_feature_names()
                X_train, feature_names_train = X_selected, selected_features
            else:
                X_train, feature_names_train = X, feature_names
            
            # Train classifier
            logger.info("Training Random Forest classifier")
            rf_params = {
                'n_estimators': args.n_estimators,
                'max_depth': args.max_depth,
                'class_weight': 'balanced' if args.balance_classes else None,
                'random_state': args.random_state
            }
            
            classifier = IoTClassifier(rf_params=rf_params, random_state=args.random_state)
            classifier.fit(X_train, y, feature_names_train, validation_split=args.test_size)
            
            # Cross-validation evaluation
            cv_results = classifier.cross_validate(X_train, y, cv=args.cv_folds)
            logger.info(f"Cross-validation results: {cv_results}")
            
            # Save model
            model_path = Path(args.model_output)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            classifier.save_model(model_path)
            logger.info(f"IoT classifier saved to {model_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"IoT classifier training failed: {e}")
        return 1


def train_device_classifier_command(args):
    """Train device type classifier."""
    setup_cli_logging(args.verbose)
    set_seed(args.random_state)
    
    logger.info("Training device type classifier")
    
    try:
        with Timer("Device classifier training", logger):
            # Load dataset
            if args.dataset_files:
                # Load from specific files
                features_df = pd.concat([load_data(f) for f in args.dataset_files], ignore_index=True)
                
                if args.label_column:
                    y = features_df[args.label_column].values
                    X = features_df.drop(columns=[args.label_column]).values
                    feature_names = features_df.drop(columns=[args.label_column]).columns.tolist()
                else:
                    logger.error("Label column not specified for custom dataset files")
                    return 1
            else:
                # Load from dataset loader
                loader = create_dataset_loader(args.dataset, args.data_dir)
                features_df, labels_df = loader.load()
                
                # Filter only IoT devices
                iot_mask = labels_df['is_iot'] == 1
                features_iot = features_df[iot_mask]
                labels_iot = labels_df[iot_mask]
                
                # Extract features and labels
                feature_extractor = HybridFeatureExtractor()
                expected_features = feature_extractor.feature_names
                available_features = [col for col in expected_features if col in features_iot.columns]
                
                X = features_iot[available_features].values
                y = labels_iot['device_type'].values
                feature_names = available_features
            
            logger.info(f"Dataset loaded: {len(X)} samples, {len(feature_names)} features")
            logger.info(f"Device types: {np.unique(y)}")
            logger.info(f"Device type distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
            
            # Apply feature selection if selector is available
            if args.selector_path and Path(args.selector_path).exists():
                logger.info(f"Loading feature selector from {args.selector_path}")
                selector = load_model(args.selector_path)
                X = selector.transform(X)
                feature_names = selector.get_selected_feature_names()
            
            # Train classifier
            logger.info("Training Random Forest device classifier")
            rf_params = {
                'n_estimators': args.n_estimators,
                'max_depth': args.max_depth,
                'class_weight': 'balanced' if args.balance_classes else None,
                'random_state': args.random_state
            }
            
            classifier = DeviceTypeClassifier(rf_params=rf_params, random_state=args.random_state)
            classifier.fit(X, y, feature_names, validation_split=args.test_size)
            
            # Cross-validation evaluation
            cv_results = classifier.cross_validate(X, y, cv=args.cv_folds)
            logger.info(f"Cross-validation results: {cv_results}")
            
            # Save model
            model_path = Path(args.model_output)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            classifier.save_model(model_path)
            logger.info(f"Device classifier saved to {model_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Device classifier training failed: {e}")
        return 1


def identify_devices_command(args):
    """Identify devices from extracted features."""
    setup_cli_logging(args.verbose)
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    logger.info(f"Identifying devices from {input_path}")
    
    try:
        with Timer("Device identification", logger):
            # Load features
            features_df = load_data(input_path)
            
            # Prepare features
            feature_extractor = HybridFeatureExtractor()
            expected_features = feature_extractor.feature_names
            available_features = [col for col in expected_features if col in features_df.columns]
            
            # Add missing features with default values
            for feature in expected_features:
                if feature not in features_df.columns:
                    features_df[feature] = 0.0
            
            X = features_df[expected_features].values
            
            # Apply feature selection if available
            if args.selector_path and Path(args.selector_path).exists():
                selector = load_model(args.selector_path)
                X = selector.transform(X)
            
            results = {}
            
            # IoT vs Non-IoT classification
            if args.iot_model_path:
                iot_model_path = Path(args.iot_model_path)
                if not iot_model_path.exists():
                    logger.error(f"IoT model not found: {iot_model_path}")
                    return 1
                
                logger.info("Loading IoT classifier...")
                iot_classifier = IoTClassifier.load_model(iot_model_path)
                
                iot_predictions = iot_classifier.predict(X)
                iot_probabilities = iot_classifier.predict_proba(X)
                
                results['iot_predictions'] = iot_predictions
                results['iot_probabilities'] = iot_probabilities
                results['iot_count'] = int(np.sum(iot_predictions))
                results['non_iot_count'] = int(len(iot_predictions) - np.sum(iot_predictions))
                
                logger.info(f"IoT devices: {results['iot_count']}, Non-IoT: {results['non_iot_count']}")
            
            # Device type classification
            if args.device_model_path and 'iot_predictions' in results:
                device_model_path = Path(args.device_model_path)
                if not device_model_path.exists():
                    logger.error(f"Device model not found: {device_model_path}")
                    return 1
                
                logger.info("Loading device type classifier...")
                device_classifier = DeviceTypeClassifier.load_model(device_model_path)
                
                iot_mask = iot_predictions == 1
                if np.any(iot_mask):
                    X_iot = X[iot_mask]
                    device_predictions = device_classifier.predict(X_iot)
                    device_probabilities = device_classifier.predict_proba(X_iot)
                    
                    # Create full predictions array
                    full_device_predictions = np.full(len(X), 'Non-IoT', dtype=object)
                    full_device_predictions[iot_mask] = device_predictions
                    
                    results['device_predictions'] = full_device_predictions
                    results['device_type_counts'] = dict(zip(*np.unique(device_predictions, return_counts=True)))
                    
                    logger.info(f"Device type distribution: {results['device_type_counts']}")
            
            # Save results
            output_path = Path(args.output) if args.output else input_path.with_suffix('.results.json')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_data(results, output_path, format='json')
            
            # Also save as CSV for easy viewing
            csv_output = output_path.with_suffix('.csv')
            results_df = features_df.copy()
            
            if 'iot_predictions' in results:
                results_df['is_iot'] = results['iot_predictions']
            if 'device_predictions' in results:
                results_df['device_type'] = results['device_predictions']
            
            save_data(results_df, csv_output, format='csv')
            
            logger.info(f"Results saved to {output_path} and {csv_output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Device identification failed: {e}")
        return 1


def evaluate_model_command(args):
    """Evaluate trained model performance."""
    setup_cli_logging(args.verbose)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1
    
    logger.info(f"Evaluating model: {model_path}")
    
    try:
        with Timer("Model evaluation", logger):
            # Load test data
            if args.test_data:
                test_df = load_data(args.test_data)
                
                if args.label_column:
                    y_true = test_df[args.label_column].values
                    X_test = test_df.drop(columns=[args.label_column]).values
                else:
                    logger.error("Label column not specified for test data")
                    return 1
            else:
                # Load from dataset
                loader = create_dataset_loader(args.dataset, args.data_dir)
                features_df, labels_df = loader.load()
                
                feature_extractor = HybridFeatureExtractor()
                expected_features = feature_extractor.feature_names
                available_features = [col for col in expected_features if col in features_df.columns]
                
                X_test = features_df[available_features].values
                
                if args.model_type == 'iot':
                    y_true = labels_df['is_iot'].values
                else:
                    # Device type classification - filter IoT only
                    iot_mask = labels_df['is_iot'] == 1
                    X_test = X_test[iot_mask]
                    y_true = labels_df[iot_mask]['device_type'].values
            
            # Load model
            if args.model_type == 'iot':
                model = IoTClassifier.load_model(model_path)
            else:
                model = DeviceTypeClassifier.load_model(model_path)
            
            # Apply feature selection if available
            if args.selector_path and Path(args.selector_path).exists():
                selector = load_model(args.selector_path)
                X_test = selector.transform(X_test)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            # Calculate metrics
            if args.model_type == 'iot':
                metrics = calculate_metrics(y_true, y_pred, y_prob, 
                                          labels=['Non-IoT', 'IoT'], average='binary')
            else:
                metrics = calculate_metrics(y_true, y_pred, y_prob, average='macro')
            
            # Log results
            logger.info("Evaluation Results:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
            
            # Save detailed results
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save metrics
                save_data(metrics, output_path.with_suffix('.metrics.json'), format='json')
                
                # Save predictions
                results_df = pd.DataFrame({
                    'true_label': y_true,
                    'predicted_label': y_pred,
                })
                
                # Add probabilities
                if y_prob.ndim == 2:
                    for i in range(y_prob.shape[1]):
                        results_df[f'prob_class_{i}'] = y_prob[:, i]
                else:
                    results_df['probability'] = y_prob
                
                save_data(results_df, output_path.with_suffix('.predictions.csv'), format='csv')
                
                # Generate confusion matrix plot
                if args.plot_confusion:
                    cm_fig = confusion_matrix_plot(
                        y_true, y_pred, 
                        title=f"Confusion Matrix - {args.model_type.title()} Classifier",
                        save_path=output_path.with_suffix('.confusion_matrix.png')
                    )
                
                logger.info(f"Detailed results saved to {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return 1


def compare_feature_selectors_command(args):
    """Compare different feature selection methods."""
    setup_cli_logging(args.verbose)
    set_seed(args.random_state)
    
    logger.info("Comparing feature selection methods")
    
    try:
        with Timer("Feature selector comparison", logger):
            # Load dataset
            loader = create_dataset_loader(args.dataset, args.data_dir)
            features_df, labels_df = loader.load()
            
            # Extract features
            feature_extractor = HybridFeatureExtractor()
            expected_features = feature_extractor.feature_names
            available_features = [col for col in expected_features if col in features_df.columns]
            
            X = features_df[available_features].values
            y = labels_df['is_iot'].values
            
            logger.info(f"Dataset: {len(X)} samples, {len(available_features)} features")
            
            # Compare selectors
            results = compare_selectors(X, y, available_features, args.n_features)
            
            # Log comparison results
            logger.info("Feature Selector Comparison Results:")
            for method, result in results.items():
                logger.info(f"  {method:15s}: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
            
            # Save results
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                save_data(results, output_path, format='json')
                logger.info(f"Comparison results saved to {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Feature selector comparison failed: {e}")
        return 1


def create_parser():
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Phase 1 IoT Device Profiling CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract features command
    extract_parser = subparsers.add_parser('extract', help='Extract features from PCAP file')
    extract_parser.add_argument('pcap', help='Path to PCAP file')
    extract_parser.add_argument('--output', '-o', help='Output CSV file path')
    extract_parser.add_argument('--window-size', type=int, default=60,
                               help='Time window size in seconds')
    extract_parser.add_argument('--flow-timeout', type=int, default=120,
                               help='Flow timeout in seconds')
    extract_parser.set_defaults(func=extract_features_command)
    
    # Train IoT classifier command
    iot_train_parser = subparsers.add_parser('train-iot', help='Train IoT vs Non-IoT classifier')
    iot_train_parser.add_argument('--dataset', default='iot_sentinel',
                                 help='Dataset name to use')
    iot_train_parser.add_argument('--data-dir', default='data/raw',
                                 help='Directory containing dataset files')
    iot_train_parser.add_argument('--dataset-files', nargs='+',
                                 help='Specific dataset files to use')
    iot_train_parser.add_argument('--label-column', default='is_iot',
                                 help='Column name containing labels')
    iot_train_parser.add_argument('--model-output', default='data/models/iot_classifier.joblib',
                                 help='Output path for trained model')
    iot_train_parser.add_argument('--feature-selection', action='store_true',
                                 help='Enable feature selection')
    iot_train_parser.add_argument('--n-features', type=int, default=35,
                                 help='Number of features to select')
    iot_train_parser.add_argument('--selector-alpha', type=float, default=0.003,
                                 help='Feature importance threshold')
    iot_train_parser.add_argument('--n-estimators', type=int, default=300,
                                 help='Number of trees in Random Forest')
    iot_train_parser.add_argument('--max-depth', type=int,
                                 help='Maximum depth of trees')
    iot_train_parser.add_argument('--balance-classes', action='store_true',
                                 help='Balance class weights')
    iot_train_parser.add_argument('--test-size', type=float, default=0.2,
                                 help='Fraction of data for testing')
    iot_train_parser.add_argument('--cv-folds', type=int, default=5,
                                 help='Number of cross-validation folds')
    iot_train_parser.add_argument('--random-state', type=int, default=42,
                                 help='Random seed')
    iot_train_parser.set_defaults(func=train_iot_classifier_command)
    
    # Train device classifier command
    device_train_parser = subparsers.add_parser('train-device', help='Train device type classifier')
    device_train_parser.add_argument('--dataset', default='iot_sentinel',
                                    help='Dataset name to use')
    device_train_parser.add_argument('--data-dir', default='data/raw',
                                    help='Directory containing dataset files')
    device_train_parser.add_argument('--dataset-files', nargs='+',
                                    help='Specific dataset files to use')
    device_train_parser.add_argument('--label-column', default='device_type',
                                    help='Column name containing device type labels')
    device_train_parser.add_argument('--model-output', default='data/models/device_classifier.joblib',
                                    help='Output path for trained model')
    device_train_parser.add_argument('--selector-path',
                                    help='Path to feature selector model')
    device_train_parser.add_argument('--n-estimators', type=int, default=300,
                                    help='Number of trees in Random Forest')
    device_train_parser.add_argument('--max-depth', type=int,
                                    help='Maximum depth of trees')
    device_train_parser.add_argument('--balance-classes', action='store_true',
                                    help='Balance class weights')
    device_train_parser.add_argument('--test-size', type=float, default=0.2,
                                    help='Fraction of data for testing')
    device_train_parser.add_argument('--cv-folds', type=int, default=5,
                                    help='Number of cross-validation folds')
    device_train_parser.add_argument('--random-state', type=int, default=42,
                                    help='Random seed')
    device_train_parser.set_defaults(func=train_device_classifier_command)
    
    # Identify devices command
    identify_parser = subparsers.add_parser('identify', help='Identify devices from features')
    identify_parser.add_argument('input', help='Input CSV file with extracted features')
    identify_parser.add_argument('--output', '-o', help='Output file path')
    identify_parser.add_argument('--iot-model-path', required=True,
                                help='Path to IoT classifier model')
    identify_parser.add_argument('--device-model-path',
                                help='Path to device type classifier model')
    identify_parser.add_argument('--selector-path',
                                help='Path to feature selector model')
    identify_parser.set_defaults(func=identify_devices_command)
    
    # Evaluate model command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('model_path', help='Path to trained model')
    eval_parser.add_argument('--model-type', choices=['iot', 'device'], required=True,
                            help='Type of model to evaluate')
    eval_parser.add_argument('--test-data', help='Path to test data CSV')
    eval_parser.add_argument('--dataset', default='iot_sentinel',
                            help='Dataset name for evaluation')
    eval_parser.add_argument('--data-dir', default='data/raw',
                            help='Directory containing dataset files')
    eval_parser.add_argument('--label-column',
                            help='Column name containing labels')
    eval_parser.add_argument('--selector-path',
                            help='Path to feature selector model')
    eval_parser.add_argument('--output', '-o',
                            help='Output path for detailed results')
    eval_parser.add_argument('--plot-confusion', action='store_true',
                            help='Generate confusion matrix plot')
    eval_parser.set_defaults(func=evaluate_model_command)
    
    # Compare feature selectors command
    compare_parser = subparsers.add_parser('compare-selectors',
                                          help='Compare feature selection methods')
    compare_parser.add_argument('--dataset', default='iot_sentinel',
                               help='Dataset name to use')
    compare_parser.add_argument('--data-dir', default='data/raw',
                               help='Directory containing dataset files')
    compare_parser.add_argument('--n-features', type=int, default=35,
                               help='Number of features to select')
    compare_parser.add_argument('--output', '-o',
                               help='Output file for comparison results')
    compare_parser.add_argument('--random-state', type=int, default=42,
                               help='Random seed')
    compare_parser.set_defaults(func=compare_feature_selectors_command)
    
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

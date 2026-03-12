#!/usr/bin/env python3
"""
Memory-Efficient Phase 1 Training for Complete BoT-IoT Dataset
Uses batch processing and incremental learning to handle large datasets
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import gc
from collections import Counter
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_in_batches(file_path, batch_size=500000):
    """Load data in batches to manage memory"""
    logger.info(f"Loading data in batches of {batch_size:,} samples")
    
    # First pass: get total size and class distribution
    total_samples = 0
    class_counts = Counter()
    
    for chunk in pd.read_csv(file_path, chunksize=batch_size):
        total_samples += len(chunk)
        class_counts.update(chunk['category'].values)
        del chunk
        gc.collect()
    
    logger.info(f"Total samples: {total_samples:,}")
    logger.info("Class distribution:")
    for class_name, count in class_counts.items():
        logger.info(f"  {class_name}: {count:,} ({count/total_samples*100:.2f}%)")
    
    return total_samples, class_counts

def create_stratified_sample(file_path, sample_size=1000000, batch_size=500000):
    """Create a stratified sample from the complete dataset"""
    logger.info(f"Creating stratified sample of {sample_size:,} samples")
    
    # Get class distribution
    total_samples, class_counts = load_data_in_batches(file_path, batch_size)
    
    # Calculate samples per class for stratified sampling
    samples_per_class = {}
    for class_name, count in class_counts.items():
        proportion = count / total_samples
        samples_per_class[class_name] = int(sample_size * proportion)
    
    logger.info("Target samples per class:")
    for class_name, target_count in samples_per_class.items():
        logger.info(f"  {class_name}: {target_count:,}")
    
    # Collect stratified sample
    collected_samples = {class_name: [] for class_name in class_counts.keys()}
    collected_counts = {class_name: 0 for class_name in class_counts.keys()}
    
    for chunk in pd.read_csv(file_path, chunksize=batch_size):
        # Sample from each class in this chunk
        for class_name in class_counts.keys():
            if collected_counts[class_name] < samples_per_class[class_name]:
                class_data = chunk[chunk['category'] == class_name]
                
                if len(class_data) > 0:
                    # Calculate how many more samples we need for this class
                    needed = samples_per_class[class_name] - collected_counts[class_name]
                    take = min(needed, len(class_data))
                    
                    # Randomly sample from this chunk
                    if take < len(class_data):
                        sampled = class_data.sample(n=take, random_state=42)
                    else:
                        sampled = class_data
                    
                    collected_samples[class_name].append(sampled)
                    collected_counts[class_name] += len(sampled)
        
        # Check if we have enough samples
        if all(collected_counts[class_name] >= samples_per_class[class_name] 
               for class_name in class_counts.keys()):
            logger.info("Collected enough samples for all classes")
            break
        
        del chunk
        gc.collect()
    
    # Combine all collected samples
    all_samples = []
    for class_name, class_samples in collected_samples.items():
        if class_samples:
            class_df = pd.concat(class_samples, ignore_index=True)
            all_samples.append(class_df)
            logger.info(f"Collected {len(class_df):,} samples for {class_name}")
    
    if all_samples:
        final_sample = pd.concat(all_samples, ignore_index=True)
        # Shuffle the final sample
        final_sample = final_sample.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Final stratified sample: {len(final_sample):,} samples")
        return final_sample
    else:
        logger.error("No samples collected!")
        return None

def train_phase1_classifier(sample_data, output_dir):
    """Train Phase 1 device classifier on sampled data"""
    logger.info("Training Phase 1 device classifier")
    
    # Prepare features and labels
    feature_cols = [col for col in sample_data.columns if col != 'category']
    X = sample_data[feature_cols].values
    y = sample_data['category'].values
    
    logger.info(f"Training data: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    logger.info("Label encoding:")
    for i, class_name in enumerate(le.classes_):
        logger.info(f"  {class_name}: {i}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    logger.info(f"Train set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest classifier
    logger.info("Training Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate
    logger.info("Evaluating classifier...")
    y_pred = rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    class_names = le.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    logger.info("Classification Report:")
    for class_name in class_names:
        metrics = report[class_name]
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {metrics['precision']:.4f}")
        logger.info(f"    Recall: {metrics['recall']:.4f}")
        logger.info(f"    F1-score: {metrics['f1-score']:.4f}")
    
    logger.info(f"Macro avg F1: {report['macro avg']['f1-score']:.4f}")
    logger.info(f"Weighted avg F1: {report['weighted avg']['f1-score']:.4f}")
    
    # Save models and results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save classifier
    classifier_path = output_dir / "device_classifier.joblib"
    joblib.dump(rf, classifier_path)
    logger.info(f"Saved classifier: {classifier_path}")
    
    # Save scaler
    scaler_path = output_dir / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler: {scaler_path}")
    
    # Save label encoder
    encoder_path = output_dir / "label_encoder.joblib"
    joblib.dump(le, encoder_path)
    logger.info(f"Saved label encoder: {encoder_path}")
    
    # Save evaluation results
    results = {
        "model_type": "RandomForestClassifier",
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": len(feature_cols),
        "classes": class_names.tolist(),
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importance": {
            feature_cols[i]: float(importance) 
            for i, importance in enumerate(rf.feature_importances_)
        }
    }
    
    results_path = output_dir / "phase1_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results: {results_path}")
    
    return results

def evaluate_on_federated_data(classifier_path, scaler_path, encoder_path, federated_dir):
    """Evaluate trained classifier on federated test data"""
    logger.info("Evaluating on federated test data...")
    
    # Load models
    rf = joblib.load(classifier_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(encoder_path)
    
    test_dir = Path(federated_dir) / "test"
    if not test_dir.exists():
        logger.warning("Federated test data not found")
        return None
    
    # Evaluate on each client
    client_results = {}
    all_predictions = []
    all_true_labels = []
    
    for client_file in sorted(test_dir.glob("*.npz")):
        client_id = client_file.stem
        
        # Load client test data
        data = np.load(client_file)
        X_test = data['x']
        y_test = data['y']
        
        if len(X_test) == 0:
            continue
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Predict
        y_pred = rf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        client_results[client_id] = {
            "samples": len(X_test),
            "accuracy": float(accuracy)
        }
        
        all_predictions.extend(y_pred)
        all_true_labels.extend(y_test)
        
        logger.info(f"Client {client_id}: {len(X_test):,} samples, accuracy: {accuracy:.4f}")
    
    # Overall federated evaluation
    if all_predictions:
        overall_accuracy = accuracy_score(all_true_labels, all_predictions)
        logger.info(f"Overall federated accuracy: {overall_accuracy:.4f}")
        
        # Classification report
        class_names = le.classes_
        report = classification_report(
            all_true_labels, all_predictions, 
            target_names=class_names, output_dict=True
        )
        
        logger.info("Federated Classification Report:")
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                logger.info(f"  {class_name}:")
                logger.info(f"    Precision: {metrics['precision']:.4f}")
                logger.info(f"    Recall: {metrics['recall']:.4f}")
                logger.info(f"    F1-score: {metrics['f1-score']:.4f}")
        
        federated_results = {
            "overall_accuracy": float(overall_accuracy),
            "total_samples": len(all_predictions),
            "client_results": client_results,
            "classification_report": report
        }
        
        return federated_results
    
    return None

def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("PHASE 1: MEMORY-EFFICIENT BoT-IoT DEVICE CLASSIFICATION")
    logger.info("=" * 60)
    
    # Paths
    data_file = Path("../dataset/BoT-IoT/botiot_complete.csv")
    federated_dir = Path("../dataset/BoT-IoT")
    output_dir = Path("models/botiot_complete")
    
    if not data_file.exists():
        logger.error(f"Dataset not found: {data_file}")
        return False
    
    try:
        # Step 1: Create stratified sample (1M samples for training)
        logger.info("Step 1: Creating stratified sample...")
        sample_data = create_stratified_sample(data_file, sample_size=1000000)
        
        if sample_data is None:
            logger.error("Failed to create sample data")
            return False
        
        # Step 2: Train classifier on sample
        logger.info("Step 2: Training Phase 1 classifier...")
        results = train_phase1_classifier(sample_data, output_dir)
        
        # Step 3: Evaluate on federated data
        logger.info("Step 3: Evaluating on federated test data...")
        classifier_path = output_dir / "device_classifier.joblib"
        scaler_path = output_dir / "feature_scaler.joblib"
        encoder_path = output_dir / "label_encoder.joblib"
        
        federated_results = evaluate_on_federated_data(
            classifier_path, scaler_path, encoder_path, federated_dir
        )
        
        if federated_results:
            # Save federated results
            fed_results_path = output_dir / "federated_evaluation.json"
            with open(fed_results_path, 'w') as f:
                json.dump(federated_results, f, indent=2)
            logger.info(f"Saved federated results: {fed_results_path}")
        
        logger.info("=" * 60)
        logger.info("PHASE 1 EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"📊 Training Accuracy: {results['accuracy']:.4f}")
        if federated_results:
            logger.info(f"🌐 Federated Accuracy: {federated_results['overall_accuracy']:.4f}")
        logger.info(f"📁 Models saved to: {output_dir}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during Phase 1 training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)








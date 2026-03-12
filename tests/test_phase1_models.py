"""
Tests for Phase 1 model training and classification functionality.

Tests the two-stage classification approach: IoT vs Non-IoT and device type identification.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.phase1_profiling.train_identifiers import (
    IoTClassifier, DeviceTypeClassifier, TwoStageClassifier
)
from src.phase1_profiling.selectors import RandomForestSelector


class TestIoTClassifier:
    """Test IoT vs Non-IoT classifier."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create synthetic dataset
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 35
        
        # Generate features with some IoT-specific patterns
        self.X = np.random.randn(self.n_samples, self.n_features)
        
        # Make IoT devices have different patterns in first few features
        iot_mask = np.random.choice([0, 1], self.n_samples, p=[0.3, 0.7])
        self.X[iot_mask == 1, :5] += 2.0  # IoT devices have higher values in first 5 features
        self.X[iot_mask == 0, :5] -= 1.0  # Non-IoT devices have lower values
        
        self.y = iot_mask
        self.feature_names = [f'feature_{i}' for i in range(self.n_features)]
        
        self.classifier = IoTClassifier(random_state=42)
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        assert self.classifier.random_state == 42
        assert self.classifier.use_scaler == True
        assert not self.classifier.is_fitted_
    
    def test_classifier_training(self):
        """Test classifier training."""
        self.classifier.fit(self.X, self.y, self.feature_names)
        
        assert self.classifier.is_fitted_
        assert self.classifier.pipeline is not None
        assert self.classifier.feature_names_ == self.feature_names
        assert len(self.classifier.classes_) == 2
    
    def test_classifier_prediction(self):
        """Test classifier prediction."""
        self.classifier.fit(self.X, self.y, self.feature_names)
        
        predictions = self.classifier.predict(self.X)
        probabilities = self.classifier.predict_proba(self.X)
        
        assert len(predictions) == self.n_samples
        assert set(predictions).issubset({0, 1})
        assert probabilities.shape == (self.n_samples, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_classifier_evaluation(self):
        """Test classifier evaluation."""
        self.classifier.fit(self.X, self.y, self.feature_names)
        
        metrics = self.classifier.evaluate(self.X, self.y)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Should achieve reasonable accuracy on this synthetic data
        assert metrics['accuracy'] > 0.7
    
    def test_cross_validation(self):
        """Test cross-validation evaluation."""
        cv_results = self.classifier.cross_validate(self.X, self.y, cv=3)
        
        assert isinstance(cv_results, dict)
        assert 'cv_accuracy_mean' in cv_results
        assert 'cv_accuracy_std' in cv_results
        assert 'cv_scores' in cv_results
        
        assert cv_results['cv_accuracy_mean'] > 0.5
        assert len(cv_results['cv_scores']) == 3
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Train model
        self.classifier.fit(self.X, self.y, self.feature_names)
        original_predictions = self.classifier.predict(self.X[:10])
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            self.classifier.save_model(model_path)
            
            # Load model
            loaded_classifier = IoTClassifier.load_model(model_path)
            loaded_predictions = loaded_classifier.predict(self.X[:10])
            
            # Predictions should be identical
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
            
        finally:
            Path(model_path).unlink(missing_ok=True)


class TestDeviceTypeClassifier:
    """Test device type classifier."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create synthetic dataset with multiple device types
        np.random.seed(42)
        self.n_samples = 300
        self.n_features = 35
        
        self.device_types = ['camera', 'sensor', 'switch', 'hub', 'thermostat']
        
        # Generate features with device-specific patterns
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.choice(self.device_types, self.n_samples)
        
        # Add device-specific patterns
        for i, device_type in enumerate(self.device_types):
            device_mask = self.y == device_type
            # Each device type has different patterns in different feature ranges
            start_idx = i * 5
            end_idx = start_idx + 5
            self.X[device_mask, start_idx:end_idx] += (i + 1) * 1.5
        
        self.feature_names = [f'feature_{i}' for i in range(self.n_features)]
        self.classifier = DeviceTypeClassifier(random_state=42)
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        assert self.classifier.random_state == 42
        assert not self.classifier.is_fitted_
    
    def test_classifier_training(self):
        """Test classifier training."""
        self.classifier.fit(self.X, self.y, self.feature_names)
        
        assert self.classifier.is_fitted_
        assert self.classifier.pipeline is not None
        assert self.classifier.feature_names_ == self.feature_names
        assert set(self.classifier.classes_) == set(self.device_types)
    
    def test_classifier_prediction(self):
        """Test classifier prediction."""
        self.classifier.fit(self.X, self.y, self.feature_names)
        
        predictions = self.classifier.predict(self.X)
        probabilities = self.classifier.predict_proba(self.X)
        
        assert len(predictions) == self.n_samples
        assert set(predictions).issubset(set(self.device_types))
        assert probabilities.shape == (self.n_samples, len(self.device_types))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_classifier_evaluation(self):
        """Test classifier evaluation."""
        self.classifier.fit(self.X, self.y, self.feature_names)
        
        metrics = self.classifier.evaluate(self.X, self.y)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Should achieve reasonable accuracy on this synthetic data
        assert metrics['accuracy'] > 0.5
    
    def test_label_encoding(self):
        """Test label encoding functionality."""
        self.classifier.fit(self.X, self.y, self.feature_names)
        
        # Test with string labels
        predictions = self.classifier.predict(self.X[:10])
        assert all(isinstance(pred, str) for pred in predictions)
        assert all(pred in self.device_types for pred in predictions)
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Train model
        self.classifier.fit(self.X, self.y, self.feature_names)
        original_predictions = self.classifier.predict(self.X[:10])
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            self.classifier.save_model(model_path)
            
            # Load model
            loaded_classifier = DeviceTypeClassifier.load_model(model_path)
            loaded_predictions = loaded_classifier.predict(self.X[:10])
            
            # Predictions should be identical
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
            
        finally:
            Path(model_path).unlink(missing_ok=True)


class TestTwoStageClassifier:
    """Test two-stage classifier integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create synthetic dataset with both IoT/Non-IoT and device types
        np.random.seed(42)
        self.n_samples = 400
        self.n_features = 35
        
        self.device_types = ['camera', 'sensor', 'switch', 'hub', 'thermostat']
        
        # Generate features
        self.X = np.random.randn(self.n_samples, self.n_features)
        
        # Create IoT vs Non-IoT labels
        self.y_iot = np.random.choice([0, 1], self.n_samples, p=[0.25, 0.75])
        
        # Create device type labels (only for IoT devices)
        self.y_device = np.full(self.n_samples, 'non_iot', dtype=object)
        iot_mask = self.y_iot == 1
        self.y_device[iot_mask] = np.random.choice(self.device_types, np.sum(iot_mask))
        
        # Add patterns to features
        # IoT devices have different patterns
        self.X[iot_mask, :10] += 1.5
        self.X[~iot_mask, :10] -= 1.0
        
        # Each device type has specific patterns
        for i, device_type in enumerate(self.device_types):
            device_mask = self.y_device == device_type
            start_idx = 10 + i * 3
            end_idx = start_idx + 3
            self.X[device_mask, start_idx:end_idx] += (i + 1) * 1.2
        
        self.feature_names = [f'feature_{i}' for i in range(self.n_features)]
        self.classifier = TwoStageClassifier(random_state=42)
    
    def test_two_stage_training(self):
        """Test two-stage classifier training."""
        self.classifier.fit(self.X, self.y_iot, self.y_device, self.feature_names)
        
        assert self.classifier.is_fitted_
        assert self.classifier.iot_classifier.is_fitted_
        assert self.classifier.device_classifier.is_fitted_
    
    def test_two_stage_prediction(self):
        """Test two-stage classifier prediction."""
        self.classifier.fit(self.X, self.y_iot, self.y_device, self.feature_names)
        
        iot_pred, device_pred = self.classifier.predict(self.X)
        
        assert len(iot_pred) == self.n_samples
        assert len(device_pred) == self.n_samples
        assert set(iot_pred).issubset({0, 1})
        
        # Non-IoT devices should be labeled as 'Non-IoT'
        non_iot_mask = iot_pred == 0
        assert all(device_pred[non_iot_mask] == 'Non-IoT')
        
        # IoT devices should have device type labels
        iot_mask = iot_pred == 1
        if np.any(iot_mask):
            iot_device_types = set(device_pred[iot_mask])
            assert 'Non-IoT' not in iot_device_types
            assert iot_device_types.issubset(set(self.device_types))
    
    def test_model_persistence(self):
        """Test two-stage model saving and loading."""
        # Train models
        self.classifier.fit(self.X, self.y_iot, self.y_device, self.feature_names)
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as tmp_dir:
            iot_model_path = Path(tmp_dir) / 'iot_model.joblib'
            device_model_path = Path(tmp_dir) / 'device_model.joblib'
            
            # Save models
            self.classifier.save_models(iot_model_path, device_model_path)
            
            # Load models
            loaded_classifier = TwoStageClassifier.load_models(iot_model_path, device_model_path)
            
            # Test predictions are identical
            original_iot, original_device = self.classifier.predict(self.X[:20])
            loaded_iot, loaded_device = loaded_classifier.predict(self.X[:20])
            
            np.testing.assert_array_equal(original_iot, loaded_iot)
            np.testing.assert_array_equal(original_device, loaded_device)


class TestModelIntegration:
    """Integration tests for model training pipeline."""
    
    def test_feature_selection_integration(self):
        """Test integration with feature selection."""
        # Create dataset
        np.random.seed(42)
        n_samples = 200
        n_features = 50
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        
        # Make first 10 features more predictive
        X[y == 1, :10] += 2.0
        X[y == 0, :10] -= 1.0
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Feature selection
        selector = RandomForestSelector(n_features=20)
        X_selected = selector.fit_transform(X, y)
        selected_names = selector.get_selected_feature_names()
        
        # Train classifier on selected features
        classifier = IoTClassifier()
        classifier.fit(X_selected, y, selected_names)
        
        # Should work and achieve good performance
        metrics = classifier.evaluate(X_selected, y)
        assert metrics['accuracy'] > 0.8  # Should be high with good feature selection
    
    def test_realistic_workflow(self):
        """Test realistic workflow with multiple device types."""
        # Simulate realistic IoT network data
        np.random.seed(42)
        
        device_configs = {
            'camera': {'n_samples': 80, 'packet_size_pattern': [800, 200], 'traffic_volume': 'high'},
            'sensor': {'n_samples': 60, 'packet_size_pattern': [100, 30], 'traffic_volume': 'low'},
            'switch': {'n_samples': 40, 'packet_size_pattern': [200, 50], 'traffic_volume': 'medium'},
            'laptop': {'n_samples': 50, 'packet_size_pattern': [1200, 400], 'traffic_volume': 'high'},
            'phone': {'n_samples': 70, 'packet_size_pattern': [600, 200], 'traffic_volume': 'medium'}
        }
        
        all_features = []
        all_iot_labels = []
        all_device_labels = []
        
        for device_type, config in device_configs.items():
            n_samples = config['n_samples']
            
            # Generate features based on device characteristics
            features = np.random.randn(n_samples, 35)
            
            # Packet size features
            size_mean, size_std = config['packet_size_pattern']
            features[:, 0] = np.random.normal(size_mean, size_std, n_samples)  # packet_length_mean
            features[:, 1] = np.random.normal(size_std, size_std/4, n_samples)  # packet_length_std
            
            # Traffic volume features
            if config['traffic_volume'] == 'high':
                features[:, 10] = np.random.normal(100, 20, n_samples)  # packets_per_second
            elif config['traffic_volume'] == 'medium':
                features[:, 10] = np.random.normal(50, 15, n_samples)
            else:
                features[:, 10] = np.random.normal(10, 5, n_samples)
            
            all_features.append(features)
            
            # IoT vs Non-IoT labels
            is_iot = 1 if device_type in ['camera', 'sensor', 'switch'] else 0
            all_iot_labels.extend([is_iot] * n_samples)
            all_device_labels.extend([device_type] * n_samples)
        
        # Combine all data
        X = np.vstack(all_features)
        y_iot = np.array(all_iot_labels)
        y_device = np.array(all_device_labels)
        
        # Train two-stage classifier
        classifier = TwoStageClassifier()
        classifier.fit(X, y_iot, y_device)
        
        # Test predictions
        iot_pred, device_pred = classifier.predict(X)
        
        # Calculate accuracies
        iot_accuracy = np.mean(iot_pred == y_iot)
        
        # Device accuracy (only for IoT devices)
        iot_mask = y_iot == 1
        if np.any(iot_mask):
            device_accuracy = np.mean(device_pred[iot_mask] == y_device[iot_mask])
        else:
            device_accuracy = 0
        
        print(f"IoT classification accuracy: {iot_accuracy:.3f}")
        print(f"Device type accuracy: {device_accuracy:.3f}")
        
        # Should achieve reasonable performance
        assert iot_accuracy > 0.7
        assert device_accuracy > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

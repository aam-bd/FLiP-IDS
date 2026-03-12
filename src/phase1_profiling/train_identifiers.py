"""
Two-stage IoT device identification and classification.

Implements the two-stage classification approach:
1. IoT vs Non-IoT classification using Random Forest
2. Device type identification for IoT devices using Random Forest

Both stages use Random Forest as recommended by Safi et al. for achieving
90%+ accuracy with minimal training time.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from ..common.logging import get_logger, MetricsLogger
from ..common.metrics import calculate_metrics, confusion_matrix_plot
from ..common.io import save_model, load_model
from ..common.utils import Timer, set_seed

logger = get_logger(__name__)


class IoTClassifier:
    """
    Binary classifier for IoT vs Non-IoT device identification.
    
    First stage of the two-stage classification pipeline. Uses Random Forest
    with optimized hyperparameters to distinguish IoT devices from traditional
    computing devices based on network traffic features.
    """
    
    def __init__(self, 
                 rf_params: Optional[Dict] = None,
                 use_scaler: bool = True,
                 random_state: int = 42):
        """
        Initialize IoT classifier.
        
        Args:
            rf_params: Random Forest parameters
            use_scaler: Whether to use feature scaling
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.use_scaler = use_scaler
        
        # Default RF parameters optimized for IoT classification
        default_rf_params = {
            'n_estimators': 300,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced',
            'random_state': random_state,
            'n_jobs': -1
        }
        
        if rf_params:
            default_rf_params.update(rf_params)
        
        self.rf_params = default_rf_params
        self.pipeline = None
        self.feature_names_ = None
        self.classes_ = None
        self.is_fitted_ = False
        
        self.logger = logger
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, List],
            feature_names: Optional[List[str]] = None,
            validation_split: float = 0.2) -> 'IoTClassifier':
        """
        Train the IoT classifier.
        
        Args:
            X: Feature matrix
            y: Binary labels (1 for IoT, 0 for Non-IoT)
            feature_names: Optional feature names
            validation_split: Fraction of data for validation
            
        Returns:
            Self for method chaining
        """
        set_seed(self.random_state)
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values
        
        y = np.array(y)
        self.feature_names_ = feature_names
        self.classes_ = np.unique(y)
        
        self.logger.info(f"Training IoT classifier with {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Class distribution: {np.bincount(y)}")
        
        # Split data for validation
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, 
                random_state=self.random_state, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Create pipeline
        steps = []
        
        if self.use_scaler:
            steps.append(('scaler', StandardScaler()))
        
        steps.append(('classifier', RandomForestClassifier(**self.rf_params)))
        
        self.pipeline = Pipeline(steps)
        
        # Train model
        with Timer("IoT classifier training", self.logger):
            self.pipeline.fit(X_train, y_train)
        
        self.is_fitted_ = True
        
        # Validation
        if X_val is not None:
            self._validate_model(X_val, y_val)
        
        # Feature importance analysis
        self._analyze_feature_importance()
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict IoT vs Non-IoT labels."""
        self._check_fitted()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        self._check_fitted()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, List]) -> Dict[str, float]:
        """
        Evaluate classifier performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        self._check_fitted()
        
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        
        metrics = calculate_metrics(
            y, y_pred, y_prob, 
            labels=['Non-IoT', 'IoT'],
            average='binary'
        )
        
        return metrics
    
    def cross_validate(self, X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, List], cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation evaluation.
        
        Args:
            X: Feature matrix
            y: True labels
            cv: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        y = np.array(y)
        
        # Create fresh pipeline for CV
        pipeline = Pipeline([
            ('scaler', StandardScaler()) if self.use_scaler else ('passthrough', 'passthrough'),
            ('classifier', RandomForestClassifier(**self.rf_params))
        ])
        
        # Remove passthrough step if not using scaler
        if not self.use_scaler:
            pipeline = Pipeline([('classifier', RandomForestClassifier(**self.rf_params))])
        
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        
        results = {
            'cv_accuracy_mean': scores.mean(),
            'cv_accuracy_std': scores.std(),
            'cv_scores': scores.tolist()
        }
        
        self.logger.info(f"Cross-validation accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
        
        return results
    
    def _validate_model(self, X_val: np.ndarray, y_val: np.ndarray):
        """Validate model on held-out data."""
        y_pred = self.predict(X_val)
        y_prob = self.predict_proba(X_val)
        
        metrics = calculate_metrics(y_val, y_pred, y_prob, average='binary')
        
        self.logger.info("Validation results:")
        self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {metrics['recall']:.4f}")
        self.logger.info(f"  F1-score: {metrics['f1_score']:.4f}")
        
        if 'auc_roc' in metrics:
            self.logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    
    def _analyze_feature_importance(self):
        """Analyze and log feature importance."""
        if not hasattr(self.pipeline.named_steps['classifier'], 'feature_importances_'):
            return
        
        importances = self.pipeline.named_steps['classifier'].feature_importances_
        
        if self.feature_names_ is not None:
            # Get top 10 most important features
            top_indices = np.argsort(importances)[::-1][:10]
            
            self.logger.info("Top 10 most important features for IoT classification:")
            for i, idx in enumerate(top_indices):
                feature_name = self.feature_names_[idx]
                importance = importances[idx]
                self.logger.info(f"  {i+1:2d}. {feature_name:25s} {importance:.6f}")
    
    def save_model(self, filepath: Union[str, Path]):
        """Save trained model to disk."""
        self._check_fitted()
        
        metadata = {
            'model_type': 'IoTClassifier',
            'rf_params': self.rf_params,
            'use_scaler': self.use_scaler,
            'feature_names': self.feature_names_,
            'classes': self.classes_.tolist(),
            'random_state': self.random_state
        }
        
        save_model(self.pipeline, filepath, metadata)
        self.logger.info(f"IoT classifier saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'IoTClassifier':
        """Load trained model from disk."""
        pipeline, metadata = load_model(filepath)
        
        # Create classifier instance
        classifier = cls(
            rf_params=metadata.get('rf_params'),
            use_scaler=metadata.get('use_scaler', True),
            random_state=metadata.get('random_state', 42)
        )
        
        # Set loaded attributes
        classifier.pipeline = pipeline
        classifier.feature_names_ = metadata.get('feature_names')
        classifier.classes_ = np.array(metadata.get('classes', [0, 1]))
        classifier.is_fitted_ = True
        
        logger.info(f"IoT classifier loaded from {filepath}")
        return classifier
    
    def _check_fitted(self):
        """Check if model is fitted."""
        if not self.is_fitted_ or self.pipeline is None:
            raise ValueError("Model not fitted. Call fit() first.")


class DeviceTypeClassifier:
    """
    Multi-class classifier for IoT device type identification.
    
    Second stage of the two-stage classification pipeline. Identifies specific
    IoT device types (e.g., camera, sensor, smart switch) from network traffic
    features using Random Forest.
    """
    
    def __init__(self,
                 rf_params: Optional[Dict] = None,
                 use_scaler: bool = True,
                 random_state: int = 42):
        """
        Initialize device type classifier.
        
        Args:
            rf_params: Random Forest parameters
            use_scaler: Whether to use feature scaling
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.use_scaler = use_scaler
        
        # Default RF parameters for multi-class classification
        default_rf_params = {
            'n_estimators': 300,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced',
            'random_state': random_state,
            'n_jobs': -1
        }
        
        if rf_params:
            default_rf_params.update(rf_params)
        
        self.rf_params = default_rf_params
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.feature_names_ = None
        self.classes_ = None
        self.is_fitted_ = False
        
        self.logger = logger
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, List],
            feature_names: Optional[List[str]] = None,
            validation_split: float = 0.2) -> 'DeviceTypeClassifier':
        """
        Train the device type classifier.
        
        Args:
            X: Feature matrix (should contain only IoT device samples)
            y: Device type labels (strings or integers)
            feature_names: Optional feature names
            validation_split: Fraction of data for validation
            
        Returns:
            Self for method chaining
        """
        set_seed(self.random_state)
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values
        
        # Encode string labels to integers
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.feature_names_ = feature_names
        self.classes_ = self.label_encoder.classes_
        
        self.logger.info(f"Training device type classifier with {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Device types: {list(self.classes_)}")
        self.logger.info(f"Class distribution: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")
        
        # Split data for validation
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=validation_split,
                random_state=self.random_state, stratify=y_encoded
            )
        else:
            X_train, y_train = X, y_encoded
            X_val, y_val = None, None
        
        # Create pipeline
        steps = []
        
        if self.use_scaler:
            steps.append(('scaler', StandardScaler()))
        
        steps.append(('classifier', RandomForestClassifier(**self.rf_params)))
        
        self.pipeline = Pipeline(steps)
        
        # Train model
        with Timer("Device type classifier training", self.logger):
            self.pipeline.fit(X_train, y_train)
        
        self.is_fitted_ = True
        
        # Validation
        if X_val is not None:
            self._validate_model(X_val, y_val)
        
        # Feature importance analysis
        self._analyze_feature_importance()
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict device type labels."""
        self._check_fitted()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        y_pred_encoded = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        self._check_fitted()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame],
                y: Union[np.ndarray, List]) -> Dict[str, float]:
        """
        Evaluate classifier performance.
        
        Args:
            X: Feature matrix
            y: True device type labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        self._check_fitted()
        
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        
        # Convert string labels to encoded for metrics calculation
        y_encoded = self.label_encoder.transform(y)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        metrics = calculate_metrics(
            y_encoded, y_pred_encoded, y_prob,
            labels=list(self.classes_),
            average='macro'
        )
        
        return metrics
    
    def cross_validate(self, X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, List], cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation evaluation.
        
        Args:
            X: Feature matrix
            y: True device type labels
            cv: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y) if not self.is_fitted_ else self.label_encoder.transform(y)
        
        # Create fresh pipeline for CV
        pipeline = Pipeline([
            ('scaler', StandardScaler()) if self.use_scaler else ('passthrough', 'passthrough'),
            ('classifier', RandomForestClassifier(**self.rf_params))
        ])
        
        if not self.use_scaler:
            pipeline = Pipeline([('classifier', RandomForestClassifier(**self.rf_params))])
        
        scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring='accuracy')
        
        results = {
            'cv_accuracy_mean': scores.mean(),
            'cv_accuracy_std': scores.std(),
            'cv_scores': scores.tolist()
        }
        
        self.logger.info(f"Cross-validation accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
        
        return results
    
    def _validate_model(self, X_val: np.ndarray, y_val: np.ndarray):
        """Validate model on held-out data."""
        y_pred = self.pipeline.predict(X_val)
        y_prob = self.predict_proba(X_val)
        
        metrics = calculate_metrics(y_val, y_pred, y_prob, average='macro')
        
        self.logger.info("Validation results:")
        self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {metrics['recall']:.4f}")
        self.logger.info(f"  F1-score: {metrics['f1_score']:.4f}")
    
    def _analyze_feature_importance(self):
        """Analyze and log feature importance."""
        if not hasattr(self.pipeline.named_steps['classifier'], 'feature_importances_'):
            return
        
        importances = self.pipeline.named_steps['classifier'].feature_importances_
        
        if self.feature_names_ is not None:
            # Get top 10 most important features
            top_indices = np.argsort(importances)[::-1][:10]
            
            self.logger.info("Top 10 most important features for device type classification:")
            for i, idx in enumerate(top_indices):
                feature_name = self.feature_names_[idx]
                importance = importances[idx]
                self.logger.info(f"  {i+1:2d}. {feature_name:25s} {importance:.6f}")
    
    def save_model(self, filepath: Union[str, Path]):
        """Save trained model to disk."""
        self._check_fitted()
        
        metadata = {
            'model_type': 'DeviceTypeClassifier',
            'rf_params': self.rf_params,
            'use_scaler': self.use_scaler,
            'feature_names': self.feature_names_,
            'classes': self.classes_.tolist(),
            'label_encoder_classes': self.label_encoder.classes_.tolist(),
            'random_state': self.random_state
        }
        
        # Save both pipeline and label encoder
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder
        }
        
        save_model(model_data, filepath, metadata)
        self.logger.info(f"Device type classifier saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'DeviceTypeClassifier':
        """Load trained model from disk."""
        model_data, metadata = load_model(filepath)
        
        # Create classifier instance
        classifier = cls(
            rf_params=metadata.get('rf_params'),
            use_scaler=metadata.get('use_scaler', True),
            random_state=metadata.get('random_state', 42)
        )
        
        # Set loaded attributes
        classifier.pipeline = model_data['pipeline']
        classifier.label_encoder = model_data['label_encoder']
        classifier.feature_names_ = metadata.get('feature_names')
        classifier.classes_ = np.array(metadata.get('classes'))
        classifier.is_fitted_ = True
        
        logger.info(f"Device type classifier loaded from {filepath}")
        return classifier
    
    def _check_fitted(self):
        """Check if model is fitted."""
        if not self.is_fitted_ or self.pipeline is None:
            raise ValueError("Model not fitted. Call fit() first.")


class TwoStageClassifier:
    """
    Combined two-stage classifier for complete IoT device identification.
    
    Combines IoT vs Non-IoT classification with device type identification
    in a single interface for end-to-end device profiling.
    """
    
    def __init__(self,
                 iot_classifier_params: Optional[Dict] = None,
                 device_classifier_params: Optional[Dict] = None,
                 random_state: int = 42):
        """
        Initialize two-stage classifier.
        
        Args:
            iot_classifier_params: Parameters for IoT classifier
            device_classifier_params: Parameters for device type classifier
            random_state: Random seed
        """
        self.iot_classifier = IoTClassifier(
            **(iot_classifier_params or {}), 
            random_state=random_state
        )
        self.device_classifier = DeviceTypeClassifier(
            **(device_classifier_params or {}),
            random_state=random_state
        )
        
        self.is_fitted_ = False
        self.logger = logger
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y_iot: Union[np.ndarray, List],
            y_device: Union[np.ndarray, List],
            feature_names: Optional[List[str]] = None) -> 'TwoStageClassifier':
        """
        Train both stages of the classifier.
        
        Args:
            X: Feature matrix
            y_iot: IoT vs Non-IoT labels
            y_device: Device type labels (for IoT samples only)
            feature_names: Optional feature names
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Training two-stage classifier...")
        
        # Stage 1: IoT vs Non-IoT
        self.iot_classifier.fit(X, y_iot, feature_names)
        
        # Stage 2: Device type classification (IoT samples only)
        iot_mask = np.array(y_iot) == 1
        if np.any(iot_mask):
            X_iot = X[iot_mask] if isinstance(X, np.ndarray) else X.iloc[iot_mask]
            y_device_iot = np.array(y_device)[iot_mask]
            
            self.device_classifier.fit(X_iot, y_device_iot, feature_names)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict both IoT classification and device types.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (iot_predictions, device_type_predictions)
        """
        if not self.is_fitted_:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        # Stage 1: IoT vs Non-IoT
        iot_pred = self.iot_classifier.predict(X)
        
        # Stage 2: Device type for IoT devices
        device_pred = np.full(len(iot_pred), 'Non-IoT', dtype=object)
        iot_mask = iot_pred == 1
        
        if np.any(iot_mask):
            X_iot = X[iot_mask] if isinstance(X, np.ndarray) else X.iloc[iot_mask]
            device_pred[iot_mask] = self.device_classifier.predict(X_iot)
        
        return iot_pred, device_pred
    
    def save_models(self, iot_model_path: Union[str, Path], 
                   device_model_path: Union[str, Path]):
        """Save both models."""
        self.iot_classifier.save_model(iot_model_path)
        self.device_classifier.save_model(device_model_path)
    
    @classmethod
    def load_models(cls, iot_model_path: Union[str, Path],
                   device_model_path: Union[str, Path]) -> 'TwoStageClassifier':
        """Load both models."""
        classifier = cls()
        classifier.iot_classifier = IoTClassifier.load_model(iot_model_path)
        classifier.device_classifier = DeviceTypeClassifier.load_model(device_model_path)
        classifier.is_fitted_ = True
        return classifier

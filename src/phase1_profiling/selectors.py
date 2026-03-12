"""
Feature selection utilities for Phase 1 profiling.

Implements Random Forest-based feature importance selection to reduce
the 58-feature hybrid set to a more manageable subset (default 35 features)
while maintaining high classification accuracy as recommended by Safi et al.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from pathlib import Path

from ..common.logging import get_logger
from ..common.utils import Timer

logger = get_logger(__name__)


class FeatureSelector(ABC):
    """Abstract base class for feature selection methods."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FeatureSelector':
        """Fit the feature selector."""
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted selector."""
        pass
    
    @abstractmethod
    def get_selected_features(self) -> List[int]:
        """Get indices of selected features."""
        pass
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit selector and transform features."""
        return self.fit(X, y).transform(X)


class RandomForestSelector(FeatureSelector):
    """
    Feature selection using Random Forest feature importance.
    
    This is the recommended approach from Safi et al., using Random Forest
    importance scores with a threshold alpha to select dominant features.
    """
    
    def __init__(self, 
                 n_features: int = 35,
                 threshold_alpha: float = 0.003,
                 rf_params: Optional[Dict] = None,
                 cv_folds: int = 5):
        """
        Initialize Random Forest feature selector.
        
        Args:
            n_features: Number of features to select
            threshold_alpha: Importance threshold for feature selection
            rf_params: Random Forest parameters
            cv_folds: Cross-validation folds for evaluation
        """
        self.n_features = n_features
        self.threshold_alpha = threshold_alpha
        self.cv_folds = cv_folds
        
        # Default RF parameters optimized for feature selection
        default_rf_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if rf_params:
            default_rf_params.update(rf_params)
        
        self.rf_params = default_rf_params
        self.rf_model = None
        self.feature_importances_ = None
        self.selected_features_ = None
        self.feature_names_ = None
        
        self.logger = logger
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> 'RandomForestSelector':
        """
        Fit Random Forest and select important features.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Fitting Random Forest for feature selection...")
        
        with Timer("Random Forest feature selection", self.logger):
            # Fit Random Forest
            self.rf_model = RandomForestClassifier(**self.rf_params)
            self.rf_model.fit(X, y)
            
            # Get feature importances
            self.feature_importances_ = self.rf_model.feature_importances_
            self.feature_names_ = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            
            # Select features using threshold or top-k
            self.selected_features_ = self._select_features()
            
            self.logger.info(f"Selected {len(self.selected_features_)} features out of {X.shape[1]}")
            
            # Log top features
            self._log_top_features()
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features by selecting only important ones."""
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        return X[:, self.selected_features_]
    
    def get_selected_features(self) -> List[int]:
        """Get indices of selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        return self.selected_features_.tolist()
    
    def get_selected_feature_names(self) -> List[str]:
        """Get names of selected features."""
        if self.selected_features_ is None or self.feature_names_ is None:
            raise ValueError("Selector not fitted or feature names not provided.")
        
        return [self.feature_names_[i] for i in self.selected_features_]
    
    def _select_features(self) -> np.ndarray:
        """Select features based on importance threshold and top-k."""
        # Method 1: Threshold-based selection
        threshold_features = np.where(self.feature_importances_ >= self.threshold_alpha)[0]
        
        # Method 2: Top-k selection
        top_k_indices = np.argsort(self.feature_importances_)[::-1][:self.n_features]
        
        # Use threshold method if it gives reasonable number of features
        if len(threshold_features) >= 10 and len(threshold_features) <= self.n_features * 1.5:
            selected = threshold_features
            self.logger.info(f"Using threshold-based selection (alpha={self.threshold_alpha})")
        else:
            selected = top_k_indices
            self.logger.info(f"Using top-{self.n_features} selection")
        
        return selected
    
    def _log_top_features(self, top_n: int = 10):
        """Log top N most important features."""
        if self.feature_importances_ is None or self.feature_names_ is None:
            return
        
        # Get top features with their importance scores
        top_indices = np.argsort(self.feature_importances_)[::-1][:top_n]
        
        self.logger.info(f"Top {top_n} most important features:")
        for i, idx in enumerate(top_indices):
            feature_name = self.feature_names_[idx]
            importance = self.feature_importances_[idx]
            selected = "✓" if idx in self.selected_features_ else "✗"
            self.logger.info(f"  {i+1:2d}. {feature_name:25s} {importance:.6f} {selected}")
    
    def evaluate_selection(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate feature selection quality using cross-validation.
        
        Args:
            X: Original feature matrix
            y: Target labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        self.logger.info("Evaluating feature selection quality...")
        
        # Test with all features
        rf_all = RandomForestClassifier(**self.rf_params)
        scores_all = cross_val_score(rf_all, X, y, cv=self.cv_folds, scoring='accuracy')
        
        # Test with selected features only
        X_selected = X[:, self.selected_features_]
        rf_selected = RandomForestClassifier(**self.rf_params)
        scores_selected = cross_val_score(rf_selected, X_selected, y, cv=self.cv_folds, scoring='accuracy')
        
        results = {
            'accuracy_all_features': scores_all.mean(),
            'accuracy_all_features_std': scores_all.std(),
            'accuracy_selected_features': scores_selected.mean(),
            'accuracy_selected_features_std': scores_selected.std(),
            'accuracy_ratio': scores_selected.mean() / scores_all.mean(),
            'feature_reduction_ratio': len(self.selected_features_) / X.shape[1],
            'num_original_features': X.shape[1],
            'num_selected_features': len(self.selected_features_)
        }
        
        self.logger.info(f"All features accuracy: {results['accuracy_all_features']:.4f} ± {results['accuracy_all_features_std']:.4f}")
        self.logger.info(f"Selected features accuracy: {results['accuracy_selected_features']:.4f} ± {results['accuracy_selected_features_std']:.4f}")
        self.logger.info(f"Accuracy ratio: {results['accuracy_ratio']:.4f}")
        self.logger.info(f"Feature reduction: {results['num_selected_features']}/{results['num_original_features']} ({results['feature_reduction_ratio']:.2%})")
        
        return results
    
    def plot_feature_importance(self, save_path: Optional[Union[str, Path]] = None,
                               top_n: int = 20, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot feature importance scores.
        
        Args:
            save_path: Path to save the plot
            top_n: Number of top features to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.feature_importances_ is None or self.feature_names_ is None:
            raise ValueError("Selector not fitted or feature names not provided.")
        
        # Get top features
        top_indices = np.argsort(self.feature_importances_)[::-1][:top_n]
        top_names = [self.feature_names_[i] for i in top_indices]
        top_importances = self.feature_importances_[top_indices]
        
        # Create colors (green for selected, gray for not selected)
        colors = ['green' if i in self.selected_features_ else 'gray' for i in top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(range(len(top_names)), top_importances, color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importances (Random Forest)')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add threshold line if used
        if any(self.feature_importances_ >= self.threshold_alpha):
            ax.axvline(x=self.threshold_alpha, color='red', linestyle='--', 
                      label=f'Threshold (α={self.threshold_alpha})')
            ax.legend()
        
        # Add importance values on bars
        for i, (bar, importance) in enumerate(zip(bars, top_importances)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance:.4f}', va='center', ha='left', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig
    
    def get_feature_importance_dataframe(self) -> pd.DataFrame:
        """Get feature importances as DataFrame."""
        if self.feature_importances_ is None or self.feature_names_ is None:
            raise ValueError("Selector not fitted or feature names not provided.")
        
        df = pd.DataFrame({
            'feature_name': self.feature_names_,
            'importance': self.feature_importances_,
            'selected': [i in self.selected_features_ for i in range(len(self.feature_names_))],
            'rank': np.argsort(np.argsort(self.feature_importances_)[::-1]) + 1
        })
        
        return df.sort_values('importance', ascending=False).reset_index(drop=True)


class MutualInfoSelector(FeatureSelector):
    """Feature selection using mutual information."""
    
    def __init__(self, n_features: int = 35, random_state: int = 42):
        self.n_features = n_features
        self.random_state = random_state
        self.selector = SelectKBest(mutual_info_classif, k=n_features)
        self.selected_features_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MutualInfoSelector':
        self.selector.fit(X, y)
        self.selected_features_ = self.selector.get_support(indices=True)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.selector.transform(X)
    
    def get_selected_features(self) -> List[int]:
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted.")
        return self.selected_features_.tolist()


class RecursiveFeatureSelector(FeatureSelector):
    """Recursive Feature Elimination selector."""
    
    def __init__(self, n_features: int = 35, estimator=None):
        self.n_features = n_features
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        
        self.selector = RFE(estimator, n_features_to_select=n_features)
        self.selected_features_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RecursiveFeatureSelector':
        self.selector.fit(X, y)
        self.selected_features_ = self.selector.get_support(indices=True)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.selector.transform(X)
    
    def get_selected_features(self) -> List[int]:
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted.")
        return self.selected_features_.tolist()


def compare_selectors(X: np.ndarray, y: np.ndarray, 
                     feature_names: Optional[List[str]] = None,
                     n_features: int = 35) -> Dict[str, Dict]:
    """
    Compare different feature selection methods.
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: Optional feature names
        n_features: Number of features to select
        
    Returns:
        Dictionary with results for each selector
    """
    logger.info(f"Comparing feature selectors with {n_features} features...")
    
    selectors = {
        'random_forest': RandomForestSelector(n_features=n_features),
        'mutual_info': MutualInfoSelector(n_features=n_features),
        'recursive': RecursiveFeatureSelector(n_features=n_features)
    }
    
    results = {}
    
    for name, selector in selectors.items():
        logger.info(f"Testing {name} selector...")
        
        with Timer(f"{name} selector", logger):
            # Fit selector
            if hasattr(selector, 'fit') and name == 'random_forest':
                selector.fit(X, y, feature_names)
            else:
                selector.fit(X, y)
            
            # Evaluate performance
            rf_eval = RandomForestClassifier(n_estimators=100, random_state=42)
            X_selected = selector.transform(X)
            
            scores = cross_val_score(rf_eval, X_selected, y, cv=5, scoring='accuracy')
            
            results[name] = {
                'accuracy_mean': scores.mean(),
                'accuracy_std': scores.std(),
                'selected_features': selector.get_selected_features(),
                'num_features': len(selector.get_selected_features())
            }
            
            if hasattr(selector, 'get_selected_feature_names') and feature_names:
                results[name]['selected_feature_names'] = selector.get_selected_feature_names()
    
    # Log comparison results
    logger.info("Feature selector comparison results:")
    for name, result in results.items():
        logger.info(f"  {name:15s}: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
    
    return results


def create_feature_selector(method: str = 'random_forest', **kwargs) -> FeatureSelector:
    """
    Factory function to create feature selectors.
    
    Args:
        method: Selector method ('random_forest', 'mutual_info', 'recursive')
        **kwargs: Additional arguments for the selector
        
    Returns:
        Feature selector instance
    """
    if method == 'random_forest':
        return RandomForestSelector(**kwargs)
    elif method == 'mutual_info':
        return MutualInfoSelector(**kwargs)
    elif method == 'recursive':
        return RecursiveFeatureSelector(**kwargs)
    else:
        raise ValueError(f"Unknown selector method: {method}")

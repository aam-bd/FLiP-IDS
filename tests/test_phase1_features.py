"""
Tests for Phase 1 feature extraction functionality.

Tests the 58-feature hybrid extraction, feature selection,
and data validation components.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.phase1_profiling.feature_extractor import HybridFeatureExtractor
from src.phase1_profiling.selectors import RandomForestSelector
from src.phase1_profiling.pcap_reader import NetworkFlow, FlowTuple


class TestHybridFeatureExtractor:
    """Test hybrid feature extraction functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = HybridFeatureExtractor()
        self.flow_tuple = FlowTuple('192.168.1.100', '10.0.0.1', 12345, 80, 'TCP')
        
    def create_test_flow(self) -> NetworkFlow:
        """Create a test network flow."""
        flow = NetworkFlow(self.flow_tuple, 1000.0)
        
        # Add some packets
        for i in range(10):
            flow.add_packet(
                packet_size=100 + i * 10,
                timestamp=1000.0 + i * 0.1,
                is_forward=(i % 2 == 0),
                tcp_flags=0x18 if i == 0 else 0x10,  # SYN+ACK for first, ACK for others
                protocol_info={'http_request': True} if i == 0 else None
            )
        
        return flow
    
    def test_feature_extractor_initialization(self):
        """Test feature extractor initialization."""
        assert self.extractor.num_features == 58
        assert len(self.extractor.feature_names) == 58
        assert 'packet_length_mean' in self.extractor.feature_names
        assert 'flow_duration' in self.extractor.feature_names
        assert 'tcp_flag_count' in self.extractor.feature_names
    
    def test_single_flow_feature_extraction(self):
        """Test feature extraction from a single flow."""
        flow = self.create_test_flow()
        features = self.extractor.extract_features(flow)
        
        # Check that we get all expected features
        assert isinstance(features, dict)
        assert len(features) == 58
        
        # Validate specific features
        assert features['packet_length_min'] > 0
        assert features['packet_length_max'] >= features['packet_length_min']
        assert features['flow_duration'] > 0
        assert features['packet_count_total'] == 10
        assert features['has_tcp'] == 1.0
        assert features['has_udp'] == 0.0
    
    def test_multiple_flows_feature_extraction(self):
        """Test feature extraction from multiple flows."""
        flows = [self.create_test_flow() for _ in range(5)]
        features_list = self.extractor.extract_features(flows)
        
        assert isinstance(features_list, list)
        assert len(features_list) == 5
        
        for features in features_list:
            assert isinstance(features, dict)
            assert len(features) == 58
    
    def test_feature_extraction_dataframe(self):
        """Test feature extraction returning DataFrame."""
        flows = [self.create_test_flow() for _ in range(3)]
        features_df = self.extractor.extract_features_dataframe(flows)
        
        assert isinstance(features_df, pd.DataFrame)
        assert features_df.shape == (3, 58)
        assert list(features_df.columns) == self.extractor.feature_names
        
        # Check for no NaN values
        assert not features_df.isnull().any().any()
    
    def test_feature_categories(self):
        """Test feature categorization."""
        categories = self.extractor.get_feature_categories()
        
        expected_categories = ['size', 'time', 'protocol', 'service', 'statistical', 'dns']
        assert set(categories.keys()) == set(expected_categories)
        
        # Check that all features are categorized
        all_categorized = []
        for cat_features in categories.values():
            all_categorized.extend(cat_features)
        
        assert len(all_categorized) == 58
        assert set(all_categorized) == set(self.extractor.feature_names)
    
    def test_feature_validation(self):
        """Test feature validation functionality."""
        flows = [self.create_test_flow()]
        features_df = self.extractor.extract_features_dataframe(flows)
        
        # Should validate successfully
        assert self.extractor.validate_features(features_df)
        
        # Should fail with missing features
        incomplete_df = features_df.drop(columns=['packet_length_mean'])
        assert not self.extractor.validate_features(incomplete_df)
    
    def test_feature_value_ranges(self):
        """Test that extracted features have reasonable value ranges."""
        flow = self.create_test_flow()
        features = self.extractor.extract_features(flow)
        
        # Size features should be positive
        assert features['packet_length_min'] >= 0
        assert features['packet_length_max'] >= features['packet_length_min']
        assert features['total_bytes_forward'] >= 0
        assert features['total_bytes_backward'] >= 0
        
        # Time features should be positive
        assert features['flow_duration'] >= 0
        
        # Count features should be non-negative integers (or close)
        assert features['packet_count_total'] >= 0
        assert features['tcp_flag_count'] >= 0
        
        # Binary features should be 0 or 1
        assert features['has_tcp'] in [0.0, 1.0]
        assert features['has_udp'] in [0.0, 1.0]
        assert features['has_icmp'] in [0.0, 1.0]


class TestFeatureSelector:
    """Test feature selection functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.selector = RandomForestSelector(n_features=10, threshold_alpha=0.01)
        
        # Create synthetic data
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 20
        
        # Create features with some having higher importance
        self.X = np.random.randn(self.n_samples, self.n_features)
        
        # Make first 5 features more predictive
        self.y = (self.X[:, 0] + self.X[:, 1] + self.X[:, 2] > 0).astype(int)
        
        self.feature_names = [f'feature_{i}' for i in range(self.n_features)]
    
    def test_selector_initialization(self):
        """Test feature selector initialization."""
        assert self.selector.n_features == 10
        assert self.selector.threshold_alpha == 0.01
        assert self.selector.selected_features_ is None
    
    def test_feature_selection_fitting(self):
        """Test feature selector fitting."""
        self.selector.fit(self.X, self.y, self.feature_names)
        
        assert self.selector.selected_features_ is not None
        assert len(self.selector.selected_features_) <= 10
        assert self.selector.feature_importances_ is not None
        assert len(self.selector.feature_importances_) == self.n_features
    
    def test_feature_transformation(self):
        """Test feature transformation."""
        self.selector.fit(self.X, self.y, self.feature_names)
        X_transformed = self.selector.transform(self.X)
        
        assert X_transformed.shape[0] == self.n_samples
        assert X_transformed.shape[1] == len(self.selector.selected_features_)
    
    def test_fit_transform(self):
        """Test combined fit and transform."""
        X_transformed = self.selector.fit_transform(self.X, self.y)
        
        assert X_transformed.shape[0] == self.n_samples
        assert X_transformed.shape[1] <= 10
    
    def test_get_selected_features(self):
        """Test getting selected feature indices."""
        self.selector.fit(self.X, self.y, self.feature_names)
        selected_indices = self.selector.get_selected_features()
        
        assert isinstance(selected_indices, list)
        assert len(selected_indices) == len(self.selector.selected_features_)
        assert all(0 <= idx < self.n_features for idx in selected_indices)
    
    def test_get_selected_feature_names(self):
        """Test getting selected feature names."""
        self.selector.fit(self.X, self.y, self.feature_names)
        selected_names = self.selector.get_selected_feature_names()
        
        assert isinstance(selected_names, list)
        assert len(selected_names) == len(self.selector.selected_features_)
        assert all(name in self.feature_names for name in selected_names)
    
    def test_evaluation(self):
        """Test feature selection evaluation."""
        self.selector.fit(self.X, self.y, self.feature_names)
        evaluation = self.selector.evaluate_selection(self.X, self.y)
        
        assert isinstance(evaluation, dict)
        assert 'accuracy_all_features' in evaluation
        assert 'accuracy_selected_features' in evaluation
        assert 'feature_reduction_ratio' in evaluation
        
        # Selected features should maintain reasonable accuracy
        assert evaluation['accuracy_selected_features'] > 0.5
        assert evaluation['feature_reduction_ratio'] < 1.0
    
    def test_feature_importance_dataframe(self):
        """Test feature importance DataFrame generation."""
        self.selector.fit(self.X, self.y, self.feature_names)
        importance_df = self.selector.get_feature_importance_dataframe()
        
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == self.n_features
        assert 'feature_name' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert 'selected' in importance_df.columns
        assert 'rank' in importance_df.columns
        
        # Should be sorted by importance (descending)
        assert importance_df['importance'].is_monotonic_decreasing


class TestIntegration:
    """Integration tests combining feature extraction and selection."""
    
    def test_end_to_end_feature_pipeline(self):
        """Test complete feature extraction and selection pipeline."""
        # Create extractor
        extractor = HybridFeatureExtractor()
        
        # Create test flows
        flows = []
        for i in range(20):
            flow_tuple = FlowTuple(f'192.168.1.{i}', '10.0.0.1', 12345, 80, 'TCP')
            flow = NetworkFlow(flow_tuple, 1000.0 + i)
            
            # Add packets with some variation
            for j in range(5 + i % 10):
                flow.add_packet(
                    packet_size=50 + j * 20 + i * 5,
                    timestamp=1000.0 + i + j * 0.1,
                    is_forward=(j % 2 == 0),
                    tcp_flags=0x18 if j == 0 else 0x10
                )
            flows.append(flow)
        
        # Extract features
        features_df = extractor.extract_features_dataframe(flows)
        
        # Create synthetic labels (some flows are "attacks")
        y = (np.arange(len(flows)) % 3 == 0).astype(int)
        
        # Select features
        selector = RandomForestSelector(n_features=15)
        X_selected = selector.fit_transform(features_df.values, y)
        
        # Verify pipeline
        assert features_df.shape == (20, 58)
        assert X_selected.shape == (20, 15)
        assert selector.feature_importances_ is not None
        assert len(selector.get_selected_features()) == 15
    
    def test_reproducibility(self):
        """Test that feature extraction is reproducible."""
        # Create identical flows
        flow_tuple = FlowTuple('192.168.1.100', '10.0.0.1', 12345, 80, 'TCP')
        
        def create_identical_flow():
            flow = NetworkFlow(flow_tuple, 1000.0)
            for i in range(5):
                flow.add_packet(100, 1000.0 + i * 0.1, i % 2 == 0)
            return flow
        
        extractor = HybridFeatureExtractor()
        
        # Extract features twice
        flow1 = create_identical_flow()
        flow2 = create_identical_flow()
        
        features1 = extractor.extract_features(flow1)
        features2 = extractor.extract_features(flow2)
        
        # Should be identical
        for key in features1.keys():
            assert abs(features1[key] - features2[key]) < 1e-10, f"Feature {key} not reproducible"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

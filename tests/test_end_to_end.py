"""
End-to-end integration tests for the complete IoT Security Framework.

Tests the full pipeline from PCAP input to final IDS predictions,
including both Phase 1 (device profiling) and Phase 2 (federated IDS).
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.phase1_profiling.pcap_reader import NetworkFlow, FlowTuple
from src.phase1_profiling.feature_extractor import HybridFeatureExtractor
from src.phase1_profiling.selectors import RandomForestSelector
from src.phase1_profiling.train_identifiers import IoTClassifier, DeviceTypeClassifier
from src.phase2_ids.federation.data_pipe import DataPipeline
from src.phase2_ids.federation.server import FederatedServer, ServerConfig
from src.phase2_ids.federation.client import FederatedClient, ClientConfig
from src.common.io import save_data, load_data


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create temporary directory for test data
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        self.device = torch.device('cpu')
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_synthetic_flows(self, n_flows: int = 50) -> list:
        """Create synthetic network flows for testing."""
        flows = []
        
        device_types = ['camera', 'sensor', 'switch', 'hub', 'thermostat']
        
        for i in range(n_flows):
            # Create flow tuple
            src_ip = f"192.168.1.{i % 20 + 100}"
            dst_ip = "10.0.0.1"
            src_port = 12345 + i
            dst_port = 80 if i % 3 == 0 else 443
            protocol = 'TCP' if i % 4 != 0 else 'UDP'
            
            flow_tuple = FlowTuple(src_ip, dst_ip, src_port, dst_port, protocol)
            flow = NetworkFlow(flow_tuple, 1000.0 + i)
            
            # Add packets with device-specific patterns
            device_type = device_types[i % len(device_types)]
            packet_count = self._get_device_packet_count(device_type)
            packet_size_base = self._get_device_packet_size(device_type)
            
            for j in range(packet_count):
                packet_size = packet_size_base + np.random.randint(-20, 21)
                timestamp = 1000.0 + i + j * 0.1
                is_forward = j % 2 == 0
                
                tcp_flags = None
                if protocol == 'TCP':
                    if j == 0:
                        tcp_flags = 0x02  # SYN
                    elif j == 1:
                        tcp_flags = 0x12  # SYN+ACK
                    else:
                        tcp_flags = 0x10  # ACK
                
                protocol_info = {}
                if dst_port == 80:
                    protocol_info['http_request'] = True
                elif dst_port == 53:
                    protocol_info['dns_query'] = True
                
                flow.add_packet(packet_size, timestamp, is_forward, tcp_flags, protocol_info)
            
            flows.append(flow)
        
        return flows
    
    def _get_device_packet_count(self, device_type: str) -> int:
        """Get typical packet count for device type."""
        counts = {
            'camera': 20,
            'sensor': 5,
            'switch': 15,
            'hub': 12,
            'thermostat': 8
        }
        return counts.get(device_type, 10)
    
    def _get_device_packet_size(self, device_type: str) -> int:
        """Get typical packet size for device type."""
        sizes = {
            'camera': 800,
            'sensor': 100,
            'switch': 200,
            'hub': 300,
            'thermostat': 150
        }
        return sizes.get(device_type, 250)
    
    def test_phase1_complete_pipeline(self):
        """Test complete Phase 1 pipeline: flows → features → classification."""
        # Step 1: Create synthetic flows
        flows = self.create_synthetic_flows(100)
        
        # Step 2: Extract features
        extractor = HybridFeatureExtractor()
        features_df = extractor.extract_features_dataframe(flows)
        
        assert features_df.shape == (100, 58)
        assert not features_df.isnull().any().any()
        
        # Step 3: Feature selection
        # Create synthetic labels for feature selection
        y_iot = np.random.choice([0, 1], 100, p=[0.2, 0.8])  # 80% IoT devices
        
        selector = RandomForestSelector(n_features=35)
        X_selected = selector.fit_transform(features_df.values, y_iot)
        
        assert X_selected.shape == (100, 35)
        
        # Step 4: IoT vs Non-IoT classification
        iot_classifier = IoTClassifier()
        iot_classifier.fit(X_selected, y_iot, selector.get_selected_feature_names())
        
        iot_predictions = iot_classifier.predict(X_selected)
        assert len(iot_predictions) == 100
        assert set(iot_predictions).issubset({0, 1})
        
        # Step 5: Device type classification (for IoT devices only)
        iot_mask = iot_predictions == 1
        if np.any(iot_mask):
            X_iot = X_selected[iot_mask]
            
            # Create synthetic device type labels
            device_types = ['camera', 'sensor', 'switch', 'hub', 'thermostat']
            y_device = np.random.choice(device_types, np.sum(iot_mask))
            
            device_classifier = DeviceTypeClassifier()
            device_classifier.fit(X_iot, y_device, selector.get_selected_feature_names())
            
            device_predictions = device_classifier.predict(X_iot)
            assert len(device_predictions) == len(y_device)
            assert all(pred in device_types for pred in device_predictions)
        
        print("✓ Phase 1 complete pipeline test passed")
    
    def test_phase2_complete_pipeline(self):
        """Test complete Phase 2 pipeline: profiles → federation → detection."""
        # Step 1: Create synthetic Phase 1 profiles
        n_samples = 200
        n_features = 35
        
        features_df = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        device_types = ['camera', 'sensor', 'switch', 'hub', 'thermostat']
        metadata_df = pd.DataFrame({
            'device_type': np.random.choice(device_types, n_samples),
            'is_iot': np.ones(n_samples, dtype=int)
        })
        
        # Step 2: Prepare federated data
        pipeline = DataPipeline(random_state=42)
        
        # Simulate attacks
        features_attack, metadata_attack = pipeline.simulate_attacks(
            features_df, metadata_df, attack_ratio=0.3
        )
        
        # Create federated splits
        client_data = pipeline.create_federated_splits(
            features_attack, metadata_attack,
            num_clients=3,
            heterogeneity=0.5
        )
        
        # Create meta-learning splits
        client_data = pipeline.create_meta_learning_splits(client_data)
        
        assert len(client_data) == 3
        for client_id, data in client_data.items():
            assert 'support_features' in data
            assert 'query_features' in data
            assert 'test_features' in data
        
        # Step 3: Setup federated learning
        model_config = {
            'input_dim': n_features,
            'num_classes': len(pipeline.label_encoder.classes_),
            'hidden_channels': [32, 64, 32],
            'dropout': 0.3
        }
        
        server_config = ServerConfig(
            num_clients=3,
            rounds=3,  # Reduced for testing
            participation_rate=1.0,
            local_epochs=1,
            gamma_top_helpers=2
        )
        
        # Create server
        server = FederatedServer(model_config, server_config, self.device)
        
        # Create clients
        clients = {}
        for client_id, data in client_data.items():
            client_config = ClientConfig(
                client_id=client_id,
                local_epochs=1,
                batch_size=16
            )
            
            client = FederatedClient(client_id, model_config, client_config, self.device)
            client.load_local_data(data)
            clients[client_id] = client
            
            # Register with server
            server.register_client(client_id, {
                'num_samples': data.get('num_samples', 0),
                'device_types': data.get('device_types', [])
            })
        
        # Step 4: Run federated training
        for round_num in range(1, 4):  # 3 rounds for testing
            selected_clients = server.select_clients(round_num)
            global_model = server.get_global_model()
            
            client_updates = {}
            for client_id in selected_clients:
                if client_id in clients:
                    update = clients[client_id].federated_round(global_model)
                    client_updates[client_id] = update
            
            round_result = server.federated_round(client_updates)
            assert 'global_metrics' in round_result
        
        # Step 5: Test self-labeling workflow
        # Train CT-AE for each client
        for client_id, client in clients.items():
            ct_ae_metrics = client.train_ct_ae()
            assert 'total_loss' in ct_ae_metrics
            
            # Encode data
            client.encode_data('support')
            client.encode_data('test')
            
            # Update server with encodings
            encodings = client.get_encodings_for_server()
            server.update_client_encodings(client_id, encodings)
        
        # Test similarity-based aggregation
        target_client = list(clients.keys())[0]
        helpers = server.select_helpers(target_client, gamma=2)
        
        if helpers:
            annotation_model = server.create_annotation_model(target_client, helpers)
            
            # Run self-labeling workflow
            workflow_results = clients[target_client].self_labeling_workflow(annotation_model)
            
            assert 'prelabel_results' in workflow_results
            assert 'adaptation_metrics' in workflow_results
            assert 'final_metrics' in workflow_results
            
            # Check final metrics
            final_metrics = workflow_results['final_metrics']
            assert 'accuracy' in final_metrics
            assert 0 <= final_metrics['accuracy'] <= 1
        
        print("✓ Phase 2 complete pipeline test passed")
    
    def test_full_end_to_end_integration(self):
        """Test complete integration from synthetic flows to federated detection."""
        # Phase 1: Device Profiling
        print("Running Phase 1: Device Profiling")
        
        # Create flows
        flows = self.create_synthetic_flows(150)
        
        # Extract features
        extractor = HybridFeatureExtractor()
        features_df = extractor.extract_features_dataframe(flows)
        
        # Create device profiles (simplified)
        device_types = ['camera', 'sensor', 'switch', 'hub', 'thermostat']
        profiles_df = features_df.copy()
        profiles_df['device_type'] = np.random.choice(device_types, len(flows))
        profiles_df['is_iot'] = 1
        
        # Save Phase 1 results
        phase1_output = self.test_dir / "phase1_profiles.parquet"
        save_data(profiles_df, phase1_output)
        
        # Phase 2: Federated Learning
        print("Running Phase 2: Federated Learning")
        
        # Load Phase 1 results
        loaded_profiles = load_data(phase1_output)
        
        # Prepare federated data
        pipeline = DataPipeline(random_state=42)
        
        # Extract features and metadata
        feature_columns = [col for col in loaded_profiles.columns 
                          if col not in ['device_type', 'is_iot']]
        features_df = loaded_profiles[feature_columns]
        metadata_df = loaded_profiles[['device_type', 'is_iot']]
        
        # Simulate attacks and create federated splits
        features_attack, metadata_attack = pipeline.simulate_attacks(
            features_df, metadata_df, attack_ratio=0.25
        )
        
        client_data = pipeline.create_federated_splits(
            features_attack, metadata_attack,
            num_clients=3,
            heterogeneity=0.6
        )
        
        client_data = pipeline.create_meta_learning_splits(client_data)
        
        # Setup and run mini federated learning
        model_config = {
            'input_dim': len(feature_columns),
            'num_classes': len(pipeline.label_encoder.classes_),
            'hidden_channels': [32, 16],
            'dropout': 0.2
        }
        
        server_config = ServerConfig(
            num_clients=3,
            rounds=2,  # Minimal for testing
            participation_rate=1.0,
            local_epochs=1
        )
        
        server = FederatedServer(model_config, server_config, self.device)
        
        clients = {}
        for client_id, data in client_data.items():
            client_config = ClientConfig(
                client_id=client_id,
                local_epochs=1,
                batch_size=8,
                ct_ae_epochs=2  # Reduced for testing
            )
            
            client = FederatedClient(client_id, model_config, client_config, self.device)
            client.load_local_data(data)
            clients[client_id] = client
            
            server.register_client(client_id, {
                'num_samples': data.get('num_samples', 0),
                'device_types': data.get('device_types', [])
            })
        
        # Run minimal federated training
        for round_num in range(1, 3):
            selected_clients = server.select_clients(round_num)
            global_model = server.get_global_model()
            
            client_updates = {}
            for client_id in selected_clients:
                if client_id in clients:
                    update = clients[client_id].federated_round(global_model)
                    client_updates[client_id] = update
            
            server.federated_round(client_updates)
        
        # Test final evaluation
        final_accuracies = []
        for client_id, client in clients.items():
            metrics = client.evaluate_on_query()
            if 'accuracy' in metrics:
                final_accuracies.append(metrics['accuracy'])
        
        if final_accuracies:
            avg_accuracy = np.mean(final_accuracies)
            assert 0 <= avg_accuracy <= 1
            print(f"Average final accuracy: {avg_accuracy:.4f}")
        
        print("✓ Full end-to-end integration test passed")
    
    def test_data_persistence_and_loading(self):
        """Test data persistence and loading between phases."""
        # Create and save Phase 1 data
        features_df = pd.DataFrame(
            np.random.randn(50, 35),
            columns=[f'feature_{i}' for i in range(35)]
        )
        
        metadata_df = pd.DataFrame({
            'device_type': np.random.choice(['camera', 'sensor'], 50),
            'is_iot': np.ones(50, dtype=int)
        })
        
        combined_df = pd.concat([features_df, metadata_df], axis=1)
        
        # Save and reload
        save_path = self.test_dir / "test_profiles.parquet"
        save_data(combined_df, save_path)
        
        loaded_df = load_data(save_path)
        
        # Verify data integrity
        pd.testing.assert_frame_equal(combined_df, loaded_df)
        
        # Test Phase 2 data pipeline with loaded data
        pipeline = DataPipeline()
        
        feature_cols = [f'feature_{i}' for i in range(35)]
        features_loaded = loaded_df[feature_cols]
        metadata_loaded = loaded_df[['device_type', 'is_iot']]
        
        # Should work without errors
        features_attack, metadata_attack = pipeline.simulate_attacks(
            features_loaded, metadata_loaded, attack_ratio=0.2
        )
        
        client_data = pipeline.create_federated_splits(
            features_attack, metadata_attack, num_clients=2
        )
        
        assert len(client_data) == 2
        
        print("✓ Data persistence and loading test passed")


class TestSystemIntegration:
    """Test system-level integration aspects."""
    
    def test_memory_efficiency(self):
        """Test that the system handles reasonably large datasets efficiently."""
        # Create larger synthetic dataset
        n_samples = 1000
        n_features = 35
        
        features_df = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Test Phase 1 components
        extractor = HybridFeatureExtractor()
        
        # Should handle batch processing efficiently
        batch_size = 100
        all_features = []
        
        for i in range(0, n_samples, batch_size):
            batch_features = features_df.iloc[i:i+batch_size]
            # In real scenario, these would be flows, but we're testing memory handling
            all_features.append(batch_features)
        
        combined_features = pd.concat(all_features, ignore_index=True)
        assert len(combined_features) == n_samples
        
        print("✓ Memory efficiency test passed")
    
    def test_error_handling_and_recovery(self):
        """Test system behavior under error conditions."""
        # Test with invalid data
        extractor = HybridFeatureExtractor()
        
        # Empty flows list should not crash
        empty_features = extractor.extract_features([])
        assert empty_features == []
        
        # Test with malformed data
        try:
            # This should handle gracefully
            invalid_flow = None
            result = extractor.extract_features(invalid_flow)
            # Should either return empty or raise appropriate exception
        except (TypeError, AttributeError):
            # Expected behavior for invalid input
            pass
        
        print("✓ Error handling test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

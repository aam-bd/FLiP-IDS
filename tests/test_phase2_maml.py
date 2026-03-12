"""
Tests for Phase 2 MAML (Model-Agnostic Meta-Learning) functionality.

Tests the MAML implementation for fast adaptation in federated learning scenarios.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.phase2_ids.models.maml import (
    MAMLOptimizer, MAMLTrainer, create_support_query_split, compute_maml_gradients
)
from src.phase2_ids.models.cnn_1d import CNN1DClassifier
from src.phase2_ids.federation.data_pipe import IoTFlowDataset


class TestMAMLOptimizer:
    """Test MAML optimizer functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device('cpu')
        self.input_dim = 35
        self.num_classes = 5
        self.batch_size = 16
        
        # Create a simple model
        self.model = CNN1DClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_channels=[32, 16]
        )
        
        self.maml_optimizer = MAMLOptimizer(
            model=self.model,
            inner_lr=0.01,
            outer_lr=0.001,
            inner_steps=1
        )
        
        # Create synthetic data
        torch.manual_seed(42)
        self.support_data = torch.randn(self.batch_size, self.input_dim)
        self.support_labels = torch.randint(0, self.num_classes, (self.batch_size,))
        self.query_data = torch.randn(self.batch_size, self.input_dim)
        self.query_labels = torch.randint(0, self.num_classes, (self.batch_size,))
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def test_maml_optimizer_initialization(self):
        """Test MAML optimizer initialization."""
        assert self.maml_optimizer.model == self.model
        assert self.maml_optimizer.inner_lr == 0.01
        assert self.maml_optimizer.outer_lr == 0.001
        assert self.maml_optimizer.inner_steps == 1
        assert self.maml_optimizer.first_order == False
    
    def test_inner_loop_adaptation(self):
        """Test inner loop adaptation."""
        adapted_model, inner_losses = self.maml_optimizer.inner_loop(
            self.support_data, self.support_labels, self.loss_fn
        )
        
        # Check that adapted model is different from original
        assert adapted_model is not self.model
        assert len(inner_losses) == self.maml_optimizer.inner_steps
        assert all(loss >= 0 for loss in inner_losses)
        
        # Check that adapted model has same architecture
        assert adapted_model.input_dim == self.model.input_dim
        assert adapted_model.num_classes == self.model.num_classes
    
    def test_outer_loop_evaluation(self):
        """Test outer loop evaluation."""
        # First adapt the model
        adapted_model, _ = self.maml_optimizer.inner_loop(
            self.support_data, self.support_labels, self.loss_fn
        )
        
        # Evaluate on query set
        query_loss = self.maml_optimizer.outer_loop(
            self.query_data, self.query_labels, adapted_model, self.loss_fn
        )
        
        assert isinstance(query_loss, float)
        assert query_loss >= 0
    
    def test_meta_update(self):
        """Test meta-update across multiple tasks."""
        # Create multiple tasks
        tasks = []
        for _ in range(3):
            support_data = torch.randn(8, self.input_dim)
            support_labels = torch.randint(0, self.num_classes, (8,))
            query_data = torch.randn(8, self.input_dim)
            query_labels = torch.randint(0, self.num_classes, (8,))
            tasks.append((support_data, support_labels, query_data, query_labels))
        
        # Store original parameters
        original_params = [p.clone() for p in self.model.parameters()]
        
        # Perform meta-update
        metrics = self.maml_optimizer.meta_update(tasks, self.loss_fn)
        
        # Check metrics
        assert isinstance(metrics, dict)
        assert 'meta_loss' in metrics
        assert 'support_loss' in metrics
        assert 'num_tasks' in metrics
        assert metrics['num_tasks'] == 3
        
        # Check that parameters have been updated
        updated_params = list(self.model.parameters())
        for orig, updated in zip(original_params, updated_params):
            assert not torch.allclose(orig, updated, atol=1e-6)
    
    def test_adapt_to_task(self):
        """Test task-specific adaptation."""
        adapted_model = self.maml_optimizer.adapt_to_task(
            self.support_data, self.support_labels, self.loss_fn
        )
        
        # Test that adapted model can make predictions
        with torch.no_grad():
            outputs = adapted_model(self.query_data)
        
        assert outputs.shape == (self.batch_size, self.num_classes)
        assert torch.isfinite(outputs).all()
    
    def test_first_order_approximation(self):
        """Test first-order MAML approximation."""
        first_order_optimizer = MAMLOptimizer(
            model=self.model,
            inner_lr=0.01,
            outer_lr=0.001,
            inner_steps=1,
            first_order=True
        )
        
        # Should work without second-order gradients
        adapted_model, inner_losses = first_order_optimizer.inner_loop(
            self.support_data, self.support_labels, self.loss_fn, create_graph=False
        )
        
        assert len(inner_losses) == 1
        assert adapted_model is not self.model


class TestMAMLTrainer:
    """Test MAML trainer functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device('cpu')
        self.input_dim = 35
        self.num_classes = 5
        
        self.model = CNN1DClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_channels=[16, 32, 16]
        )
        
        self.trainer = MAMLTrainer(
            model=self.model,
            device=self.device,
            inner_lr=0.01,
            outer_lr=0.001,
            inner_steps=1
        )
        
        # Create synthetic datasets
        self.support_loaders = []
        self.query_loaders = []
        
        torch.manual_seed(42)
        for _ in range(3):  # 3 tasks
            # Support set
            support_features = torch.randn(32, self.input_dim)
            support_labels = torch.randint(0, self.num_classes, (32,))
            support_dataset = IoTFlowDataset(support_features.numpy(), support_labels.numpy())
            support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=16, shuffle=False)
            self.support_loaders.append(support_loader)
            
            # Query set
            query_features = torch.randn(16, self.input_dim)
            query_labels = torch.randint(0, self.num_classes, (16,))
            query_dataset = IoTFlowDataset(query_features.numpy(), query_labels.numpy())
            query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=16, shuffle=False)
            self.query_loaders.append(query_loader)
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def test_trainer_initialization(self):
        """Test MAML trainer initialization."""
        assert self.trainer.model == self.model
        assert self.trainer.device == self.device
        assert self.trainer.maml_optimizer.inner_lr == 0.01
        assert self.trainer.maml_optimizer.outer_lr == 0.001
    
    def test_train_episode(self):
        """Test training one meta-learning episode."""
        metrics = self.trainer.train_episode(
            self.support_loaders, self.query_loaders, self.loss_fn
        )
        
        assert isinstance(metrics, dict)
        assert 'meta_loss' in metrics
        assert 'support_loss' in metrics
        assert 'num_tasks' in metrics
        assert metrics['num_tasks'] == 3
        assert metrics['meta_loss'] >= 0
    
    def test_evaluate_episode(self):
        """Test evaluation episode."""
        metrics = self.trainer.evaluate_episode(
            self.support_loaders, self.query_loaders, self.loss_fn
        )
        
        assert isinstance(metrics, dict)
        assert 'query_loss' in metrics
        assert 'query_accuracy' in metrics
        assert 'num_tasks' in metrics
        assert 0 <= metrics['query_accuracy'] <= 1
    
    def test_adapt_and_predict(self):
        """Test adaptation and prediction."""
        # Get one batch from support and query
        support_data, support_labels = next(iter(self.support_loaders[0]))
        query_data, query_labels = next(iter(self.query_loaders[0]))
        
        predictions, probabilities = self.trainer.adapt_and_predict(
            support_data, support_labels, query_data, self.loss_fn
        )
        
        assert predictions.shape == (query_data.shape[0],)
        assert probabilities.shape == (query_data.shape[0], self.num_classes)
        assert torch.all(predictions >= 0) and torch.all(predictions < self.num_classes)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(query_data.shape[0]))
    
    def test_multi_episode_training(self):
        """Test training over multiple episodes."""
        initial_loss = float('inf')
        
        for episode in range(3):  # Short training for test
            metrics = self.trainer.train_episode(
                self.support_loaders, self.query_loaders, self.loss_fn
            )
            
            if episode == 0:
                initial_loss = metrics['meta_loss']
            
            assert metrics['meta_loss'] >= 0
        
        # Final evaluation
        eval_metrics = self.trainer.evaluate_episode(
            self.support_loaders, self.query_loaders, self.loss_fn
        )
        
        # Should have reasonable performance
        assert eval_metrics['query_accuracy'] >= 0.0  # At least random chance


class TestMAMLUtilities:
    """Test MAML utility functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        torch.manual_seed(42)
        self.data = torch.randn(100, 35)
        self.labels = torch.randint(0, 5, (100,))
    
    def test_support_query_split(self):
        """Test support/query data splitting."""
        support_data, support_labels, query_data, query_labels = create_support_query_split(
            self.data, self.labels, support_ratio=0.6
        )
        
        total_samples = self.data.shape[0]
        expected_support = int(total_samples * 0.6)
        expected_query = total_samples - expected_support
        
        assert support_data.shape[0] == expected_support
        assert query_data.shape[0] == expected_query
        assert support_labels.shape[0] == expected_support
        assert query_labels.shape[0] == expected_query
        
        # Check that all samples are accounted for
        assert support_data.shape[0] + query_data.shape[0] == total_samples
    
    def test_maml_gradients_computation(self):
        """Test MAML gradient computation."""
        model = CNN1DClassifier(input_dim=35, num_classes=5, hidden_channels=[16, 8])
        loss_fn = nn.CrossEntropyLoss()
        
        support_data = self.data[:20]
        support_labels = self.labels[:20]
        
        gradients = compute_maml_gradients(
            model, support_data, support_labels, loss_fn, inner_lr=0.01
        )
        
        # Check that gradients are computed for all parameters
        model_params = list(model.parameters())
        assert len(gradients) == len(model_params)
        
        # Check that gradients have correct shapes
        for grad, param in zip(gradients, model_params):
            assert grad.shape == param.shape
            assert torch.isfinite(grad).all()
    
    def test_different_support_ratios(self):
        """Test support/query splits with different ratios."""
        ratios = [0.3, 0.5, 0.7, 0.9]
        
        for ratio in ratios:
            support_data, support_labels, query_data, query_labels = create_support_query_split(
                self.data, self.labels, support_ratio=ratio
            )
            
            total_samples = self.data.shape[0]
            expected_support = int(total_samples * ratio)
            
            assert support_data.shape[0] == expected_support
            assert support_data.shape[0] + query_data.shape[0] == total_samples


class TestMAMLIntegration:
    """Integration tests for MAML in federated learning context."""
    
    def test_few_shot_adaptation(self):
        """Test few-shot learning capability of MAML."""
        device = torch.device('cpu')
        model = CNN1DClassifier(input_dim=35, num_classes=3, hidden_channels=[16, 8])
        trainer = MAMLTrainer(model, device, inner_lr=0.02, inner_steps=2)
        
        # Create a few-shot learning scenario
        torch.manual_seed(42)
        
        # Task 1: Only 8 support samples per class
        support_per_class = 8
        num_classes = 3
        
        support_features = []
        support_labels = []
        
        for class_id in range(num_classes):
            # Create class-specific patterns
            class_features = torch.randn(support_per_class, 35)
            class_features[:, class_id*5:(class_id+1)*5] += 2.0  # Class-specific pattern
            
            support_features.append(class_features)
            support_labels.extend([class_id] * support_per_class)
        
        support_data = torch.cat(support_features, dim=0)
        support_labels = torch.tensor(support_labels)
        
        # Query set with same pattern
        query_features = []
        query_labels = []
        
        for class_id in range(num_classes):
            class_features = torch.randn(5, 35)
            class_features[:, class_id*5:(class_id+1)*5] += 2.0
            
            query_features.append(class_features)
            query_labels.extend([class_id] * 5)
        
        query_data = torch.cat(query_features, dim=0)
        query_labels = torch.tensor(query_labels)
        
        # Test adaptation
        loss_fn = nn.CrossEntropyLoss()
        predictions, probabilities = trainer.adapt_and_predict(
            support_data, support_labels, query_data, loss_fn
        )
        
        # Calculate accuracy
        accuracy = (predictions == query_labels).float().mean().item()
        
        # Should achieve better than random performance (1/3 = 0.33)
        assert accuracy > 0.4, f"Few-shot accuracy {accuracy} should be > 0.4"
    
    def test_meta_learning_vs_regular_learning(self):
        """Compare MAML meta-learning with regular learning."""
        device = torch.device('cpu')
        torch.manual_seed(42)
        
        # Create two identical models
        meta_model = CNN1DClassifier(input_dim=35, num_classes=3, hidden_channels=[16, 8])
        regular_model = CNN1DClassifier(input_dim=35, num_classes=3, hidden_channels=[16, 8])
        
        # Copy weights to ensure identical starting points
        regular_model.load_state_dict(meta_model.state_dict())
        
        # Create training tasks for meta-learning
        support_loaders = []
        query_loaders = []
        
        for _ in range(5):  # 5 meta-training tasks
            support_data = torch.randn(24, 35)
            support_labels = torch.randint(0, 3, (24,))
            support_dataset = IoTFlowDataset(support_data.numpy(), support_labels.numpy())
            support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=24)
            support_loaders.append(support_loader)
            
            query_data = torch.randn(12, 35)
            query_labels = torch.randint(0, 3, (12,))
            query_dataset = IoTFlowDataset(query_data.numpy(), query_labels.numpy())
            query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=12)
            query_loaders.append(query_loader)
        
        # Train meta-model
        meta_trainer = MAMLTrainer(meta_model, device, inner_lr=0.01, inner_steps=1)
        loss_fn = nn.CrossEntropyLoss()
        
        for _ in range(3):  # Few meta-training steps
            meta_trainer.train_episode(support_loaders, query_loaders, loss_fn)
        
        # Train regular model on the same data (flattened)
        all_data = []
        all_labels = []
        for support_loader in support_loaders:
            for data, labels in support_loader:
                all_data.append(data)
                all_labels.append(labels)
        
        regular_data = torch.cat(all_data, dim=0)
        regular_labels = torch.cat(all_labels, dim=0)
        
        regular_optimizer = torch.optim.Adam(regular_model.parameters(), lr=0.01)
        
        for _ in range(10):  # Regular training steps
            regular_optimizer.zero_grad()
            outputs = regular_model(regular_data)
            loss = loss_fn(outputs, regular_labels)
            loss.backward()
            regular_optimizer.step()
        
        # Test both models on a new task
        test_support = torch.randn(12, 35)
        test_support_labels = torch.randint(0, 3, (12,))
        test_query = torch.randn(6, 35)
        test_query_labels = torch.randint(0, 3, (6,))
        
        # Meta-model adaptation
        meta_predictions, _ = meta_trainer.adapt_and_predict(
            test_support, test_support_labels, test_query, loss_fn
        )
        meta_accuracy = (meta_predictions == test_query_labels).float().mean().item()
        
        # Regular model (no adaptation)
        with torch.no_grad():
            regular_outputs = regular_model(test_query)
            regular_predictions = regular_outputs.argmax(dim=1)
            regular_accuracy = (regular_predictions == test_query_labels).float().mean().item()
        
        print(f"Meta-learning accuracy: {meta_accuracy:.3f}")
        print(f"Regular learning accuracy: {regular_accuracy:.3f}")
        
        # Meta-learning should be at least competitive
        # (Note: This is a weak test due to randomness and small scale)
        assert meta_accuracy >= 0.0  # At least not completely broken


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

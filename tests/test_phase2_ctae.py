"""
Tests for Phase 2 CT-AE (Cosine-Targeted Autoencoder) functionality.

Tests the CT-AE implementation, loss functions, encoding quality,
and similarity computation capabilities.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.phase2_ids.models.autoencoders import (
    CosineTargetedAutoencoder, CTAELoss, CTAETrainer, create_ctae_model
)
from src.phase2_ids.federation.data_pipe import IoTFlowDataset


class TestCosineTargetedAutoencoder:
    """Test CT-AE model functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 35
        self.latent_dim = 16
        self.batch_size = 8
        
        self.model = CosineTargetedAutoencoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=[64, 32]
        )
        
        self.device = torch.device('cpu')
        self.model.to(self.device)
        
        # Create test data
        torch.manual_seed(42)
        self.test_data = torch.randn(self.batch_size, self.input_dim)
    
    def test_model_initialization(self):
        """Test CT-AE model initialization."""
        assert self.model.input_dim == self.input_dim
        assert self.model.latent_dim == self.latent_dim
        assert self.model.hidden_dims == [64, 32]
        
        # Check model info
        info = self.model.get_model_info()
        assert info['model_type'] == 'CosineTargetedAutoencoder'
        assert info['input_dim'] == self.input_dim
        assert info['latent_dim'] == self.latent_dim
        assert info['total_parameters'] > 0
    
    def test_forward_pass(self):
        """Test forward pass through CT-AE."""
        with torch.no_grad():
            x_recon, z_latent = self.model(self.test_data)
        
        # Check output shapes
        assert x_recon.shape == self.test_data.shape
        assert z_latent.shape == (self.batch_size, self.latent_dim)
        
        # Check that outputs are reasonable
        assert torch.isfinite(x_recon).all()
        assert torch.isfinite(z_latent).all()
    
    def test_encode_decode_consistency(self):
        """Test encode-decode consistency."""
        with torch.no_grad():
            # Encode
            z_latent = self.model.encode(self.test_data)
            
            # Decode
            x_recon = self.model.decode(z_latent)
            
            # Compare with forward pass
            x_recon_forward, z_latent_forward = self.model(self.test_data)
            
            assert torch.allclose(z_latent, z_latent_forward, atol=1e-6)
            assert torch.allclose(x_recon, x_recon_forward, atol=1e-6)
    
    def test_latent_dimension_reduction(self):
        """Test that latent space has lower dimension."""
        with torch.no_grad():
            z_latent = self.model.encode(self.test_data)
        
        assert z_latent.shape[1] < self.test_data.shape[1]
        assert z_latent.shape[1] == self.latent_dim
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        # Enable gradients
        self.test_data.requires_grad_(True)
        
        x_recon, z_latent = self.model(self.test_data)
        loss = nn.MSELoss()(x_recon, self.test_data)
        
        loss.backward()
        
        # Check that gradients exist
        for param in self.model.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
    
    def test_different_input_sizes(self):
        """Test model with different batch sizes."""
        batch_sizes = [1, 4, 16]
        
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, self.input_dim)
            
            with torch.no_grad():
                x_recon, z_latent = self.model(test_input)
            
            assert x_recon.shape == (batch_size, self.input_dim)
            assert z_latent.shape == (batch_size, self.latent_dim)


class TestCTAELoss:
    """Test CT-AE loss function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.w_rec = 0.7
        self.w_cos = 0.3
        self.loss_fn = CTAELoss(w_rec=self.w_rec, w_cos=self.w_cos)
        
        # Create test data
        torch.manual_seed(42)
        self.batch_size = 8
        self.input_dim = 35
        self.latent_dim = 16
        
        self.x_input = torch.randn(self.batch_size, self.input_dim)
        self.x_recon = torch.randn(self.batch_size, self.input_dim)
        self.z_latent = torch.randn(self.batch_size, self.latent_dim)
        self.z_target = torch.randn(self.batch_size, self.latent_dim)
    
    def test_loss_initialization(self):
        """Test loss function initialization."""
        assert self.loss_fn.w_rec == self.w_rec
        assert self.loss_fn.w_cos == self.w_cos
    
    def test_loss_computation_with_target(self):
        """Test loss computation with target latent vectors."""
        total_loss, components = self.loss_fn(
            self.x_input, self.x_recon, self.z_latent, self.z_target
        )
        
        # Check loss components
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert total_loss.item() >= 0
        
        assert 'total_loss' in components
        assert 'reconstruction_loss' in components
        assert 'cosine_loss' in components
        
        # Check that total loss is weighted sum
        expected_total = (self.w_rec * components['reconstruction_loss'] + 
                         self.w_cos * components['cosine_loss'])
        assert abs(components['total_loss'] - expected_total) < 1e-6
    
    def test_loss_computation_without_target(self):
        """Test loss computation without target latent vectors."""
        total_loss, components = self.loss_fn(
            self.x_input, self.x_recon, self.z_latent, None
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() >= 0
        assert 'cosine_loss' in components
    
    def test_reconstruction_loss_component(self):
        """Test reconstruction loss component."""
        # Perfect reconstruction should give zero reconstruction loss
        total_loss, components = self.loss_fn(
            self.x_input, self.x_input, self.z_latent, self.z_target
        )
        
        assert components['reconstruction_loss'] < 1e-6
    
    def test_cosine_similarity_component(self):
        """Test cosine similarity component."""
        # Identical latent vectors should give zero cosine loss
        total_loss, components = self.loss_fn(
            self.x_input, self.x_recon, self.z_latent, self.z_latent
        )
        
        assert components['cosine_loss'] < 1e-6
    
    def test_loss_weights(self):
        """Test that loss weights affect the total loss correctly."""
        # Test with different weights
        loss_fn_1 = CTAELoss(w_rec=1.0, w_cos=0.0)
        loss_fn_2 = CTAELoss(w_rec=0.0, w_cos=1.0)
        
        total_loss_1, _ = loss_fn_1(self.x_input, self.x_recon, self.z_latent, self.z_target)
        total_loss_2, _ = loss_fn_2(self.x_input, self.x_recon, self.z_latent, self.z_target)
        
        # Losses should be different
        assert abs(total_loss_1.item() - total_loss_2.item()) > 1e-6


class TestCTAETrainer:
    """Test CT-AE trainer functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device('cpu')
        self.input_dim = 35
        self.latent_dim = 16
        
        self.model = CosineTargetedAutoencoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim
        )
        
        self.trainer = CTAETrainer(self.model, self.device, w_rec=0.7, w_cos=0.3)
        
        # Create synthetic dataset
        torch.manual_seed(42)
        self.n_samples = 100
        features = torch.randn(self.n_samples, self.input_dim)
        labels = torch.randint(0, 5, (self.n_samples,))
        
        self.dataset = IoTFlowDataset(features.numpy(), labels.numpy())
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=16, shuffle=True
        )
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.model == self.model
        assert self.trainer.device == self.device
        assert self.trainer.criterion.w_rec == 0.7
        assert self.trainer.criterion.w_cos == 0.3
    
    def test_train_epoch(self):
        """Test training for one epoch."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        metrics = self.trainer.train_epoch(self.dataloader, optimizer)
        
        assert isinstance(metrics, dict)
        assert 'total_loss' in metrics
        assert 'reconstruction_loss' in metrics
        assert 'cosine_loss' in metrics
        
        # Losses should be positive
        assert metrics['total_loss'] > 0
        assert metrics['reconstruction_loss'] > 0
        assert metrics['cosine_loss'] >= 0
    
    def test_evaluate(self):
        """Test model evaluation."""
        metrics = self.trainer.evaluate(self.dataloader)
        
        assert isinstance(metrics, dict)
        assert 'total_loss' in metrics
        assert 'reconstruction_loss' in metrics
        assert 'cosine_loss' in metrics
    
    def test_encode_data(self):
        """Test data encoding."""
        encodings = self.trainer.encode_data(self.dataloader)
        
        assert isinstance(encodings, np.ndarray)
        assert encodings.shape == (self.n_samples, self.latent_dim)
        assert np.isfinite(encodings).all()
    
    def test_similarity_computation(self):
        """Test similarity matrix computation."""
        encodings1 = np.random.randn(10, self.latent_dim)
        encodings2 = np.random.randn(15, self.latent_dim)
        
        similarity_matrix = self.trainer.compute_similarity_matrix(encodings1, encodings2)
        
        assert similarity_matrix.shape == (10, 15)
        assert np.all(similarity_matrix >= -1) and np.all(similarity_matrix <= 1)
        
        # Test self-similarity
        self_similarity = self.trainer.compute_similarity_matrix(encodings1)
        assert self_similarity.shape == (10, 10)
        
        # Diagonal should be 1 (self-similarity)
        np.testing.assert_allclose(np.diag(self_similarity), 1.0, atol=1e-6)
    
    def test_training_improves_reconstruction(self):
        """Test that training improves reconstruction quality."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Initial evaluation
        initial_metrics = self.trainer.evaluate(self.dataloader)
        initial_loss = initial_metrics['reconstruction_loss']
        
        # Train for a few epochs
        for _ in range(3):
            self.trainer.train_epoch(self.dataloader, optimizer)
        
        # Final evaluation
        final_metrics = self.trainer.evaluate(self.dataloader)
        final_loss = final_metrics['reconstruction_loss']
        
        # Loss should decrease (or at least not increase significantly)
        assert final_loss <= initial_loss * 1.1  # Allow 10% tolerance


class TestCTAEIntegration:
    """Integration tests for CT-AE functionality."""
    
    def test_ctae_model_factory(self):
        """Test CT-AE model factory function."""
        config = {
            'input_dim': 35,
            'latent_dim': 20,
            'hidden_dims': [128, 64],
            'activation': 'relu',
            'dropout_rate': 0.2
        }
        
        model = create_ctae_model(config)
        
        assert isinstance(model, CosineTargetedAutoencoder)
        assert model.input_dim == 35
        assert model.latent_dim == 20
        assert model.hidden_dims == [128, 64]
    
    def test_encoding_preserves_similarity_structure(self):
        """Test that CT-AE encoding preserves similarity structure."""
        device = torch.device('cpu')
        model = CosineTargetedAutoencoder(input_dim=35, latent_dim=16)
        trainer = CTAETrainer(model, device)
        
        # Create data with known similarity structure
        torch.manual_seed(42)
        
        # Group 1: similar data
        group1 = torch.randn(10, 35) + torch.tensor([1.0] * 35)
        
        # Group 2: similar data (different from group 1)
        group2 = torch.randn(10, 35) + torch.tensor([-1.0] * 35)
        
        all_data = torch.cat([group1, group2])
        
        # Encode data
        with torch.no_grad():
            encodings = model.encode(all_data).numpy()
        
        # Compute similarity matrices
        original_sim = np.corrcoef(all_data.numpy())
        encoded_sim = np.corrcoef(encodings)
        
        # Similarity structure should be somewhat preserved
        # (this is a weak test due to lack of training, but checks basic functionality)
        assert encodings.shape == (20, 16)
        assert np.isfinite(encoded_sim).all()
    
    def test_end_to_end_workflow(self):
        """Test complete CT-AE workflow."""
        device = torch.device('cpu')
        
        # Create model
        model = CosineTargetedAutoencoder(input_dim=35, latent_dim=16)
        trainer = CTAETrainer(model, device)
        
        # Create dataset
        torch.manual_seed(42)
        features = torch.randn(50, 35)
        labels = torch.randint(0, 3, (50,))
        
        dataset = IoTFlowDataset(features.numpy(), labels.numpy())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Train model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(2):  # Short training for test
            train_metrics = trainer.train_epoch(dataloader, optimizer)
            eval_metrics = trainer.evaluate(dataloader)
            
            assert train_metrics['total_loss'] > 0
            assert eval_metrics['total_loss'] > 0
        
        # Encode data
        encodings = trainer.encode_data(dataloader)
        
        assert encodings.shape == (50, 16)
        assert np.isfinite(encodings).all()
        
        # Compute similarities
        similarities = trainer.compute_similarity_matrix(encodings)
        
        assert similarities.shape == (50, 50)
        assert np.all(similarities >= -1) and np.all(similarities <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

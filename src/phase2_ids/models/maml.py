"""
Model-Agnostic Meta-Learning (MAML) utilities for personalized federated learning.

Implements MAML optimization for the SOH-FL framework, enabling fast adaptation
to new tasks with minimal gradient steps. The implementation follows the
meta-learning objective: min f(ω - α∇f(ω)) where α is the inner learning rate.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Callable, Any
import copy
from collections import OrderedDict
import numpy as np

from ...common.logging import get_logger

logger = get_logger(__name__)


class MAMLOptimizer:
    """
    MAML optimizer for meta-learning in federated settings.
    
    Implements the MAML algorithm with inner and outer loop optimization
    as described in the SOH-FL paper for personalized federated learning.
    """
    
    def __init__(self,
                 model: nn.Module,
                 inner_lr: float = 0.001,
                 outer_lr: float = 0.005,
                 inner_steps: int = 1,
                 first_order: bool = False):
        """
        Initialize MAML optimizer.
        
        Args:
            model: Base model to optimize
            inner_lr: Inner loop learning rate (alpha)
            outer_lr: Outer loop learning rate (beta)
            inner_steps: Number of inner loop gradient steps
            first_order: Whether to use first-order approximation
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
        # Outer loop optimizer
        self.meta_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
        
        self.logger = logger
    
    def inner_loop(self,
                   support_data: torch.Tensor,
                   support_labels: torch.Tensor,
                   loss_fn: Callable,
                   create_graph: bool = True) -> Tuple[nn.Module, List[float]]:
        """
        Perform inner loop adaptation.
        
        Args:
            support_data: Support set data
            support_labels: Support set labels
            loss_fn: Loss function
            create_graph: Whether to create computation graph for second-order derivatives
            
        Returns:
            Tuple of (adapted_model, inner_losses)
        """
        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self.model)
        
        inner_losses = []
        
        for step in range(self.inner_steps):
            # Forward pass on support set
            support_logits = adapted_model(support_data)
            support_loss = loss_fn(support_logits, support_labels)
            
            inner_losses.append(support_loss.item())
            
            # Compute gradients
            grads = torch.autograd.grad(
                support_loss,
                adapted_model.parameters(),
                create_graph=create_graph and not self.first_order,
                retain_graph=True
            )
            
            # Update model parameters using gradient descent
            with torch.no_grad():
                for param, grad in zip(adapted_model.parameters(), grads):
                    if grad is not None:
                        param.data = param.data - self.inner_lr * grad
        
        return adapted_model, inner_losses
    
    def outer_loop(self,
                   query_data: torch.Tensor,
                   query_labels: torch.Tensor,
                   adapted_model: nn.Module,
                   loss_fn: Callable) -> float:
        """
        Perform outer loop meta-update.
        
        Args:
            query_data: Query set data
            query_labels: Query set labels
            adapted_model: Model adapted on support set
            loss_fn: Loss function
            
        Returns:
            Query loss value
        """
        # Forward pass on query set with adapted model
        query_logits = adapted_model(query_data)
        query_loss = loss_fn(query_logits, query_labels)
        
        return query_loss.item()
    
    def meta_update(self,
                   tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                   loss_fn: Callable) -> Dict[str, float]:
        """
        Perform full MAML meta-update across multiple tasks.
        
        Args:
            tasks: List of (support_data, support_labels, query_data, query_labels) tuples
            loss_fn: Loss function
            
        Returns:
            Training metrics
        """
        self.meta_optimizer.zero_grad()
        
        total_query_loss = 0.0
        total_support_loss = 0.0
        
        for task_idx, (support_data, support_labels, query_data, query_labels) in enumerate(tasks):
            # Inner loop adaptation
            adapted_model, inner_losses = self.inner_loop(
                support_data, support_labels, loss_fn, create_graph=not self.first_order
            )
            
            # Outer loop evaluation
            query_logits = adapted_model(query_data)
            query_loss = loss_fn(query_logits, query_labels)
            
            # Accumulate gradients
            query_loss.backward()
            
            total_query_loss += query_loss.item()
            total_support_loss += inner_losses[-1] if inner_losses else 0.0
        
        # Meta-optimization step
        self.meta_optimizer.step()
        
        metrics = {
            'meta_loss': total_query_loss / len(tasks),
            'support_loss': total_support_loss / len(tasks),
            'num_tasks': len(tasks)
        }
        
        return metrics
    
    def adapt_to_task(self,
                     support_data: torch.Tensor,
                     support_labels: torch.Tensor,
                     loss_fn: Callable) -> nn.Module:
        """
        Adapt the meta-model to a specific task.
        
        Args:
            support_data: Task-specific support data
            support_labels: Task-specific support labels
            loss_fn: Loss function
            
        Returns:
            Task-adapted model
        """
        adapted_model, _ = self.inner_loop(
            support_data, support_labels, loss_fn, create_graph=False
        )
        return adapted_model


class MAMLTrainer:
    """
    High-level trainer for MAML in federated learning context.
    
    Manages the training process for personalized federated learning
    using MAML meta-learning approach.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 inner_lr: float = 0.001,
                 outer_lr: float = 0.005,
                 inner_steps: int = 1):
        """
        Initialize MAML trainer.
        
        Args:
            model: Base model
            device: Training device
            inner_lr: Inner loop learning rate
            outer_lr: Outer loop learning rate
            inner_steps: Inner loop steps
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.maml_optimizer = MAMLOptimizer(
            model, inner_lr, outer_lr, inner_steps
        )
        
        self.logger = logger
    
    def train_episode(self,
                     support_loaders: List[torch.utils.data.DataLoader],
                     query_loaders: List[torch.utils.data.DataLoader],
                     loss_fn: Callable) -> Dict[str, float]:
        """
        Train one meta-learning episode.
        
        Args:
            support_loaders: List of support set data loaders
            query_loaders: List of query set data loaders
            loss_fn: Loss function
            
        Returns:
            Episode metrics
        """
        self.model.train()
        
        # Collect tasks from data loaders
        tasks = []
        
        for support_loader, query_loader in zip(support_loaders, query_loaders):
            # Get one batch from each loader
            support_batch = next(iter(support_loader))
            query_batch = next(iter(query_loader))
            
            support_data, support_labels = support_batch
            query_data, query_labels = query_batch
            
            # Move to device
            support_data = support_data.to(self.device)
            support_labels = support_labels.to(self.device)
            query_data = query_data.to(self.device)
            query_labels = query_labels.to(self.device)
            
            tasks.append((support_data, support_labels, query_data, query_labels))
        
        # Perform meta-update
        metrics = self.maml_optimizer.meta_update(tasks, loss_fn)
        
        return metrics
    
    def evaluate_episode(self,
                        support_loaders: List[torch.utils.data.DataLoader],
                        query_loaders: List[torch.utils.data.DataLoader],
                        loss_fn: Callable) -> Dict[str, float]:
        """
        Evaluate meta-learning performance.
        
        Args:
            support_loaders: List of support set data loaders
            query_loaders: List of query set data loaders
            loss_fn: Loss function
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        total_query_loss = 0.0
        total_query_accuracy = 0.0
        num_tasks = 0
        
        with torch.no_grad():
            for support_loader, query_loader in zip(support_loaders, query_loaders):
                # Get batches
                support_batch = next(iter(support_loader))
                query_batch = next(iter(query_loader))
                
                support_data, support_labels = support_batch
                query_data, query_labels = query_batch
                
                # Move to device
                support_data = support_data.to(self.device)
                support_labels = support_labels.to(self.device)
                query_data = query_data.to(self.device)
                query_labels = query_labels.to(self.device)
                
                # Adapt to task
                adapted_model = self.maml_optimizer.adapt_to_task(
                    support_data, support_labels, loss_fn
                )
                
                # Evaluate on query set
                query_logits = adapted_model(query_data)
                query_loss = loss_fn(query_logits, query_labels)
                
                # Calculate accuracy
                _, predicted = query_logits.max(1)
                correct = predicted.eq(query_labels).sum().item()
                accuracy = correct / query_labels.size(0)
                
                total_query_loss += query_loss.item()
                total_query_accuracy += accuracy
                num_tasks += 1
        
        metrics = {
            'query_loss': total_query_loss / num_tasks,
            'query_accuracy': total_query_accuracy / num_tasks,
            'num_tasks': num_tasks
        }
        
        return metrics
    
    def adapt_and_predict(self,
                         support_data: torch.Tensor,
                         support_labels: torch.Tensor,
                         query_data: torch.Tensor,
                         loss_fn: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adapt to new data and make predictions.
        
        Args:
            support_data: Support set for adaptation
            support_labels: Support set labels
            query_data: Query data for prediction
            loss_fn: Loss function
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            support_data = support_data.to(self.device)
            support_labels = support_labels.to(self.device)
            query_data = query_data.to(self.device)
            
            # Adapt to task
            adapted_model = self.maml_optimizer.adapt_to_task(
                support_data, support_labels, loss_fn
            )
            
            # Make predictions
            logits = adapted_model(query_data)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
        
        return predictions, probabilities


def create_support_query_split(data: torch.Tensor,
                              labels: torch.Tensor,
                              support_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split data into support and query sets for meta-learning.
    
    Args:
        data: Input data
        labels: Labels
        support_ratio: Fraction of data for support set
        
    Returns:
        Tuple of (support_data, support_labels, query_data, query_labels)
    """
    n_samples = data.size(0)
    n_support = int(n_samples * support_ratio)
    
    # Random permutation
    indices = torch.randperm(n_samples)
    support_indices = indices[:n_support]
    query_indices = indices[n_support:]
    
    support_data = data[support_indices]
    support_labels = labels[support_indices]
    query_data = data[query_indices]
    query_labels = labels[query_indices]
    
    return support_data, support_labels, query_data, query_labels


def compute_maml_gradients(model: nn.Module,
                          support_data: torch.Tensor,
                          support_labels: torch.Tensor,
                          loss_fn: Callable,
                          inner_lr: float = 0.01,
                          create_graph: bool = True) -> List[torch.Tensor]:
    """
    Compute MAML gradients for one inner step.
    
    Args:
        model: Model to compute gradients for
        support_data: Support set data
        support_labels: Support set labels
        loss_fn: Loss function
        inner_lr: Inner learning rate
        create_graph: Whether to create computation graph
        
    Returns:
        List of gradients
    """
    # Forward pass
    logits = model(support_data)
    loss = loss_fn(logits, support_labels)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        loss,
        model.parameters(),
        create_graph=create_graph,
        retain_graph=True
    )
    
    return list(gradients)


# Example usage and testing
if __name__ == "__main__":
    from .cnn_1d import CNN1DClassifier
    
    # Create model
    model = CNN1DClassifier(input_dim=35, num_classes=5)
    device = torch.device('cpu')
    
    # Create MAML trainer
    trainer = MAMLTrainer(model, device, inner_lr=0.01, outer_lr=0.001)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test support/query split
    batch_size = 32
    input_dim = 35
    num_classes = 5
    
    data = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    support_data, support_labels, query_data, query_labels = create_support_query_split(
        data, labels, support_ratio=0.6
    )
    
    print(f"Support set: {support_data.shape}, Query set: {query_data.shape}")
    
    # Test adaptation
    loss_fn = nn.CrossEntropyLoss()
    predictions, probabilities = trainer.adapt_and_predict(
        support_data, support_labels, query_data, loss_fn
    )
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")

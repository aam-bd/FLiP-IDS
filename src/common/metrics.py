"""
Metrics calculation utilities for model evaluation.

Provides comprehensive evaluation metrics for both classification tasks:
- Phase 1: IoT vs Non-IoT and device type classification
- Phase 2: Intrusion detection classification
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from .logging import get_logger

logger = get_logger(__name__)


def calculate_metrics(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    y_prob: Optional[Union[np.ndarray, List]] = None,
    labels: Optional[List[str]] = None,
    average: str = 'macro',
    return_dict: bool = True
) -> Union[Dict[str, float], Tuple]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        labels: Class labels (optional)
        average: Averaging strategy for multi-class metrics
        return_dict: Whether to return results as dictionary
        
    Returns:
        Dictionary of metrics or tuple of individual metric values
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': len(y_true)
    }
    
    # Add AUC metrics if probabilities are provided
    if y_prob is not None:
        y_prob = np.array(y_prob)
        
        try:
            # For binary classification
            if len(np.unique(y_true)) == 2:
                if y_prob.ndim == 1:
                    auc_roc = roc_auc_score(y_true, y_prob)
                    auc_pr = average_precision_score(y_true, y_prob)
                else:
                    # Take probabilities for positive class
                    auc_roc = roc_auc_score(y_true, y_prob[:, 1])
                    auc_pr = average_precision_score(y_true, y_prob[:, 1])
            else:
                # For multi-class classification
                if y_prob.ndim == 2 and y_prob.shape[1] > 2:
                    auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
                    # Average precision for multi-class is more complex
                    auc_pr = None
                else:
                    auc_roc = None
                    auc_pr = None
            
            if auc_roc is not None:
                metrics['auc_roc'] = auc_roc
            if auc_pr is not None:
                metrics['auc_pr'] = auc_pr
                
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not calculate AUC metrics: {e}")
    
    # Per-class metrics if labels provided
    if labels is not None:
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, label in enumerate(labels):
            if i < len(per_class_precision):
                metrics[f'precision_{label}'] = per_class_precision[i]
                metrics[f'recall_{label}'] = per_class_recall[i]
                metrics[f'f1_{label}'] = per_class_f1[i]
    
    if return_dict:
        return metrics
    else:
        return accuracy, precision, recall, f1


def confusion_matrix_plot(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Create and optionally save a confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title += ' (Normalized)'
    else:
        fmt = 'd'
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    if labels is None:
        labels = [f'Class {i}' for i in range(cm.shape[0])]
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig


def roc_curve_plot(
    y_true: Union[np.ndarray, List],
    y_prob: Union[np.ndarray, List],
    labels: Optional[List[str]] = None,
    title: str = "ROC Curves",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Create ROC curve plot for binary or multi-class classification.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        labels: Class labels
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Binary classification
    if len(np.unique(y_true)) == 2:
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]  # Take positive class probabilities
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    
    # Multi-class classification
    else:
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        # Binarize labels
        unique_labels = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=unique_labels)
        
        if labels is None:
            labels = [f'Class {i}' for i in unique_labels]
        
        # Plot ROC curve for each class
        for i, label in enumerate(labels[:len(unique_labels)]):
            if i < y_prob.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    
    return fig


def precision_recall_plot(
    y_true: Union[np.ndarray, List],
    y_prob: Union[np.ndarray, List],
    labels: Optional[List[str]] = None,
    title: str = "Precision-Recall Curves",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Create precision-recall curve plot.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        labels: Class labels
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Binary classification
    if len(np.unique(y_true)) == 2:
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]  # Take positive class probabilities
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        ax.plot(recall, precision, linewidth=2, 
               label=f'PR Curve (AP = {avg_precision:.3f})')
    
    # Multi-class classification
    else:
        from sklearn.preprocessing import label_binarize
        
        # Binarize labels
        unique_labels = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=unique_labels)
        
        if labels is None:
            labels = [f'Class {i}' for i in unique_labels]
        
        # Plot PR curve for each class
        for i, label in enumerate(labels[:len(unique_labels)]):
            if i < y_prob.shape[1] and i < y_true_bin.shape[1]:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                avg_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                ax.plot(recall, precision, linewidth=2, 
                       label=f'{label} (AP = {avg_precision:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-recall curve saved to {save_path}")
    
    return fig


def classification_report_dict(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    labels: Optional[List[str]] = None,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate classification report as dictionary.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Labels to include in report
        target_names: Display names for labels
        
    Returns:
        Classification report as dictionary
    """
    return classification_report(
        y_true, y_pred, 
        labels=labels, 
        target_names=target_names, 
        output_dict=True,
        zero_division=0
    )


def federated_metrics_summary(
    client_metrics: Dict[str, Dict[str, float]],
    global_metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Summarize metrics across federated learning clients.
    
    Args:
        client_metrics: Dictionary mapping client_id to metrics dict
        global_metrics: Optional global model metrics
        save_path: Path to save summary
        
    Returns:
        Summary statistics dictionary
    """
    if not client_metrics:
        return {}
    
    # Extract metric names
    metric_names = set()
    for metrics in client_metrics.values():
        metric_names.update(metrics.keys())
    
    summary = {
        'num_clients': len(client_metrics),
        'metric_names': list(metric_names),
        'per_metric_stats': {},
        'client_rankings': {}
    }
    
    # Calculate statistics for each metric
    for metric_name in metric_names:
        values = []
        for client_id, metrics in client_metrics.items():
            if metric_name in metrics:
                values.append(metrics[metric_name])
        
        if values:
            summary['per_metric_stats'][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
            
            # Rank clients by this metric
            client_scores = [(client_id, metrics.get(metric_name, 0)) 
                           for client_id, metrics in client_metrics.items()]
            client_scores.sort(key=lambda x: x[1], reverse=True)
            summary['client_rankings'][metric_name] = client_scores
    
    # Add global metrics if provided
    if global_metrics:
        summary['global_metrics'] = global_metrics
    
    # Save summary if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame for better formatting
        df_summary = pd.DataFrame(summary['per_metric_stats']).T
        df_summary.to_csv(save_path.with_suffix('.csv'))
        
        # Save full summary as JSON
        import json
        with open(save_path.with_suffix('.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Federated metrics summary saved to {save_path}")
    
    return summary


def plot_federated_metrics(
    client_metrics: Dict[str, Dict[str, float]],
    metric_name: str = 'f1_score',
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot federated learning metrics across clients.
    
    Args:
        client_metrics: Dictionary mapping client_id to metrics dict
        metric_name: Metric to plot
        title: Plot title
        save_path: Path to save plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Extract metric values
    client_ids = []
    metric_values = []
    
    for client_id, metrics in client_metrics.items():
        if metric_name in metrics:
            client_ids.append(client_id)
            metric_values.append(metrics[metric_name])
    
    if not metric_values:
        logger.warning(f"No values found for metric: {metric_name}")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    bars = ax1.bar(range(len(client_ids)), metric_values, alpha=0.7)
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel(metric_name.replace('_', ' ').title())
    ax1.set_title(f'{metric_name.replace("_", " ").title()} by Client')
    ax1.set_xticks(range(len(client_ids)))
    ax1.set_xticklabels(client_ids, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Box plot
    ax2.boxplot(metric_values, labels=[metric_name.replace('_', ' ').title()])
    ax2.set_ylabel(metric_name.replace('_', ' ').title())
    ax2.set_title(f'{metric_name.replace("_", " ").title()} Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(metric_values):.3f}\n'
    stats_text += f'Std: {np.std(metric_values):.3f}\n'
    stats_text += f'Min: {np.min(metric_values):.3f}\n'
    stats_text += f'Max: {np.max(metric_values):.3f}'
    
    ax2.text(0.7, 0.95, stats_text, transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Federated metrics plot saved to {save_path}")
    
    return fig

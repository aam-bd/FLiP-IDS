"""
I/O utilities for saving and loading models, data, and checkpoints.

Provides consistent interfaces for persisting various data types used
throughout the IoT Security Framework.
"""

import os
import pickle
import joblib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import torch
import yaml
from datetime import datetime

from .logging import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model: Any, filepath: Union[str, Path], metadata: Optional[Dict] = None) -> None:
    """
    Save a model to disk with optional metadata.
    
    Supports scikit-learn models (joblib), PyTorch models, and generic Python objects.
    
    Args:
        model: Model object to save
        filepath: Path to save the model
        metadata: Optional metadata dictionary to save alongside model
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    try:
        # Determine save method based on model type
        if hasattr(model, 'state_dict'):  # PyTorch model
            if metadata:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'metadata': metadata,
                    'timestamp': datetime.now().isoformat()
                }, filepath)
            else:
                torch.save(model.state_dict(), filepath)
            logger.info(f"PyTorch model saved to {filepath}")
            
        elif hasattr(model, 'fit') and hasattr(model, 'predict'):  # Scikit-learn model
            if metadata:
                # Save model and metadata separately for sklearn
                joblib.dump(model, filepath)
                metadata_path = filepath.with_suffix('.metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump({**metadata, 'timestamp': datetime.now().isoformat()}, f, indent=2)
                logger.info(f"Scikit-learn model saved to {filepath} with metadata")
            else:
                joblib.dump(model, filepath)
                logger.info(f"Scikit-learn model saved to {filepath}")
                
        else:  # Generic Python object
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'metadata': metadata or {},
                    'timestamp': datetime.now().isoformat()
                }, f)
            logger.info(f"Generic model saved to {filepath}")
            
    except Exception as e:
        logger.error(f"Failed to save model to {filepath}: {e}")
        raise


def load_model(filepath: Union[str, Path], model_class: Optional[Any] = None) -> Union[Any, tuple]:
    """
    Load a model from disk.
    
    Args:
        filepath: Path to the saved model
        model_class: Optional model class for PyTorch models
        
    Returns:
        Model object, or tuple of (model, metadata) if metadata was saved
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    try:
        # Try to load as PyTorch model first
        try:
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                if model_class is None:
                    logger.warning("PyTorch model loaded but no model_class provided. Returning state_dict only.")
                    return checkpoint['model_state_dict'], checkpoint.get('metadata', {})
                else:
                    model = model_class()
                    model.load_state_dict(checkpoint['model_state_dict'])
                    return model, checkpoint.get('metadata', {})
            else:
                # Direct state dict
                if model_class is None:
                    return checkpoint
                else:
                    model = model_class()
                    model.load_state_dict(checkpoint)
                    return model
                    
        except (RuntimeError, KeyError):
            pass  # Not a PyTorch model, try other methods
            
        # Try to load as scikit-learn model
        try:
            model = joblib.load(filepath)
            metadata_path = filepath.with_suffix('.metadata.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return model, metadata
            else:
                return model
                
        except (joblib.externals.loky.process_executor.TerminatedWorkerError, ValueError):
            pass  # Not a joblib model
            
        # Try to load as pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict) and 'model' in data:
                return data['model'], data.get('metadata', {})
            else:
                return data
                
    except Exception as e:
        logger.error(f"Failed to load model from {filepath}: {e}")
        raise


def save_data(data: Union[pd.DataFrame, np.ndarray, Dict, List], 
              filepath: Union[str, Path],
              format: Optional[str] = None) -> None:
    """
    Save data in various formats.
    
    Args:
        data: Data to save (DataFrame, array, dict, list)
        filepath: Path to save the data
        format: Optional format specification ('csv', 'parquet', 'json', 'npy', 'npz')
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    # Auto-detect format from extension if not specified
    if format is None:
        format = filepath.suffix.lower().lstrip('.')
    
    try:
        if isinstance(data, pd.DataFrame):
            if format in ['csv']:
                data.to_csv(filepath, index=False)
            elif format in ['parquet']:
                data.to_parquet(filepath, index=False)
            elif format in ['json']:
                data.to_json(filepath, orient='records', indent=2)
            else:
                # Default to parquet for DataFrames
                data.to_parquet(filepath.with_suffix('.parquet'), index=False)
                
        elif isinstance(data, np.ndarray):
            if format in ['npy']:
                np.save(filepath, data)
            elif format in ['npz']:
                np.savez_compressed(filepath, data=data)
            elif format in ['csv']:
                np.savetxt(filepath, data, delimiter=',')
            else:
                np.save(filepath.with_suffix('.npy'), data)
                
        elif isinstance(data, (dict, list)):
            if format in ['json']:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif format in ['yaml', 'yml']:
                with open(filepath, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
            else:
                # Default to JSON for structured data
                with open(filepath.with_suffix('.json'), 'w') as f:
                    json.dump(data, f, indent=2, default=str)
        else:
            # Generic pickle fallback
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
                
        logger.info(f"Data saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save data to {filepath}: {e}")
        raise


def load_data(filepath: Union[str, Path], 
              format: Optional[str] = None) -> Union[pd.DataFrame, np.ndarray, Dict, List]:
    """
    Load data from various formats.
    
    Args:
        filepath: Path to the data file
        format: Optional format specification
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Auto-detect format from extension if not specified
    if format is None:
        format = filepath.suffix.lower().lstrip('.')
    
    try:
        if format in ['csv']:
            return pd.read_csv(filepath)
        elif format in ['parquet']:
            return pd.read_parquet(filepath)
        elif format in ['json']:
            with open(filepath, 'r') as f:
                return json.load(f)
        elif format in ['yaml', 'yml']:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        elif format in ['npy']:
            return np.load(filepath)
        elif format in ['npz']:
            return np.load(filepath)['data']
        else:
            # Try common formats in order
            try:
                return pd.read_parquet(filepath)
            except:
                try:
                    return pd.read_csv(filepath)
                except:
                    try:
                        with open(filepath, 'r') as f:
                            return json.load(f)
                    except:
                        with open(filepath, 'rb') as f:
                            return pickle.load(f)
                            
    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {e}")
        raise


def save_config(config: Dict, filepath: Union[str, Path]) -> None:
    """Save configuration dictionary as YAML."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration saved to {filepath}")


def load_config(filepath: Union[str, Path]) -> Dict:
    """Load configuration from YAML file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {filepath}")
    return config


def create_checkpoint(model: Any, optimizer: Optional[Any] = None, 
                     epoch: int = 0, loss: float = 0.0, 
                     metrics: Optional[Dict] = None,
                     filepath: Union[str, Path] = None) -> None:
    """
    Create a training checkpoint.
    
    Args:
        model: Model to checkpoint
        optimizer: Optional optimizer state
        epoch: Current epoch
        loss: Current loss
        metrics: Optional metrics dictionary
        filepath: Path to save checkpoint
    """
    if filepath is None:
        filepath = f"checkpoint_epoch_{epoch}.pt"
    
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model,
        'loss': loss,
        'metrics': metrics or {},
        'timestamp': datetime.now().isoformat()
    }
    
    if optimizer is not None and hasattr(optimizer, 'state_dict'):
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: Union[str, Path]) -> Dict:
    """Load training checkpoint."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    logger.info(f"Checkpoint loaded from {filepath}")
    return checkpoint

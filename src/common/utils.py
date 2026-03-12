"""
General utilities for the IoT Security Framework.

Provides common helper functions used across multiple modules including
random seed management, device detection, data chunking, and time utilities.
"""

import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import torch

from .logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_device(use_cuda: bool = True, device_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate PyTorch device (CPU/CUDA).
    
    Args:
        use_cuda: Whether to use CUDA if available
        device_id: Specific CUDA device ID (optional)
        
    Returns:
        PyTorch device object
    """
    if use_cuda and torch.cuda.is_available():
        if device_id is not None:
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cuda")
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name()})")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
    
    return device


def create_chunks(data: List[Any], chunk_size: int) -> Generator[List[Any], None, None]:
    """
    Split data into chunks of specified size.
    
    Args:
        data: List of data items to chunk
        chunk_size: Size of each chunk
        
    Yields:
        Chunks of data
    """
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_bytes(bytes_value: int) -> str:
    """
    Format byte count to human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted bytes string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f}PB"


def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format_str: Timestamp format string
        
    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime(format_str)


def parse_timestamp(timestamp_str: str, format_str: str = "%Y%m%d_%H%M%S") -> datetime:
    """
    Parse timestamp string to datetime object.
    
    Args:
        timestamp_str: Timestamp string
        format_str: Expected format
        
    Returns:
        Parsed datetime object
    """
    return datetime.strptime(timestamp_str, format_str)


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, description: str = "Operation", logger=None):
        self.description = description
        self.logger = logger or get_logger(__name__)
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.description}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        if exc_type is None:
            self.logger.info(f"{self.description} completed in {format_duration(duration)}")
        else:
            self.logger.error(f"{self.description} failed after {format_duration(duration)}")
    
    @property
    def elapsed(self) -> Optional[float]:
        """Get elapsed time if timer has been used."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing", 
                 update_interval: int = 10, logger=None):
        self.total = total
        self.description = description
        self.update_interval = update_interval
        self.logger = logger or get_logger(__name__)
        
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, increment: int = 1) -> None:
        """Update progress counter."""
        self.current += increment
        
        # Update log if interval reached or completed
        if (self.current - self.last_update >= self.update_interval or 
            self.current >= self.total):
            
            elapsed = time.time() - self.start_time
            progress_pct = (self.current / self.total) * 100
            
            if self.current < self.total and elapsed > 0:
                rate = self.current / elapsed
                eta = (self.total - self.current) / rate
                self.logger.info(
                    f"{self.description}: {self.current}/{self.total} "
                    f"({progress_pct:.1f}%) - ETA: {format_duration(eta)}"
                )
            else:
                self.logger.info(
                    f"{self.description}: {self.current}/{self.total} "
                    f"({progress_pct:.1f}%) - Total time: {format_duration(elapsed)}"
                )
            
            self.last_update = self.current
    
    def complete(self) -> None:
        """Mark as complete."""
        self.current = self.total
        elapsed = time.time() - self.start_time
        self.logger.info(f"{self.description} completed in {format_duration(elapsed)}")


def ensure_list(value: Union[Any, List[Any]]) -> List[Any]:
    """Ensure value is a list."""
    if isinstance(value, list):
        return value
    elif isinstance(value, (tuple, set)):
        return list(value)
    else:
        return [value]


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict, sep: str = '_') -> Dict:
    """
    Unflatten dictionary with separated keys.
    
    Args:
        d: Flattened dictionary
        sep: Key separator
        
    Returns:
        Nested dictionary
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value for zero denominator.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero
        
    Returns:
        Division result or default value
    """
    return numerator / denominator if denominator != 0 else default


def moving_average(data: List[float], window_size: int) -> List[float]:
    """
    Calculate moving average of data.
    
    Args:
        data: List of numeric values
        window_size: Size of moving window
        
    Returns:
        List of moving averages
    """
    if len(data) < window_size:
        return data
    
    result = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        result.append(sum(window) / window_size)
    
    return result


def exponential_moving_average(data: List[float], alpha: float = 0.1) -> List[float]:
    """
    Calculate exponential moving average.
    
    Args:
        data: List of numeric values
        alpha: Smoothing factor (0 < alpha <= 1)
        
    Returns:
        List of exponential moving averages
    """
    if not data:
        return []
    
    result = [data[0]]
    for value in data[1:]:
        ema = alpha * value + (1 - alpha) * result[-1]
        result.append(ema)
    
    return result


def normalize_features(data: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Dict]:
    """
    Normalize feature data.
    
    Args:
        data: Feature array (samples x features)
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    if method == 'minmax':
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        normalized = (data - min_vals) / range_vals
        params = {'method': 'minmax', 'min': min_vals, 'max': max_vals}
        
    elif method == 'zscore':
        mean_vals = np.mean(data, axis=0)
        std_vals = np.std(data, axis=0)
        
        # Avoid division by zero
        std_vals[std_vals == 0] = 1
        
        normalized = (data - mean_vals) / std_vals
        params = {'method': 'zscore', 'mean': mean_vals, 'std': std_vals}
        
    elif method == 'robust':
        median_vals = np.median(data, axis=0)
        mad_vals = np.median(np.abs(data - median_vals), axis=0)
        
        # Avoid division by zero
        mad_vals[mad_vals == 0] = 1
        
        normalized = (data - median_vals) / mad_vals
        params = {'method': 'robust', 'median': median_vals, 'mad': mad_vals}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def apply_normalization(data: np.ndarray, params: Dict) -> np.ndarray:
    """
    Apply normalization using saved parameters.
    
    Args:
        data: Data to normalize
        params: Normalization parameters from normalize_features
        
    Returns:
        Normalized data
    """
    method = params['method']
    
    if method == 'minmax':
        min_vals = params['min']
        max_vals = params['max']
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        return (data - min_vals) / range_vals
        
    elif method == 'zscore':
        mean_vals = params['mean']
        std_vals = params['std']
        std_vals[std_vals == 0] = 1
        return (data - mean_vals) / std_vals
        
    elif method == 'robust':
        median_vals = params['median']
        mad_vals = params['mad']
        mad_vals[mad_vals == 0] = 1
        return (data - median_vals) / mad_vals
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_time_windows(timestamps: List[datetime], 
                       window_size: timedelta,
                       overlap: timedelta = None) -> List[Tuple[datetime, datetime]]:
    """
    Create time windows from timestamps.
    
    Args:
        timestamps: List of timestamps
        window_size: Size of each window
        overlap: Overlap between windows (default: no overlap)
        
    Returns:
        List of (start_time, end_time) tuples
    """
    if not timestamps:
        return []
    
    timestamps = sorted(timestamps)
    start_time = timestamps[0]
    end_time = timestamps[-1]
    
    if overlap is None:
        overlap = timedelta(0)
    
    step_size = window_size - overlap
    windows = []
    
    current_start = start_time
    while current_start < end_time:
        current_end = min(current_start + window_size, end_time)
        windows.append((current_start, current_end))
        current_start += step_size
    
    return windows


def validate_config(config: Dict, required_keys: List[str], 
                   optional_keys: List[str] = None) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        optional_keys: List of optional keys
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    if optional_keys:
        unknown_keys = set(config.keys()) - set(required_keys) - set(optional_keys)
        if unknown_keys:
            logger.warning(f"Unknown configuration keys: {unknown_keys}")
    
    return True


def memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage info
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident set size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual memory size
            'percent': process.memory_percent()
        }
    except ImportError:
        logger.warning("psutil not available, cannot get memory usage")
        return {}


def disk_usage(path: Union[str, Path]) -> Dict[str, float]:
    """
    Get disk usage statistics for a path.
    
    Args:
        path: Path to check
        
    Returns:
        Dictionary with disk usage info
    """
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        
        return {
            'total_gb': total / (1024**3),
            'used_gb': used / (1024**3),
            'free_gb': free / (1024**3),
            'used_percent': (used / total) * 100
        }
    except Exception as e:
        logger.warning(f"Cannot get disk usage for {path}: {e}")
        return {}

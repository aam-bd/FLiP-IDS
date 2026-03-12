"""
Logging utilities for the IoT Security Framework.

Provides centralized logging configuration and utilities for consistent
logging across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in console output
        
    Returns:
        Configured logger instance
    """
    # Default format string
    if format_string is None:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ProgressLogger:
    """Logger wrapper for tracking progress of long-running operations."""
    
    def __init__(self, logger: logging.Logger, operation: str, total: Optional[int] = None):
        self.logger = logger
        self.operation = operation
        self.total = total
        self.current = 0
        self.start_time = datetime.now()
        
        self.logger.info(f"Starting {operation}...")
    
    def update(self, increment: int = 1, message: Optional[str] = None) -> None:
        """Update progress counter and optionally log a message."""
        self.current += increment
        
        if message:
            if self.total:
                progress = (self.current / self.total) * 100
                self.logger.info(f"{self.operation} - {message} ({self.current}/{self.total}, {progress:.1f}%)")
            else:
                self.logger.info(f"{self.operation} - {message} (step {self.current})")
    
    def complete(self, message: Optional[str] = None) -> None:
        """Mark operation as complete and log duration."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        if message:
            self.logger.info(f"{self.operation} completed - {message} (took {duration})")
        else:
            self.logger.info(f"{self.operation} completed in {duration}")


class MetricsLogger:
    """Logger for tracking and reporting metrics."""
    
    def __init__(self, logger: logging.Logger, experiment_name: str):
        self.logger = logger
        self.experiment_name = experiment_name
        self.metrics = {}
        self.start_time = datetime.now()
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric value."""
        if step is not None:
            self.logger.info(f"{self.experiment_name} - Step {step}: {name} = {value:.4f}")
        else:
            self.logger.info(f"{self.experiment_name} - {name} = {value:.4f}")
        
        # Store for later reporting
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((step, value, datetime.now()))
    
    def log_metrics(self, metrics_dict: dict, step: Optional[int] = None) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)
    
    def log_model_summary(self, model_name: str, num_parameters: int, 
                         architecture: Optional[str] = None) -> None:
        """Log model architecture summary."""
        self.logger.info(f"{self.experiment_name} - Model: {model_name}")
        self.logger.info(f"{self.experiment_name} - Parameters: {num_parameters:,}")
        if architecture:
            self.logger.info(f"{self.experiment_name} - Architecture: {architecture}")
    
    def log_training_summary(self, epoch: int, train_loss: float, 
                           val_loss: Optional[float] = None,
                           val_metrics: Optional[dict] = None) -> None:
        """Log training epoch summary."""
        summary = f"{self.experiment_name} - Epoch {epoch}: train_loss={train_loss:.4f}"
        
        if val_loss is not None:
            summary += f", val_loss={val_loss:.4f}"
        
        if val_metrics:
            for name, value in val_metrics.items():
                summary += f", {name}={value:.4f}"
        
        self.logger.info(summary)
    
    def get_best_metric(self, metric_name: str, maximize: bool = True) -> tuple:
        """Get the best value for a specific metric."""
        if metric_name not in self.metrics:
            return None, None
        
        values = self.metrics[metric_name]
        if maximize:
            best_idx = max(range(len(values)), key=lambda i: values[i][1])
        else:
            best_idx = min(range(len(values)), key=lambda i: values[i][1])
        
        step, value, timestamp = values[best_idx]
        return step, value
    
    def summary_report(self) -> str:
        """Generate a summary report of all logged metrics."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        report = [
            f"\n{'='*60}",
            f"EXPERIMENT SUMMARY: {self.experiment_name}",
            f"{'='*60}",
            f"Duration: {duration}",
            f"Start Time: {self.start_time}",
            f"End Time: {end_time}",
            f"{'='*60}"
        ]
        
        for metric_name, values in self.metrics.items():
            if values:
                final_value = values[-1][1]
                best_step, best_value = self.get_best_metric(metric_name, maximize=True)
                worst_step, worst_value = self.get_best_metric(metric_name, maximize=False)
                
                report.extend([
                    f"{metric_name}:",
                    f"  Final: {final_value:.4f}",
                    f"  Best:  {best_value:.4f} (step {best_step})",
                    f"  Worst: {worst_value:.4f} (step {worst_step})",
                    ""
                ])
        
        report.append(f"{'='*60}")
        return "\n".join(report)


def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """Decorator to log function calls with arguments."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            args_str = ", ".join([str(arg) for arg in args])
            kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            all_args = ", ".join(filter(None, [args_str, kwargs_str]))
            
            logger.log(level, f"Calling {func_name}({all_args})")
            
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func_name} failed with error: {e}")
                raise
        
        return wrapper
    return decorator


def log_execution_time(logger: logging.Logger, level: int = logging.INFO):
    """Decorator to log function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = end_time - start_time
                logger.log(level, f"{func_name} executed in {duration}")
                return result
            except Exception as e:
                end_time = datetime.now()
                duration = end_time - start_time
                logger.error(f"{func_name} failed after {duration}: {e}")
                raise
        
        return wrapper
    return decorator

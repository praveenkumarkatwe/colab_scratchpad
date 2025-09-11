"""Utility functions for the EHI package."""

import logging
import os
import random
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    structured: bool = True,
) -> logging.Logger:
    """
    Set up structured logging with rich formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs
        structured: Whether to use structured (JSON-like) logging
    
    Returns:
        Configured logger instance
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up rich console handler
    console = Console()
    
    handlers = [
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            show_time=True,
        )
    ]
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        if structured:
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers,
        force=True,
    )
    
    return logging.getLogger("ehi")


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior on CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """
    Normalize Unicode text using the specified normalization form.
    
    Args:
        text: Input text to normalize
        form: Unicode normalization form (NFC, NFD, NFKC, NFKD)
    
    Returns:
        Normalized text
    """
    if not text:
        return text
    
    return unicodedata.normalize(form, text)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config or {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse config file {config_path}: {e}") from e


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration dictionary to YAML file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the config file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device() -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.
    
    Args:
        seconds: Time duration in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


class MetricsTracker:
    """Simple metrics tracking utility for training loops."""
    
    def __init__(self) -> None:
        self.metrics: Dict[str, list] = {}
        self.step = 0
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update metrics for current step."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        self.step += 1
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get the latest value for a metric."""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return None
    
    def get_average(self, key: str, window: Optional[int] = None) -> Optional[float]:
        """Get average value for a metric over a window."""
        if key not in self.metrics or not self.metrics[key]:
            return None
        
        values = self.metrics[key]
        if window is not None:
            values = values[-window:]
        
        return sum(values) / len(values)
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.step = 0
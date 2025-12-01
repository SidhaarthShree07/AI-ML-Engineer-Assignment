"""
Simple Dataset Utilities for HybridAutoMLE

This module provides basic dataset loading utilities.
The main dataset handling is done by dataset_handler.py.
This file exists for backwards compatibility only.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class CompetitionInfo:
    """Basic competition information."""
    id: str
    name: str
    metric: str = "accuracy"


# Known competition metrics (for reference only)
COMPETITION_METRICS = {
    "siim-isic-melanoma-classification": "auc",
    "spooky-author-identification": "log_loss",
    "tabular-playground-series-may-2022": "auc",
    "text-normalization-challenge-english-language": "accuracy",
    "the-icml-2013-whale-challenge-right-whale-redux": "auc",
}


def get_competition_metric(competition_id: str) -> str:
    """
    Get the evaluation metric for a known competition.
    
    Args:
        competition_id: Competition identifier
        
    Returns:
        Metric name (defaults to 'accuracy' if unknown)
    """
    return COMPETITION_METRICS.get(competition_id, "accuracy")


def load_csv_safe(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Safely load a CSV file with error handling.
    
    Args:
        path: Path to CSV file
        nrows: Optional number of rows to load
        
    Returns:
        DataFrame or empty DataFrame if loading fails
    """
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return pd.DataFrame()


def find_csv_in_dir(directory: str, name_pattern: str = "*.csv") -> Optional[str]:
    """
    Find a CSV file in a directory.
    
    Args:
        directory: Directory to search
        name_pattern: Glob pattern for filename
        
    Returns:
        Path to first matching CSV or None
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return None
    
    csv_files = list(dir_path.glob(name_pattern))
    return str(csv_files[0]) if csv_files else None


# Version info
__version__ = "1.0.0"

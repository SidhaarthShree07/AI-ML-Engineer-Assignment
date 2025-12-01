"""Performance monitoring for regression detection"""

import logging
from typing import List, Optional
from src.models.data_models import TrainingMetrics

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitor training performance and detect issues.
    
    Implements Tier 1 of self-improvement: Performance Monitoring
    Detects:
    - Significant performance regression (>5% drop from best)
    - Minor performance regression (>2% drop from best)
    - Overfitting (train-val gap > 0.15)
    - Underfitting (train score < 0.7)
    - Resource inefficiency (GPU utilization < 60%)
    """
    
    def __init__(self):
        """Initialize performance monitor with empty history"""
        self.history: List[TrainingMetrics] = []
        logger.info("PerformanceMonitor initialized")
    
    def add_metrics(self, metrics: TrainingMetrics):
        """
        Add metrics to history.
        
        Args:
            metrics: Training metrics to add
        """
        self.history.append(metrics)
        logger.debug(f"Added metrics: train={metrics.train_score:.4f}, "
                    f"val={metrics.val_score:.4f}, epoch={metrics.epoch}")
    
    def detect_issues(self, current_metrics: TrainingMetrics) -> str:
        """
        Detect performance issues from current metrics.
        
        Implements threshold-based detection for:
        - Performance regression (validation score drops)
        - Overfitting (large train-val gap)
        - Underfitting (low training score)
        - Resource inefficiency (low GPU utilization)
        
        Args:
            current_metrics: Current training metrics
            
        Returns:
            Issue type: "significant_regression", "minor_regression", 
                       "overfitting", "underfitting", "resource_inefficiency",
                       "insufficient_data", or "healthy"
        """
        # Need at least 2 data points for regression detection
        if len(self.history) < 2:
            logger.debug("Insufficient data for issue detection")
            return "insufficient_data"
        
        current_score = current_metrics.val_score
        best_score = max([h.val_score for h in self.history])
        
        # Check for significant regression (>5% drop)
        if current_score < best_score - 0.05:
            logger.warning(f"Significant regression detected: {current_score:.4f} < {best_score:.4f} - 0.05")
            return "significant_regression"
        
        # Check for minor regression (>2% drop)
        if current_score < best_score - 0.02:
            logger.info(f"Minor regression detected: {current_score:.4f} < {best_score:.4f} - 0.02")
            return "minor_regression"
        
        # Check for overfitting (train-val gap > 0.15)
        train_val_gap = current_metrics.train_score - current_metrics.val_score
        if train_val_gap > 0.15:
            logger.warning(f"Overfitting detected: train-val gap = {train_val_gap:.4f}")
            return "overfitting"
        
        # Check for underfitting (train score < 0.7)
        if current_metrics.train_score < 0.7:
            logger.warning(f"Underfitting detected: train score = {current_metrics.train_score:.4f}")
            return "underfitting"
        
        # Check for resource inefficiency (GPU utilization < 60%)
        if current_metrics.gpu_utilization < 60.0:
            logger.info(f"Resource inefficiency detected: GPU utilization = {current_metrics.gpu_utilization:.1f}%")
            return "resource_inefficiency"
        
        logger.debug("No issues detected - healthy performance")
        return "healthy"
    
    def get_best_score(self) -> Optional[float]:
        """
        Get best validation score from history.
        
        Returns:
            Best validation score, or None if no history
        """
        if not self.history:
            return None
        return max([h.val_score for h in self.history])
    
    def get_history(self) -> List[TrainingMetrics]:
        """
        Get complete metrics history.
        
        Returns:
            List of all training metrics
        """
        return self.history.copy()
    
    def clear_history(self):
        """Clear metrics history"""
        self.history.clear()
        logger.info("Performance history cleared")

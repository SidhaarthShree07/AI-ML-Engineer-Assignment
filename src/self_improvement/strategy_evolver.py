"""Strategy evolution for autonomous improvement"""

import logging
from typing import Dict, Any
from src.models.data_models import Strategy, PerformanceData, ResourceConstraints

logger = logging.getLogger(__name__)


class StrategyEvolver:
    """
    Autonomous strategy evolution based on performance feedback.
    
    Implements Tier 3 of self-improvement: Strategy Evolution
    Evolves strategies to address:
    - Overfitting (train-val gap > 0.15)
    - Underfitting (train score < 0.7)
    - Resource inefficiency (GPU utilization < 60%)
    """
    
    def __init__(self):
        """Initialize strategy evolver"""
        logger.info("StrategyEvolver initialized")
    
    def evolve(
        self,
        current: Strategy,
        performance: PerformanceData
    ) -> Strategy:
        """
        Evolve strategy based on performance data.
        
        Applies evolution rules to address detected issues:
        - Overfitting: Increase regularization (dropout, weight decay, augmentation)
        - Underfitting: Increase model complexity and training duration
        - Resource inefficiency: Optimize batch size for better GPU utilization
        
        Args:
            current: Current strategy
            performance: Performance data with issue type
            
        Returns:
            Evolved strategy with improvements
        """
        logger.info(f"Evolving strategy to address: {performance.issue_type}")
        
        # Create a copy to evolve
        evolved = self._copy_strategy(current)
        
        # Apply evolution rules based on issue type
        if performance.issue_type == "overfitting":
            evolved = self._handle_overfitting(evolved, performance)
            
        elif performance.issue_type == "underfitting":
            evolved = self._handle_underfitting(evolved, performance)
            
        elif performance.issue_type == "resource_inefficiency":
            evolved = self._handle_resource_inefficiency(evolved, performance)
            
        elif performance.issue_type in ["significant_regression", "minor_regression"]:
            # For regression, apply conservative improvements
            evolved = self._handle_regression(evolved, performance)
        
        logger.info("Strategy evolution complete")
        self._log_changes(current, evolved)
        
        return evolved
    
    def _copy_strategy(self, strategy: Strategy) -> Strategy:
        """
        Create a deep copy of strategy.
        
        Args:
            strategy: Strategy to copy
            
        Returns:
            Copy of strategy
        """
        return Strategy(
            modality=strategy.modality,
            primary_model=strategy.primary_model,
            fallback_model=strategy.fallback_model,
            preprocessing=strategy.preprocessing.copy(),
            augmentation=strategy.augmentation.copy() if strategy.augmentation else None,
            loss_function=strategy.loss_function,
            optimizer=strategy.optimizer,
            batch_size=strategy.batch_size,
            max_epochs=strategy.max_epochs,
            early_stopping_patience=strategy.early_stopping_patience,
            hyperparameters=strategy.hyperparameters.copy(),
            resource_constraints=strategy.resource_constraints,
            learning_rate=strategy.learning_rate,
            weight_decay=strategy.weight_decay,
            dropout=strategy.dropout,
            mixed_precision=strategy.mixed_precision,
            gradient_accumulation_steps=strategy.gradient_accumulation_steps,
            gradient_clip_norm=strategy.gradient_clip_norm,
            model_size=strategy.model_size,
            augmentation_strength=strategy.augmentation_strength
        )
    
    def _handle_overfitting(
        self,
        strategy: Strategy,
        performance: PerformanceData
    ) -> Strategy:
        """
        Handle overfitting by increasing regularization.
        
        Requirement 7.3: When train-validation gap exceeds 0.15,
        the evolved strategy must include increased regularization.
        
        Args:
            strategy: Current strategy
            performance: Performance data
            
        Returns:
            Strategy with increased regularization
        """
        logger.info(f"Handling overfitting (train-val gap: {performance.train_val_gap:.4f})")
        
        # Increase dropout (cap at 0.5)
        old_dropout = strategy.dropout
        strategy.dropout = min(strategy.dropout + 0.1, 0.5)
        logger.info(f"Increased dropout: {old_dropout:.2f} -> {strategy.dropout:.2f}")
        
        # Increase weight decay (ensure minimum increase even if starting from 0)
        old_weight_decay = strategy.weight_decay
        if strategy.weight_decay == 0.0:
            strategy.weight_decay = 0.01  # Set to minimum non-zero value
        else:
            strategy.weight_decay = strategy.weight_decay * 2.0
        logger.info(f"Increased weight decay: {old_weight_decay:.4f} -> {strategy.weight_decay:.4f}")
        
        # Increase augmentation strength if applicable
        if strategy.augmentation_strength is not None:
            old_aug = strategy.augmentation_strength
            strategy.augmentation_strength = min(strategy.augmentation_strength * 1.5, 2.0)
            logger.info(f"Increased augmentation: {old_aug:.2f} -> {strategy.augmentation_strength:.2f}")
        
        # Reduce learning rate slightly to stabilize
        old_lr = strategy.learning_rate
        strategy.learning_rate = strategy.learning_rate * 0.8
        logger.info(f"Reduced learning rate: {old_lr:.6f} -> {strategy.learning_rate:.6f}")
        
        return strategy
    
    def _handle_underfitting(
        self,
        strategy: Strategy,
        performance: PerformanceData
    ) -> Strategy:
        """
        Handle underfitting by increasing model complexity.
        
        Requirement 7.4: When training score remains below 0.7,
        the evolved strategy must include increased model complexity.
        
        Args:
            strategy: Current strategy
            performance: Performance data
            
        Returns:
            Strategy with increased complexity
        """
        logger.info(f"Handling underfitting (train score: {performance.train_score:.4f})")
        
        # Increase model size
        old_size = strategy.model_size
        if strategy.model_size == "small":
            strategy.model_size = "medium"
        elif strategy.model_size == "medium":
            strategy.model_size = "large"
        # If already large, keep it
        
        if old_size != strategy.model_size:
            logger.info(f"Increased model size: {old_size} -> {strategy.model_size}")
        
        # Increase max epochs
        old_epochs = strategy.max_epochs
        strategy.max_epochs = int(strategy.max_epochs * 1.5)
        logger.info(f"Increased max epochs: {old_epochs} -> {strategy.max_epochs}")
        
        # Increase learning rate
        old_lr = strategy.learning_rate
        strategy.learning_rate = strategy.learning_rate * 1.2
        logger.info(f"Increased learning rate: {old_lr:.6f} -> {strategy.learning_rate:.6f}")
        
        # Reduce dropout to allow more learning
        if strategy.dropout > 0.1:
            old_dropout = strategy.dropout
            strategy.dropout = max(strategy.dropout - 0.1, 0.1)
            logger.info(f"Reduced dropout: {old_dropout:.2f} -> {strategy.dropout:.2f}")
        
        return strategy
    
    def _handle_resource_inefficiency(
        self,
        strategy: Strategy,
        performance: PerformanceData
    ) -> Strategy:
        """
        Handle resource inefficiency by optimizing batch size.
        
        Requirement 7.5: When GPU utilization remains below 60%,
        the system must optimize batch size for better resource usage.
        
        Args:
            strategy: Current strategy
            performance: Performance data
            
        Returns:
            Strategy with optimized resource usage
        """
        logger.info(f"Handling resource inefficiency (GPU util: {performance.gpu_utilization:.1f}%)")
        
        # Increase batch size to improve GPU utilization
        old_batch = strategy.batch_size
        strategy.batch_size = int(strategy.batch_size * 1.5)
        logger.info(f"Increased batch size: {old_batch} -> {strategy.batch_size}")
        
        # Adjust learning rate proportionally (linear scaling rule)
        old_lr = strategy.learning_rate
        lr_scale = strategy.batch_size / old_batch
        strategy.learning_rate = strategy.learning_rate * lr_scale
        logger.info(f"Scaled learning rate: {old_lr:.6f} -> {strategy.learning_rate:.6f}")
        
        # Enable mixed precision if not already enabled
        if not strategy.mixed_precision:
            strategy.mixed_precision = True
            logger.info("Enabled mixed precision for better GPU utilization")
        
        return strategy
    
    def _handle_regression(
        self,
        strategy: Strategy,
        performance: PerformanceData
    ) -> Strategy:
        """
        Handle performance regression with conservative improvements.
        
        Args:
            strategy: Current strategy
            performance: Performance data
            
        Returns:
            Strategy with conservative adjustments
        """
        logger.info("Handling performance regression")
        
        # Reduce learning rate to stabilize
        old_lr = strategy.learning_rate
        strategy.learning_rate = strategy.learning_rate * 0.5
        logger.info(f"Reduced learning rate: {old_lr:.6f} -> {strategy.learning_rate:.6f}")
        
        # Add gradient clipping if not present
        if strategy.gradient_clip_norm is None:
            strategy.gradient_clip_norm = 1.0
            logger.info("Added gradient clipping: norm=1.0")
        
        return strategy
    
    def _log_changes(self, old: Strategy, new: Strategy):
        """
        Log changes between old and new strategy.
        
        Args:
            old: Old strategy
            new: New strategy
        """
        changes = []
        
        if old.dropout != new.dropout:
            changes.append(f"dropout: {old.dropout:.2f} -> {new.dropout:.2f}")
        
        if old.weight_decay != new.weight_decay:
            changes.append(f"weight_decay: {old.weight_decay:.4f} -> {new.weight_decay:.4f}")
        
        if old.learning_rate != new.learning_rate:
            changes.append(f"learning_rate: {old.learning_rate:.6f} -> {new.learning_rate:.6f}")
        
        if old.batch_size != new.batch_size:
            changes.append(f"batch_size: {old.batch_size} -> {new.batch_size}")
        
        if old.max_epochs != new.max_epochs:
            changes.append(f"max_epochs: {old.max_epochs} -> {new.max_epochs}")
        
        if old.model_size != new.model_size:
            changes.append(f"model_size: {old.model_size} -> {new.model_size}")
        
        if old.mixed_precision != new.mixed_precision:
            changes.append(f"mixed_precision: {old.mixed_precision} -> {new.mixed_precision}")
        
        if changes:
            logger.info(f"Strategy changes: {', '.join(changes)}")
        else:
            logger.info("No strategy changes applied")

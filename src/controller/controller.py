"""Controller: Central reasoning engine for HybridAutoMLE agent"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from src.models.data_models import (
    DatasetProfile,
    Strategy,
    RecoveryPlan,
    TrainingMetrics,
    PerformanceData,
    TargetType,
    Modality
)
from src.detector.modality_detector import HybridModalityDetector
from src.strategies.strategy_system import StrategySystem
from src.generator.code_generator import HybridCodeGenerator
from src.utils.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class Controller:
    """
    Central reasoning engine that maintains world model and generates execution plans.
    
    The Controller is responsible for:
    - Dataset profiling and modality detection
    - Strategy selection and code generation
    - Error analysis and recovery planning
    - Strategy evolution based on feedback
    
    CRITICAL: The Controller NEVER executes code directly. All code execution
    must be delegated to the Executor component.
    """
    
    def __init__(self, gemini_client: GeminiClient, config: Any):
        """
        Initialize controller with Gemini client and configuration.
        
        Args:
            gemini_client: Gemini client for LLM-based reasoning
            config: Agent configuration (AgentConfig)
        """
        self.gemini_client = gemini_client
        self.config = config
        
        # Initialize components
        self.modality_detector = HybridModalityDetector(gemini_client)
        self.strategy_system = StrategySystem(config.resource_constraints)
        self.code_generator = HybridCodeGenerator(gemini_client)
        
        logger.info("Controller initialized")
    
    def analyze_dataset(self, dataset_path: str) -> DatasetProfile:
        """
        Analyze dataset and return comprehensive profile.
        
        This method orchestrates the modality detection and data profiling
        to create a complete understanding of the dataset.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            DatasetProfile with comprehensive dataset information
        """
        logger.info(f"Analyzing dataset at: {dataset_path}")
        
        # Detect modality using hybrid approach
        modality_result = self.modality_detector.detect_modality(dataset_path)
        logger.info(f"Detected modality: {modality_result.modality} (confidence: {modality_result.confidence})")
        
        # Get statistical profile
        data_profile = self.modality_detector.data_profiling(dataset_path)
        logger.info(f"Dataset profile: {data_profile.num_samples} samples, {data_profile.num_features} features")
        
        # Infer target type
        target_type = self._infer_target_type(data_profile, modality_result.modality)
        
        # Calculate class imbalance ratio
        class_imbalance_ratio = self._calculate_class_imbalance(data_profile)
        
        # Estimate GPU memory requirements
        estimated_gpu_memory_gb = self._estimate_gpu_memory(
            modality_result.modality,
            data_profile.num_samples,
            data_profile.num_features
        )
        
        # Determine if dataset has metadata (for multimodal)
        has_metadata = modality_result.modality == Modality.MULTIMODAL.value
        
        # Create comprehensive dataset profile
        dataset_profile = DatasetProfile(
            modality=modality_result.modality,
            confidence=modality_result.confidence,
            memory_gb=data_profile.memory_usage_gb,
            num_samples=data_profile.num_samples,
            num_features=data_profile.num_features,
            target_type=target_type,
            class_imbalance_ratio=class_imbalance_ratio,
            missing_percentage=data_profile.missing_values,
            feature_correlations=data_profile.feature_correlations,
            has_metadata=has_metadata,
            estimated_gpu_memory_gb=estimated_gpu_memory_gb
        )
        
        logger.info(f"Dataset analysis complete: {dataset_profile.modality} modality, "
                   f"{dataset_profile.target_type} task")
        
        return dataset_profile
    
    def select_strategy(self, profile: DatasetProfile) -> Strategy:
        """
        Select optimal strategy based on dataset profile.
        
        Uses the StrategySystem to select modality-specific strategies
        with appropriate resource adaptations.
        
        Args:
            profile: Comprehensive dataset profile
            
        Returns:
            Strategy configuration optimized for the dataset
        """
        logger.info(f"Selecting strategy for {profile.modality} modality")
        
        # Use strategy system to get optimal strategy
        strategy = self.strategy_system.get_strategy(profile.modality, profile)
        
        logger.info(f"Selected strategy: {strategy.primary_model} "
                   f"(fallback: {strategy.fallback_model})")
        logger.info(f"Strategy config: batch_size={strategy.batch_size}, "
                   f"max_epochs={strategy.max_epochs}, loss={strategy.loss_function}")
        
        return strategy
    
    def generate_code(self, strategy: Strategy, profile: DatasetProfile, 
                     dataset_info: Dict[str, Any]) -> str:
        """
        Generate training code using Gemini and templates.
        
        IMPORTANT: This method generates code but NEVER executes it.
        Code execution must be delegated to the Executor component.
        
        Args:
            strategy: Selected ML strategy
            profile: Dataset profile
            dataset_info: Dataset-specific information (paths, columns, etc.)
            
        Returns:
            Generated training code as string
        """
        logger.info("Generating training code")
        
        # Use code generator to create training code
        code = self.code_generator.generate_training_code(
            strategy=strategy,
            modality=profile.modality,
            profile=profile,
            dataset_info=dataset_info
        )
        
        logger.info(f"Generated {len(code)} characters of training code")
        
        # CRITICAL: Controller never executes code
        # The generated code must be passed to Executor for execution
        
        return code
    
    def analyze_error(self, error_log: str, strategy: Strategy) -> RecoveryPlan:
        """
        Analyze execution errors and generate recovery plan.
        
        Uses Gemini to perform root cause analysis and determine
        appropriate recovery actions.
        
        Args:
            error_log: Error log from execution
            strategy: Current strategy that failed
            
        Returns:
            RecoveryPlan with recovery actions
        """
        logger.info("Analyzing error for recovery plan")
        
        # Classify error type
        error_type = self._classify_error(error_log)
        logger.info(f"Error classified as: {error_type}")
        
        # Use Gemini for root cause analysis
        analysis = self._gemini_root_cause_analysis(error_log, strategy, error_type)
        
        # Generate recovery plan based on analysis
        recovery_plan = self._generate_recovery_plan(error_type, analysis, strategy)
        
        logger.info(f"Recovery plan generated: {recovery_plan.action}")
        
        return recovery_plan
    
    def evolve_strategy(self, current: Strategy, metrics: TrainingMetrics) -> Strategy:
        """
        Evolve strategy based on performance feedback.
        
        Implements self-improvement by analyzing performance issues
        and adapting the strategy accordingly.
        
        Args:
            current: Current strategy
            metrics: Performance metrics from training
            
        Returns:
            Evolved strategy with improvements
        """
        logger.info("Evolving strategy based on performance feedback")
        
        # Analyze performance
        performance_data = PerformanceData(
            train_score=metrics.train_score,
            val_score=metrics.val_score,
            train_val_gap=metrics.train_score - metrics.val_score,
            gpu_utilization=metrics.gpu_utilization,
            issue_type=self._detect_performance_issue(metrics)
        )
        
        logger.info(f"Performance issue detected: {performance_data.issue_type}")
        
        # Create evolved strategy
        evolved = self._apply_evolution_rules(current, performance_data)
        
        # Use Gemini for additional insights if available
        if performance_data.issue_type != "healthy":
            gemini_suggestions = self._gemini_strategy_suggestions(
                current, performance_data
            )
            evolved = self._apply_gemini_suggestions(evolved, gemini_suggestions)
        
        logger.info("Strategy evolution complete")
        
        return evolved
    
    def _infer_target_type(self, profile: Any, modality: str) -> str:
        """
        Infer target type from data profile and modality.
        
        Args:
            profile: Data profile
            modality: Dataset modality
            
        Returns:
            Target type (classification, regression, or sequence)
        """
        # Check target distribution
        if profile.target_distribution:
            # If target has discrete values, likely classification
            num_unique = len(profile.target_distribution)
            if num_unique < 20:
                return TargetType.CLASSIFICATION.value
            else:
                return TargetType.REGRESSION.value
        
        # Infer from modality
        if modality == Modality.TIME_SERIES.value:
            return TargetType.SEQUENCE.value
        elif modality in [Modality.IMAGE.value, Modality.TEXT.value, Modality.MULTIMODAL.value]:
            return TargetType.CLASSIFICATION.value
        else:
            # Default to classification for tabular
            return TargetType.CLASSIFICATION.value
    
    def _calculate_class_imbalance(self, profile: Any) -> float:
        """
        Calculate class imbalance ratio from target distribution.
        
        Args:
            profile: Data profile
            
        Returns:
            Class imbalance ratio (max_class_count / min_class_count)
        """
        if not profile.target_distribution:
            return 1.0
        
        counts = list(profile.target_distribution.values())
        if not counts or len(counts) < 2:
            return 1.0
        
        max_count = max(counts)
        min_count = min(counts)
        
        if min_count == 0:
            return float('inf')
        
        return max_count / min_count
    
    def _estimate_gpu_memory(self, modality: str, num_samples: int, 
                            num_features: int) -> float:
        """
        Estimate GPU memory requirements based on dataset characteristics.
        
        Args:
            modality: Dataset modality
            num_samples: Number of samples
            num_features: Number of features
            
        Returns:
            Estimated GPU memory in GB
        """
        # Base estimates by modality
        base_memory = {
            Modality.TABULAR.value: 2.0,
            Modality.IMAGE.value: 8.0,
            Modality.TEXT.value: 6.0,
            Modality.TIME_SERIES.value: 4.0,
            Modality.MULTIMODAL.value: 12.0
        }
        
        base = base_memory.get(modality, 4.0)
        
        # Scale based on dataset size
        sample_factor = min(num_samples / 10000, 2.0)
        feature_factor = min(num_features / 100, 1.5)
        
        estimated = base * sample_factor * feature_factor
        
        # Cap at reasonable maximum
        return min(estimated, 20.0)
    
    def _classify_error(self, error_log: str) -> str:
        """
        Classify error type from error log.
        
        Args:
            error_log: Error log text
            
        Returns:
            Error type (resource, data, model, or code)
        """
        error_lower = error_log.lower()
        
        # Resource errors
        if any(keyword in error_lower for keyword in [
            'out of memory', 'oom', 'cuda', 'memory error', 'timeout'
        ]):
            return "resource"
        
        # Data errors
        if any(keyword in error_lower for keyword in [
            'filenotfound', 'no such file', 'missing', 'corrupt', 'invalid data'
        ]):
            return "data"
        
        # Model/training errors
        if any(keyword in error_lower for keyword in [
            'nan', 'inf', 'gradient', 'loss', 'convergence', 'training'
        ]):
            return "model"
        
        # Code errors
        if any(keyword in error_lower for keyword in [
            'syntax', 'import', 'attribute', 'type error', 'name error'
        ]):
            return "code"
        
        return "unknown"
    
    def _gemini_root_cause_analysis(self, error_log: str, strategy: Strategy, 
                                   error_type: str) -> Dict[str, Any]:
        """
        Use Gemini to perform root cause analysis.
        
        Args:
            error_log: Error log text
            strategy: Current strategy
            error_type: Classified error type
            
        Returns:
            Analysis dictionary with root causes and fixes
        """
        prompt = f"""Perform root cause analysis for this ML training error:

ERROR TYPE: {error_type}

STRATEGY:
- Model: {strategy.primary_model}
- Batch Size: {strategy.batch_size}
- Learning Rate: {strategy.learning_rate}
- Mixed Precision: {strategy.mixed_precision}

ERROR LOG (last 1000 chars):
{error_log[-1000:]}

Analyze and provide:
1. Top 3 likely root causes (with confidence > 80%)
2. Specific fixes for each root cause
3. Priority order for applying fixes

Response format: JSON with keys "root_causes", "fixes", "priority_order"
"""
        
        try:
            analysis = self.gemini_client.generate_json(prompt)
            return analysis
        except Exception as e:
            logger.error(f"Gemini root cause analysis failed: {e}")
            # Return default analysis
            return {
                "root_causes": [f"{error_type} error"],
                "fixes": ["retry with adjusted parameters"],
                "priority_order": [0]
            }
    
    def _generate_recovery_plan(self, error_type: str, analysis: Dict[str, Any], 
                               strategy: Strategy) -> RecoveryPlan:
        """
        Generate recovery plan based on error analysis.
        
        Args:
            error_type: Classified error type
            analysis: Gemini analysis results
            strategy: Current strategy
            
        Returns:
            RecoveryPlan with ordered actions
        """
        # Get fixes from analysis
        fixes = analysis.get("fixes", [])
        
        if error_type == "resource":
            # Resource error recovery
            return RecoveryPlan(
                action="reduce_batch_size",
                params={
                    "new_batch_size": strategy.batch_size // 2,
                    "enable_gradient_accumulation": True,
                    "enable_mixed_precision": True
                },
                priority=1
            )
        elif error_type == "data":
            # Data error recovery
            return RecoveryPlan(
                action="skip_missing_data",
                params={
                    "threshold": 0.05,
                    "alternative_paths": []
                },
                priority=2
            )
        elif error_type == "model":
            # Model/training error recovery
            return RecoveryPlan(
                action="adjust_learning_rate",
                params={
                    "new_learning_rate": strategy.learning_rate * 0.1,
                    "add_gradient_clipping": True,
                    "gradient_clip_norm": 1.0
                },
                priority=1
            )
        elif error_type == "code":
            # Code error recovery
            return RecoveryPlan(
                action="request_gemini_fix",
                params={
                    "fixes": fixes
                },
                priority=0
            )
        else:
            # Unknown error - use Gemini suggestions
            return RecoveryPlan(
                action="apply_gemini_suggestions",
                params={
                    "suggestions": fixes
                },
                priority=3
            )
    
    def _detect_performance_issue(self, metrics: TrainingMetrics) -> str:
        """
        Detect performance issues from training metrics.
        
        Args:
            metrics: Training metrics
            
        Returns:
            Issue type (overfitting, underfitting, resource_inefficiency, or healthy)
        """
        train_val_gap = metrics.train_score - metrics.val_score
        
        # Check for overfitting
        if train_val_gap > 0.15:
            return "overfitting"
        
        # Check for underfitting
        if metrics.train_score < 0.7:
            return "underfitting"
        
        # Check for resource inefficiency
        if metrics.gpu_utilization < 60.0:
            return "resource_inefficiency"
        
        return "healthy"
    
    def _apply_evolution_rules(self, current: Strategy, 
                              performance: PerformanceData) -> Strategy:
        """
        Apply evolution rules based on performance data.
        
        Args:
            current: Current strategy
            performance: Performance data
            
        Returns:
            Evolved strategy
        """
        # Create a copy to evolve
        evolved = Strategy(
            modality=current.modality,
            primary_model=current.primary_model,
            fallback_model=current.fallback_model,
            preprocessing=current.preprocessing.copy(),
            augmentation=current.augmentation.copy() if current.augmentation else None,
            loss_function=current.loss_function,
            optimizer=current.optimizer,
            batch_size=current.batch_size,
            max_epochs=current.max_epochs,
            early_stopping_patience=current.early_stopping_patience,
            hyperparameters=current.hyperparameters.copy(),
            resource_constraints=current.resource_constraints,
            learning_rate=current.learning_rate,
            weight_decay=current.weight_decay,
            dropout=current.dropout,
            mixed_precision=current.mixed_precision,
            gradient_accumulation_steps=current.gradient_accumulation_steps,
            gradient_clip_norm=current.gradient_clip_norm,
            model_size=current.model_size,
            augmentation_strength=current.augmentation_strength
        )
        
        # Apply rules based on issue type
        if performance.issue_type == "overfitting":
            # Increase regularization
            evolved.dropout = min(evolved.dropout + 0.1, 0.5)
            evolved.weight_decay = evolved.weight_decay * 2
            if evolved.augmentation_strength:
                evolved.augmentation_strength = min(evolved.augmentation_strength * 1.5, 2.0)
            logger.info("Applied overfitting mitigation: increased regularization")
            
        elif performance.issue_type == "underfitting":
            # Increase model complexity
            if evolved.model_size == "small":
                evolved.model_size = "medium"
            elif evolved.model_size == "medium":
                evolved.model_size = "large"
            evolved.max_epochs = int(evolved.max_epochs * 1.5)
            evolved.learning_rate = evolved.learning_rate * 1.2
            logger.info("Applied underfitting mitigation: increased model complexity")
            
        elif performance.issue_type == "resource_inefficiency":
            # Optimize resource usage
            evolved.batch_size = int(evolved.batch_size * 1.5)
            logger.info("Applied resource optimization: increased batch size")
        
        return evolved
    
    def _gemini_strategy_suggestions(self, current: Strategy, 
                                    performance: PerformanceData) -> Dict[str, Any]:
        """
        Get strategy improvement suggestions from Gemini.
        
        Args:
            current: Current strategy
            performance: Performance data
            
        Returns:
            Dictionary with improvement suggestions
        """
        prompt = f"""Suggest improvements for this ML strategy based on performance issues:

CURRENT STRATEGY:
- Model: {current.primary_model}
- Batch Size: {current.batch_size}
- Learning Rate: {current.learning_rate}
- Dropout: {current.dropout}
- Weight Decay: {current.weight_decay}

PERFORMANCE:
- Train Score: {performance.train_score}
- Val Score: {performance.val_score}
- Train-Val Gap: {performance.train_val_gap}
- GPU Utilization: {performance.gpu_utilization}%
- Issue: {performance.issue_type}

Provide specific suggestions to improve performance.

Response format: JSON with key "suggestions" containing list of actionable improvements
"""
        
        try:
            suggestions = self.gemini_client.generate_json(prompt)
            return suggestions
        except Exception as e:
            logger.error(f"Failed to get Gemini suggestions: {e}")
            return {"suggestions": []}
    
    def _apply_gemini_suggestions(self, strategy: Strategy, 
                                 suggestions: Dict[str, Any]) -> Strategy:
        """
        Apply Gemini suggestions to strategy.
        
        Args:
            strategy: Current strategy
            suggestions: Gemini suggestions
            
        Returns:
            Strategy with suggestions applied
        """
        # For now, just log the suggestions
        # In a full implementation, we would parse and apply specific suggestions
        suggestion_list = suggestions.get("suggestions", [])
        if suggestion_list:
            logger.info(f"Gemini suggestions: {suggestion_list}")
        
        return strategy

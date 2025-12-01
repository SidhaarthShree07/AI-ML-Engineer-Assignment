"""Comprehensive error handling system for HybridAutoMLE agent"""

import logging
import traceback
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from src.models.data_models import Strategy, RecoveryPlan, ResourceConstraints
from src.utils.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Error classification types"""
    RESOURCE = "resource"
    DATA = "data"
    MODEL = "model"
    CODE = "code"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    error_log: str
    error_type: ErrorType
    strategy: Strategy
    attempt_count: int
    stack_trace: Optional[str] = None


class ErrorHandler:
    """
    Comprehensive error handling system with classification, recovery, and retry logic.
    
    Handles four main error categories:
    - Resource errors (OOM, timeout)
    - Data errors (missing files, format issues)
    - Model errors (NaN loss, training failures)
    - Code errors (syntax, import issues)
    
    Implements:
    - Error classification
    - Recovery action generation
    - Retry limit enforcement (3 attempts)
    - Fallback strategy activation
    """
    
    MAX_RETRY_ATTEMPTS = 3
    
    def __init__(self, controller=None, state_manager=None):
        """
        Initialize error handler.
        
        Args:
            controller: Controller instance for error analysis
            state_manager: StateManager for logging recovery attempts
        """
        self.controller = controller
        self.state_manager = state_manager
        self.gemini_client = controller.gemini_client if controller else None
        self.error_history: List[ErrorContext] = []
        self.retry_counts: Dict[str, int] = {}
        
        logger.info("ErrorHandler initialized")
    
    def handle_execution_error(self, error_log: str, strategy: Strategy, 
                               executor, max_retries: int = 3) -> bool:
        """
        Handle execution error with retry logic.
        
        Args:
            error_log: Error log from execution
            strategy: Current strategy
            executor: Executor instance for re-execution
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if recovery successful, False otherwise
        """
        for attempt in range(max_retries):
            logger.info(f"Recovery attempt {attempt + 1}/{max_retries}")
            
            # Get recovery plan from controller
            if self.controller:
                recovery_plan = self.controller.analyze_error(error_log, strategy)
            else:
                # Fallback to basic error handling
                recovery_plan, _ = self.handle_error(error_log, strategy)
            
            # Log recovery attempt
            if self.state_manager:
                self.state_manager.log_action(
                    phase="execution",
                    action="recovery_attempt",
                    input_data={
                        "attempt": attempt + 1,
                        "error_type": self.classify_error(error_log).value,
                        "recovery_action": recovery_plan.action
                    },
                    output_data={"params": recovery_plan.params}
                )
            
            # Apply recovery (this would modify strategy and retry)
            # For now, we just log and return False to indicate recovery needed
            logger.warning(f"Recovery plan: {recovery_plan.action}")
            logger.warning(f"Recovery params: {recovery_plan.params}")
            
            # In a full implementation, we would:
            # 1. Apply recovery plan to strategy
            # 2. Regenerate code with modified strategy
            # 3. Re-execute with executor
            # 4. Check if successful
            
            # For now, return False to indicate recovery not fully implemented
            break
        
        return False
    
    def handle_error(self, error_log: str, strategy: Strategy, 
                    error_id: Optional[str] = None) -> Tuple[RecoveryPlan, bool]:
        """
        Main error handling entry point.
        
        Classifies error, generates recovery plan, and tracks retry attempts.
        
        Args:
            error_log: Error log text
            strategy: Current strategy
            error_id: Optional identifier for tracking retries
            
        Returns:
            Tuple of (RecoveryPlan, should_activate_fallback)
            should_activate_fallback is True if retry limit exceeded
        """
        # Capture full stack trace if available
        stack_trace = self._extract_stack_trace(error_log)
        
        # Classify error type
        error_type = self.classify_error(error_log)
        logger.info(f"Error classified as: {error_type.value}")
        
        # Track retry attempts
        if error_id is None:
            error_id = f"{error_type.value}_{hash(error_log[:100])}"
        
        attempt_count = self.retry_counts.get(error_id, 0) + 1
        self.retry_counts[error_id] = attempt_count
        
        logger.info(f"Error attempt {attempt_count}/{self.MAX_RETRY_ATTEMPTS}")
        
        # Create error context
        context = ErrorContext(
            error_log=error_log,
            error_type=error_type,
            strategy=strategy,
            attempt_count=attempt_count,
            stack_trace=stack_trace
        )
        
        # Store in history
        self.error_history.append(context)
        
        # Check if retry limit exceeded
        should_activate_fallback = attempt_count >= self.MAX_RETRY_ATTEMPTS
        
        if should_activate_fallback:
            logger.warning(f"Retry limit exceeded for {error_id}, activating fallback")
            recovery_plan = self._create_fallback_plan(strategy)
        else:
            # Generate recovery plan based on error type
            recovery_plan = self._generate_recovery_plan(context)
        
        return recovery_plan, should_activate_fallback
    
    def classify_error(self, error_log: str) -> ErrorType:
        """
        Classify error type from error log.
        
        Args:
            error_log: Error log text
            
        Returns:
            ErrorType classification
        """
        error_lower = error_log.lower()
        
        # Resource errors (OOM, timeout, memory)
        resource_keywords = [
            'out of memory', 'oom', 'cuda', 'memory error', 
            'timeout', 'killed', 'resource', 'vram'
        ]
        if any(keyword in error_lower for keyword in resource_keywords):
            return ErrorType.RESOURCE
        
        # Data errors (file not found, missing data, corrupt)
        data_keywords = [
            'filenotfound', 'no such file', 'missing', 'corrupt', 
            'invalid data', 'cannot open', 'file does not exist',
            'permission denied', 'io error'
        ]
        if any(keyword in error_lower for keyword in data_keywords):
            return ErrorType.DATA
        
        # Model/training errors (NaN, loss issues, convergence)
        model_keywords = [
            'nan', 'inf', 'gradient', 'loss', 'convergence', 
            'training', 'exploding', 'vanishing', 'overflow'
        ]
        if any(keyword in error_lower for keyword in model_keywords):
            return ErrorType.MODEL
        
        # Code errors (syntax, import, attribute)
        code_keywords = [
            'syntax', 'import', 'attribute', 'type error', 
            'name error', 'indentation', 'module', 'undefined'
        ]
        if any(keyword in error_lower for keyword in code_keywords):
            return ErrorType.CODE
        
        return ErrorType.UNKNOWN
    
    def _generate_recovery_plan(self, context: ErrorContext) -> RecoveryPlan:
        """
        Generate recovery plan based on error context.
        
        Args:
            context: Error context
            
        Returns:
            RecoveryPlan with ordered actions
        """
        if context.error_type == ErrorType.RESOURCE:
            return handle_resource_error(context.error_log, context.strategy)
        elif context.error_type == ErrorType.DATA:
            return handle_data_error(context.error_log, context.strategy)
        elif context.error_type == ErrorType.MODEL:
            return handle_training_error(context.error_log, context.strategy)
        elif context.error_type == ErrorType.CODE:
            return handle_code_error(
                context.error_log, 
                context.strategy, 
                self.gemini_client
            )
        else:
            # Unknown error - use Gemini if available
            if self.gemini_client:
                return self._gemini_recovery_plan(context)
            else:
                return RecoveryPlan(
                    action="log_and_continue",
                    params={"error": context.error_log[:500]},
                    priority=5
                )
    
    def _create_fallback_plan(self, strategy: Strategy) -> RecoveryPlan:
        """
        Create fallback activation plan when retry limit exceeded.
        
        Args:
            strategy: Current strategy
            
        Returns:
            RecoveryPlan for fallback activation
        """
        return RecoveryPlan(
            action="activate_fallback",
            params={
                "fallback_model": strategy.fallback_model,
                "reason": "retry_limit_exceeded",
                "max_attempts": self.MAX_RETRY_ATTEMPTS
            },
            priority=0  # Highest priority
        )
    
    def _extract_stack_trace(self, error_log: str) -> Optional[str]:
        """
        Extract stack trace from error log.
        
        Args:
            error_log: Error log text
            
        Returns:
            Stack trace if found, None otherwise
        """
        # Look for common stack trace patterns
        if 'Traceback' in error_log:
            lines = error_log.split('\n')
            trace_start = None
            for i, line in enumerate(lines):
                if 'Traceback' in line:
                    trace_start = i
                    break
            
            if trace_start is not None:
                return '\n'.join(lines[trace_start:])
        
        return None
    
    def _gemini_recovery_plan(self, context: ErrorContext) -> RecoveryPlan:
        """
        Use Gemini to generate recovery plan for unknown errors.
        
        Args:
            context: Error context
            
        Returns:
            RecoveryPlan from Gemini analysis
        """
        prompt = f"""Analyze this error and suggest recovery actions:

ERROR TYPE: {context.error_type.value}
ATTEMPT: {context.attempt_count}/{self.MAX_RETRY_ATTEMPTS}

ERROR LOG:
{context.error_log[-1000:]}

CURRENT STRATEGY:
- Model: {context.strategy.primary_model}
- Batch Size: {context.strategy.batch_size}
- Learning Rate: {context.strategy.learning_rate}

Provide:
1. Root cause analysis
2. Specific recovery action
3. Parameters for recovery

Response format: JSON with keys "action", "params", "priority"
"""
        
        try:
            result = self.gemini_client.generate_json(prompt)
            return RecoveryPlan(
                action=result.get("action", "retry_with_defaults"),
                params=result.get("params", {}),
                priority=result.get("priority", 3)
            )
        except Exception as e:
            logger.error(f"Gemini recovery plan failed: {e}")
            return RecoveryPlan(
                action="retry_with_defaults",
                params={},
                priority=3
            )
    
    def reset_retry_count(self, error_id: str):
        """
        Reset retry count for a specific error.
        
        Args:
            error_id: Error identifier
        """
        if error_id in self.retry_counts:
            del self.retry_counts[error_id]
            logger.info(f"Reset retry count for {error_id}")
    
    def get_error_history(self) -> List[ErrorContext]:
        """
        Get error history.
        
        Returns:
            List of error contexts
        """
        return self.error_history.copy()
    
    def clear_history(self):
        """Clear error history and retry counts."""
        self.error_history.clear()
        self.retry_counts.clear()
        logger.info("Error history cleared")


def handle_resource_error(error_log: str, strategy: Strategy) -> RecoveryPlan:
    """
    Handle resource errors (OOM, timeout) with multi-stage recovery.
    
    Recovery stages (in order):
    1. Reduce batch size by 50%
    2. Enable gradient accumulation
    3. Switch to mixed precision (fp16)
    4. Downsize model architecture
    5. Apply aggressive feature selection (tabular)
    
    Args:
        error_log: Error log text
        strategy: Current strategy
        
    Returns:
        RecoveryPlan with resource optimization actions
    """
    error_lower = error_log.lower()
    
    # Determine specific resource issue
    is_oom = 'out of memory' in error_lower or 'oom' in error_lower
    is_timeout = 'timeout' in error_lower
    is_cuda = 'cuda' in error_lower
    
    params = {}
    
    # Stage 1: Reduce batch size
    new_batch_size = max(strategy.batch_size // 2, 1)
    params['new_batch_size'] = new_batch_size
    
    # Stage 2: Enable gradient accumulation
    params['enable_gradient_accumulation'] = True
    params['gradient_accumulation_steps'] = 2
    
    # Stage 3: Enable mixed precision
    if not strategy.mixed_precision:
        params['enable_mixed_precision'] = True
    
    # Stage 4: Model downsizing
    if strategy.model_size == "large":
        params['new_model_size'] = "medium"
    elif strategy.model_size == "medium":
        params['new_model_size'] = "small"
    
    # Stage 5: Feature selection for tabular
    if strategy.modality == "tabular":
        params['apply_feature_selection'] = True
        params['max_features'] = 50
    
    # Timeout-specific: reduce epochs
    if is_timeout:
        params['new_max_epochs'] = max(strategy.max_epochs // 2, 5)
        params['enable_early_stopping'] = True
    
    return RecoveryPlan(
        action="reduce_resource_usage",
        params=params,
        priority=1  # High priority
    )


def handle_data_error(error_log: str, strategy: Strategy) -> RecoveryPlan:
    """
    Handle data errors (missing files, format issues) with recovery strategies.
    
    Recovery actions:
    1. Attempt to locate files in alternative paths
    2. Skip missing samples if < 5% of dataset
    3. Infer missing values using statistical methods
    4. Request Gemini analysis for format issues
    
    Args:
        error_log: Error log text
        strategy: Current strategy
        
    Returns:
        RecoveryPlan with data recovery actions
    """
    error_lower = error_log.lower()
    
    # Determine specific data issue
    is_file_missing = 'filenotfound' in error_lower or 'no such file' in error_lower
    is_corrupt = 'corrupt' in error_lower
    is_format = 'invalid data' in error_lower or 'format' in error_lower
    
    params = {}
    
    if is_file_missing:
        # Try alternative paths
        params['action_type'] = 'locate_files'
        params['search_alternative_paths'] = True
        params['skip_threshold'] = 0.05  # Skip if < 5% missing
        
        return RecoveryPlan(
            action="handle_missing_files",
            params=params,
            priority=2
        )
    
    elif is_corrupt or is_format:
        # Handle corrupt/format issues
        params['action_type'] = 'fix_format'
        params['infer_missing_values'] = True
        params['drop_corrupt_rows'] = True
        params['max_drop_percentage'] = 0.05
        
        return RecoveryPlan(
            action="fix_data_format",
            params=params,
            priority=2
        )
    
    else:
        # Generic data error
        params['action_type'] = 'generic'
        params['validate_data'] = True
        params['clean_data'] = True
        
        return RecoveryPlan(
            action="validate_and_clean_data",
            params=params,
            priority=2
        )


def handle_training_error(error_log: str, strategy: Strategy) -> RecoveryPlan:
    """
    Handle training errors (NaN loss, convergence issues) with recovery strategies.
    
    Recovery actions:
    1. Reduce learning rate by 10x
    2. Add gradient clipping
    3. Switch optimizer (Adam → SGD)
    4. Reinitialize model weights
    5. Switch to fallback model
    
    Args:
        error_log: Error log text
        strategy: Current strategy
        
    Returns:
        RecoveryPlan with training recovery actions
    """
    error_lower = error_log.lower()
    
    # Determine specific training issue
    is_nan = 'nan' in error_lower
    is_inf = 'inf' in error_lower
    is_exploding = 'exploding' in error_lower or 'overflow' in error_lower
    is_vanishing = 'vanishing' in error_lower
    
    params = {}
    
    if is_nan or is_inf:
        # NaN/Inf loss - aggressive fixes
        params['new_learning_rate'] = strategy.learning_rate * 0.1
        params['add_gradient_clipping'] = True
        params['gradient_clip_norm'] = 1.0
        params['reinitialize_weights'] = True
        
        return RecoveryPlan(
            action="fix_nan_loss",
            params=params,
            priority=1
        )
    
    elif is_exploding:
        # Exploding gradients
        params['new_learning_rate'] = strategy.learning_rate * 0.1
        params['gradient_clip_norm'] = 0.5
        params['switch_optimizer'] = True
        params['new_optimizer'] = 'SGD'
        params['momentum'] = 0.9
        
        return RecoveryPlan(
            action="fix_exploding_gradients",
            params=params,
            priority=1
        )
    
    elif is_vanishing:
        # Vanishing gradients
        params['new_learning_rate'] = strategy.learning_rate * 2.0
        params['use_batch_normalization'] = True
        params['activation'] = 'relu'
        
        return RecoveryPlan(
            action="fix_vanishing_gradients",
            params=params,
            priority=1
        )
    
    else:
        # Generic training error
        params['new_learning_rate'] = strategy.learning_rate * 0.5
        params['add_gradient_clipping'] = True
        params['gradient_clip_norm'] = 1.0
        
        return RecoveryPlan(
            action="adjust_training_params",
            params=params,
            priority=1
        )


def handle_code_error(error_log: str, strategy: Strategy, 
                     gemini_client: Optional[GeminiClient] = None) -> RecoveryPlan:
    """
    Handle code errors (syntax, import issues) with Gemini-based fixes.
    
    Recovery actions:
    1. Request Gemini to fix syntax errors
    2. Add missing imports from template library
    3. Simplify code structure
    4. Fall back to verified template
    
    Args:
        error_log: Error log text
        strategy: Current strategy
        gemini_client: Optional Gemini client for code fixes
        
    Returns:
        RecoveryPlan with code fix actions
    """
    error_lower = error_log.lower()
    
    # Determine specific code issue
    is_syntax = 'syntax' in error_lower
    is_import = 'import' in error_lower or 'module' in error_lower
    is_attribute = 'attribute' in error_lower
    is_type = 'type error' in error_lower
    
    params = {}
    
    if is_syntax:
        # Syntax error - request Gemini fix
        params['error_type'] = 'syntax'
        params['request_gemini_fix'] = True
        params['error_message'] = error_log[-500:]
        
        return RecoveryPlan(
            action="fix_syntax_error",
            params=params,
            priority=0  # Highest priority - must fix before execution
        )
    
    elif is_import:
        # Import error - add missing imports
        params['error_type'] = 'import'
        params['add_missing_imports'] = True
        params['error_message'] = error_log[-500:]
        
        # Try to extract missing module name
        if 'no module named' in error_lower:
            # Extract module name
            parts = error_log.split("'")
            if len(parts) >= 2:
                params['missing_module'] = parts[1]
        
        return RecoveryPlan(
            action="fix_import_error",
            params=params,
            priority=0
        )
    
    elif is_attribute or is_type:
        # Attribute/type error - request Gemini fix
        params['error_type'] = 'attribute' if is_attribute else 'type'
        params['request_gemini_fix'] = True
        params['error_message'] = error_log[-500:]
        
        return RecoveryPlan(
            action="fix_code_error",
            params=params,
            priority=0
        )
    
    else:
        # Generic code error
        params['error_type'] = 'generic'
        params['request_gemini_fix'] = True
        params['fallback_to_template'] = True
        params['error_message'] = error_log[-500:]
        
        return RecoveryPlan(
            action="fix_generic_code_error",
            params=params,
            priority=0
        )

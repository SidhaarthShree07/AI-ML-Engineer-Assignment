"""
Code Enhancement using LLM for Self-Improvement.

This module connects the self-improvement phase to the LLM to:
1. Analyze training code execution results (errors, performance)
2. Generate enhanced code if needed
3. Decide whether to retry training or move on
"""

import logging
import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.utils.gemini_client import GeminiClient
from src.models.data_models import Strategy, DatasetProfile

logger = logging.getLogger(__name__)


@dataclass
class EnhancementDecision:
    """Decision from the code enhancer about whether to retry training."""
    should_retry: bool
    enhanced_code: Optional[str]
    reason: str
    improvements_made: list
    performance_acceptable: bool


class CodeEnhancer:
    """
    Connects self-improvement phase to LLM for code enhancement.
    
    Workflow:
    1. Analyze execution results (success, errors, performance metrics)
    2. If no errors and decent performance → move on (don't retry)
    3. If errors or poor performance → use LLM to enhance code → retry
    4. Track improvement iterations to avoid infinite loops
    """
    
    # Performance thresholds for "decent" performance
    MIN_ACCEPTABLE_SCORE = 0.5  # Minimum validation score to be considered acceptable
    MAX_ACCEPTABLE_TRAIN_VAL_GAP = 0.25  # Maximum train-val gap before overfitting concern
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None, max_iterations: int = 3):
        """
        Initialize code enhancer.
        
        Args:
            gemini_client: Gemini client for LLM-based enhancement
            max_iterations: Maximum number of enhancement iterations
        """
        self.gemini_client = gemini_client
        self.max_iterations = max_iterations
        self.current_iteration = 0
        logger.info(f"CodeEnhancer initialized (max_iterations={max_iterations})")
    
    def analyze_and_decide(
        self,
        execution_success: bool,
        stdout: str,
        stderr: str,
        exit_code: int,
        current_code: str,
        strategy: Strategy,
        profile: DatasetProfile
    ) -> EnhancementDecision:
        """
        Analyze execution results and decide whether to enhance and retry.
        
        Args:
            execution_success: Whether execution completed without errors
            stdout: Standard output from execution
            stderr: Standard error from execution  
            exit_code: Exit code from execution
            current_code: Current training code
            strategy: ML strategy being used
            profile: Dataset profile
            
        Returns:
            EnhancementDecision with retry decision and enhanced code if needed
        """
        self.current_iteration += 1
        logger.info(f"Analyzing execution (iteration {self.current_iteration}/{self.max_iterations})")
        
        # Check if we've exceeded max iterations
        if self.current_iteration > self.max_iterations:
            logger.warning(f"Maximum iterations ({self.max_iterations}) reached, moving on")
            return EnhancementDecision(
                should_retry=False,
                enhanced_code=None,
                reason=f"Maximum enhancement iterations ({self.max_iterations}) reached",
                improvements_made=[],
                performance_acceptable=True  # Accept whatever we have
            )
        
        # Extract performance metrics from output
        metrics = self._extract_metrics(stdout)
        
        # Check for critical errors
        has_critical_error = self._has_critical_error(stderr, exit_code)
        
        # Check if performance is acceptable (with large dataset awareness)
        performance_acceptable = self._is_performance_acceptable(metrics, execution_success, profile)
        
        logger.info(f"Analysis: success={execution_success}, critical_error={has_critical_error}, "
                   f"performance_acceptable={performance_acceptable}, metrics={metrics}")
        
        # Decision logic:
        # 1. If no errors and decent performance → move on
        # 2. If errors or poor performance → enhance and retry
        
        if execution_success and not has_critical_error and performance_acceptable:
            logger.info("Execution successful with acceptable performance - moving on")
            return EnhancementDecision(
                should_retry=False,
                enhanced_code=None,
                reason="Training completed successfully with acceptable performance",
                improvements_made=[],
                performance_acceptable=True
            )
        
        # Need to enhance the code
        if not self.gemini_client:
            logger.warning("No LLM client available for enhancement - moving on")
            return EnhancementDecision(
                should_retry=False,
                enhanced_code=None,
                reason="No LLM client available for code enhancement",
                improvements_made=[],
                performance_acceptable=performance_acceptable
            )
        
        # Use LLM to enhance the code
        enhanced_code, improvements = self._enhance_code_with_llm(
            current_code=current_code,
            stdout=stdout,
            stderr=stderr,
            metrics=metrics,
            strategy=strategy,
            profile=profile,
            has_error=has_critical_error
        )
        
        if enhanced_code and enhanced_code != current_code:
            logger.info(f"Code enhanced with {len(improvements)} improvements - will retry")
            return EnhancementDecision(
                should_retry=True,
                enhanced_code=enhanced_code,
                reason=f"Enhanced code to address: {', '.join(improvements[:3])}",
                improvements_made=improvements,
                performance_acceptable=False
            )
        else:
            logger.info("LLM could not improve the code - moving on")
            return EnhancementDecision(
                should_retry=False,
                enhanced_code=None,
                reason="LLM could not identify improvements",
                improvements_made=[],
                performance_acceptable=performance_acceptable
            )
    
    def _extract_metrics(self, stdout: str) -> Dict[str, Any]:
        """Extract performance metrics from training output."""
        metrics = {
            'train_score': None,
            'val_score': None,
            'best_model': None,
            'best_loss': None,
            'execution_time': None
        }
        
        # Common patterns to look for
        patterns = {
            'val_score': [
                r'Best validation score[:\s]+([0-9.]+)',
                r'val[_\s]?score[:\s]+([0-9.]+)',
                r'validation[_\s]?accuracy[:\s]+([0-9.]+)',
                r'auc[:\s]+([0-9.]+)',
                r'roc_auc[:\s]+([0-9.]+)'
            ],
            'train_score': [
                r'train[_\s]?score[:\s]+([0-9.]+)',
                r'training[_\s]?accuracy[:\s]+([0-9.]+)'
            ],
            'best_model': [
                r'Best model[:\s]+(\w+)',
                r'best_estimator[:\s]+(\w+)'
            ],
            'best_loss': [
                r'best_loss[:\s]+([0-9.]+)',
                r'Best loss[:\s]+([0-9.]+)'
            ]
        }
        
        stdout_lower = stdout.lower()
        
        for metric_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, stdout, re.IGNORECASE)
                if match:
                    try:
                        value = match.group(1)
                        if metric_name in ['val_score', 'train_score', 'best_loss']:
                            metrics[metric_name] = float(value)
                        else:
                            metrics[metric_name] = value
                        break
                    except (ValueError, IndexError):
                        continue
        
        # Check for completion indicators
        metrics['completed'] = any([
            'training complete' in stdout_lower,
            'submission.csv' in stdout_lower,
            'saved' in stdout_lower
        ])
        
        return metrics
    
    def _has_critical_error(self, stderr: str, exit_code: int) -> bool:
        """Check if there are critical errors that need fixing."""
        if exit_code != 0:
            return True
        
        critical_patterns = [
            r'error',
            r'exception',
            r'traceback',
            r'failed',
            r'cuda out of memory',
            r'oom',
            r'killed',
            r'segmentation fault'
        ]
        
        stderr_lower = stderr.lower()
        for pattern in critical_patterns:
            if re.search(pattern, stderr_lower):
                return True
        
        return False
    
    def _is_performance_acceptable(self, metrics: Dict[str, Any], execution_success: bool, profile: DatasetProfile = None) -> bool:
        """Check if model performance is acceptable."""
        if not execution_success:
            return False
        
        # Adjust thresholds for large datasets (they tend to have lower performance)
        is_large_dataset = profile and profile.num_samples and profile.num_samples > 500000
        min_score_threshold = 0.4 if is_large_dataset else self.MIN_ACCEPTABLE_SCORE
        max_gap_threshold = 0.30 if is_large_dataset else self.MAX_ACCEPTABLE_TRAIN_VAL_GAP
        
        if is_large_dataset:
            logger.info(f"Large dataset detected ({profile.num_samples:,} rows) - using relaxed thresholds")
        
        # If we couldn't extract metrics, assume acceptable if completed
        if metrics.get('completed') and metrics.get('val_score') is None:
            return True
        
        val_score = metrics.get('val_score')
        train_score = metrics.get('train_score')
        
        # Check minimum validation score
        if val_score is not None and val_score < min_score_threshold:
            logger.warning(f"Validation score {val_score:.4f} below threshold {min_score_threshold}")
            return False
        
        # Check for overfitting
        if val_score is not None and train_score is not None:
            gap = train_score - val_score
            if gap > max_gap_threshold:
                logger.warning(f"Train-val gap {gap:.4f} exceeds threshold {max_gap_threshold}")
                return False
        
        return True
    
    def _enhance_code_with_llm(
        self,
        current_code: str,
        stdout: str,
        stderr: str,
        metrics: Dict[str, Any],
        strategy: Strategy,
        profile: DatasetProfile,
        has_error: bool
    ) -> Tuple[Optional[str], list]:
        """
        Use LLM to enhance the training code based on execution results.
        
        Returns:
            Tuple of (enhanced_code, list_of_improvements)
        """
        logger.info("Requesting LLM to enhance training code")
        
        # Build comprehensive prompt
        prompt = self._build_enhancement_prompt(
            current_code=current_code,
            stdout=stdout[-2000:],  # Last 2000 chars
            stderr=stderr[-1500:],  # Last 1500 chars
            metrics=metrics,
            strategy=strategy,
            profile=profile,
            has_error=has_error
        )
        
        try:
            response = self.gemini_client.generate_content(prompt)
            result = self._parse_enhancement_response(response.text, current_code)
            return result
        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            return None, []
    
    def _build_enhancement_prompt(
        self,
        current_code: str,
        stdout: str,
        stderr: str,
        metrics: Dict[str, Any],
        strategy: Strategy,
        profile: DatasetProfile,
        has_error: bool
    ) -> str:
        """Build prompt for LLM code enhancement."""
        
        issue_type = "ERRORS" if has_error else "PERFORMANCE ISSUES"
        
        # Detect large dataset constraints
        is_large_dataset = profile.num_samples > 500000
        has_many_features = profile.num_features > 50 if profile.num_features else False
        is_very_large = profile.num_samples > 900000
        
        # Build large dataset optimization instructions
        large_dataset_instructions = ""
        if is_large_dataset:
            large_dataset_instructions = f"""
⚠️ LARGE DATASET OPTIMIZATION REQUIRED ⚠️
This dataset has {profile.num_samples:,} rows - you MUST apply lightweight optimizations:

MANDATORY CHANGES FOR LARGE DATASETS:
1. **REMOVE ALL CROSS-VALIDATION** - Replace with simple train/val split (80/20)
   - Remove any StratifiedKFold, KFold, cross_val_score
   - Use train_test_split with test_size=0.2
   
2. **REDUCE FLAML TIME BUDGET** - Set time_budget=300-600 max
   
3. **USE ONLY FAST MODELS**:
   - estimator_list=['lgbm', 'xgboost'] only
   - Remove 'catboost', 'rf', 'extra_tree' (too slow)
   - Set max_depth=6-8, n_estimators=100-200
   
4. **NO FEATURE ENGINEERING**:
   - Remove PolynomialFeatures
   - Remove interaction features
   - Keep only basic preprocessing
   
5. **MEMORY OPTIMIZATION**:
   - Add: df = df.astype({{col: 'float32' for col in numeric_cols}})
   - Drop unused columns immediately after loading
   
6. **SIMPLIFY PIPELINE**:
   - Remove ensemble methods
   - Use single model training
   - Set early_stopping_rounds=15

{"7. **EXTREME MODE (900K+ rows)**: Train on 30% stratified sample, use fixed LightGBM parameters, no hyperparameter search." if is_very_large else ""}
"""
        
        prompt = f"""You are an expert ML engineer. Analyze this training code execution and enhance the code to fix issues.

## EXECUTION RESULTS

**Status**: {"FAILED with errors" if has_error else "Completed but with performance concerns"}
**Issue Type**: {issue_type}
{large_dataset_instructions}

**Performance Metrics**:
- Validation Score: {metrics.get('val_score', 'N/A')}
- Train Score: {metrics.get('train_score', 'N/A')}
- Best Model: {metrics.get('best_model', 'N/A')}
- Best Loss: {metrics.get('best_loss', 'N/A')}

**Standard Error (last 1500 chars)**:
```
{stderr if stderr else "No errors"}
```

**Standard Output (last 2000 chars)**:
```
{stdout if stdout else "No output"}
```

## DATASET INFO
- Modality: {profile.modality}
- Samples: {profile.num_samples}
- Features: {profile.num_features}
- Target Type: {profile.target_type}

## STRATEGY
- Model: {strategy.primary_model}
- Batch Size: {strategy.batch_size}
- Learning Rate: {strategy.learning_rate}

## CURRENT CODE
```python
{current_code}
```

## YOUR TASK

{"Fix the errors in the code. Focus on:" if has_error else "Improve the code performance. Focus on:"}
{self._get_fix_instructions(has_error, stderr, metrics, profile)}

## RESPONSE FORMAT

Provide your response in this EXACT format:

IMPROVEMENTS:
- [List each improvement you're making, one per line]

ENHANCED_CODE:
```python
[Your enhanced Python code here - complete and runnable]
```

IMPORTANT:
1. Return COMPLETE, RUNNABLE code (not snippets)
2. Preserve all existing functionality that works
3. Only fix/enhance what's broken or underperforming
4. Keep data preprocessing and submission generation intact
5. The code must generate submission.csv
"""
        return prompt
    
    def _get_fix_instructions(self, has_error: bool, stderr: str, metrics: Dict[str, Any], profile: DatasetProfile = None) -> str:
        """Get specific fix instructions based on the issues detected."""
        instructions = []
        
        # Check for large dataset first
        is_large_dataset = profile and profile.num_samples and profile.num_samples > 500000
        is_very_large = profile and profile.num_samples and profile.num_samples > 900000
        
        if is_large_dataset:
            instructions.append("- ⚠️ LARGE DATASET: Remove all cross-validation, use holdout split only")
            instructions.append("- ⚠️ LARGE DATASET: Set FLAML time_budget to 300-600 seconds max")
            instructions.append("- ⚠️ LARGE DATASET: Use only ['lgbm', 'xgboost'] in estimator_list")
            instructions.append("- ⚠️ LARGE DATASET: Remove polynomial features and interaction terms")
            instructions.append("- ⚠️ LARGE DATASET: Convert float64 to float32 for memory savings")
            if is_very_large:
                instructions.append("- ⚠️ VERY LARGE (900K+): Consider training on 30% stratified sample")
                instructions.append("- ⚠️ VERY LARGE (900K+): Use fixed LightGBM parameters, skip hyperparameter search")
        
        if has_error:
            stderr_lower = stderr.lower()
            if 'memory' in stderr_lower or 'oom' in stderr_lower:
                instructions.append("- Reduce batch size or add memory optimization")
                instructions.append("- Consider using gradient checkpointing")
                if is_large_dataset:
                    instructions.append("- ⚠️ OOM on large data: Sample 30-50% of training data")
                    instructions.append("- ⚠️ OOM on large data: Use chunked data loading")
            if 'cuda' in stderr_lower:
                instructions.append("- Add proper CUDA availability checks")
                instructions.append("- Add fallback to CPU if GPU unavailable")
            if 'syntax' in stderr_lower or 'indentation' in stderr_lower:
                instructions.append("- Fix syntax/indentation errors")
            if 'import' in stderr_lower or 'module' in stderr_lower:
                instructions.append("- Fix import statements")
            if 'keyerror' in stderr_lower or 'column' in stderr_lower:
                instructions.append("- Fix column name issues")
                instructions.append("- Add defensive checks for missing columns")
            if 'nan' in stderr_lower or 'inf' in stderr_lower:
                instructions.append("- Add NaN/Inf handling in data")
                instructions.append("- Add gradient clipping if needed")
        else:
            val_score = metrics.get('val_score')
            train_score = metrics.get('train_score')
            
            if val_score and val_score < 0.5:
                instructions.append("- Improve feature engineering")
                instructions.append("- Increase FLAML time budget")
                instructions.append("- Try different preprocessing strategies")
            
            if train_score and val_score:
                gap = train_score - val_score
                if gap > 0.15:
                    instructions.append("- Add regularization (increase dropout, weight decay)")
                    instructions.append("- Reduce model complexity")
                    instructions.append("- Add more aggressive data augmentation")
        
        if not instructions:
            instructions.append("- Review and optimize the overall approach")
            instructions.append("- Check for data quality issues")
        
        return '\n'.join(instructions)
    
    def _parse_enhancement_response(self, response_text: str, original_code: str) -> Tuple[Optional[str], list]:
        """Parse LLM response to extract improvements and enhanced code."""
        improvements = []
        enhanced_code = None
        
        # Extract improvements
        improvements_match = re.search(r'IMPROVEMENTS:\s*\n((?:[-•*]\s*.+\n?)+)', response_text)
        if improvements_match:
            improvements_text = improvements_match.group(1)
            for line in improvements_text.split('\n'):
                line = line.strip()
                if line and line[0] in '-•*':
                    improvements.append(line.lstrip('-•* '))
        
        # Extract enhanced code
        code_match = re.search(r'ENHANCED_CODE:\s*\n```python\s*\n(.*?)```', response_text, re.DOTALL)
        if code_match:
            enhanced_code = code_match.group(1).strip()
        else:
            # Try alternative patterns
            code_match = re.search(r'```python\s*\n(.*?)```', response_text, re.DOTALL)
            if code_match:
                enhanced_code = code_match.group(1).strip()
        
        # Validate the enhanced code is substantially different and not empty
        if enhanced_code:
            if len(enhanced_code) < 100:
                logger.warning("Enhanced code too short, rejecting")
                enhanced_code = None
            elif enhanced_code == original_code:
                logger.warning("Enhanced code identical to original, rejecting")
                enhanced_code = None
        
        return enhanced_code, improvements
    
    def reset(self):
        """Reset iteration counter for a new training session."""
        self.current_iteration = 0
        logger.info("CodeEnhancer reset for new session")

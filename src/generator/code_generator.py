"""Code generator with Gemini integration for dynamic training code generation"""

import ast
import json
import logging
import tempfile
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path

from src.models.data_models import Strategy, DatasetProfile, ValidationResult
from src.templates.template_manager import get_template_manager
from src.utils.gemini_client import GeminiClient
from src.detector.modality_detector import detect_submission_format

logger = logging.getLogger(__name__)


class HybridCodeGenerator:
    """
    Generates runnable training code from templates and Gemini enhancement.
    
    Combines modality-specific templates with Gemini's contextual understanding
    to produce dataset-specific training code with proper error handling and
    resource management.
    """
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """
        Initialize code generator with Gemini client and template manager.
        
        Args:
            gemini_client: Gemini client for code enhancement (optional)
        """
        self.gemini_client = gemini_client
        self.template_manager = get_template_manager()
        logger.info("HybridCodeGenerator initialized")
    
    def load_template(
        self, 
        modality: str, 
        profile: DatasetProfile, 
        strategy: Strategy
    ) -> str:
        """
        Load appropriate template for the given modality and constraints.
        
        Args:
            modality: Dataset modality (tabular, image, text, sequence, multimodal)
            profile: Dataset profile with characteristics
            strategy: Selected strategy configuration
            
        Returns:
            Template string with placeholders
            
        Raises:
            ValueError: If modality is not supported
        """
        logger.info(f"Loading template for modality: {modality}")
        
        # Determine if resource-constrained variant is needed
        resource_constrained = (
            profile.estimated_gpu_memory_gb > 20.0 or
            profile.memory_gb > 400.0 or
            strategy.batch_size < 16 or
            profile.num_samples > 500000  # Large dataset detection
        )
        
        # Log large dataset detection
        if profile.num_samples > 500000:
            logger.info(f"Large dataset detected: {profile.num_samples:,} rows - using lightweight template")
        
        # Get appropriate template
        if modality == 'tabular':
            template = self.template_manager.get_template(
                modality=modality,
                memory_gb=profile.memory_gb,
                resource_constrained=resource_constrained,
                num_samples=profile.num_samples
            )
        elif modality == 'text':
            use_fallback = strategy.fallback_model is not None and 'tfidf' in strategy.fallback_model.lower()
            template = self.template_manager.get_template(
                modality=modality,
                use_fallback=use_fallback,
                resource_constrained=resource_constrained,
                num_samples=profile.num_samples
            )
        else:
            template = self.template_manager.get_template(
                modality=modality,
                resource_constrained=resource_constrained,
                num_samples=profile.num_samples
            )
        
        logger.info(f"Template loaded successfully (resource_constrained={resource_constrained})")
        return template
    
    def enhance_with_gemini(
        self,
        template: str,
        strategy: Strategy,
        profile: DatasetProfile
    ) -> str:
        """
        Enhance template with Gemini for dataset-specific adaptation.
        
        Args:
            template: Base template string
            strategy: Selected strategy configuration
            profile: Dataset profile with characteristics
            
        Returns:
            Enhanced code string
            
        Raises:
            RuntimeError: If Gemini client is not available
        """
        if not self.gemini_client:
            logger.warning("No Gemini client available, returning template as-is")
            return template
        
        logger.info(f"Enhancing template with Gemini")
        
        # Construct comprehensive prompt
        prompt = self._build_enhancement_prompt(
            template, 
            strategy, 
            profile
        )
        
        try:
            # Generate enhanced code
            response = self.gemini_client.generate_content(prompt)
            enhanced_code = self._extract_code_from_response(response.text)
            
            logger.info("Template enhanced successfully with Gemini")
            return enhanced_code
        except Exception as e:
            logger.error(f"Gemini enhancement failed: {e}")
            logger.warning("Falling back to original template")
            return template
    
    def _build_enhancement_prompt(
        self,
        template: str,
        strategy: Strategy,
        profile: DatasetProfile
    ) -> str:
        """
        Build comprehensive prompt for Gemini enhancement.
        
        Args:
            template: Base template string
            strategy: Selected strategy configuration
            profile: Dataset profile
            
        Returns:
            Formatted prompt string
        """
        # Build dataset context
        dataset_context = f"""
DATASET SOURCE: Custom Dataset
Dataset Path: {profile.dataset_path if hasattr(profile, 'dataset_path') else 'N/A'}

IMPORTANT - DATASET CHARACTERISTICS:
- This is a custom dataset that may require preprocessing
- Data cleaning and validation are critical
- Handle missing values, outliers, and data type inconsistencies
- Implement robust preprocessing pipelines
- Add extensive error handling for data quality issues
"""
        
        # Determine if this is a large dataset requiring lightweight approach
        is_large_dataset = profile.num_samples > 500000
        has_many_features = profile.num_features > 50 if profile.num_features else False
        is_very_large = profile.num_samples > 900000
        
        # Build large dataset warning if applicable
        large_dataset_warning = ""
        if is_large_dataset:
            large_dataset_warning = f"""
⚠️ LARGE DATASET DETECTED - CRITICAL OPTIMIZATION REQUIRED ⚠️
Dataset Size: {profile.num_samples:,} rows x {profile.num_features} features

MANDATORY LIGHTWEIGHT OPTIMIZATIONS - YOU MUST FOLLOW THESE:
1. **NO CROSS-VALIDATION**: Use simple train/validation split (80/20 or 90/10)
   - Replace any k-fold CV with a single holdout split
   - Set n_splits=1 or use train_test_split directly
   
2. **REDUCE FLAML TIME BUDGET**: Use max 300-600 seconds for large data
   - Reduce time_budget parameter significantly
   
3. **USE LIGHTER MODELS ONLY**: 
   - Prefer LightGBM, XGBoost with shallow trees (max_depth=6-8)
   - Avoid CatBoost (slower on large data)
   - Avoid RandomForest (memory intensive)
   - Set estimator_list=['lgbm', 'xgboost'] only
   
4. **LIMIT FEATURE ENGINEERING**:
   - NO polynomial features (exponential memory growth)
   - NO interaction features (too many combinations)
   - Only basic preprocessing (imputation, encoding)
   
5. **MEMORY OPTIMIZATION**:
   - Convert float64 to float32
   - Convert int64 to int32 where possible
   - Drop unnecessary columns immediately
   - Use chunked processing if needed
   
6. **REDUCE HYPERPARAMETER SEARCH**:
   - Limit n_estimators to 100-200 max
   - Use larger learning rates (0.1-0.3)
   - Limit tree depth to 6-8
   - Set early_stopping_rounds=10-20
   
7. **SAMPLING STRATEGY**:
   - Consider training on a stratified sample (30-50% of data)
   - Then predict on full test set

{"8. **EXTREME OPTIMIZATION FOR 900K+ ROWS**: Use only 30% sample for training, single model (lgbm), no hyperparameter search, fixed parameters." if is_very_large else ""}
"""
        
        prompt = f"""You are an expert ML engineer specializing in automated machine learning and code optimization.
Enhance the following training code template with SIMPLE, FOCUSED improvements.
{large_dataset_warning}

IMPORTANT: This template uses FLAML AutoML which already handles:
- Model selection (lgbm, xgboost, catboost, rf, etc.)
- Hyperparameter optimization
- Ensemble creation
- {"⚠️ Cross-validation - BUT DISABLE THIS FOR LARGE DATASETS" if is_large_dataset else "Cross-validation"}
- Task type detection (classification vs regression)

Your job is to add SIMPLE enhancements like feature engineering, NOT to add complex ensemble code or manual model training.
{"⚠️ CRITICAL: This is a LARGE DATASET - prioritize speed and memory efficiency over accuracy!" if is_large_dataset else ""}

{dataset_context}

DATASET PROFILE:
- Modality: {profile.modality}
- Samples: {profile.num_samples}
- Features: {profile.num_features}
- Target Type: {profile.target_type}
- Class Imbalance Ratio: {profile.class_imbalance_ratio}
- Memory Usage: {profile.memory_gb:.2f} GB
- Estimated GPU Memory: {profile.estimated_gpu_memory_gb:.2f} GB
- Has Metadata: {profile.has_metadata}

TASK TYPE DETECTION:
- The template includes automatic task type detection (classification vs regression)
- For classification: Uses appropriate metrics (roc_auc for binary, log_loss for multi-class)
- For regression: Uses appropriate metrics (r2, rmse)
- You can enhance the task detection logic if needed, but the basic detection is already in place

STRATEGY:
- Primary Model: {strategy.primary_model}
- Fallback Model: {strategy.fallback_model}
- Loss Function: {strategy.loss_function}
- Optimizer: {strategy.optimizer}
- Batch Size: {strategy.batch_size}
- Max Epochs: {strategy.max_epochs}
- Learning Rate: {strategy.learning_rate}
- Weight Decay: {strategy.weight_decay}
- Mixed Precision: {strategy.mixed_precision}
- Gradient Accumulation: {strategy.gradient_accumulation_steps}

RESOURCE CONSTRAINTS:
- Max VRAM: {strategy.resource_constraints.max_vram_gb} GB
- Max RAM: {strategy.resource_constraints.max_ram_gb} GB
- Max Runtime: {strategy.resource_constraints.max_runtime_hours} hours

TEMPLATE CODE:
```python
{template}
```

CRITICAL REQUIREMENTS - DO NOT VIOLATE THESE:
1. **PRESERVE ALL EXISTING DATA TYPE HANDLING** - The template already correctly separates numeric and categorical columns
2. **NEVER REMOVE** the lines that do: `numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()`
3. **NEVER REMOVE** the lines that do: `categorical_cols = X.select_dtypes(include=['object']).columns.tolist()`
4. **NEVER APPLY** median imputation to categorical/string columns - keep separate imputers
5. **PRESERVE** the separate imputation logic: SimpleImputer(strategy='median') for numeric, SimpleImputer(strategy='most_frequent') for categorical
6. **PRESERVE** the label encoding loop for categorical features
7. **PRESERVE** the column name string conversion: `X_imputed.columns = X_imputed.columns.astype(str)` - this is critical for sklearn compatibility
8. **DO NOT MODIFY** the ID column handling logic - it's already correct
9. **DO NOT CHANGE** how test_ids are saved or how the submission DataFrame is created
10. **KEEP** all existing error handling and data validation

WHAT YOU CAN ENHANCE (BE CREATIVE BUT SIMPLE!):
{"1. ONLY basic preprocessing - no feature engineering for large datasets" if is_large_dataset else "1. Add simple feature engineering (polynomial features degree 2, basic interactions)"}
2. Improve FLAML configuration (time budget, estimator list, ensemble settings)
3. Add basic logging for debugging
{"4. USE HOLDOUT VALIDATION ONLY - no cross-validation for large datasets" if is_large_dataset else "4. Add simple validation strategies if helpful"}
{"5. ADD MEMORY OPTIMIZATION - convert dtypes, drop columns early" if is_large_dataset else "5. Optimize memory usage if needed"}

WHAT YOU MUST NOT DO - CRITICAL:
1. DO NOT consolidate the separate numeric/categorical imputation into a single imputer
2. DO NOT remove the data type separation logic
3. DO NOT change the core preprocessing pipeline structure
4. DO NOT modify test_ids or submission generation code
5. DO NOT add if/else blocks for classification vs regression - the template already handles this automatically
6. DO NOT add complex ensemble code with VotingClassifier/VotingRegressor - FLAML already does ensembling
7. DO NOT train multiple models manually (lgbm, xgb, catboost) - FLAML handles this
8. DO NOT add stacking, blending, or manual ensemble logic - keep it simple
9. DO NOT add model calibration code - unnecessary complexity
10. DO NOT add complex cross-validation loops - FLAML handles this internally
{"11. ⚠️ DO NOT USE CROSS-VALIDATION - use holdout split only for this large dataset" if is_large_dataset else ""}
{"12. ⚠️ DO NOT ADD POLYNOMIAL FEATURES - memory will explode with large data" if is_large_dataset else ""}
{"13. ⚠️ DO NOT USE CATBOOST OR RANDOM FOREST - too slow/memory intensive for large data" if is_large_dataset else ""}

INSTRUCTIONS:
1. Keep the existing preprocessing pipeline EXACTLY as is - it already handles mixed types correctly
2. The template already detects task type (classification vs regression) - DO NOT add if/else blocks for this
3. FLAML already handles model selection and ensembling - DO NOT manually train multiple models
4. Keep enhancements SIMPLE - focus on feature engineering and FLAML configuration only
5. DO NOT add complex ensemble code, stacking, or manual model training
6. Return ONLY the enhanced Python code, no explanations

REMEMBER: The goal is to make the code BETTER, not MORE COMPLEX. Simple is better than complex.

Enhanced code:"""
        
        return prompt
    
    def _extract_code_from_response(self, response_text: str) -> str:
        """
        Extract Python code from Gemini response.
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Extracted code string
        """
        # Try to find code blocks
        if "```python" in response_text:
            start = response_text.find("```python") + 9
            end = response_text.find("```", start)
            if end != -1:
                return response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end != -1:
                return response_text[start:end].strip()
        
        # If no code blocks found, return the whole response
        return response_text.strip()
    
    def validate_code(self, code: str) -> ValidationResult:
        """
        Validate generated code for syntax and common errors.
        
        Args:
            code: Python code string to validate
            
        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        logger.info("Validating generated code")
        
        errors = []
        warnings = []
        
        # Check for syntax errors using AST
        try:
            ast.parse(code)
            logger.info("Code syntax is valid")
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        # Check for common issues
        code_lower = code.lower()
        
        # Check for required imports
        required_imports = {
            'pandas': 'import pandas',
            'numpy': 'import numpy',
        }
        
        for lib, import_stmt in required_imports.items():
            if lib in code_lower and import_stmt not in code_lower:
                warnings.append(f"Missing import statement for {lib}")
        
        # Check for submission file generation
        if 'submission.csv' not in code:
            errors.append("Code does not generate submission.csv file")
        
        # Check for proper error handling in critical sections
        if 'try:' not in code and 'except' not in code:
            warnings.append("Code lacks error handling (try/except blocks)")
        
        # Check for GPU usage if CUDA is mentioned
        if 'cuda' in code_lower and 'torch.cuda.is_available()' not in code:
            warnings.append("Code uses CUDA but doesn't check availability")
        
        # Check for data loading
        if 'read_csv' not in code and 'load' not in code_lower:
            errors.append("Code does not appear to load data")
        
        is_valid = len(errors) == 0
        
        logger.info(f"Validation complete: valid={is_valid}, errors={len(errors)}, warnings={len(warnings)}")
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
    
    def generate_training_code(
        self,
        strategy: Strategy,
        modality: str,
        profile: DatasetProfile,
        dataset_info: Dict[str, Any]
    ) -> str:
        """
        Generate complete training code for the given strategy and dataset.
        
        This is the main orchestration method that:
        1. Loads the appropriate template
        2. Enhances it with Gemini (if available)
        3. Fills in dataset-specific values
        4. Validates the generated code
        5. Requests fixes if validation fails
        
        Args:
            strategy: Selected ML strategy
            modality: Dataset modality
            profile: Dataset profile
            dataset_info: Dictionary with dataset-specific information
                (paths, column names, etc.)
            
        Returns:
            Complete, validated training code
            
        Raises:
            ValueError: If code generation or validation fails after retries
        """
        logger.info(f"Generating training code for {modality} modality")
        
        # Step 1: Load template
        template = self.load_template(modality, profile, strategy)
        
        # Step 2: Fill in dataset-specific values FIRST (before Gemini)
        # This ensures placeholders are filled before Gemini sees them
        filled_template = self._fill_template_values(template, strategy, dataset_info)
        
        # Step 3: Enhance with Gemini (if available)
        # Gemini enhances the already-filled code
        # Templates already have robust data type handling - Gemini should preserve it
        if self.gemini_client:
            filled_code = self.enhance_with_gemini(
                filled_template, 
                strategy, 
                profile
            )
        else:
            filled_code = filled_template
        
        # Step 4: Validate the code
        validation_result = self.validate_code(filled_code)
        
        # Step 5: Request fixes if validation fails
        max_fix_attempts = 3
        attempt = 0
        
        while not validation_result.is_valid and attempt < max_fix_attempts:
            attempt += 1
            logger.warning(f"Validation failed (attempt {attempt}/{max_fix_attempts})")
            logger.warning(f"Errors: {validation_result.errors}")
            
            if self.gemini_client:
                filled_code = self._request_gemini_fix(filled_code, validation_result)
                validation_result = self.validate_code(filled_code)
            else:
                # Without Gemini, we can't fix automatically
                break
        
        if not validation_result.is_valid:
            error_msg = f"Code validation failed after {max_fix_attempts} attempts: {validation_result.errors}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if validation_result.warnings:
            logger.warning(f"Code has warnings: {validation_result.warnings}")
        
        logger.info("Training code generated successfully")
        return filled_code
    
    def _fill_template_values(
        self,
        template: str,
        strategy: Strategy,
        dataset_info: Dict[str, Any]
    ) -> str:
        """
        Fill template placeholders with actual values.
        
        Args:
            template: Template string with {placeholder} markers
            strategy: Strategy configuration
            dataset_info: Dataset-specific information
            
        Returns:
            Filled template string
        """
        # Normalize paths to use forward slashes (works on both Windows and Unix)
        def normalize_path(path: str) -> str:
            """Convert Windows backslashes to forward slashes for cross-platform compatibility"""
            if path:
                return path.replace('\\', '/')
            return path
        
        # Auto-detect submission format from sample_submission.csv
        submission_format = {}
        dataset_path = dataset_info.get('dataset_path', './data')
        try:
            submission_format = detect_submission_format(dataset_path)
            logger.info(f"Auto-detected submission format: {submission_format}")
        except Exception as e:
            logger.warning(f"Could not auto-detect submission format: {e}")
        
        # Combine strategy and dataset info with comprehensive defaults
        values = {
            # Strategy values
            'batch_size': strategy.batch_size,
            'max_epochs': strategy.max_epochs,
            'learning_rate': strategy.learning_rate,
            'weight_decay': strategy.weight_decay,
            'early_stopping_patience': strategy.early_stopping_patience,
            'loss_function': strategy.loss_function,
            'gradient_accumulation_steps': strategy.gradient_accumulation_steps,
            'primary_model': strategy.primary_model,
            'optimizer': strategy.optimizer,
            
            # Common defaults for tabular
            # FLAML time budget: if no limit (0), use 1 hour default; otherwise use 60% of max runtime
            'time_budget': 3600 if strategy.resource_constraints.max_runtime_hours == 0 else max(300, int(strategy.resource_constraints.max_runtime_hours * 3600 * 0.6)),
            'max_depth': 10,
            'seed': 42,
            
            # Task and metric defaults
            'metric': 'accuracy',
            'task_type': 'classification',
            'objective': 'binary',
            
            # Auto-detected submission format values (with fallbacks)
            'prediction_column': submission_format.get('prediction_column', dataset_info.get('prediction_column', 'prediction')),
            'id_column': submission_format.get('id_column', dataset_info.get('id_column', 'id')),
            
            # Common defaults for paths (normalized)
            'train_path': normalize_path(dataset_info.get('train_path', dataset_info.get('dataset_path', './data') + '/train.csv')),
            'test_path': normalize_path(dataset_info.get('test_path', dataset_info.get('dataset_path', './data') + '/test.csv')),
            'output_dir': normalize_path(dataset_info.get('output_dir', './output')),
            'dataset_path': normalize_path(dataset_info.get('dataset_path', './data')),
            
            # Common defaults for columns
            'target_column': submission_format.get('prediction_column', dataset_info.get('target_column', 'target')),
        }
        
        # Add submission format metadata if detected
        if submission_format.get('expected_format'):
            values['expected_format'] = submission_format.get('expected_format')
        if submission_format.get('num_rows'):
            values['expected_num_rows'] = submission_format.get('num_rows')
        
        # Add dataset_info values, normalizing any path-like strings
        for key, value in dataset_info.items():
            if key not in values:  # Don't override already normalized values
                if isinstance(value, str) and ('path' in key.lower() or 'dir' in key.lower()):
                    values[key] = normalize_path(value)
                else:
                    values[key] = value
        
        # Add optional values
        if strategy.gradient_clip_norm:
            values['gradient_clip_norm'] = strategy.gradient_clip_norm
        
        # Try to fill template, handling missing keys gracefully
        try:
            filled = template.format(**values)
            return filled
        except KeyError as e:
            logger.error(f"Missing template value: {e}")
            # Add the missing value with a default and try again
            missing_key = str(e).strip("'")
            if missing_key not in values:
                # Provide sensible defaults for common missing keys
                defaults = {
                    'metric': 'accuracy',
                    'task_type': 'classification',
                    'prediction_column': 'prediction',
                    'model_name': 'model',
                    'n_jobs': -1
                }
                if missing_key in defaults:
                    values[missing_key] = defaults[missing_key]
                    logger.warning(f"Using default value for {missing_key}: {defaults[missing_key]}")
                    try:
                        filled = template.format(**values)
                        return filled
                    except KeyError:
                        pass  # Try safe_substitute below
            
            # Try to fill with safe_substitute approach (uses $variable instead of {variable})
            import string
            try:
                # Convert {variable} to $variable for safe_substitute
                import re
                template_dollar = re.sub(r'\{(\w+)\}', r'$\1', template)
                filled = string.Template(template_dollar).safe_substitute(**values)
                logger.warning(f"Used safe_substitute for template filling")
                return filled
            except Exception as e2:
                logger.error(f"Safe substitute also failed: {e2}")
                # Last resort: try to fill what we can
                for key, value in values.items():
                    template = template.replace(f'{{{key}}}', str(value))
                return template
    
    def _request_gemini_fix(self, code: str, validation_result: ValidationResult) -> str:
        """
        Request Gemini to fix validation errors.
        
        Args:
            code: Code with validation errors
            validation_result: Validation result with errors
            
        Returns:
            Fixed code string
        """
        logger.info("Requesting Gemini to fix validation errors")
        
        prompt = f"""The following Python code has validation errors. Please fix them.

ERRORS:
{chr(10).join(f"- {error}" for error in validation_result.errors)}

WARNINGS:
{chr(10).join(f"- {warning}" for warning in validation_result.warnings)}

CODE:
```python
{code}
```

Please provide the corrected code. Return ONLY the fixed Python code, no explanations.

Fixed code:"""
        
        try:
            response = self.gemini_client.generate_content(prompt)
            fixed_code = self._extract_code_from_response(response.text)
            logger.info("Received fixed code from Gemini")
            return fixed_code
        except Exception as e:
            logger.error(f"Failed to get fix from Gemini: {e}")
            return code
    
    def execute_dry_run(
        self,
        code: str,
        dataset_path: str,
        subset_size: int = 100
    ) -> tuple[bool, str, str]:
        """
        Execute a dry run of the code on a data subset.
        
        Args:
            code: Training code to execute
            dataset_path: Path to dataset
            subset_size: Number of samples to use for dry run
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        logger.info(f"Executing dry run with subset_size={subset_size}")
        
        # Create temporary directory for dry run
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Write code to file
            code_file = tmpdir_path / "train.py"
            code_file.write_text(code)
            
            # TODO: Create subset of data
            # For now, we'll just try to run the code with a timeout
            
            try:
                # Run with timeout
                result = subprocess.run(
                    ['python', str(code_file)],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=60  # 1 minute timeout for dry run
                )
                
                success = result.returncode == 0
                stdout = result.stdout
                stderr = result.stderr
                
                if success:
                    logger.info("Dry run completed successfully")
                else:
                    logger.warning(f"Dry run failed with return code {result.returncode}")
                
                return success, stdout, stderr
                
            except subprocess.TimeoutExpired:
                logger.warning("Dry run timed out after 60 seconds")
                return False, "", "Dry run timed out"
            except Exception as e:
                logger.error(f"Dry run execution failed: {e}")
                return False, "", str(e)

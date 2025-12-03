"""Code generator with Gemini integration for dynamic training code generation"""

import ast
import json
import logging
import tempfile
import subprocess
import os
from typing import Dict, Any, Optional
from pathlib import Path

from src.models.data_models import Strategy, DatasetProfile, ValidationResult
from src.templates.template_manager import get_template_manager
from src.utils.gemini_client import GeminiClient
from src.detector.modality_detector import detect_submission_format

logger = logging.getLogger(__name__)


def get_competition_context(dataset_path: str) -> Dict[str, Any]:
    """
    Read competition context from mlebench data directory.
    
    Reads description.md, sample_submission.csv, and data files to provide
    competition-specific context to Gemini for better code generation.
    
    Args:
        dataset_path: Path to dataset directory (mlebench public dir)
        
    Returns:
        Dictionary with competition context:
        - description: Competition description text (truncated)
        - sample_submission: Sample submission format info
        - evaluation_metric: Detected evaluation metric if mentioned
        - train_head: First few rows of training data (to understand columns)
        - test_head: First few rows of test data (to see what columns are available at inference)
    """
    context = {
        "description": None,
        "sample_submission_preview": None,
        "sample_submission_columns": None,
        "evaluation_metric": None,
        "competition_type": None,
        "train_head": None,
        "train_columns": None,
        "test_head": None,
        "test_columns": None
    }
    
    dataset_path = Path(dataset_path)
    
    # Read description.md
    description_path = dataset_path / "description.md"
    if description_path.exists():
        try:
            # Try multiple encodings for description.md
            desc = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(description_path, 'r', encoding=encoding) as f:
                        desc = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if desc is None:
                raise ValueError("Could not decode description.md with any encoding")
            
            # Truncate to reasonable size for prompt (first 2000 chars)
            context["description"] = desc[:2000] if len(desc) > 2000 else desc
            
            # Try to detect evaluation metric from description
            desc_lower = desc.lower()
            if 'auc' in desc_lower or 'roc' in desc_lower:
                context["evaluation_metric"] = "AUC-ROC"
            elif 'accuracy' in desc_lower:
                context["evaluation_metric"] = "accuracy"
            elif 'rmse' in desc_lower or 'root mean squared' in desc_lower:
                context["evaluation_metric"] = "RMSE"
            elif 'mae' in desc_lower or 'mean absolute' in desc_lower:
                context["evaluation_metric"] = "MAE"
            elif 'log loss' in desc_lower or 'logloss' in desc_lower:
                context["evaluation_metric"] = "log_loss"
            elif 'f1' in desc_lower:
                context["evaluation_metric"] = "F1"
                
            # Detect competition type
            if 'classification' in desc_lower or 'binary' in desc_lower:
                context["competition_type"] = "classification"
            elif 'regression' in desc_lower:
                context["competition_type"] = "regression"
            elif 'segmentation' in desc_lower:
                context["competition_type"] = "segmentation"
                
            logger.info(f"Loaded competition description ({len(desc)} chars)")
        except Exception as e:
            logger.warning(f"Could not read description.md: {e}")
    
    # Read sample_submission.csv
    sample_sub_path = dataset_path / "sample_submission.csv"
    if sample_sub_path.exists():
        try:
            import pandas as pd
            sample_sub = pd.read_csv(sample_sub_path, nrows=5)
            context["sample_submission_columns"] = list(sample_sub.columns)
            context["sample_submission_preview"] = sample_sub.to_string(index=False)
            logger.info(f"Loaded sample submission: columns={sample_sub.columns.tolist()}")
        except Exception as e:
            logger.warning(f"Could not read sample_submission.csv: {e}")
    
    # Read train.csv head - CRITICAL for understanding what columns are available for training
    train_path = dataset_path / "train.csv"
    if train_path.exists():
        try:
            import pandas as pd
            train_df = pd.read_csv(train_path, nrows=5)
            context["train_columns"] = list(train_df.columns)
            context["train_head"] = train_df.to_string(index=False)
            logger.info(f"Loaded train.csv head: columns={train_df.columns.tolist()}")
        except Exception as e:
            logger.warning(f"Could not read train.csv: {e}")
    
    # Read test.csv head - CRITICAL for understanding what columns are available at inference time
    # This is important because test data often has fewer columns (no target, no class labels, etc.)
    test_path = dataset_path / "test.csv"
    if test_path.exists():
        try:
            import pandas as pd
            test_df = pd.read_csv(test_path, nrows=5)
            context["test_columns"] = list(test_df.columns)
            context["test_head"] = test_df.to_string(index=False)
            logger.info(f"Loaded test.csv head: columns={test_df.columns.tolist()}")
            
            # Highlight column differences between train and test
            if context.get("train_columns"):
                train_cols = set(context["train_columns"])
                test_cols = set(context["test_columns"])
                context["columns_only_in_train"] = list(train_cols - test_cols)
                context["columns_only_in_test"] = list(test_cols - train_cols)
                if context["columns_only_in_train"]:
                    logger.info(f"Columns ONLY in train (not in test): {context['columns_only_in_train']}")
        except Exception as e:
            logger.warning(f"Could not read test.csv: {e}")
    
    return context


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
            # Detect if this is a text normalization task (for seq2seq modality)
            is_text_norm = False
            if modality in ['seq2seq', 'sequence']:
                # Check for text normalization columns using data_types keys
                data_types = getattr(profile, 'data_types', None) or {}
                columns = list(data_types.keys()) if data_types else []
                columns_lower = {c.lower() for c in columns}
                has_before_after = 'before' in columns_lower and 'after' in columns_lower
                has_class = 'class' in columns_lower
                is_text_norm = has_before_after or has_class
                if is_text_norm:
                    logger.info("Text normalization task detected - using specialized template")
            
            template = self.template_manager.get_template(
                modality=modality,
                resource_constrained=resource_constrained,
                num_samples=profile.num_samples,
                is_text_normalization=is_text_norm
            )
        
        logger.info(f"Template loaded successfully (resource_constrained={resource_constrained})")
        return template
    
    def enhance_with_gemini(
        self,
        template: str,
        strategy: Strategy,
        profile: DatasetProfile,
        dataset_path: str = None
    ) -> str:
        """
        Enhance template with Gemini for dataset-specific adaptation.

        Args:
            template: Base template string
            strategy: Selected strategy configuration
            profile: Dataset profile with characteristics
            dataset_path: Path to dataset for reading competition context

        Returns:
            Enhanced code string

        Raises:
            RuntimeError: If Gemini client is not available
        """
        if not self.gemini_client:
            error_msg = "Gemini client is required but not available! Cannot generate code without LLM."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        print("[GEMINI] Preparing prompt for Gemini...")
        logger.info("Enhancing template with Gemini")
        
        # Load competition context from mlebench data if available
        competition_context = None
        if dataset_path:
            competition_context = get_competition_context(dataset_path)
            if competition_context.get("description"):
                print("[GEMINI] Loaded competition description for context")
            if competition_context.get("sample_submission_columns"):
                print(f"[GEMINI] Sample submission columns: {competition_context['sample_submission_columns']}")

        # Build prompt and call Gemini client
        prompt = self._build_enhancement_prompt(template, strategy, profile, competition_context)
        try:
            response = self.gemini_client.generate_content(prompt)
            generated_code = self._extract_code_from_response(response.text)
            print(f"[GEMINI] Received LLM-generated code ({len(generated_code)} chars)")
            logger.info("Code generated successfully by Gemini LLM")
            return generated_code
        except Exception as e:
            error_msg = f"Gemini code generation failed: {e}. LLM is required, no fallback."
            logger.error(error_msg)
            print(f"[GEMINI ERROR] {error_msg}")
            raise RuntimeError(error_msg)
    
    def _build_enhancement_prompt(
        self,
        template: str,
        strategy: Strategy,
        profile: DatasetProfile,
        competition_context: Dict[str, Any] = None
    ) -> str:
        """
        Build JSON-based prompt for Gemini code generation.

        Uses structured context with high signal-to-noise ratio.
        Includes competition description and sample submission if available.

        Args:
            template: Base template string
            strategy: Selected strategy configuration
            profile: Dataset profile
            competition_context: Competition context from mlebench (description, sample submission)

        Returns:
            Formatted prompt string (JSON)
        """

        payload = {
            "task": "enhance_template_code",
            "description": "Enhance the provided training template for this ML competition. Return ONLY the Python code string.",
            "dataset": {
                "modality": profile.modality,
                "num_samples": int(profile.num_samples or 0),
                "num_features": int(profile.num_features or 0),
                "memory_gb": float(getattr(profile, 'memory_gb', 0.0) or 0.0),
                "target_type": getattr(profile, 'target_type', None),
                "class_imbalance_ratio": getattr(profile, 'class_imbalance_ratio', None)
            },
            "strategy": {
                "primary_model": strategy.primary_model,
                "fallback_model": strategy.fallback_model,
                "time_budget": int(getattr(strategy, 'resource_constraints').max_runtime_hours * 3600) if getattr(strategy, 'resource_constraints', None) else None,
                "seed": getattr(strategy, 'seed', None)
            },
            "template": template,
            "rules": {
                "output_format": "python_code_only"
            }
        }
        
        # Add competition context if available - this gives LLM more freedom to adapt
        has_competition_context = competition_context and competition_context.get("description")
        
        if competition_context:
            payload["competition"] = {}
            if competition_context.get("description"):
                payload["competition"]["description"] = competition_context["description"]
            if competition_context.get("evaluation_metric"):
                payload["competition"]["evaluation_metric"] = competition_context["evaluation_metric"]
            if competition_context.get("competition_type"):
                payload["competition"]["type"] = competition_context["competition_type"]
            if competition_context.get("sample_submission_columns"):
                payload["competition"]["submission_columns"] = competition_context["sample_submission_columns"]
            if competition_context.get("sample_submission_preview"):
                payload["competition"]["submission_preview"] = competition_context["sample_submission_preview"]
            
            # CRITICAL: Add train and test data heads so LLM knows column differences
            if competition_context.get("train_head"):
                payload["competition"]["train_data_preview"] = competition_context["train_head"]
                payload["competition"]["train_columns"] = competition_context["train_columns"]
            if competition_context.get("test_head"):
                payload["competition"]["test_data_preview"] = competition_context["test_head"]
                payload["competition"]["test_columns"] = competition_context["test_columns"]
            
            # Explicitly highlight column differences - this is critical for seq2seq/text tasks
            if competition_context.get("columns_only_in_train"):
                payload["competition"]["columns_only_in_train"] = competition_context["columns_only_in_train"]
                payload["competition"]["WARNING"] = (
                    f"These columns are in train.csv but NOT in test.csv: {competition_context['columns_only_in_train']}. "
                    "You cannot use these columns during inference! Design your model accordingly."
                )
        
        # Build instruction based on whether we have competition context
        if has_competition_context:
            # WITH competition description: Give LLM freedom to redesign for better score
            instruction = (
                "You are an expert ML engineer competing in a Kaggle-style competition. "
                "Your goal is to achieve the HIGHEST POSSIBLE SCORE on the leaderboard. "
                "\n\n"
                "CRITICAL: Read the competition description carefully and understand:\n"
                "1. What the evaluation metric is and how to optimize for it\n"
                "2. What the expected output format is from sample_submission\n"
                "3. What domain-specific techniques could improve performance\n"
                "\n"
                "**IMPORTANT DATA INSIGHT:**\n"
                "- Look at train_data_preview and test_data_preview to see actual data samples\n"
                "- Look at columns_only_in_train - these columns exist in training but NOT in test\n"
                "- Your inference code CANNOT use columns that don't exist in test data!\n"
                "- Design your model to only use features available in test.csv\n"
                "\n"
                "**CRITICAL - FOLLOW THE TEMPLATE STRUCTURE:**\n"
                "- You MUST use the SAME ML framework/library shown in the template\n"
                "- If template uses LightAutoML (TabularAutoML), you MUST use LightAutoML\n"
                "- If template uses FLAML, you MUST use FLAML\n"
                "- If template uses T5, you MUST use T5\n"
                "- Keep the overall code structure and flow from the template\n"
                "- DO NOT switch between AutoML frameworks!\n"
                "\n"
                "YOU CAN adapt:\n"
                "- Fill in column names based on train/test data preview\n"
                "- Add feature engineering specific to the data\n"
                "- Adjust hyperparameters (timeout, folds, learning_rate, etc.)\n"
                "- Add data preprocessing as needed\n"
                "- Modify the feature engineering function\n"
                "- The template provided is a STARTING POINT only. You can restructure it significantly to improve performance.\n"
                "\n"
                "YOU CANNOT:\n"
                "- Switch to a different ML framework (e.g., FLAML instead of LightAutoML)\n"
                "- Completely replace the model training section\n"
                "- Remove core template components\n"
                "\n"
                "REQUIREMENTS:\n"
                "- Output must be a complete, runnable Python script\n"
                "- Must generate submission.csv with correct column names from sample_submission\n"
                "- Must handle the data paths provided in the template\n"
                "- Return ONLY the Python code, no markdown or explanations\n"
            )
        else:
            # WITHOUT competition description: Make minimal changes
            instruction = (
                "You are an ML engineer. Receive the JSON payload and return ONLY an enhanced Python code string. "
                "Make MINIMAL changes (<=5% of lines). Preserve hyperparameters and feature engineering. "
                "Do not add contest-specific hardcoding. If hyperparameter tuning is desired, prefer AutoML calls (FLAML/Optuna) rather than replacing constants. "
                "Return the full valid Python script as plain text (no markdown)."
            )

        prompt = json.dumps({"payload": payload, "instruction": instruction})
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
        1. Loads the appropriate template as reference
        2. Uses Gemini LLM to generate dataset-specific code (REQUIRED)
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
            RuntimeError: If Gemini client is not available (LLM is required)
        """
        logger.info(f"Generating training code for {modality} modality with LLM")
        
        # Step 1: Verify LLM is available (REQUIRED - no hardcoded templates)
        if not self.gemini_client:
            error_msg = "Gemini LLM client is REQUIRED for code generation. No hardcoded fallback allowed."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Step 2: Load template as reference/starting point for LLM
        template = self.load_template(modality, profile, strategy)
        
        # Step 3: Fill in dataset-specific values in template (for LLM context)
        filled_template = self._fill_template_values(template, strategy, dataset_info)
        
        # Step 4: Generate code with Gemini LLM (REQUIRED for all modalities)
        print("\n" + "=" * 60)
        print("[GEMINI] Generating code with LLM (no hardcoded templates)...")
        print(f"[GEMINI] Modality: {modality}, Samples: {profile.num_samples}")
        print("=" * 60)
        logger.info("[GEMINI] Calling Gemini LLM for code generation...")
        
        # Pass dataset_path for competition context (mlebench description/sample_submission)
        dataset_path = dataset_info.get('dataset_path')
        generated_code = self.enhance_with_gemini(
            filled_template, 
            strategy, 
            profile,
            dataset_path=dataset_path
        )
        
        print("[GEMINI] Code successfully generated by LLM!")
        print("=" * 60 + "\n")
        logger.info("[GEMINI] LLM code generation completed")
        
        # Step 5: Validate the code
        validation_result = self.validate_code(generated_code)
        
        # Step 6: Request fixes if validation fails
        max_fix_attempts = 3
        attempt = 0
        # Loop requesting fixes from Gemini until code is valid or max attempts reached
        while not validation_result.is_valid and attempt < max_fix_attempts:
            attempt += 1
            logger.info(f"Validation failed (attempt {attempt}/{max_fix_attempts}). Requesting fix from Gemini...")
            print(f"[GEMINI] Requesting fix attempt {attempt}/{max_fix_attempts}...")
            generated_code = self._request_gemini_fix(generated_code, validation_result)
            validation_result = self.validate_code(generated_code)
        
        if not validation_result.is_valid:
            error_msg = f"Code validation failed after {max_fix_attempts} attempts: {validation_result.errors}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if validation_result.warnings:
            logger.warning(f"Code has warnings: {validation_result.warnings}")
        
        logger.info("Training code generated successfully by LLM")
        return generated_code
    
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
            'seed': dataset_info.get('seed', 42),  # Use seed from dataset_info (passed from config)
            
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
            
            # Target column: use from dataset_info, NOT from submission format
            # The target column in training data may have a different name than the prediction column in submission
            'target_column': dataset_info.get('target_column', 'target'),
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

"""Submission file generation and validation for MLEbench competitions"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.models.data_models import Strategy


@dataclass
class CompetitionFormat:
    """Competition-specific submission format specification"""
    competition_id: str
    required_columns: List[str]
    id_column: str
    prediction_columns: List[str]
    file_format: str = "csv"
    header_required: bool = True
    index_required: bool = False
    
    
class SubmissionGenerator:
    """Generate and validate submission files for MLEbench competitions"""
    
    # Competition format specifications for 5 MLEbench competitions
    COMPETITION_FORMATS = {
        "siim-isic-melanoma-classification": CompetitionFormat(
            competition_id="siim-isic-melanoma-classification",
            required_columns=["image_name", "target"],
            id_column="image_name",
            prediction_columns=["target"],
            file_format="csv",
            header_required=True,
            index_required=False
        ),
        "spooky-author-identification": CompetitionFormat(
            competition_id="spooky-author-identification",
            required_columns=["id", "EAP", "HPL", "MWS"],
            id_column="id",
            prediction_columns=["EAP", "HPL", "MWS"],
            file_format="csv",
            header_required=True,
            index_required=False
        ),
        "tabular-playground-series-may-2022": CompetitionFormat(
            competition_id="tabular-playground-series-may-2022",
            required_columns=["id", "target"],
            id_column="id",
            prediction_columns=["target"],
            file_format="csv",
            header_required=True,
            index_required=False
        ),
        "text-normalization-challenge-english-language": CompetitionFormat(
            competition_id="text-normalization-challenge-english-language",
            required_columns=["id", "after"],
            id_column="id",
            prediction_columns=["after"],
            file_format="csv",
            header_required=True,
            index_required=False
        ),
        "the-icml-2013-whale-challenge-right-whale-redux": CompetitionFormat(
            competition_id="the-icml-2013-whale-challenge-right-whale-redux",
            required_columns=["Image", "whale_00", "whale_01", "whale_02", "whale_03", "whale_04",
                            "whale_05", "whale_06", "whale_07", "whale_08", "whale_09",
                            "whale_10", "whale_11", "whale_12", "whale_13", "whale_14",
                            "whale_15", "whale_16", "whale_17", "whale_18", "whale_19",
                            "whale_20", "whale_21", "whale_22", "whale_23", "whale_24",
                            "whale_25", "whale_26", "whale_27", "whale_28", "whale_29",
                            "whale_30", "whale_31", "whale_32", "whale_33", "whale_34",
                            "whale_35", "whale_36", "whale_37", "whale_38", "whale_39",
                            "whale_40", "whale_41", "whale_42", "whale_43", "whale_44",
                            "whale_45", "whale_46", "whale_47", "whale_48", "whale_49"],
            id_column="Image",
            prediction_columns=[f"whale_{i:02d}" for i in range(50)],
            file_format="csv",
            header_required=True,
            index_required=False
        )
    }
    
    def __init__(self, competition_id: str, output_dir: str, dataset_path: Optional[str] = None):
        """
        Initialize submission generator
        
        Args:
            competition_id: Competition identifier
            output_dir: Directory to save submission files
            dataset_path: Optional path to dataset for format inference
        """
        self.competition_id = competition_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = dataset_path
        
        # Get competition format (use hardcoded if available, otherwise infer)
        if competition_id in self.COMPETITION_FORMATS:
            self.format = self.COMPETITION_FORMATS[competition_id]
        else:
            # Infer format from dataset (truly autonomous!)
            self.format = self._infer_format_from_dataset()
    
    def _infer_format_from_dataset(self) -> CompetitionFormat:
        """
        Infer submission format from dataset structure
        
        Returns:
            CompetitionFormat inferred from data
        """
        if not self.dataset_path:
            # Create a generic format
            return CompetitionFormat(
                competition_id=self.competition_id,
                required_columns=["id", "target"],
                id_column="id",
                prediction_columns=["target"],
                file_format="csv",
                header_required=True,
                index_required=False
            )
        
        try:
            from src.dataset.dataset_handler import DatasetHandler
            
            handler = DatasetHandler(self.dataset_path, self.competition_id)
            
            # Try to load sample submission
            sample_sub = handler.load_sample_submission()
            if sample_sub is not None:
                # Infer from sample submission
                required_columns = list(sample_sub.columns)
                id_column = required_columns[0] if required_columns else "id"
                prediction_columns = required_columns[1:] if len(required_columns) > 1 else ["target"]
                
                return CompetitionFormat(
                    competition_id=self.competition_id,
                    required_columns=required_columns,
                    id_column=id_column,
                    prediction_columns=prediction_columns,
                    file_format="csv",
                    header_required=True,
                    index_required=False
                )
            
            # No sample submission - infer from test data
            test_df = handler.load_test_data(nrows=10)
            id_column = handler.get_id_column(test_df)
            
            # Try to get target column from train data
            try:
                train_df = handler.load_train_data(nrows=10)
                target_column = handler.get_target_column(train_df)
            except:
                target_column = "target"
            
            return CompetitionFormat(
                competition_id=self.competition_id,
                required_columns=[id_column, target_column],
                id_column=id_column,
                prediction_columns=[target_column],
                file_format="csv",
                header_required=True,
                index_required=False
            )
            
        except Exception as e:
            # Fallback to generic format
            return CompetitionFormat(
                competition_id=self.competition_id,
                required_columns=["id", "target"],
                id_column="id",
                prediction_columns=["target"],
                file_format="csv",
                header_required=True,
                index_required=False
            )
    
    def generate_predictions(
        self,
        model_path: str,
        test_data_path: str,
        strategy: Strategy
    ) -> pd.DataFrame:
        """
        Generate predictions on test data using trained model
        
        Args:
            model_path: Path to trained model file
            test_data_path: Path to test dataset
            strategy: Strategy used for training
            
        Returns:
            DataFrame with predictions
        """
        # Load test data
        test_df = pd.read_csv(test_data_path)
        
        # This is a placeholder - actual implementation would load the model
        # and generate predictions based on the modality and strategy
        # For now, we'll create dummy predictions with the correct structure
        
        predictions = {}
        
        # Add ID column
        if self.format.id_column in test_df.columns:
            predictions[self.format.id_column] = test_df[self.format.id_column]
        else:
            # Generate IDs if not present
            predictions[self.format.id_column] = range(len(test_df))
        
        # Generate prediction columns
        for pred_col in self.format.prediction_columns:
            if len(self.format.prediction_columns) == 1:
                # Single prediction column (regression or binary classification)
                predictions[pred_col] = np.random.rand(len(test_df))
            else:
                # Multiple prediction columns (multi-class probabilities)
                predictions[pred_col] = np.random.rand(len(test_df))
        
        # Normalize probabilities if multi-class
        if len(self.format.prediction_columns) > 1:
            pred_cols = self.format.prediction_columns
            pred_df = pd.DataFrame({col: predictions[col] for col in pred_cols})
            # Normalize to sum to 1
            row_sums = pred_df.sum(axis=1)
            for col in pred_cols:
                predictions[col] = pred_df[col] / row_sums
        
        return pd.DataFrame(predictions)
    
    def format_submission(
        self,
        predictions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Format predictions according to competition requirements
        
        Args:
            predictions: Raw predictions DataFrame
            
        Returns:
            Formatted submission DataFrame
        """
        # Ensure all required columns are present
        missing_cols = set(self.format.required_columns) - set(predictions.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Required: {self.format.required_columns}"
            )
        
        # Select only required columns in correct order
        submission = predictions[self.format.required_columns].copy()
        
        # Ensure correct data types
        # ID column should be string or int
        if self.format.id_column in submission.columns:
            # Keep original type if it's already string or int
            pass
        
        # Prediction columns should be numeric
        for col in self.format.prediction_columns:
            if col in submission.columns:
                submission.loc[:, col] = pd.to_numeric(submission[col], errors='coerce')
        
        return submission
    
    def validate_submission(
        self,
        submission: pd.DataFrame,
        test_data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate submission file against competition requirements
        
        Args:
            submission: Submission DataFrame to validate
            test_data_path: Optional path to test data for row count validation
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required columns
        missing_cols = set(self.format.required_columns) - set(submission.columns)
        if missing_cols:
            validation_result["is_valid"] = False
            validation_result["errors"].append(
                f"Missing required columns: {missing_cols}"
            )
        
        # Check for extra columns
        extra_cols = set(submission.columns) - set(self.format.required_columns)
        if extra_cols:
            validation_result["warnings"].append(
                f"Extra columns present (will be ignored): {extra_cols}"
            )
        
        # Check for missing values in prediction columns
        for col in self.format.prediction_columns:
            if col in submission.columns:
                null_count = submission[col].isnull().sum()
                if null_count > 0:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(
                        f"Column '{col}' has {null_count} missing values"
                    )
        
        # Check row count matches test set if provided
        if test_data_path:
            try:
                test_df = pd.read_csv(test_data_path)
                expected_rows = len(test_df)
                actual_rows = len(submission)
                
                if actual_rows != expected_rows:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(
                        f"Row count mismatch: expected {expected_rows}, got {actual_rows}"
                    )
            except Exception as e:
                validation_result["warnings"].append(
                    f"Could not validate row count: {str(e)}"
                )
        
        # Check for duplicate IDs
        if self.format.id_column in submission.columns:
            duplicate_count = submission[self.format.id_column].duplicated().sum()
            if duplicate_count > 0:
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"Found {duplicate_count} duplicate IDs"
                )
        
        # Check probability columns sum to 1 (for multi-class)
        if len(self.format.prediction_columns) > 1:
            pred_cols = [col for col in self.format.prediction_columns if col in submission.columns]
            if pred_cols:
                row_sums = submission[pred_cols].sum(axis=1)
                # Allow small tolerance for floating point errors
                tolerance = 0.01
                invalid_rows = ((row_sums < 1 - tolerance) | (row_sums > 1 + tolerance)).sum()
                if invalid_rows > 0:
                    validation_result["warnings"].append(
                        f"{invalid_rows} rows have probabilities not summing to 1"
                    )
        
        return validation_result
    
    def save_submission(
        self,
        submission: pd.DataFrame,
        strategy_id: Optional[str] = None,
        seed: Optional[int] = None
    ) -> str:
        """
        Save submission file with timestamp and strategy identifier
        
        Args:
            submission: Submission DataFrame to save
            strategy_id: Optional strategy identifier
            seed: Optional random seed used
            
        Returns:
            Path to saved submission file
        """
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename_parts = ["submission"]
        
        if strategy_id:
            filename_parts.append(strategy_id)
        
        if seed is not None:
            filename_parts.append(f"seed{seed}")
        
        filename_parts.append(timestamp)
        
        filename = "_".join(filename_parts) + ".csv"
        filepath = self.output_dir / filename
        
        # Save with appropriate settings
        submission.to_csv(
            filepath,
            index=self.format.index_required,
            header=self.format.header_required
        )
        
        return str(filepath)
    
    def generate_and_save_submission(
        self,
        model_path: str,
        test_data_path: str,
        strategy: Strategy,
        strategy_id: Optional[str] = None,
        seed: Optional[int] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Complete workflow: generate, format, validate, and save submission
        
        Args:
            model_path: Path to trained model
            test_data_path: Path to test data
            strategy: Strategy used for training
            strategy_id: Optional strategy identifier
            seed: Optional random seed
            validate: Whether to validate before saving
            
        Returns:
            Dictionary with submission path and validation results
        """
        # Generate predictions
        predictions = self.generate_predictions(model_path, test_data_path, strategy)
        
        # Format submission
        submission = self.format_submission(predictions)
        
        # Validate if requested
        validation_result = None
        if validate:
            validation_result = self.validate_submission(submission, test_data_path)
            
            if not validation_result["is_valid"]:
                raise ValueError(
                    f"Submission validation failed: {validation_result['errors']}"
                )
        
        # Save submission
        filepath = self.save_submission(submission, strategy_id, seed)
        
        return {
            "filepath": filepath,
            "validation": validation_result,
            "row_count": len(submission),
            "columns": list(submission.columns)
        }

"""Hybrid Modality Detector for dataset classification"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np

from src.models.data_models import (
    ModalityResult,
    DataProfile,
    VerificationResult,
    Modality
)
from src.utils.gemini_client import GeminiClient
from src.dataset.dataset_handler import DatasetHandler

logger = logging.getLogger(__name__)


class HybridModalityDetector:
    """Detects dataset modality using hybrid approach combining heuristics, profiling, and LLM consensus"""
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """Initialize detector
        
        Args:
            gemini_client: Optional Gemini client for consensus resolution
        """
        self.gemini_client = gemini_client
        
        # Heuristic patterns
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        self.audio_extensions = {'.aif', '.aiff', '.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        self.text_columns = {'text', 'content', 'description', 'comment', 'review', 'message', 'body'}
        self.image_columns = {'image', 'img', 'photo', 'picture', 'file', 'path', 'filename'}
        self.audio_columns = {'clip', 'audio', 'sound', 'recording', 'wav', 'aiff', 'file', 'filename'}
        self.time_columns = {'date', 'time', 'timestamp', 'datetime', 'year', 'month', 'day'}
        
        # Seq2seq/Text normalization patterns
        self.seq2seq_columns = {'before', 'after', 'source', 'target', 'input', 'output'}
        self.text_norm_classes = {'PLAIN', 'PUNCT', 'DATE', 'LETTERS', 'CARDINAL', 'VERBATIM', 
                                   'DECIMAL', 'MEASURE', 'MONEY', 'ORDINAL', 'TIME', 'ELECTRONIC',
                                   'DIGIT', 'FRACTION', 'TELEPHONE', 'ADDRESS'}
    
    def detect_modality(self, dataset_path: str) -> ModalityResult:
        """Detect modality using hybrid approach
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            ModalityResult with detection details
        """
        logger.info(f"Starting modality detection for: {dataset_path}")
        
        # Initialize dataset handler to understand structure
        dataset_handler = DatasetHandler(dataset_path)
        
        # Validate dataset
        is_valid, issues = dataset_handler.validate_dataset()
        if not is_valid:
            logger.warning(f"Dataset validation issues: {issues}")
        
        # Phase 1: Heuristic detection
        heuristic_result = self.heuristic_detection(dataset_path, dataset_handler)
        logger.info(f"Heuristic detection result: {heuristic_result}")
        
        # Phase 2: Data profiling
        profile = self.data_profiling(dataset_path, dataset_handler)
        profiling_result = self._infer_modality_from_profile(profile)
        logger.info(f"Profiling detection result: {profiling_result}")
        
        # Phase 3: File path verification (if applicable)
        verification = self._verify_paths_if_needed(dataset_path, heuristic_result, profiling_result, dataset_handler)
        
        # Phase 4: Submission format detection
        submission_info = self._detect_submission_format(dataset_path)
        if submission_info:
            logger.info(f"Detected submission format: {submission_info}")
        
        # Phase 5: Consensus resolution
        if heuristic_result == profiling_result:
            final_modality = heuristic_result
            confidence = 0.95 if verification.all_exist else 0.85
            gemini_consensus = None
        else:
            # Use Gemini for consensus
            if self.gemini_client:
                final_modality = self.gemini_consensus(heuristic_result, profile)
                confidence = 0.80
                gemini_consensus = final_modality
            else:
                # Fallback: trust profiling over heuristics
                final_modality = profiling_result
                confidence = 0.70
                gemini_consensus = None
        
        # Store dataset handler info in result for later use
        result = ModalityResult(
            modality=final_modality,
            confidence=confidence,
            heuristic_result=heuristic_result,
            profiling_result=profiling_result,
            gemini_consensus=gemini_consensus,
            verification_status=verification.all_exist,
            verification_message=f"Verified {verification.verified_files}/{verification.total_files} files"
        )
        
        # Attach profile with dataset info
        result.profile = profile
        
        # Attach submission info if detected
        if submission_info:
            result.submission_info = submission_info
        
        return result
    
    def _detect_submission_format(self, dataset_path: str) -> Optional[Dict]:
        """
        Detect submission format from sample_submission.csv
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Dictionary with submission format info or None
        """
        dataset_path = Path(dataset_path)
        
        # Search for sample submission file
        submission_patterns = [
            'sample_submission.csv',
            'sampleSubmission.csv', 
            'sample_submission*.csv',
            'submission_format.csv'
        ]
        
        submission_file = None
        for pattern in submission_patterns:
            matches = list(dataset_path.glob(pattern))
            if matches:
                submission_file = matches[0]
                break
            # Also check in subdirectories
            matches = list(dataset_path.glob(f'**/{pattern}'))
            if matches:
                submission_file = matches[0]
                break
        
        if not submission_file or not submission_file.exists():
            return None
        
        try:
            sample_sub = pd.read_csv(submission_file, nrows=10)
            
            # Analyze submission format
            columns = list(sample_sub.columns)
            dtypes = {col: str(sample_sub[col].dtype) for col in columns}
            
            # Detect ID column
            id_column = None
            for col in columns:
                if col.lower() in ['id', 'row_id', 'index', 'image_id', 'image', 'sample_id']:
                    id_column = col
                    break
            if id_column is None and len(columns) > 0:
                id_column = columns[0]
            
            # Detect prediction column(s)
            pred_columns = [c for c in columns if c != id_column]
            
            # Detect if probabilities or classes expected
            needs_probability = False
            if pred_columns:
                pred_col = pred_columns[0]
                sample_values = sample_sub[pred_col].dropna()
                if len(sample_values) > 0:
                    # Check if values are floats between 0 and 1
                    if sample_sub[pred_col].dtype in ['float64', 'float32']:
                        if sample_values.min() >= 0 and sample_values.max() <= 1:
                            needs_probability = True
            
            return {
                'file': str(submission_file),
                'columns': columns,
                'id_column': id_column,
                'prediction_columns': pred_columns,
                'needs_probability': needs_probability,
                'dtypes': dtypes,
                'sample_rows': len(sample_sub)
            }
            
        except Exception as e:
            logger.warning(f"Could not parse sample submission: {e}")
            return None
    
    def _detect_text_normalization(self, df: pd.DataFrame) -> bool:
        """
        Check if dataset is a text normalization task
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if text normalization dataset detected
        """
        columns_lower = {col.lower() for col in df.columns}
        
        # Check for before/after pattern
        has_before_after = ('before' in columns_lower and 'after' in columns_lower) or \
                          ('source' in columns_lower and 'target' in columns_lower) or \
                          ('input' in columns_lower and 'output' in columns_lower)
        
        if not has_before_after:
            return False
        
        # Check for class column with text normalization classes
        for col in df.columns:
            if col.lower() == 'class':
                unique_values = set(str(v).upper() for v in df[col].dropna().unique())
                # Check if values match text normalization classes
                if unique_values & self.text_norm_classes:
                    return True
        
        return has_before_after
    
    def heuristic_detection(self, dataset_path: str, dataset_handler: Optional[DatasetHandler] = None) -> str:
        """Fast heuristic-based detection using file extensions and column names
        
        Args:
            dataset_path: Path to dataset directory
            dataset_handler: Optional DatasetHandler for structured access
            
        Returns:
            Detected modality string
        """
        if dataset_handler is None:
            dataset_handler = DatasetHandler(dataset_path)
        
        # Check for image directories
        train_img_dir, test_img_dir = dataset_handler.get_image_directories()
        has_image_dirs = train_img_dir is not None or test_img_dir is not None
        
        # Read training data to analyze
        try:
            df = dataset_handler.load_train_data(nrows=100)  # Read more rows for better detection
            columns_lower = {col.lower() for col in df.columns}
            
            # Check for seq2seq/text normalization FIRST (before text classification)
            if self._detect_text_normalization(df):
                logger.info("Detected text normalization dataset (before/after pattern with class column)")
                return "seq2seq"
            
            # Check for before/after columns (generic seq2seq)
            has_seq2seq_cols = bool(columns_lower & {'before', 'after', 'source', 'target'})
            if has_seq2seq_cols:
                logger.info("Detected seq2seq dataset (before/after or source/target columns)")
                return "seq2seq"
            
            # Check for image-related columns
            has_image_cols = bool(columns_lower & self.image_columns)
            
            # Check for audio-related columns
            has_audio_cols = bool(columns_lower & self.audio_columns)
            
            # Check for text-related columns
            has_text_cols = bool(columns_lower & self.text_columns)
            
            # Check for time-related columns
            has_time_cols = bool(columns_lower & self.time_columns)
            
            # Check for file paths in data (distinguish image vs audio)
            has_image_paths = False
            has_audio_paths = False
            for col in df.columns:
                if df[col].dtype == object:
                    sample_values = df[col].dropna().head(3).astype(str)
                    for val in sample_values:
                        val_lower = str(val).lower()
                        if any(ext in val_lower for ext in self.image_extensions):
                            has_image_paths = True
                        if any(ext in val_lower for ext in self.audio_extensions):
                            has_audio_paths = True
            
            # Audio detection: check for .aif/.aiff/.wav extensions or audio columns
            if has_audio_paths or (has_audio_cols and 'clip' in columns_lower):
                logger.info("Detected audio dataset (audio file extensions or clip column)")
                return "audio"
            
            # Determine modality
            modality_indicators = {
                'image': has_image_cols or has_image_paths or has_image_dirs,
                'text': has_text_cols,
                'time': has_time_cols
            }
            
            active_modalities = [k for k, v in modality_indicators.items() if v]
            
            if len(active_modalities) > 1:
                return Modality.MULTIMODAL.value
            elif 'image' in active_modalities:
                return Modality.IMAGE.value
            elif 'text' in active_modalities:
                return Modality.TEXT.value
            elif 'time' in active_modalities:
                return Modality.TIME_SERIES.value
            else:
                return Modality.TABULAR.value
                
        except Exception as e:
            logger.warning(f"Heuristic detection failed: {e}")
            return Modality.TABULAR.value
    
    def data_profiling(self, dataset_path: str, dataset_handler: Optional[DatasetHandler] = None) -> DataProfile:
        """Statistical profiling of dataset
        
        Args:
            dataset_path: Path to dataset directory
            dataset_handler: Optional DatasetHandler for structured access
            
        Returns:
            DataProfile with statistical information
        """
        if dataset_handler is None:
            dataset_handler = DatasetHandler(dataset_path)
        
        try:
            # Load training data using dataset handler
            df = dataset_handler.load_train_data()
            
            # Calculate missing values
            missing_values = {}
            for col in df.columns:
                missing_pct = df[col].isna().sum() / len(df)
                if missing_pct > 0:
                    missing_values[col] = float(missing_pct)
            
            # Target distribution - use dataset handler to detect target column
            target_col = dataset_handler.get_target_column(df)
            
            target_distribution = {}
            if target_col and target_col in df.columns:
                if df[target_col].dtype in ['int64', 'object', 'category']:
                    target_distribution = df[target_col].value_counts().to_dict()
                    # Convert to string keys for JSON serialization
                    target_distribution = {str(k): int(v) for k, v in target_distribution.items()}
            
            # Feature correlations (for numeric columns only)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_correlations = {}
            if len(numeric_cols) > 1:
                try:
                    # Create a copy to avoid access violations on Windows
                    numeric_df = df[numeric_cols].copy()
                    # Drop columns with all NaN values
                    numeric_df = numeric_df.dropna(axis=1, how='all')
                    if len(numeric_df.columns) > 1:
                        corr_matrix = numeric_df.corr()
                        # Get top correlations
                        for i, col1 in enumerate(corr_matrix.columns):
                            for col2 in corr_matrix.columns[i+1:]:
                                try:
                                    corr_val = corr_matrix.at[col1, col2]
                                    if not np.isnan(corr_val) and abs(corr_val) > 0.5:  # Only store significant correlations
                                        feature_correlations[f"{col1}_{col2}"] = float(corr_val)
                                except (KeyError, ValueError):
                                    continue
                except Exception as e:
                    logger.warning(f"Failed to compute correlations: {e}")
                    feature_correlations = {}
            
            # Memory usage
            memory_usage_gb = df.memory_usage(deep=True).sum() / (1024**3)
            
            # Data types
            data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            return DataProfile(
                missing_values=missing_values,
                target_distribution=target_distribution,
                feature_correlations=feature_correlations,
                memory_usage_gb=float(memory_usage_gb),
                num_samples=len(df),
                num_features=len(df.columns),
                data_types=data_types
            )
            
        except Exception as e:
            logger.error(f"Data profiling failed: {e}")
            return DataProfile(
                missing_values={},
                target_distribution={},
                feature_correlations={},
                memory_usage_gb=0.0,
                num_samples=0,
                num_features=0,
                data_types={}
            )
    
    def verify_file_paths(self, csv_path: str, dataset_dir: str) -> VerificationResult:
        """Verify file paths in CSV actually exist
        
        Args:
            csv_path: Path to CSV file
            dataset_dir: Base directory for dataset
            
        Returns:
            VerificationResult with verification details
        """
        try:
            df = pd.read_csv(csv_path)
            dataset_path = Path(dataset_dir)
            
            all_files = []
            missing_files = []
            
            # Check each column for file paths
            for col in df.columns:
                if df[col].dtype == object:
                    sample_values = df[col].dropna().head(100).astype(str)
                    
                    # Check if this column contains file paths
                    has_extensions = any(
                        any(ext in str(val).lower() for ext in self.image_extensions)
                        for val in sample_values
                    )
                    
                    if has_extensions:
                        # This column likely contains file paths
                        for val in df[col].dropna():
                            file_path = dataset_path / str(val)
                            all_files.append(str(val))
                            if not file_path.exists():
                                missing_files.append(str(val))
            
            if not all_files:
                # No file paths found
                return VerificationResult(
                    all_exist=True,
                    missing_files=[],
                    total_files=0,
                    verified_files=0
                )
            
            return VerificationResult(
                all_exist=len(missing_files) == 0,
                missing_files=missing_files[:10],  # Limit to first 10
                total_files=len(all_files),
                verified_files=len(all_files) - len(missing_files)
            )
            
        except Exception as e:
            logger.error(f"File path verification failed: {e}")
            return VerificationResult(
                all_exist=False,
                missing_files=[],
                total_files=0,
                verified_files=0
            )
    
    def gemini_consensus(self, heuristic: str, profile: DataProfile) -> str:
        """Use Gemini to resolve detection conflicts
        
        Args:
            heuristic: Heuristic detection result
            profile: Data profile
            
        Returns:
            Final modality determination
        """
        if not self.gemini_client:
            logger.warning("Gemini client not available, using heuristic result")
            return heuristic
        
        prompt = f"""
Analyze this dataset and determine its modality (tabular, image, text, time_series, multimodal, seq2seq, or audio).

HEURISTIC DETECTION: {heuristic}

DATA PROFILE:
- Number of samples: {profile.num_samples}
- Number of features: {profile.num_features}
- Memory usage: {profile.memory_usage_gb:.2f} GB
- Missing values: {len(profile.missing_values)} columns with missing data
- Data types: {profile.data_types}
- Target distribution: {profile.target_distribution}

MODALITY HINTS:
- seq2seq: Datasets with 'before'/'after' or 'source'/'target' columns (e.g., text normalization, translation)
- audio: Datasets with audio file paths (.wav, .aif, .mp3) or 'clip' column (e.g., whale detection)
- text: Datasets with 'text', 'review', 'comment' columns for classification
- tabular: Standard numeric/categorical feature datasets
- image: Datasets with image file paths or image directories
- time_series: Datasets with temporal patterns or date/time columns

Based on this information, what is the most accurate modality classification?

Respond with ONLY one word: tabular, image, text, time_series, multimodal, seq2seq, or audio
"""
        
        try:
            response = self.gemini_client.generate_content(prompt)
            result = response.text.strip().lower()
            
            # Validate response
            valid_modalities = {m.value for m in Modality}
            if result in valid_modalities:
                return result
            else:
                logger.warning(f"Invalid Gemini response: {result}, using heuristic")
                return heuristic
                
        except Exception as e:
            logger.error(f"Gemini consensus failed: {e}")
            return heuristic
    
    def _infer_modality_from_profile(self, profile: DataProfile) -> str:
        """Infer modality from statistical profile
        
        Args:
            profile: Data profile
            
        Returns:
            Inferred modality
        """
        # Check data types
        object_cols = sum(1 for dtype in profile.data_types.values() if 'object' in dtype)
        numeric_cols = sum(1 for dtype in profile.data_types.values() 
                          if any(t in dtype for t in ['int', 'float']))
        
        # Check for seq2seq patterns in column names
        column_names = [col.lower() for col in profile.data_types.keys()]
        has_before_after = ('before' in column_names and 'after' in column_names) or \
                          ('source' in column_names and 'target' in column_names) or \
                          ('input' in column_names and 'output' in column_names)
        has_class_column = 'class' in column_names
        
        # Seq2seq detection: before/after or source/target pattern
        if has_before_after:
            return Modality.SEQ2SEQ.value
        
        # High ratio of object columns suggests text or image paths
        if object_cols > numeric_cols and object_cols > 2:
            # Check if likely text (many unique values) or paths (fewer unique values)
            if profile.num_samples > 0:
                return Modality.TEXT.value
            return Modality.IMAGE.value
        
        # Check for time series indicators
        time_indicators = ['date', 'time', 'timestamp']
        has_time = any(indicator in col.lower() 
                      for col in profile.data_types.keys() 
                      for indicator in time_indicators)
        
        if has_time:
            return Modality.TIME_SERIES.value
        
        # Default to tabular
        return Modality.TABULAR.value
    
    def _verify_paths_if_needed(
        self, 
        dataset_path: str, 
        heuristic: str, 
        profiling: str,
        dataset_handler: Optional[DatasetHandler] = None
    ) -> VerificationResult:
        """Verify file paths if modality suggests files
        
        Args:
            dataset_path: Path to dataset
            heuristic: Heuristic result
            profiling: Profiling result
            dataset_handler: Optional DatasetHandler for structured access
            
        Returns:
            VerificationResult
        """
        needs_verification = (
            Modality.IMAGE.value in [heuristic, profiling] or
            Modality.MULTIMODAL.value in [heuristic, profiling]
        )
        
        if not needs_verification:
            return VerificationResult(
                all_exist=True,
                missing_files=[],
                total_files=0,
                verified_files=0
            )
        
        if dataset_handler is None:
            dataset_handler = DatasetHandler(dataset_path)
        
        # Get train CSV path
        try:
            train_csv_path = dataset_handler.get_train_data_path()
            if not train_csv_path.endswith('.csv'):
                # If it's a directory, find CSV inside
                csv_files = list(Path(train_csv_path).glob("*.csv"))
                if csv_files:
                    train_csv_path = str(csv_files[0])
                else:
                    return VerificationResult(
                        all_exist=True,
                        missing_files=[],
                        total_files=0,
                        verified_files=0
                    )
            
            return self.verify_file_paths(train_csv_path, dataset_path)
        except Exception as e:
            logger.warning(f"Could not verify file paths: {e}")
            return VerificationResult(
                all_exist=True,
                missing_files=[],
                total_files=0,
                verified_files=0
            )


def detect_submission_format(dataset_path: str) -> Dict:
    """
    Standalone function to detect submission format from sample_submission.csv
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dictionary with submission format info including:
        - id_column: Name of ID column
        - prediction_column: Name of prediction column
        - expected_format: Expected format type
        - num_rows: Expected number of rows
    """
    dataset_path = Path(dataset_path)
    
    # Search for sample submission file
    submission_patterns = [
        'sample_submission.csv',
        'sampleSubmission.csv', 
        'submission_format.csv'
    ]
    
    submission_file = None
    for pattern in submission_patterns:
        # Check root directory
        candidate = dataset_path / pattern
        if candidate.exists():
            submission_file = candidate
            break
        # Check subdirectories
        matches = list(dataset_path.glob(f'**/{pattern}'))
        if matches:
            submission_file = matches[0]
            break
    
    if not submission_file or not submission_file.exists():
        logger.warning(f"No sample submission file found in {dataset_path}")
        return {}
    
    try:
        sample_sub = pd.read_csv(submission_file, nrows=100)
        
        columns = list(sample_sub.columns)
        
        # Detect ID column
        id_column = None
        id_patterns = ['id', 'row_id', 'index', 'image_id', 'image', 'sample_id', 'clip', 'sentence_id']
        for col in columns:
            if col.lower() in id_patterns:
                id_column = col
                break
        if id_column is None and len(columns) > 0:
            id_column = columns[0]
        
        # Detect prediction column(s)
        pred_columns = [c for c in columns if c != id_column]
        
        # Get the main prediction column name
        prediction_column = pred_columns[0] if pred_columns else 'prediction'
        
        # Detect if probabilities or classes expected
        expected_format = 'class'
        if pred_columns:
            pred_col = pred_columns[0]
            sample_values = sample_sub[pred_col].dropna()
            if len(sample_values) > 0:
                if sample_sub[pred_col].dtype in ['float64', 'float32']:
                    if sample_values.min() >= 0 and sample_values.max() <= 1:
                        expected_format = 'probability'
        
        # Get expected number of rows
        num_rows = len(sample_sub)
        
        return {
            'id_column': id_column,
            'prediction_column': prediction_column,
            'prediction_columns': pred_columns,
            'expected_format': expected_format,
            'num_rows': num_rows,
            'columns': columns
        }
        
    except Exception as e:
        logger.warning(f"Could not parse sample submission: {e}")
        return {}

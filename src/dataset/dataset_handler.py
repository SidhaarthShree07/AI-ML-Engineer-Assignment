"""
Dataset Handler for HybridAutoMLE Agent
Handles various dataset structures including train/test folders
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json

logger = logging.getLogger(__name__)


class DatasetStructure:
    """Represents the structure of a dataset"""
    
    def __init__(self, base_path: str):
        """
        Initialize dataset structure
        
        Args:
            base_path: Base path to dataset
        """
        self.base_path = Path(base_path)
        self.train_path: Optional[Path] = None
        self.test_path: Optional[Path] = None
        self.train_csv: Optional[Path] = None
        self.test_csv: Optional[Path] = None
        self.train_images_dir: Optional[Path] = None
        self.test_images_dir: Optional[Path] = None
        self.metadata_file: Optional[Path] = None
        self.sample_submission: Optional[Path] = None
        self.structure_type: str = "unknown"
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "base_path": str(self.base_path),
            "train_path": str(self.train_path) if self.train_path else None,
            "test_path": str(self.test_path) if self.test_path else None,
            "train_csv": str(self.train_csv) if self.train_csv else None,
            "test_csv": str(self.test_csv) if self.test_csv else None,
            "train_images_dir": str(self.train_images_dir) if self.train_images_dir else None,
            "test_images_dir": str(self.test_images_dir) if self.test_images_dir else None,
            "metadata_file": str(self.metadata_file) if self.metadata_file else None,
            "sample_submission": str(self.sample_submission) if self.sample_submission else None,
            "structure_type": self.structure_type
        }


class DatasetHandler:
    """
    Handles dataset loading and structure detection
    
    Supports multiple dataset structures:
    1. Flat structure: train.csv, test.csv in base directory
    2. Folder structure: train/ and test/ subdirectories
    3. Image datasets: train_images/, test_images/ with CSV metadata
    4. Mixed structures: combinations of above
    """
    
    def __init__(self, dataset_path: str, competition_id: Optional[str] = None):
        """
        Initialize dataset handler
        
        Args:
            dataset_path: Path to dataset directory
            competition_id: Optional competition ID
        """
        self.dataset_path_str = dataset_path
        self.dataset_path = Path(dataset_path)
        self.competition_id = competition_id
        self.structure: Optional[DatasetStructure] = None
        
        logger.info(f"Initializing DatasetHandler for: {dataset_path}")
        
        # Detect dataset structure
        self.structure = self._detect_structure()
        logger.info(f"Detected structure type: {self.structure.structure_type}")
    
    def _detect_structure(self) -> DatasetStructure:
        """
        Detect the structure of the dataset
        
        Returns:
            DatasetStructure object with detected paths
        """
        structure = DatasetStructure(str(self.dataset_path))
        
        # Check for train/test subdirectories
        train_dir = self.dataset_path / "train"
        test_dir = self.dataset_path / "test"
        
        if train_dir.exists() and train_dir.is_dir():
            structure.train_path = train_dir
            structure.structure_type = "folder_structure"
            logger.info(f"Found train directory: {train_dir}")
            
        if test_dir.exists() and test_dir.is_dir():
            structure.test_path = test_dir
            logger.info(f"Found test directory: {test_dir}")
        
        # Check for CSV files in base directory
        train_csv_candidates = [
            self.dataset_path / "train.csv",
            self.dataset_path / "training.csv",
            self.dataset_path / "train_data.csv"
        ]
        
        for candidate in train_csv_candidates:
            if candidate.exists():
                structure.train_csv = candidate
                if structure.structure_type == "unknown":
                    structure.structure_type = "flat_csv"
                logger.info(f"Found train CSV: {candidate}")
                break
        
        test_csv_candidates = [
            self.dataset_path / "test.csv",
            self.dataset_path / "testing.csv",
            self.dataset_path / "test_data.csv"
        ]
        
        for candidate in test_csv_candidates:
            if candidate.exists():
                structure.test_csv = candidate
                logger.info(f"Found test CSV: {candidate}")
                break
        
        # Check for CSV files in train/test subdirectories
        if structure.train_path:
            train_csv_in_dir = list(structure.train_path.glob("*.csv"))
            if train_csv_in_dir:
                structure.train_csv = train_csv_in_dir[0]
                logger.info(f"Found train CSV in subdirectory: {structure.train_csv}")
        
        if structure.test_path:
            test_csv_in_dir = list(structure.test_path.glob("*.csv"))
            if test_csv_in_dir:
                structure.test_csv = test_csv_in_dir[0]
                logger.info(f"Found test CSV in subdirectory: {structure.test_csv}")
        
        # Check for image directories
        image_dir_candidates = [
            ("train_images", "test_images"),
            ("train_imgs", "test_imgs"),
            ("images/train", "images/test"),
            ("train", "test")  # Could be image directories
        ]
        
        for train_name, test_name in image_dir_candidates:
            train_img_dir = self.dataset_path / train_name
            test_img_dir = self.dataset_path / test_name
            
            if train_img_dir.exists() and train_img_dir.is_dir():
                # Check if it contains images
                image_files = list(train_img_dir.glob("*.jpg")) + \
                             list(train_img_dir.glob("*.png")) + \
                             list(train_img_dir.glob("*.jpeg"))
                
                if image_files:
                    structure.train_images_dir = train_img_dir
                    structure.structure_type = "image_dataset"
                    logger.info(f"Found train images directory: {train_img_dir}")
            
            if test_img_dir.exists() and test_img_dir.is_dir():
                image_files = list(test_img_dir.glob("*.jpg")) + \
                             list(test_img_dir.glob("*.png")) + \
                             list(test_img_dir.glob("*.jpeg"))
                
                if image_files:
                    structure.test_images_dir = test_img_dir
                    logger.info(f"Found test images directory: {test_img_dir}")
        
        # Check for sample submission
        submission_candidates = [
            self.dataset_path / "sample_submission.csv",
            self.dataset_path / "submission.csv",
            self.dataset_path / "sample_sub.csv"
        ]
        
        for candidate in submission_candidates:
            if candidate.exists():
                structure.sample_submission = candidate
                logger.info(f"Found sample submission: {candidate}")
                break
        
        # Check for metadata file
        metadata_candidates = [
            self.dataset_path / "metadata.json",
            self.dataset_path / "info.json",
            self.dataset_path / "dataset_info.json"
        ]
        
        for candidate in metadata_candidates:
            if candidate.exists():
                structure.metadata_file = candidate
                logger.info(f"Found metadata file: {candidate}")
                break
        
        # Validate structure
        if not structure.train_csv and not structure.train_path:
            logger.warning("No train data found in dataset!")
        
        if not structure.test_csv and not structure.test_path:
            logger.warning("No test data found in dataset!")
        
        return structure
    
    def get_train_data_path(self) -> str:
        """
        Get path to training data
        
        Returns:
            Path to train CSV or train directory
        """
        if self.structure.train_csv:
            return str(self.structure.train_csv)
        elif self.structure.train_path:
            return str(self.structure.train_path)
        else:
            raise FileNotFoundError("No training data found in dataset")
    
    def get_test_data_path(self) -> str:
        """
        Get path to test data
        
        Returns:
            Path to test CSV or test directory
        """
        if self.structure.test_csv:
            return str(self.structure.test_csv)
        elif self.structure.test_path:
            return str(self.structure.test_path)
        else:
            raise FileNotFoundError("No test data found in dataset")
    
    def get_image_directories(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get paths to image directories
        
        Returns:
            Tuple of (train_images_dir, test_images_dir)
        """
        train_dir = str(self.structure.train_images_dir) if self.structure.train_images_dir else None
        test_dir = str(self.structure.test_images_dir) if self.structure.test_images_dir else None
        return train_dir, test_dir
    
    def load_train_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load training data
        
        Args:
            nrows: Optional number of rows to load
            
        Returns:
            DataFrame with training data
        """
        train_path = self.get_train_data_path()
        
        if train_path.endswith('.csv'):
            logger.info(f"Loading train data from CSV: {train_path}")
            return pd.read_csv(train_path, nrows=nrows)
        else:
            # If it's a directory, look for CSV inside
            csv_files = list(Path(train_path).glob("*.csv"))
            if csv_files:
                logger.info(f"Loading train data from: {csv_files[0]}")
                return pd.read_csv(csv_files[0], nrows=nrows)
            else:
                raise FileNotFoundError(f"No CSV file found in {train_path}")
    
    def load_test_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load test data
        
        Args:
            nrows: Optional number of rows to load
            
        Returns:
            DataFrame with test data
        """
        test_path = self.get_test_data_path()
        
        if test_path.endswith('.csv'):
            logger.info(f"Loading test data from CSV: {test_path}")
            return pd.read_csv(test_path, nrows=nrows)
        else:
            # If it's a directory, look for CSV inside
            csv_files = list(Path(test_path).glob("*.csv"))
            if csv_files:
                logger.info(f"Loading test data from: {csv_files[0]}")
                return pd.read_csv(csv_files[0], nrows=nrows)
            else:
                raise FileNotFoundError(f"No CSV file found in {test_path}")
    
    def load_sample_submission(self) -> Optional[pd.DataFrame]:
        """
        Load sample submission if available
        
        Returns:
            DataFrame with sample submission or None
        """
        if self.structure.sample_submission:
            logger.info(f"Loading sample submission: {self.structure.sample_submission}")
            return pd.read_csv(self.structure.sample_submission)
        return None
    
    def get_target_column(self, train_df: Optional[pd.DataFrame] = None) -> Optional[str]:
        """
        Automatically detect target column
        
        Args:
            train_df: Optional training DataFrame (will load if not provided)
            
        Returns:
            Name of target column or None
        """
        if train_df is None:
            try:
                train_df = self.load_train_data(nrows=100)
            except Exception as e:
                logger.warning(f"Could not load train data to detect target: {e}")
                return None
        
        # Common target column names
        target_candidates = [
            'target', 'label', 'class', 'y', 'output',
            'prediction', 'score', 'rating', 'category'
        ]
        
        for col in train_df.columns:
            if col.lower() in target_candidates:
                logger.info(f"Detected target column: {col}")
                return col
        
        # If sample submission exists, use its second column as target
        sample_sub = self.load_sample_submission()
        if sample_sub is not None and len(sample_sub.columns) >= 2:
            target_col = sample_sub.columns[1]
            if target_col in train_df.columns:
                logger.info(f"Detected target column from sample submission: {target_col}")
                return target_col
        
        # Last resort: assume last column is target
        last_col = train_df.columns[-1]
        logger.warning(f"Could not detect target column, assuming last column: {last_col}")
        return last_col
    
    def get_id_column(self, df: Optional[pd.DataFrame] = None) -> Optional[str]:
        """
        Automatically detect ID column
        
        Args:
            df: Optional DataFrame (will load test data if not provided)
            
        Returns:
            Name of ID column or None
        """
        if df is None:
            try:
                df = self.load_test_data(nrows=100)
            except Exception as e:
                logger.warning(f"Could not load data to detect ID column: {e}")
                return None
        
        # Common ID column names
        id_candidates = ['id', 'index', 'row_id', 'sample_id', 'image_id']
        
        for col in df.columns:
            if col.lower() in id_candidates:
                logger.info(f"Detected ID column: {col}")
                return col
        
        # If sample submission exists, use its first column as ID
        sample_sub = self.load_sample_submission()
        if sample_sub is not None and len(sample_sub.columns) >= 1:
            id_col = sample_sub.columns[0]
            if id_col in df.columns:
                logger.info(f"Detected ID column from sample submission: {id_col}")
                return id_col
        
        # First column is often ID
        first_col = df.columns[0]
        logger.warning(f"Could not detect ID column, assuming first column: {first_col}")
        return first_col
    
    def get_dataset_info(self) -> Dict:
        """
        Get comprehensive dataset information
        
        Returns:
            Dictionary with dataset information
        """
        info = {
            "structure": self.structure.to_dict(),
            "competition_id": self.competition_id
        }
        
        try:
            train_df = self.load_train_data(nrows=100)
            info["train_shape"] = train_df.shape
            info["train_columns"] = list(train_df.columns)
            info["target_column"] = self.get_target_column(train_df)
        except Exception as e:
            logger.warning(f"Could not load train data info: {e}")
            info["train_shape"] = None
            info["train_columns"] = None
            info["target_column"] = None
        
        try:
            test_df = self.load_test_data(nrows=100)
            info["test_shape"] = test_df.shape
            info["test_columns"] = list(test_df.columns)
            info["id_column"] = self.get_id_column(test_df)
        except Exception as e:
            logger.warning(f"Could not load test data info: {e}")
            info["test_shape"] = None
            info["test_columns"] = None
            info["id_column"] = None
        
        return info
    
    def validate_dataset(self) -> Tuple[bool, List[str]]:
        """
        Validate dataset structure and contents
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check for train data
        try:
            train_path = self.get_train_data_path()
        except FileNotFoundError as e:
            issues.append(str(e))
        
        # Check for test data
        try:
            test_path = self.get_test_data_path()
        except FileNotFoundError as e:
            issues.append(str(e))
        
        # Try to load data
        try:
            train_df = self.load_train_data(nrows=10)
            if train_df.empty:
                issues.append("Train data is empty")
        except Exception as e:
            issues.append(f"Cannot load train data: {e}")
        
        try:
            test_df = self.load_test_data(nrows=10)
            if test_df.empty:
                issues.append("Test data is empty")
        except Exception as e:
            issues.append(f"Cannot load test data: {e}")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Dataset validation passed")
        else:
            logger.warning(f"Dataset validation failed with {len(issues)} issues")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues

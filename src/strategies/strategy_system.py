"""Strategy System for selecting optimal ML strategies based on modality and dataset profile"""

from typing import Optional
from src.models.data_models import (
    DatasetProfile,
    Strategy,
    ResourceConstraints,
    Modality,
    TargetType
)


class StrategySystem:
    """
    Provides modality-specific ML strategies with dynamic adaptation.
    
    Selects optimal strategies based on:
    - Dataset modality (tabular, image, text, time-series, multimodal)
    - Resource constraints (memory, compute)
    - Dataset characteristics (size, class imbalance, etc.)
    """
    
    def __init__(self, resource_constraints: Optional[ResourceConstraints] = None):
        """
        Initialize strategy system.
        
        Args:
            resource_constraints: Resource limits for execution
        """
        self.resource_constraints = resource_constraints or ResourceConstraints()
    
    def get_strategy(self, modality: str, profile: DatasetProfile) -> Strategy:
        """
        Get optimal strategy for modality and profile.
        
        Args:
            modality: Dataset modality
            profile: Comprehensive dataset profile
            
        Returns:
            Strategy configuration
            
        Raises:
            ValueError: If modality is not supported
        """
        modality_lower = modality.lower()
        
        if modality_lower == Modality.TABULAR:
            return self.get_tabular_strategy(profile)
        elif modality_lower == Modality.IMAGE:
            return self.get_image_strategy(profile)
        elif modality_lower == Modality.TEXT:
            return self.get_text_strategy(profile)
        elif modality_lower == Modality.TIME_SERIES:
            return self.get_seq2seq_strategy(profile)
        elif modality_lower == Modality.MULTIMODAL:
            return self.get_multimodal_strategy(profile)
        else:
            raise ValueError(
                f"Unsupported modality: {modality}. "
                f"Supported: {[m.value for m in Modality]}"
            )
    
    def get_tabular_strategy(self, profile: DatasetProfile) -> Strategy:
        """
        Get tabular-specific strategy with memory-based selection.
        
        For small datasets (< 10GB): Use FLAML AutoML
        For large datasets (>= 10GB): Use LightGBM with GPU
        
        Args:
            profile: Dataset profile
            
        Returns:
            Tabular strategy configuration
        """
        # Memory-based selection
        if profile.memory_gb < 10.0:
            # Small dataset: FLAML AutoML
            primary_model = "FLAML_AutoML"
            fallback_model = "LightGBM_XGBoost_CatBoost_Ensemble"
            preprocessing = [
                "handle_missing_values",
                "target_encoding_high_cardinality",
                "robust_scaling"
            ]
            hyperparameters = {
                "time_budget": 3600,  # 1 hour
                "metric": "accuracy" if profile.target_type == TargetType.CLASSIFICATION else "r2",
                "task_type": profile.target_type,
                "feature_engineering": [
                    "polynomial_features_degree_2",
                    "interaction_features"
                ],
                "hyperparameter_tuning": "optuna_with_early_stopping"
            }
            batch_size = 256
            max_epochs = 100
        else:
            # Large dataset: LightGBM with GPU
            primary_model = "LightGBM_with_GPU"
            fallback_model = "XGBoost"
            preprocessing = [
                "handle_missing_values",
                "label_encoding"
            ]
            hyperparameters = {
                "feature_selection": "shapley_based_top_50",
                "ensemble": "VotingClassifier",
                "models": ["LightGBM", "XGBoost", "CatBoost"],
                "device": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": 0
            }
            batch_size = 512
            max_epochs = 50
        
        # Adaptive loss selection based on class imbalance
        loss_function = self._select_loss_function(profile)
        
        return Strategy(
            modality=Modality.TABULAR,
            primary_model=primary_model,
            fallback_model=fallback_model,
            preprocessing=preprocessing,
            augmentation=None,
            loss_function=loss_function,
            optimizer="Adam",
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping_patience=10,
            hyperparameters=hyperparameters,
            resource_constraints=self.resource_constraints,
            learning_rate=0.001,
            weight_decay=0.01,
            dropout=0.1,
            mixed_precision=False,
            gradient_accumulation_steps=1,
            model_size="medium"
        )
    
    def get_image_strategy(self, profile: DatasetProfile) -> Strategy:
        """
        Get image-specific strategy with resource adaptation.
        
        Uses EfficientNet backbone with albumentations augmentation.
        Adapts model size and batch size based on resource constraints.
        
        Args:
            profile: Dataset profile
            
        Returns:
            Image strategy configuration
        """
        # Resource adaptation
        if profile.estimated_gpu_memory_gb > 20.0:
            # Resource-constrained configuration
            primary_model = "EfficientNet-B3"
            batch_size = 16
            gradient_accumulation_steps = 2
            mixed_precision = True
            model_size = "small"
        else:
            # Standard configuration
            primary_model = "EfficientNet-B5"
            batch_size = 32
            gradient_accumulation_steps = 1
            mixed_precision = True
            model_size = "medium"
        
        # Albumentations augmentation
        augmentation = {
            "library": "albumentations",
            "transforms": [
                "HorizontalFlip",
                "VerticalFlip",
                "RandomRotate90",
                "ShiftScaleRotate",
                "RandomBrightnessContrast",
                "HueSaturationValue",
                "CoarseDropout"
            ]
        }
        
        # Adaptive loss selection
        loss_function = self._select_loss_function(profile)
        
        return Strategy(
            modality=Modality.IMAGE,
            primary_model=primary_model,
            fallback_model="ResNet50",
            preprocessing=["normalize", "resize"],
            augmentation=augmentation,
            loss_function=loss_function,
            optimizer="AdamW",
            batch_size=batch_size,
            max_epochs=50,
            early_stopping_patience=5,
            hyperparameters={
                "pretrained": True,
                "scheduler": "CosineAnnealingLR",
                "image_size": 224,
                "num_workers": 4
            },
            resource_constraints=self.resource_constraints,
            learning_rate=2e-4,
            weight_decay=0.01,
            dropout=0.2,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clip_norm=1.0,
            model_size=model_size,
            augmentation_strength=1.0
        )
    
    def get_text_strategy(self, profile: DatasetProfile) -> Strategy:
        """
        Get text-specific strategy with task type detection.
        
        Primary: DistilBERT fine-tuning
        Fallback: TF-IDF + classical ML ensemble
        
        Args:
            profile: Dataset profile
            
        Returns:
            Text strategy configuration
        """
        # Primary strategy: DistilBERT
        primary_model = "DistilBERT"
        fallback_model = "TF-IDF_Ensemble"
        
        preprocessing = ["lowercase", "remove_special_chars", "tokenize"]
        
        # Task type detection (classification vs sequence)
        if profile.target_type == TargetType.SEQUENCE:
            max_length = 512
            batch_size = 16
        else:
            max_length = 256
            batch_size = 32
        
        # Adaptive loss selection
        loss_function = self._select_loss_function(profile)
        
        return Strategy(
            modality=Modality.TEXT,
            primary_model=primary_model,
            fallback_model=fallback_model,
            preprocessing=preprocessing,
            augmentation=None,
            loss_function=loss_function,
            optimizer="AdamW",
            batch_size=batch_size,
            max_epochs=10,
            early_stopping_patience=3,
            hyperparameters={
                "pretrained": "distilbert-base-uncased",
                "max_length": max_length,
                "warmup_steps": 500,
                "num_workers": 4,
                "fallback_config": {
                    "vectorizer": "TF-IDF",
                    "max_features": 10000,
                    "ngram_range": (1, 3),
                    "ensemble": ["LogisticRegression", "LightGBM", "SVC"],
                    "voting": "soft"
                }
            },
            resource_constraints=self.resource_constraints,
            learning_rate=2e-5,
            weight_decay=0.01,
            dropout=0.1,
            mixed_precision=True,
            gradient_accumulation_steps=1,
            gradient_clip_norm=1.0,
            model_size="medium"
        )
    
    def get_seq2seq_strategy(self, profile: DatasetProfile) -> Strategy:
        """
        Get sequence-to-sequence strategy for text normalization and time series.
        
        Uses T5-Small model with beam search for inference.
        
        Args:
            profile: Dataset profile
            
        Returns:
            Seq2seq strategy configuration
        """
        return Strategy(
            modality=Modality.TIME_SERIES,
            primary_model="T5-Small",
            fallback_model="LSTM_Seq2Seq",
            preprocessing=["tokenize", "normalize"],
            augmentation=None,
            loss_function="CrossEntropy",
            optimizer="AdamW",
            batch_size=16,
            max_epochs=20,
            early_stopping_patience=5,
            hyperparameters={
                "max_source_length": 128,
                "max_target_length": 128,
                "beam_search_size": 4,
                "length_penalty": 0.6,
                "num_workers": 4
            },
            resource_constraints=self.resource_constraints,
            learning_rate=3e-4,
            weight_decay=0.01,
            dropout=0.1,
            mixed_precision=True,
            gradient_accumulation_steps=2,
            gradient_clip_norm=1.0,
            model_size="small"
        )
    
    def get_multimodal_strategy(self, profile: DatasetProfile) -> Strategy:
        """
        Get multimodal strategy with fusion configuration.
        
        Implements dual encoder architecture:
        - Image encoder: EfficientNet-B5
        - Tabular encoder: MLP
        - Fusion: Concatenation
        
        Args:
            profile: Dataset profile
            
        Returns:
            Multimodal strategy configuration
        """
        # Dual encoder configuration
        augmentation = {
            "library": "albumentations",
            "transforms": [
                "HorizontalFlip",
                "VerticalFlip",
                "RandomRotate90",
                "RandomBrightnessContrast"
            ]
        }
        
        # Adaptive loss selection
        loss_function = self._select_loss_function(profile)
        
        return Strategy(
            modality=Modality.MULTIMODAL,
            primary_model="DualEncoder",
            fallback_model="ImageOnly_EfficientNet",
            preprocessing=["normalize_images", "scale_tabular"],
            augmentation=augmentation,
            loss_function=loss_function,
            optimizer="AdamW",
            batch_size=32,
            max_epochs=50,
            early_stopping_patience=5,
            hyperparameters={
                "image_encoder": {
                    "backbone": "EfficientNet-B5",
                    "output_dim": 512
                },
                "tabular_encoder": {
                    "architecture": "MLP",
                    "hidden_dims": [256, 128],
                    "dropout": 0.3
                },
                "fusion": {
                    "method": "concatenate",
                    "fusion_dim": 640  # 512 + 128
                },
                "classifier": {
                    "hidden_dims": [256, 128],
                    "dropout": 0.5
                },
                "image_size": 224,
                "num_workers": 4
            },
            resource_constraints=self.resource_constraints,
            learning_rate=2e-4,
            weight_decay=0.01,
            dropout=0.3,
            mixed_precision=True,
            gradient_accumulation_steps=1,
            gradient_clip_norm=1.0,
            model_size="medium",
            augmentation_strength=1.0
        )
    
    def _select_loss_function(self, profile: DatasetProfile) -> str:
        """
        Select loss function based on class imbalance and target type.
        
        Uses FocalLoss for imbalanced classification (ratio > 5).
        Uses CrossEntropy for balanced classification.
        Uses MSE for regression.
        
        Args:
            profile: Dataset profile
            
        Returns:
            Loss function name
        """
        if profile.target_type == TargetType.REGRESSION:
            return "MSE"
        elif profile.target_type == TargetType.SEQUENCE:
            return "CrossEntropy"
        else:
            # Classification: check for class imbalance
            if profile.class_imbalance_ratio > 5.0:
                return "FocalLoss"
            else:
                return "CrossEntropy"

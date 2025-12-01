"""Template manager for accessing modality-specific strategy templates"""

from typing import Dict, Any, Optional
from src.templates.tabular_template import get_tabular_template
from src.templates.image_template import get_image_template
from src.templates.text_template import get_text_template
from src.templates.seq2seq_template import get_seq2seq_template
from src.templates.multimodal_template import get_multimodal_template
from src.templates.audio_template import get_audio_template


class TemplateManager:
    """
    Manages access to modality-specific strategy templates.
    
    Provides methods to retrieve appropriate templates based on:
    - Dataset modality (tabular, image, text, sequence, multimodal, audio)
    - Resource constraints
    - Dataset characteristics (size, complexity)
    """
    
    def __init__(self):
        """Initialize template manager"""
        self.template_getters = {
            'tabular': get_tabular_template,
            'image': get_image_template,
            'text': get_text_template,
            'time_series': get_seq2seq_template,  # Time series uses seq2seq
            'sequence': get_seq2seq_template,
            'seq2seq': get_seq2seq_template,  # Direct seq2seq mapping
            'text_normalization': get_seq2seq_template,  # Text normalization uses seq2seq
            'multimodal': get_multimodal_template,
            'audio': get_audio_template,  # Audio/spectrogram classification
        }
    
    def get_template(
        self,
        modality: str,
        memory_gb: Optional[float] = None,
        resource_constrained: bool = False,
        use_fallback: bool = False,
        is_text_normalization: bool = False,
        num_samples: Optional[int] = None
    ) -> str:
        """
        Get appropriate template for the given modality and constraints.
        
        Args:
            modality: Dataset modality (tabular, image, text, sequence, multimodal, seq2seq, audio)
            memory_gb: Estimated memory usage in GB (for tabular)
            resource_constrained: Whether to use resource-constrained variant
            use_fallback: Whether to use fallback strategy (for text)
            is_text_normalization: Whether this is a text normalization task
            num_samples: Number of samples in dataset (for large dataset detection)
            
        Returns:
            Template string with placeholders for dataset-specific values
            
        Raises:
            ValueError: If modality is not supported
        """
        # Auto-detect if resource-constrained needed based on dataset size
        is_large_dataset = num_samples and num_samples > 500000
        if is_large_dataset and not resource_constrained:
            import logging
            logging.getLogger(__name__).info(
                f"Large dataset detected ({num_samples:,} rows > 500k) - using resource-constrained template"
            )
            resource_constrained = True
        # Normalize modality name
        modality_lower = modality.lower()
        
        # Map modality aliases
        modality_map = {
            'sequence': 'seq2seq',
            'time_series': 'seq2seq',
            'text_normalization': 'seq2seq'
        }
        modality_key = modality_map.get(modality_lower, modality_lower)
        
        if modality_key not in self.template_getters:
            raise ValueError(
                f"Unsupported modality: {modality}. "
                f"Supported modalities: {list(self.template_getters.keys())}"
            )
        
        getter = self.template_getters[modality_key]
        
        # Call appropriate getter with relevant parameters
        if modality_key == 'tabular':
            return getter(memory_gb=memory_gb or 5.0, resource_constrained=resource_constrained)
        elif modality_key == 'text':
            return getter(use_fallback=use_fallback, resource_constrained=resource_constrained)
        elif modality_key == 'seq2seq':
            # Determine if this is text normalization
            is_text_norm = is_text_normalization or modality_lower in ['text_normalization', 'sequence']
            return getter(resource_constrained=resource_constrained, is_text_normalization=is_text_norm)
        elif modality_key == 'audio':
            return getter(resource_constrained=resource_constrained)
        else:
            return getter(resource_constrained=resource_constrained)
    
    def get_template_placeholders(self, modality: str) -> Dict[str, str]:
        """
        Get list of placeholders that need to be filled in the template.
        
        Args:
            modality: Dataset modality
            
        Returns:
            Dictionary mapping placeholder names to descriptions
        """
        common_placeholders = {
            'train_path': 'Path to training CSV file',
            'test_path': 'Path to test CSV file',
            'target_column': 'Name of target column',
            'id_column': 'Name of ID column',
            'prediction_column': 'Name of prediction column in submission',
            'seed': 'Random seed for reproducibility',
            'batch_size': 'Training batch size',
            'max_epochs': 'Maximum number of training epochs',
            'learning_rate': 'Learning rate',
            'weight_decay': 'Weight decay for regularization',
            'early_stopping_patience': 'Early stopping patience'
        }
        
        modality_specific = {
            'tabular': {
                'time_budget': 'Time budget for FLAML AutoML (seconds)',
                'metric': 'Evaluation metric',
                'task_type': 'Task type (classification or regression)',
                'objective': 'Objective function',
            },
            'image': {
                'image_column': 'Column containing image filenames',
                'image_dir': 'Directory containing images',
                'image_size': 'Image size for resizing',
                'num_classes': 'Number of classes',
                'loss_function': 'Loss function name',
                'gradient_accumulation_steps': 'Gradient accumulation steps'
            },
            'text': {
                'text_column': 'Column containing text data',
                'num_classes': 'Number of classes',
                'max_length': 'Maximum sequence length',
                'warmup_steps': 'Warmup steps for scheduler',
                'gradient_clip_norm': 'Gradient clipping norm',
                'gradient_accumulation_steps': 'Gradient accumulation steps'
            },
            'sequence': {
                'source_column': 'Column containing source sequences',
                'max_source_length': 'Maximum source sequence length',
                'max_target_length': 'Maximum target sequence length',
                'beam_search_size': 'Beam search size for inference',
                'length_penalty': 'Length penalty for beam search',
                'gradient_accumulation_steps': 'Gradient accumulation steps'
            },
            'seq2seq': {
                'source_column': 'Column containing source sequences (before)',
                'target_column': 'Column containing target sequences (after)',
                'max_source_length': 'Maximum source sequence length',
                'max_target_length': 'Maximum target sequence length',
                'beam_search_size': 'Beam search size for inference',
                'length_penalty': 'Length penalty for beam search',
                'gradient_accumulation_steps': 'Gradient accumulation steps'
            },
            'multimodal': {
                'image_column': 'Column containing image filenames',
                'image_dir': 'Directory containing images',
                'image_size': 'Image size for resizing',
                'num_classes': 'Number of classes',
                'gradient_accumulation_steps': 'Gradient accumulation steps'
            }
        }
        
        placeholders = common_placeholders.copy()
        modality_key = modality.lower()
        if modality_key in modality_specific:
            placeholders.update(modality_specific[modality_key])
        
        return placeholders
    
    def detect_text_normalization(self, dataset_info: Dict[str, Any]) -> bool:
        """
        Detect if dataset is a text normalization task.
        
        Args:
            dataset_info: Dictionary with dataset information
            
        Returns:
            True if text normalization task detected
        """
        # Check for common text normalization indicators
        columns = dataset_info.get('columns', [])
        columns_lower = {c.lower() for c in columns}
        
        # Text normalization has before/after columns
        has_before_after = 'before' in columns_lower and 'after' in columns_lower
        
        # And often has a 'class' column with semiotic classes
        has_class = 'class' in columns_lower
        
        # Competition ID might indicate text normalization
        competition_id = dataset_info.get('competition_id', '').lower()
        is_text_norm_competition = 'text-normalization' in competition_id or 'text_normalization' in competition_id
        
        return (has_before_after and has_class) or is_text_norm_competition
    
    def fill_template(self, template: str, values: Dict[str, Any]) -> str:
        """
        Fill template placeholders with actual values.
        
        Args:
            template: Template string with {placeholder} markers
            values: Dictionary mapping placeholder names to values
            
        Returns:
            Filled template string
        """
        return template.format(**values)
    
    def get_supported_modalities(self) -> list:
        """
        Get list of supported modalities.
        
        Returns:
            List of supported modality names
        """
        return list(self.template_getters.keys())
    
    def validate_template_values(self, modality: str, values: Dict[str, Any]) -> tuple[bool, list]:
        """
        Validate that all required placeholders are provided.
        
        Args:
            modality: Dataset modality
            values: Dictionary of values to fill template
            
        Returns:
            Tuple of (is_valid, missing_keys)
        """
        required_placeholders = self.get_template_placeholders(modality)
        missing_keys = [key for key in required_placeholders if key not in values]
        return len(missing_keys) == 0, missing_keys


# Singleton instance
_template_manager = None


def get_template_manager() -> TemplateManager:
    """
    Get singleton instance of TemplateManager.
    
    Returns:
        TemplateManager instance
    """
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager

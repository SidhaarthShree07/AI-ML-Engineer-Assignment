"""Strategy templates for different data modalities"""

from src.templates.template_manager import TemplateManager, get_template_manager
from src.templates.tabular_template import get_tabular_template
from src.templates.image_template import get_image_template
from src.templates.text_template import get_text_template
from src.templates.seq2seq_template import get_seq2seq_template
from src.templates.multimodal_template import get_multimodal_template

__all__ = [
    'TemplateManager',
    'get_template_manager',
    'get_tabular_template',
    'get_image_template',
    'get_text_template',
    'get_seq2seq_template',
    'get_multimodal_template'
]

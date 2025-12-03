"""Core data models for HybridAutoMLE agent"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class Modality(str, Enum):
    """Dataset modality types"""
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"
    TIME_SERIES = "time_series"
    MULTIMODAL = "multimodal"
    AUDIO = "audio"  # Audio/spectrogram classification (e.g., whale challenge)
    SEQ2SEQ = "seq2seq"  # Sequence-to-sequence (e.g., text normalization)


class TargetType(str, Enum):
    """Target variable types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEQUENCE = "sequence"


@dataclass
class DatasetProfile:
    """Comprehensive dataset profile"""
    modality: str
    confidence: float
    memory_gb: float
    num_samples: int
    num_features: int
    target_type: str
    class_imbalance_ratio: float
    missing_percentage: Dict[str, float]
    feature_correlations: Dict[str, float]
    has_metadata: bool
    estimated_gpu_memory_gb: float
    data_types: Optional[Dict[str, str]] = None  # Column name -> data type mapping


@dataclass
class ResourceConstraints:
    """Resource limits for execution"""
    max_vram_gb: float = 24.0
    max_ram_gb: float = 440.0
    max_runtime_hours: float = 24.0
    max_cpu_cores: int = 36


@dataclass
class Strategy:
    """ML strategy configuration"""
    modality: str
    primary_model: str
    fallback_model: Optional[str]
    preprocessing: List[str]
    augmentation: Optional[Dict[str, Any]]
    loss_function: str
    optimizer: str
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    hyperparameters: Dict[str, Any]
    resource_constraints: ResourceConstraints
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    dropout: float = 0.1
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: Optional[float] = None
    model_size: str = "medium"
    augmentation_strength: float = 1.0


@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    gpu_utilization_percent: float
    ram_used_gb: float
    ram_total_gb: float
    cpu_percent: float
    runtime_seconds: float


@dataclass
class ExecutionResult:
    """Result of code execution"""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    resource_usage: ResourceMetrics
    artifacts: List[str]


@dataclass
class ModalityResult:
    """Result of modality detection"""
    modality: str
    confidence: float
    heuristic_result: str
    profiling_result: str
    gemini_consensus: Optional[str]
    verification_status: bool
    verification_message: str


@dataclass
class DataProfile:
    """Statistical profile of dataset"""
    missing_values: Dict[str, float]
    target_distribution: Dict[str, int]
    feature_correlations: Dict[str, float]
    memory_usage_gb: float
    num_samples: int
    num_features: int
    data_types: Dict[str, str]


@dataclass
class VerificationResult:
    """File path verification result"""
    all_exist: bool
    missing_files: List[str]
    total_files: int
    verified_files: int


@dataclass
class ValidationResult:
    """Code validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


@dataclass
class RecoveryPlan:
    """Error recovery plan"""
    action: str
    params: Dict[str, Any]
    priority: int = 0


@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    train_score: float
    val_score: float
    loss: float
    epoch: int
    gpu_utilization: float


@dataclass
class PerformanceData:
    """Performance analysis data"""
    train_score: float
    val_score: float
    train_val_gap: float
    gpu_utilization: float
    issue_type: str


class ExecutionMode(str, Enum):
    """Execution mode types"""
    NORMAL = "normal"  # Local/Docker execution
    CLOUD = "cloud"    # Google Colab execution via notebook


@dataclass
class AgentConfig:
    """Agent configuration for dataset processing"""
    dataset_path: str
    competition_id: str
    output_dir: str
    max_runtime_hours: float = 24.0
    num_seeds: int = 3
    seed: int = 42  # Random seed for reproducibility
    execution_mode: str = "normal"  # "normal" or "cloud"
    gemini_model: str = "gemini-2.5-pro"
    gemini_api_key: Optional[str] = None
    eval_mode: bool = False
    competitions: List[str] = field(default_factory=list)
    docker_image: str = "hybrid-automle:latest"
    resource_constraints: Optional[ResourceConstraints] = None
    use_papermill: bool = False  # Use papermill for local kernel, False for Colab polling
    generate_only: bool = False  # If True in cloud mode, only generate notebook without waiting
    
    def __post_init__(self):
        """Initialize resource constraints with runtime from config"""
        if self.resource_constraints is None:
            self.resource_constraints = ResourceConstraints(
                max_runtime_hours=self.max_runtime_hours
            )

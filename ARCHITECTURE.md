# HybridAutoMLE Architecture & Workflow Documentation

## Overview

HybridAutoMLE is an autonomous Machine Learning Engineering agent that automatically processes datasets from MLEbench competitions. It detects data modality, selects appropriate strategies, generates training code, executes it in a sandboxed Docker environment, and produces submission files—all without human intervention.

---

## System Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                          │
│   python hybrid_agent.py --dataset_path ./data --competition_id test             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         1. INITIALIZATION PHASE                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │  hybrid_agent.py │───►│  StateManager   │───►│ ResourceManager │              │
│  │  (Entry Point)   │    │  (Session Init) │    │ (GPU/RAM Check) │              │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         2. DATA ANALYSIS PHASE                                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │ DatasetHandler  │───►│ModalityDetector │───►│  DataProfile    │              │
│  │ (Load CSV/Files)│    │ (Detect Type)   │    │ (Stats/Features)│              │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘              │
│                                                                                  │
│  Detected Modalities: TABULAR | IMAGE | TEXT | SEQ2SEQ | AUDIO | MULTIMODAL     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         3. STRATEGY SELECTION PHASE                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │ Controller      │───►│ StrategySystem  │───►│ TemplateManager │              │
│  │ (Gemini LLM)    │    │ (Select Strategy│    │ (Get Template)  │              │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘              │
│                                                                                  │
│  Templates: tabular_template | image_template | text_template |                  │
│             seq2seq_template | audio_template | multimodal_template              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         4. CODE GENERATION PHASE                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │ CodeGenerator   │───►│ Gemini API      │───►│  train.py       │              │
│  │ (Template+LLM)  │    │ (Enhancement)   │    │  (Generated)    │              │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         5. EXECUTION PHASE                                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │    Executor     │───►│  Docker Container│───►│  Training Loop  │              │
│  │ (Run Sandboxed) │    │  (ml-sandbox)   │    │  (Model Train)  │              │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘              │
│                                                                                  │
│  Monitors: GPU Usage | RAM Usage | Training Loss | Validation Score             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         6. SELF-IMPROVEMENT PHASE                                │
│  ┌─────────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │ PerformanceMonitor  │───►│RootCauseAnalyzer│───►│ StrategyEvolver │          │
│  │ (Track Metrics)     │    │ (Find Issues)   │    │ (Adapt Strategy)│          │
│  └─────────────────────┘    └─────────────────┘    └─────────────────┘          │
│                                                                                  │
│  Issues Detected: Overfitting | Underfitting | OOM | Slow Training               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         7. SUBMISSION PHASE                                      │
│  ┌─────────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │ SubmissionGenerator │───►│ EvaluationRunner│───►│  submission.csv │          │
│  │ (Format Predictions)│    │ (Validate)      │    │  (Final Output) │          │
│  └─────────────────────┘    └─────────────────┘    └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT FILES                                        │
│   submission.csv | reasoning_trace.json | model.pkl | train.py | logs/          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Hexo.ai Project/
│
├── hybrid_agent.py              # Main entry point - orchestrates entire workflow
├── requirements.txt             # Python dependencies for local development
├── Dockerfile                   # Main Docker image for full agent
├── docker-compose.yml           # Docker compose configuration
├── build_docker.sh              # Shell script to build Docker image
├── run_docker.sh                # Shell script to run Docker container
├── create_test_dataset.py       # Utility to create smaller test datasets
│
├── ARCHITECTURE.md              # This documentation file
├── README.md                    # Project overview and setup instructions
├── QUICKSTART.md                # Quick start guide for new users
│
├── src/                         # Source code - all core modules
│   ├── __init__.py
│   │
│   ├── controller/              # LLM-based decision making
│   │   ├── __init__.py
│   │   └── controller.py        # Gemini API integration for reasoning
│   │
│   ├── dataset/                 # Dataset loading and handling
│   │   ├── __init__.py
│   │   ├── dataset_handler.py   # Load train/test data, detect structure
│   │   └── mock_mle_bench.py    # Simple dataset utilities
│   │
│   ├── detector/                # Data modality detection
│   │   ├── __init__.py
│   │   └── modality_detector.py # Detect: tabular/image/text/audio/seq2seq
│   │
│   ├── error_handler/           # Error handling and recovery
│   │   ├── __init__.py
│   │   └── error_handler.py     # OOM recovery, retry logic
│   │
│   ├── evaluation/              # Model evaluation
│   │   ├── __init__.py
│   │   └── evaluation_runner.py # Run validation, compute metrics
│   │
│   ├── executor/                # Code execution engine
│   │   ├── __init__.py
│   │   └── executor.py          # Run generated code in Docker
│   │
│   ├── generator/               # Code generation
│   │   ├── __init__.py
│   │   └── code_generator.py    # Generate train.py from templates
│   │
│   ├── models/                  # Data models and types
│   │   ├── __init__.py
│   │   └── data_models.py       # Enums, dataclasses for config
│   │
│   ├── resource_manager/        # Resource monitoring
│   │   ├── __init__.py
│   │   └── resource_manager.py  # GPU/RAM monitoring, OOM prevention
│   │
│   ├── self_improvement/        # Self-improvement system
│   │   ├── __init__.py
│   │   ├── performance_monitor.py   # Track training metrics
│   │   ├── root_cause_analyzer.py   # Analyze failures
│   │   └── strategy_evolver.py      # Adapt strategies based on feedback
│   │
│   ├── state_manager/           # Session state management
│   │   ├── __init__.py
│   │   └── state_manager.py     # Reasoning trace, checkpoints
│   │
│   ├── strategies/              # ML strategy definitions
│   │   ├── __init__.py
│   │   └── strategy_system.py   # Strategy selection logic
│   │
│   ├── submission/              # Submission file generation
│   │   ├── __init__.py
│   │   └── submission_generator.py  # Format predictions as CSV
│   │
│   ├── templates/               # ML code templates
│   │   ├── __init__.py
│   │   ├── template_manager.py      # Route to correct template
│   │   ├── tabular_template.py      # FLAML AutoML for tabular
│   │   ├── image_template.py        # EfficientNet for images
│   │   ├── text_template.py         # DistilBERT for text
│   │   ├── seq2seq_template.py      # T5 for text normalization
│   │   ├── audio_template.py        # Spectrogram + CNN for audio
│   │   └── multimodal_template.py   # Dual encoder for mixed data
│   │
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── gemini_client.py     # Gemini API wrapper
│
├── executor/                    # Sandboxed executor environment
│   ├── Dockerfile               # Docker image for code execution
│   ├── requirements.txt         # Dependencies for executor
│   ├── run.sh                   # Entrypoint script
│   └── workspace/               # Mounted workspace for generated code
│       └── generated_code.py    # Placeholder for generated training code
│
├── data/                        # Full dataset (large files)
│   ├── train/
│   │   └── train.csv
│   └── test/
│       └── test.csv
│
├── data_test/                   # Smaller test dataset (for quick testing)
│   ├── train/
│   │   └── train.csv            # 50% sample of full train data
│   └── test/
│       └── test.csv             # 50% sample of full test data
│
├── output/                      # Generated outputs (per session)
│   └── session_YYYYMMDD_HHMMSS/
│       ├── submission.csv       # Final predictions
│       ├── train.py             # Generated training code
│       ├── model.pkl            # Saved model
│       ├── reasoning_trace.json # Decision log
│       └── flaml_log.txt        # Training logs
│
├── reference/                   # Reference notebooks for analysis
│   ├── text_norm_challenge.ipynb
│   └── tps-may-22-eda-lgbm-model (1).ipynb
│
└── cache/                       # Cache directory for models/data
```

---

## Core Components Explained

### 1. Entry Point

#### `hybrid_agent.py`
**Role**: Main orchestrator that coordinates all components.

**What it does**:
- Parses command-line arguments (dataset path, competition ID, output dir)
- Initializes all system components
- Runs the main workflow loop
- Handles top-level error recovery
- Reports final results

**Key Functions**:
```python
def main():
    # 1. Parse arguments
    # 2. Initialize StateManager, ResourceManager
    # 3. Load dataset via DatasetHandler
    # 4. Detect modality via ModalityDetector
    # 5. Select strategy via Controller + StrategySystem
    # 6. Generate code via CodeGenerator
    # 7. Execute via Executor
    # 8. Generate submission via SubmissionGenerator
```

---

### 2. Controller (`src/controller/`)

#### `controller.py`
**Role**: LLM-powered decision making using Gemini API.

**What it does**:
- Analyzes dataset characteristics
- Makes strategic decisions about model selection
- Handles ambiguous cases where heuristics fail
- Provides reasoning explanations for transparency

**Key Features**:
- Uses `gemini-2.0-flash-exp` model
- Maintains conversation context
- Generates structured JSON responses
- Fallback to heuristics if API fails

---

### 3. Dataset Handling (`src/dataset/`)

#### `dataset_handler.py`
**Role**: Load and parse dataset files.

**What it does**:
- Detects dataset structure (train/test folders or CSVs)
- Loads CSV files with proper encoding
- Handles large files with chunking
- Extracts column information and data types

#### `mock_mle_bench.py`
**Role**: Simple dataset utilities module.

**What it does**:
- Provides competition metric lookup for known competitions
- Safe CSV loading utilities
- Helper functions for finding CSV files in directories

**Note**: Main dataset handling is done by `dataset_handler.py`. This module exists for backwards compatibility only.

---

### 4. Modality Detection (`src/detector/`)

#### `modality_detector.py`
**Role**: Automatically detect what type of data we're working with.

**What it does**:
- **Tabular**: Pure numeric/categorical CSV data
- **Image**: Detects image paths (.jpg, .png, etc.)
- **Text**: Detects text columns for classification
- **Seq2Seq**: Detects input/output text pairs (text normalization)
- **Audio**: Detects audio files (.aif, .wav, etc.)
- **Multimodal**: Combination of above

**Detection Methods**:
1. File extension analysis
2. Column name patterns
3. Content sampling
4. Gemini consensus for edge cases

---

### 5. Strategy System (`src/strategies/`)

#### `strategy_system.py`
**Role**: Define and select ML strategies based on data characteristics.

**Strategies by Modality**:

| Modality | Primary Strategy | Fallback Strategy |
|----------|-----------------|-------------------|
| Tabular | FLAML AutoML + GPU LightGBM | XGBoost + CatBoost ensemble |
| Image | EfficientNet-B5 + Augmentation | ResNet-50 |
| Text | DistilBERT fine-tuning | TF-IDF + LogisticRegression |
| Seq2Seq | T5-small | Rule-based normalization |
| Audio | Mel-spectrogram + EfficientNet-B0 | MFCC + Random Forest |

---

### 6. Templates (`src/templates/`)

#### `template_manager.py`
**Role**: Route to correct template based on modality.

#### `tabular_template.py`
**Role**: Generate training code for tabular data.

**Features**:
- FLAML AutoML with configurable time budget
- GPU-accelerated LightGBM
- Target encoding for high-cardinality categoricals
- Feature importance analysis

#### `image_template.py`
**Role**: Generate training code for image classification.

**Features**:
- EfficientNet-B5 backbone (timm library)
- Albumentations augmentation pipeline
- Mixed precision (fp16) training
- Learning rate scheduling with warmup

#### `text_template.py`
**Role**: Generate training code for text classification.

**Features**:
- DistilBERT transformer
- Tokenization with max_length=256
- Gradient clipping for stability
- TF-IDF fallback for simple cases

#### `seq2seq_template.py`
**Role**: Generate training code for text normalization (seq2seq).

**Features**:
- T5-small encoder-decoder
- Beam search decoding
- Entity type detection (DATE, MONEY, etc.)
- Special handling for numbers and currencies

#### `audio_template.py`
**Role**: Generate training code for audio classification.

**Features**:
- Librosa mel-spectrogram extraction
- 128 mel bins, 10-second clips
- EfficientNet-B0 on spectrograms
- Binary classification for whale detection

#### `multimodal_template.py`
**Role**: Generate training code for mixed modality data.

**Features**:
- Dual encoder architecture
- Separate encoders for image/text/tabular
- Late fusion with attention
- Adaptive loss weighting

---

### 7. Code Generator (`src/generator/`)

#### `code_generator.py`
**Role**: Generate executable training code from templates.

**What it does**:
1. Gets base template for detected modality
2. Fills in dataset-specific parameters
3. Optionally enhances with Gemini suggestions
4. Adds submission format detection
5. Saves as `train.py` in output directory

---

### 8. Executor (`src/executor/`)

#### `executor.py`
**Role**: Run generated code in sandboxed Docker container.

**What it does**:
- Mounts data and output directories
- Runs `train.py` in Docker container
- Captures stdout/stderr
- Monitors resource usage
- Enforces timeout limits

**Docker Configuration**:
```python
DockerConfig(
    image="ml-sandbox:latest",
    gpu=True,
    memory_limit="24g",
    timeout_hours=24
)
```

---

### 9. Self-Improvement System (`src/self_improvement/`)

#### `performance_monitor.py`
**Role**: Track training metrics in real-time.

**Metrics Tracked**:
- Training loss per epoch
- Validation score per epoch
- GPU memory usage
- Training time per epoch

#### `root_cause_analyzer.py`
**Role**: Analyze failures and identify root causes.

**Issue Detection**:
- Overfitting: train-val gap > 0.15
- Underfitting: train score < 0.7
- OOM: CUDA out of memory errors
- Slow training: epoch time > expected

#### `strategy_evolver.py`
**Role**: Adapt strategy based on identified issues.

**Adaptations**:
| Issue | Adaptation |
|-------|-----------|
| Overfitting | Increase dropout, add weight decay |
| Underfitting | Increase model capacity, more epochs |
| OOM | Reduce batch size, enable gradient checkpointing |
| Slow training | Enable mixed precision, reduce model size |

---

### 10. State Manager (`src/state_manager/`)

#### `state_manager.py`
**Role**: Maintain session state and reasoning trace.

**What it stores**:
- Session ID and timestamps
- All decisions made with reasoning
- Execution logs and metrics
- Checkpoints for recovery

**Output**: `reasoning_trace.json`

---

### 11. Resource Manager (`src/resource_manager/`)

#### `resource_manager.py`
**Role**: Monitor and manage computational resources.

**What it monitors**:
- GPU memory usage (nvidia-smi)
- System RAM usage (psutil)
- CPU utilization
- Disk space

**OOM Prevention**:
- Pre-execution resource check
- Dynamic batch size adjustment
- Automatic garbage collection

---

### 12. Submission Generator (`src/submission/`)

#### `submission_generator.py`
**Role**: Format model predictions into submission file.

**What it does**:
1. Load test predictions from model
2. Match format from `sample_submission.csv`
3. Auto-detect ID column and prediction column
4. Validate row count matches
5. Save as `submission.csv`

---

### 13. Evaluation Runner (`src/evaluation/`)

#### `evaluation_runner.py`
**Role**: Validate submission and compute metrics.

**What it does**:
- Load submission file
- Validate format matches expected
- Compute competition metric (AUC, log_loss, accuracy)
- Multi-seed evaluation for robust scores

---

### 14. Error Handler (`src/error_handler/`)

#### `error_handler.py`
**Role**: Handle errors and implement recovery strategies.

**Error Types & Recovery**:

| Error Type | Recovery Strategy |
|------------|-------------------|
| OOM | Reduce batch size by 50%, enable fp16 |
| NaN Loss | Reduce learning rate, add gradient clipping |
| File Not Found | Search alternative paths |
| Timeout | Save checkpoint, resume with fallback |
| Import Error | Install missing package |

**Max retries**: 3 per error type

---

### 15. Utilities (`src/utils/`)

#### `gemini_client.py`
**Role**: Wrapper for Google Gemini API.

**Features**:
- Automatic retry with exponential backoff
- Rate limiting
- Response parsing and validation
- Error handling for API failures

---

### 16. Data Models (`src/models/`)

#### `data_models.py`
**Role**: Define data structures and enums.

**Key Types**:
```python
class Modality(Enum):
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"
    SEQ2SEQ = "seq2seq"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"

@dataclass
class DataProfile:
    num_rows: int
    num_cols: int
    memory_mb: float
    target_distribution: dict
    # ...

@dataclass
class DockerConfig:
    image: str
    use_docker: bool
    gpu: bool
    memory_limit: str
    # ...
```

---

## Executor Docker Environment

### `executor/Dockerfile`
**Role**: Define isolated execution environment.

**Base Image**: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`

**Installed Packages**:
- PyTorch 2.0.1 + CUDA 11.8
- Transformers 4.35.0
- FLAML, LightGBM, XGBoost, CatBoost
- Librosa, SoundFile (audio)
- Albumentations, Pillow (image)
- Pandas, NumPy, Scikit-learn

---

## Data Directories

### `data/`
Full dataset directory. Contains:
- `train/train.csv` - Training data (may be large, 50MB+)
- `test/test.csv` - Test data for predictions

### `data_test/`
Smaller test dataset for quick testing. Created by:
```bash
python create_test_dataset.py --fraction 0.5
```

---

## Output Structure

Each run creates a session directory:

```
output/session_20251201_085303/
├── submission.csv           # Final predictions (required output)
├── train.py                 # Generated training code
├── model.pkl                # Saved trained model
├── reasoning_trace.json     # Complete decision log
├── flaml_log.txt            # FLAML training output
├── checkpoints/             # Model checkpoints
│   ├── checkpoint_epoch_5.pt
│   └── checkpoint_epoch_10.pt
└── logs/
    ├── training.log         # Training stdout
    └── execution.log        # Agent execution log
```

---

## Supported Competitions (MLEbench Lite)

| Competition ID | Modality | Metric | Template Used |
|---------------|----------|--------|---------------|
| `siim-isic-melanoma-classification` | Image | AUC | `image_template.py` |
| `spooky-author-identification` | Text | Log Loss | `text_template.py` |
| `tabular-playground-series-may-2022` | Tabular | AUC | `tabular_template.py` |
| `text-normalization-challenge-english-language` | Seq2Seq | Accuracy | `seq2seq_template.py` |
| `the-icml-2013-whale-challenge-right-whale-redux` | Audio | AUC | `audio_template.py` |

---

## Quick Reference Commands

```bash
# Run on full dataset
python hybrid_agent.py --dataset_path ./data --competition_id test --output_dir ./output

# Run on smaller test dataset (faster)
python create_test_dataset.py --fraction 0.5
python hybrid_agent.py --dataset_path ./data_test --competition_id test --output_dir ./output

# Build Docker image
docker build -t ml-sandbox:latest -f executor/Dockerfile .

# Set Gemini API key (Windows PowerShell)
$env:GEMINI_API_KEY="your-api-key"
```

---

## Error Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `GEMINI_API_KEY not set` | API key missing | Set environment variable |
| `Docker image not found` | Image not built | Run `docker build` command |
| `CUDA out of memory` | Batch size too large | Reduce batch_size in template |
| `FileNotFoundError: train.csv` | Wrong dataset path | Check `--dataset_path` argument |
| `tokenizers version conflict` | Dependency mismatch | Use `tokenizers==0.14.1` |

---

## License

MIT License

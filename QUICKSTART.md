# Quick Start Guide

Get HybridAutoMLE running in 5 minutes on **Windows**, **macOS**, or **Linux**.

## Prerequisites

- **Python 3.10+** (3.8+ minimum)
- **Docker** (optional, for isolated execution)
- **Gemini API key** (optional, for LLM enhancement)
- **CUDA 11.8+** (optional, for GPU support)

## Installation

### Step 1: Install Python Dependencies

**All Platforms:**
```bash
pip install -r requirements.txt
```

### Step 2: Set Gemini API Key (Optional)

**Linux/macOS:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

**Windows (CMD):**
```cmd
set GEMINI_API_KEY=your-api-key-here
```

### Step 3: Build Docker Image (Optional)

**All Platforms:**
```bash
docker build -t ml-sandbox:latest -f executor/Dockerfile .
```

## Quick Start: Run Your First Agent

**All Platforms:**
```bash
python hybrid_agent.py --dataset_path ./data/your-dataset --competition_id your-competition-id --output_dir ./results
```

**What this does:**
- Analyzes your dataset and detects modality automatically
- Runs with default 24 hour time limit (set to 0 for no limit)
- Outputs results to `./results` directory

## What Happens Next?

1. **Initialization** (1-2 min): Agent loads dataset and initializes components
2. **Profiling** (1-5 min): Detects modality and profiles data characteristics
3. **Strategy Selection** (<1 min): Selects optimal ML strategy
4. **Code Generation** (1-2 min): Generates training code (with Gemini if available)
5. **Training** (varies): Trains model with FLAML AutoML
6. **Submission** (1-5 min): Generates predictions and submission.csv

## Check Results

**Linux/macOS:**
```bash
# View submission file
cat ./output/session_*/submission.csv

# View reasoning trace
cat ./output/session_*/reasoning_trace.json

# View training log
cat ./output/session_*/flaml_log.txt
```

**Windows (PowerShell):**
```powershell
# View submission file
Get-Content .\output\session_*\submission.csv

# View reasoning trace
Get-Content .\output\session_*\reasoning_trace.json

# View training log
Get-Content .\output\session_*\flaml_log.txt
```

**Windows (CMD):**
```cmd
# View submission file
type output\session_*\submission.csv

# View reasoning trace
type output\session_*\reasoning_trace.json

# View training log
type output\session_*\flaml_log.txt
```

## Common Issues

### Gemini API Key Not Set

**Error:** `ValueError: Gemini API key must be provided`

**Solution:**

**Linux/macOS:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

**Windows (CMD):**
```cmd
set GEMINI_API_KEY=your-api-key-here
```

Or run without Gemini (templates still work):
```bash
python hybrid_agent.py --dataset_path ./data/your-dataset --competition_id test --output_dir ./output
```

### Docker Not Running

**Linux:**
```bash
sudo systemctl start docker
```

**macOS:**
```bash
open -a Docker
```

**Windows:**
Open Docker Desktop from Start Menu

### Python Version Too Old

**Error:** `Python 3.10+ required`

**Solution:** Download Python 3.10+ from [python.org](https://www.python.org/downloads/)

## Next Steps

- Review [README.md](README.md) for comprehensive documentation
- Check [DOCKER_SETUP.md](DOCKER_SETUP.md) for Docker configuration

## Example Commands

### Quick Test with Smaller Dataset

Create a smaller dataset for fast functionality testing:

```bash
# Create 50% sample (default)
python create_test_dataset.py

# Create 25% sample for faster testing
python create_test_dataset.py --fraction 0.25

# Create 10% sample for very quick testing
python create_test_dataset.py --fraction 0.1

# Run agent on the smaller test dataset
python hybrid_agent.py --dataset_path ./data_test --competition_id test --output_dir ./output
```

### Tabular Data

**Linux/macOS:**
```bash
python hybrid_agent.py --dataset_path ./data/tabular-playground --competition_id tabular-playground-series --output_dir ./results/tabular
```

**Windows:**
```cmd
python hybrid_agent.py --dataset_path .\data\tabular-playground --competition_id tabular-playground-series --output_dir .\results\tabular
```

### Image Classification

**Linux/macOS:**
```bash
python hybrid_agent.py --dataset_path ./data/siim-isic-melanoma --competition_id siim-isic-melanoma-classification --output_dir ./results/melanoma
```

**Windows:**
```cmd
python hybrid_agent.py --dataset_path .\data\siim-isic-melanoma --competition_id siim-isic-melanoma-classification --output_dir .\results\melanoma
```

### Text Classification

**Linux/macOS:**
```bash
python hybrid_agent.py --dataset_path ./data/spooky-author --competition_id spooky-author-identification --output_dir ./results/spooky
```

**Windows:**
```cmd
python hybrid_agent.py --dataset_path .\data\spooky-author --competition_id spooky-author-identification --output_dir .\results\spooky
```

### Seq2Seq (Text Normalization)

**Linux/macOS:**
```bash
python hybrid_agent.py --dataset_path ./data/text-normalization --competition_id text-normalization-challenge-english-language --output_dir ./results/seq2seq
```

**Windows:**
```cmd
python hybrid_agent.py --dataset_path .\data\text-normalization --competition_id text-normalization-challenge-english-language --output_dir .\results\seq2seq
```

### Audio Classification (Whale Detection)

**Linux/macOS:**
```bash
python hybrid_agent.py --dataset_path ./data/whale-challenge --competition_id the-icml-2013-whale-challenge-right-whale-redux --output_dir ./results/audio
```

**Windows:**
```cmd
python hybrid_agent.py --dataset_path .\data\whale-challenge --competition_id the-icml-2013-whale-challenge-right-whale-redux --output_dir .\results\audio
```

### Multi-Seed Evaluation

**All Platforms:**
```bash
python hybrid_agent.py --dataset_path ./data/your-dataset --competition_id test_multi --output_dir ./results --num_seeds 5 --max_runtime_hours 4
```

### With Time Limit

**All Platforms:**
```bash
# 2 hour time limit
python hybrid_agent.py --dataset_path ./data/your-dataset --competition_id test_timed --output_dir ./output --max_runtime_hours 2

# No time limit (recommended for first run)
python hybrid_agent.py --dataset_path ./data/your-dataset --competition_id test_unlimited --output_dir ./output --max_runtime_hours 0
```

## Docker Setup (Optional)

### Build Docker Image

**All Platforms:**
```bash
docker build -t ml-sandbox:latest -f executor/Dockerfile .
```

This will:
- Install all ML libraries (FLAML, LightGBM, XGBoost, CatBoost, scikit-learn)
- Install PyTorch with CUDA support
- Install transformers and audio processing libraries
- Set up isolated execution environment

### Run with Docker

**All Platforms:**
```bash
python hybrid_agent.py --dataset_path ./data/your-dataset --competition_id docker_test --output_dir ./output
```

The agent automatically uses Docker if the image is available.

### Disable Docker

To run locally without Docker, modify `hybrid_agent.py`:

```python
docker_config = DockerConfig(
    image="ml-sandbox:latest",
    use_docker=False  # Set to False
)
```

## Platform-Specific Notes

### Windows

- **Use PowerShell** for better command compatibility
- **Path separators**: Use `\` or `/` (both work in Python)
- **Environment variables**: Use `$env:VAR="value"` in PowerShell
- **Docker**: Requires Docker Desktop for Windows

### macOS

- **Homebrew recommended** for installing git, Python
- **Docker**: Requires Docker Desktop for Mac
- **M1/M2 Macs**: Full support, no special configuration needed

### Linux

- **Docker**: May require `sudo` for Docker commands
- **GPU support**: Requires NVIDIA drivers and nvidia-docker
- **Permissions**: May need to add user to docker group

## Performance Tips

1. **Start with no time limit**: Use `--max_runtime_hours 0` for first runs
2. **Enable Gemini**: Better code optimization with LLM enhancement
3. **Use Docker**: Isolated environment prevents conflicts
4. **Monitor logs**: Check `flaml_log.txt` for training progress

## Verification

After running, verify your setup:

**Check submission file exists:**
```bash
# Linux/macOS
ls -la ./output/session_*/submission.csv

# Windows
dir .\output\session_*\submission.csv
```

**Check model was saved:**
```bash
# Linux/macOS
ls -la ./output/session_*/model.pkl

# Windows
dir .\output\session_*\model.pkl
```

**Check training completed:**
```bash
# Linux/macOS
tail -n 20 ./output/session_*/flaml_log.txt

# Windows (PowerShell)
Get-Content .\output\session_*\flaml_log.txt -Tail 20
```

## Support

For issues or questions:
- **Docker**: Check [DOCKER_SETUP.md](DOCKER_SETUP.md)
- **Troubleshooting**: Review `flaml_log.txt` and `reasoning_trace.json`

## Supported Competitions (MLEbench Lite)

| Competition | Type | Command Example |
|------------|------|-----------------|
| `tabular-playground-series-may-2022` | Tabular | `--dataset_path ./data/tps-may-22` |
| `siim-isic-melanoma-classification` | Image | `--dataset_path ./data/siim-melanoma` |
| `spooky-author-identification` | Text | `--dataset_path ./data/spooky-author` |
| `text-normalization-challenge-english-language` | Seq2Seq | `--dataset_path ./data/text-norm` |
| `the-icml-2013-whale-challenge-right-whale-redux` | Audio | `--dataset_path ./data/whale` |

## Quick Reference

### Essential Commands

```bash
# Create smaller test dataset (for quick testing)
python create_test_dataset.py --fraction 0.5

# Quick test (no time limit)
python hybrid_agent.py --dataset_path ./data_test --competition_id test --output_dir ./output --max_runtime_hours 0

# With Gemini (Linux/Mac)
export GEMINI_API_KEY="your-key"
python hybrid_agent.py --dataset_path ./data/your-dataset --competition_id test --output_dir ./output

# With Gemini (Windows PowerShell)
$env:GEMINI_API_KEY="your-key"
python hybrid_agent.py --dataset_path ./data/your-dataset --competition_id test --output_dir ./output

# Build Docker
docker build -t ml-sandbox:latest -f executor/Dockerfile .
```

### File Locations

- **Submission**: `./output/session_*/submission.csv`
- **Model**: `./output/session_*/model.pkl`
- **Training code**: `./output/session_*/train.py`
- **Training log**: `./output/session_*/flaml_log.txt`
- **Reasoning trace**: `./output/session_*/reasoning_trace.json`

Happy AutoML-ing! 🚀

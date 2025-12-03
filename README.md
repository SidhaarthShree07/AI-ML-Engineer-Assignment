# HybridAutoMLE: Autonomous ML Agent

An autonomous ML agent that processes any dataset (tabular, image, text, audio, seq2seq) with a single command.

## Execution Modes

- **Normal (`-n`)**: Local/Docker execution with GPU support
- **Colab (`-c`)**: Google Colab execution via notebook interface

```bash
# Normal mode (default)
python hybrid_agent.py --dataset_path ./data --competition_id test --output_dir ./output -n

# Colab mode (Google Colab)
python hybrid_agent.py --dataset_path ./data --competition_id test --output_dir ./output -c
```

## How It Understands Tasks

The agent uses **hybrid 3-phase modality detection**:
1. **Heuristic**: Analyzes file extensions and column patterns
2. **Statistical**: Profiles data types, missing values, class imbalance
3. **LLM Consensus**: Gemini resolves conflicts for edge cases

## Why It Chooses Strategies

Strategies are selected based on **modality + data characteristics**:

| Modality | Strategy | Key Features |
|----------|----------|--------------|
| **Tabular** | LightAutoML (TabularAutoML) | Auto feature engineering, StandardScaler, cross-validation |
| **Image** | EfficientNet-B0 | Mixed precision training, data augmentation, pretrained weights |
| **Text** | DistilBERT | Transformer-based, fine-tuning with AdamW optimizer |
| **Audio** | Mel-Spectrogram + EfficientNet | Audio to spectrogram conversion, CNN classification |
| **Seq2Seq** | T5-small | Text-to-text transformer, beam search decoding |

## Self-Improvement Loop

After each training run, the `CodeEnhancer` analyzes output:
- **Success + decent score (≥0.5)**: Move on
- **Errors or poor performance**: LLM enhances code and retries (max 3x)
- Fixes include: batch size reduction, regularization, memory optimization

## MLEbench Lite Evaluation

Run 3-seed evaluation with mean ± SEM reporting:
```bash
python run_mlebench_eval.py --output_dir ./mlebench_results
```

### Evaluation Report

#### Competition: Tabular Playground Series - May 2022

**Run Results (3 Seeds):**

| Seed | Score | Medal |
|:----:|:-----:|:-----:|
| 42 | 0.99518 | None |
| 43 | 0.98519 | None |
| 44 | 0.99163 | None |

**Summary:**

| Metric | Value |
|:-------|------:|
| Total Runs | 3 |
| Successful Submissions | 3/3 (100%) |
| Best Score | 0.99518 |
| Mean Score | 0.99067 |
| Total Runtime | ~13 hours |
| Hardware | CPU only |

**Medal Thresholds:**

| Medal | Threshold Required | Our Best Score | Gap |
|:-----:|:------------------:|:--------------:|:---:|
| 🥇 Gold | 0.99823 | 0.99518 | -0.00305 |
| 🥈 Silver | 0.99822 | 0.99518 | -0.00304 |
| 🥉 Bronze | 0.99818 | 0.99518 | -0.00300 |
| ✅ Above Median | 0.97268 | 0.99518 | +0.02250 |

> **Result:** Our best score (0.99518) is **above median** but falls short of bronze by ~0.003. With GPU acceleration and longer training time, scores could potentially improve.

---

### ⚠️ Important Note on Remaining Datasets

For the remaining 4 datasets in the MLEbench evaluation:

| Dataset | Status | Constraint |
|:--------|:------:|:-----------|
| `text-normalization-challenge-english-language` | 🔄 Running | Code execution in progress |
| Other image/multimodal datasets | ⏸️ Pending | Requires GPU with high VRAM (16GB+) |
| Large-scale datasets | ⏸️ Pending | Requires 200GB+ storage for dataset alone |

**Hardware Limitations:**
- Local machine lacks sufficient GPU VRAM for large vision models
- Storage constraints prevent downloading datasets exceeding 200GB
- Full evaluation requires cloud infrastructure (AWS/GCP with A100 GPUs)

*The evaluation framework is fully functional - hardware resources are the limiting factor for complete benchmark execution.*

---

## Reasoning Traces

All decisions logged to `output/session_*/reasoning_trace_*.json`.

---
See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture.

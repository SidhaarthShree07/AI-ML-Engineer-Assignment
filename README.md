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
| **Seq2Seq** | ByT5 (google/byt5-small) | Byte-level text-to-text transformer, token-level mapping with sentence aggregation & token reconstruction |

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

<img width="2046" height="305" alt="image" src="https://github.com/user-attachments/assets/ca451df3-6555-47b7-9323-ef2e41b4a1f3" />

The following table aggregates the CLI-style `mlebench grade-sample` JSON outputs into a single table (detailed version). Values are taken from existing evaluation outputs under `mlebench_results`.

| competition_id | seed | score | gold_threshold | silver_threshold | bronze_threshold | median_threshold | any_medal | gold_medal | silver_medal | bronze_medal | above_median | submission_exists | valid_submission | is_lower_better | submission_path |
|---|:---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| tabular-playground-series-may-2022 | 42 | 0.99518 | 0.99823 | 0.99822 | 0.99818 | 0.972675 | false | false | false | false | true | true | true | false | mlebench_results/tabular-playground-series-may-2022/seed_42/submission.csv |
| tabular-playground-series-may-2022 | 43 | 0.98519 | 0.99823 | 0.99822 | 0.99818 | 0.972675 | false | false | false | false | true | true | true | false | mlebench_results/tabular-playground-series-may-2022/seed_43/submission.csv |
| tabular-playground-series-may-2022 | 44 | 0.99163 | 0.99823 | 0.99822 | 0.99818 | 0.972675 | false | false | false | false | true | true | true | false | mlebench_results/tabular-playground-series-may-2022/seed_44/submission.csv |
| spooky-author-identification | 40 | 0.32924 | 0.16506 | 0.26996 | 0.29381 | 0.418785 | false | false | false | false | true | true | true | true | mlebench_results/spooky-author-identification/seed_40/submission.csv |
| spooky-author-identification | 41 | 0.37119 | 0.16506 | 0.26996 | 0.29381 | 0.418785 | false | false | false | false | true | true | true | true | mlebench_results/spooky-author-identification/seed_41/submission.csv |
| spooky-author-identification | 42 | 0.34312 | 0.16506 | 0.26996 | 0.29381 | 0.418785 | false | false | false | false | true | true | true | true | mlebench_results/spooky-author-identification/seed_42/submission.csv |
| text-normalization-challenge-english-language | 40 | 0.46183 | 0.99724 | 0.99135 | 0.99038 | 0.99037 | false | false | false | false | false | true | true | false | mlebench_results/text-normalization-challenge-english-language/seed_40/submission.csv |

> **⚠️ Important (text-normalization challenge)**  
> The results of the text-normalization challenge were poor due to a drastic change in the template base code format, which was done to make sure the code is able to run completely on limited CPU-oriented hardware. Gemini rewrote the training code to be optimized for CPU execution (to reduce memory/VRAM needs), which significantly degraded model performance. Other seeds were not executed, and the remaining two competitions were not attempted because the available lower computational power makes achieving competitive results impractical.


### Per-Submission Reports
> To reproduce the per-submission report locally, run for each submission:

```bash
mlebench grade-sample <PATH_TO_SUBMISSION> <competition_name>
# Example:
mlebench grade-sample mlebench_results/tabular-playground-series-may-2022/seed_43/submission.csv tabular-playground-series-may-2022
```


### ⚠️ Important Note on Remaining Datasets

For the remaining 2 datasets in the MLEbench evaluation:

| Dataset | Status | Constraint |
|:--------|:------:|:-----------|
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

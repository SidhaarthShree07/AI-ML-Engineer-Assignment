# HybridAutoMLE: Autonomous ML Agent

An autonomous ML agent that processes any dataset (tabular, image, text, audio, seq2seq) with a single command.

## How It Understands Tasks

The agent uses **hybrid 3-phase modality detection**:
1. **Heuristic**: Analyzes file extensions and column patterns
2. **Statistical**: Profiles data types, missing values, class imbalance
3. **LLM Consensus**: Gemini resolves conflicts for edge cases

## Why It Chooses Strategies

Strategies are selected based on **modality + data characteristics**:
- **Tabular**: FLAML AutoML (small) or GPU LightGBM (large)
- **Image**: EfficientNet with mixed precision
- **Text**: DistilBERT or TF-IDF fallback
- **Audio**: Mel-spectrogram + EfficientNet
- **Seq2Seq**: T5-small with beam search
- **Large datasets (>500k rows)**: Auto-switches to lightweight templates, drops CV

## Self-Improvement Loop

After each training run, the `CodeEnhancer` analyzes output:
- **Success + decent score (≥0.5)**: Move on
- **Errors or poor performance**: LLM enhances code and retries (max 3x)
- Fixes include: batch size reduction, regularization, memory optimization

## Quick Start

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
python hybrid_agent.py --dataset_path ./data --competition_id test --output_dir ./output
```

## Reasoning Traces

All decisions logged to `output/session_*/reasoning_trace_*.json`.

---
See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture.

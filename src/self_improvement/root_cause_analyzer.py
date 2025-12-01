"""Root cause analysis using Gemini"""

import json
import logging
from typing import Dict, Any
from src.models.data_models import Strategy, TrainingMetrics
from src.utils.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


def analyze_with_gemini(
    gemini_client: GeminiClient,
    strategy: Strategy,
    metrics: TrainingMetrics,
    error_log: str = ""
) -> Dict[str, Any]:
    """
    Perform root cause analysis using Gemini.
    
    Implements Tier 2 of self-improvement: Root Cause Analysis
    Uses Gemini to analyze performance issues and suggest fixes.
    
    Args:
        gemini_client: Gemini client for LLM analysis
        strategy: Current ML strategy
        metrics: Training metrics showing issues
        error_log: Optional error log for additional context
        
    Returns:
        Analysis dictionary with:
        - root_causes: List of likely root causes
        - fixes: List of specific fixes
        - priority_order: Priority order for applying fixes
        - next_strategy: Suggested strategy improvements
    """
    logger.info("Performing Gemini root cause analysis")
    
    # Calculate train-val gap
    train_val_gap = metrics.train_score - metrics.val_score
    
    # Build comprehensive prompt
    prompt = f"""Perform root cause analysis for ML training issue:

STRATEGY:
- Model: {strategy.primary_model}
- Modality: {strategy.modality}
- Batch Size: {strategy.batch_size}
- Learning Rate: {strategy.learning_rate}
- Dropout: {strategy.dropout}
- Weight Decay: {strategy.weight_decay}
- Mixed Precision: {strategy.mixed_precision}
- Model Size: {strategy.model_size}

PERFORMANCE:
- Train Score: {metrics.train_score:.4f}
- Val Score: {metrics.val_score:.4f}
- Train-Val Gap: {train_val_gap:.4f}
- Loss: {metrics.loss:.4f}
- Epoch: {metrics.epoch}
- GPU Utilization: {metrics.gpu_utilization:.1f}%

ERRORS (last 500 chars):
{error_log[-500:] if error_log else "No errors logged"}

Analyze and provide:
1. Top 3 likely root causes (with confidence > 80%)
2. Specific fixes for each root cause
3. Priority order for applying fixes (0=highest priority)
4. Next strategy suggestions with expected improvement

Response format: JSON with keys "root_causes", "fixes", "priority_order", "next_strategy"
"""
    
    try:
        # Get analysis from Gemini
        analysis = gemini_client.generate_json(prompt)
        
        # Validate response structure
        required_keys = ["root_causes", "fixes", "priority_order"]
        for key in required_keys:
            if key not in analysis:
                logger.warning(f"Missing key '{key}' in Gemini response, adding default")
                analysis[key] = []
        
        # Ensure next_strategy exists
        if "next_strategy" not in analysis:
            analysis["next_strategy"] = {}
        
        logger.info(f"Root cause analysis complete: {len(analysis['root_causes'])} causes identified")
        logger.debug(f"Root causes: {analysis['root_causes']}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Gemini root cause analysis failed: {e}")
        
        # Return default analysis based on metrics
        default_analysis = _generate_default_analysis(metrics, train_val_gap)
        logger.info("Using default analysis due to Gemini failure")
        
        return default_analysis


def _generate_default_analysis(
    metrics: TrainingMetrics,
    train_val_gap: float
) -> Dict[str, Any]:
    """
    Generate default analysis when Gemini is unavailable.
    
    Args:
        metrics: Training metrics
        train_val_gap: Train-validation gap
        
    Returns:
        Default analysis dictionary
    """
    root_causes = []
    fixes = []
    priority_order = []
    
    # Analyze based on metrics
    if train_val_gap > 0.15:
        root_causes.append("Overfitting: model memorizing training data")
        fixes.append("Increase regularization (dropout, weight decay)")
        priority_order.append(0)
    
    if metrics.train_score < 0.7:
        root_causes.append("Underfitting: model too simple or learning rate too low")
        fixes.append("Increase model complexity or learning rate")
        priority_order.append(1)
    
    if metrics.gpu_utilization < 60.0:
        root_causes.append("Resource inefficiency: GPU underutilized")
        fixes.append("Increase batch size to improve GPU utilization")
        priority_order.append(2)
    
    # If no specific issues, provide general suggestions
    if not root_causes:
        root_causes.append("Performance within normal range")
        fixes.append("Continue training with current strategy")
        priority_order.append(0)
    
    return {
        "root_causes": root_causes,
        "fixes": fixes,
        "priority_order": priority_order,
        "next_strategy": {
            "suggestions": fixes
        }
    }

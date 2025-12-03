#!/usr/bin/env python
"""
Compute MLE-Bench Scores

Computes mean ± standard error of Any Medal (%) for each competition
and overall, following the MLE-Bench evaluation protocol.

Usage:
    python compute_mlebench_scores.py --results_dir ./mlebench_results

Output format:
    Competition: siim-isic-melanoma-classification
    Any Medal (%): 66.67 ± 33.33 (2/3 runs)
    
    Overall Any Medal (%): 60.00 ± 12.25 (9/15 runs)
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.evaluation.mlebench_grader import MLEBenchGrader, GradeResult
except ImportError:
    # Standalone mode
    MLEBenchGrader = None


@dataclass
class CompetitionScore:
    """Score summary for a single competition."""
    competition_id: str
    num_seeds: int
    num_medals: int
    medal_rate: float  # 0-100 percentage
    mean_any_medal: float  # Mean of binary medal indicators (0 or 100)
    std_error: float  # Standard error of the mean
    individual_results: List[Dict]


@dataclass
class EvaluationSummary:
    """Overall evaluation summary."""
    num_competitions: int
    num_total_runs: int
    num_successful_runs: int
    num_medals: int
    overall_medal_rate: float
    overall_mean_any_medal: float
    overall_std_error: float
    per_competition: List[CompetitionScore]
    timestamp: str


def compute_mean_and_sem(values: List[float]) -> tuple:
    """
    Compute mean and standard error of the mean.
    
    Args:
        values: List of values
        
    Returns:
        Tuple of (mean, standard_error)
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    
    mean = sum(values) / n
    
    if n == 1:
        return mean, 0.0
    
    # Sample standard deviation
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std_dev = math.sqrt(variance)
    
    # Standard error of the mean
    std_error = std_dev / math.sqrt(n)
    
    return mean, std_error


def grade_and_compute_scores(
    results_dir: str,
    mlebench_path: str = None,
    use_run_results: bool = True
) -> EvaluationSummary:
    """
    Grade all submissions and compute Any Medal (%) scores.
    
    Args:
        results_dir: Directory containing competition/seed_N/submission.csv
        mlebench_path: Path to mle-bench installation
        use_run_results: Whether to use run_results.json if available
        
    Returns:
        EvaluationSummary with all scores
    """
    from datetime import datetime
    
    # Try to load existing run results
    run_results_path = os.path.join(results_dir, "run_results.json")
    run_results = []
    if use_run_results and os.path.exists(run_results_path):
        with open(run_results_path, 'r') as f:
            run_results = json.load(f)
        print(f"Loaded {len(run_results)} run results from {run_results_path}")
    
    # Initialize grader
    grader = None
    if MLEBenchGrader:
        grader = MLEBenchGrader(mlebench_path=mlebench_path)
    
    # Competition list
    COMPETITIONS = [
        "siim-isic-melanoma-classification",
        "spooky-author-identification",
        "tabular-playground-series-may-2022",
        "text-normalization-challenge-english-language",
        "the-icml-2013-whale-challenge-right-whale-redux"
    ]
    
    SEEDS = [0, 1, 2]
    
    per_competition_scores = []
    all_medal_indicators = []  # List of 0 or 100 for each run
    
    for comp_id in COMPETITIONS:
        print(f"\nProcessing: {comp_id}")
        
        comp_medal_indicators = []
        comp_results = []
        
        for seed in SEEDS:
            submission_path = os.path.join(results_dir, comp_id, f"seed_{seed}", "submission.csv")
            
            # Check if submission exists
            has_submission = os.path.exists(submission_path)
            
            # Get run result if available
            run_result = None
            for rr in run_results:
                if rr.get("competition_id") == comp_id and rr.get("seed") == seed:
                    run_result = rr
                    break
            
            # Grade submission
            grade_result = None
            if grader and has_submission:
                grade_result = grader.grade_submission(submission_path, comp_id)
            
            # Determine medal status
            # For now, we assume:
            # - If grader available and returned is_medal: use it
            # - If submission exists but no grader: mark as "pending"
            # - If no submission: mark as no medal
            
            if grade_result and grade_result.is_medal:
                medal_indicator = 100  # Medal achieved
            elif has_submission:
                # Submission exists but couldn't grade or no medal
                medal_indicator = 0  # Assume no medal for now
            else:
                medal_indicator = 0  # No submission = no medal
            
            comp_medal_indicators.append(medal_indicator)
            all_medal_indicators.append(medal_indicator)
            
            result_entry = {
                "seed": seed,
                "has_submission": has_submission,
                "medal_indicator": medal_indicator,
                "submission_path": submission_path if has_submission else None
            }
            
            if grade_result:
                result_entry["score"] = grade_result.score
                result_entry["medal"] = grade_result.medal
                result_entry["is_medal"] = grade_result.is_medal
            
            comp_results.append(result_entry)
            
            status = "[OK] Medal" if medal_indicator == 100 else ("[X] No medal" if has_submission else "[!] No submission")
            print(f"  Seed {seed}: {status}")
        
        # Compute competition-level statistics
        mean, sem = compute_mean_and_sem(comp_medal_indicators)
        num_medals = sum(1 for m in comp_medal_indicators if m == 100)
        
        comp_score = CompetitionScore(
            competition_id=comp_id,
            num_seeds=len(SEEDS),
            num_medals=num_medals,
            medal_rate=100 * num_medals / len(SEEDS),
            mean_any_medal=mean,
            std_error=sem,
            individual_results=comp_results
        )
        
        per_competition_scores.append(comp_score)
        print(f"  Any Medal (%): {mean:.2f} ± {sem:.2f} ({num_medals}/{len(SEEDS)} runs)")
    
    # Compute overall statistics
    overall_mean, overall_sem = compute_mean_and_sem(all_medal_indicators)
    total_runs = len(all_medal_indicators)
    total_medals = sum(1 for m in all_medal_indicators if m == 100)
    successful_runs = sum(1 for s in per_competition_scores for r in s.individual_results if r.get("has_submission"))
    
    summary = EvaluationSummary(
        num_competitions=len(COMPETITIONS),
        num_total_runs=total_runs,
        num_successful_runs=successful_runs,
        num_medals=total_medals,
        overall_medal_rate=100 * total_medals / total_runs if total_runs > 0 else 0,
        overall_mean_any_medal=overall_mean,
        overall_std_error=overall_sem,
        per_competition=per_competition_scores,
        timestamp=datetime.now().isoformat()
    )
    
    return summary


def print_summary(summary: EvaluationSummary):
    """Print formatted summary."""
    print(f"\n{'='*80}")
    print("MLE-BENCH EVALUATION RESULTS")
    print(f"{'='*80}")
    
    print(f"\nPer-Competition Results:")
    print("-" * 80)
    
    for comp in summary.per_competition:
        print(f"\n{comp.competition_id}:")
        print(f"  Any Medal (%): {comp.mean_any_medal:.2f} ± {comp.std_error:.2f}")
        print(f"  Medals: {comp.num_medals}/{comp.num_seeds} runs")
    
    print(f"\n{'='*80}")
    print("OVERALL RESULTS")
    print(f"{'='*80}")
    print(f"\n  Total Runs: {summary.num_total_runs}")
    print(f"  Successful Submissions: {summary.num_successful_runs}")
    print(f"  Total Medals: {summary.num_medals}")
    print(f"\n  *** Any Medal (%): {summary.overall_mean_any_medal:.2f} ± {summary.overall_std_error:.2f} ***")
    print(f"\n  (This is the primary metric for MLE-Bench Lite evaluation)")
    print(f"{'='*80}")


def save_summary(summary: EvaluationSummary, output_path: str):
    """Save summary to JSON file."""
    # Convert to dict
    summary_dict = {
        "num_competitions": summary.num_competitions,
        "num_total_runs": summary.num_total_runs,
        "num_successful_runs": summary.num_successful_runs,
        "num_medals": summary.num_medals,
        "overall_medal_rate": summary.overall_medal_rate,
        "overall_mean_any_medal": summary.overall_mean_any_medal,
        "overall_std_error": summary.overall_std_error,
        "timestamp": summary.timestamp,
        "per_competition": [asdict(c) for c in summary.per_competition]
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    
    print(f"\nSummary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute MLE-Bench Any Medal (%) scores with mean ± SEM"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./mlebench_results",
        help="Directory containing competition/seed_N/submission.csv structure"
    )
    parser.add_argument(
        "--mlebench_path",
        type=str,
        default=None,
        help="Path to mle-bench installation (for CLI grading)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for JSON summary (default: results_dir/evaluation_summary.json)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"ERROR: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    # Compute scores
    summary = grade_and_compute_scores(
        results_dir=args.results_dir,
        mlebench_path=args.mlebench_path
    )
    
    # Print summary
    print_summary(summary)
    
    # Save summary
    output_path = args.output or os.path.join(args.results_dir, "evaluation_summary.json")
    save_summary(summary, output_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

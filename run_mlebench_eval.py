#!/usr/bin/env python
"""
MLE-Bench Style Evaluation Runner

Runs the HybridAutoMLE agent on 5 MLEbench lite competitions with 3 seeds each,
producing 15 submission.csv files for grading with mlebench grade-sample.

Usage:
    python run_mlebench_eval.py --data_root ./data --output_root ./mlebench_results

Competitions:
    1. siim-isic-melanoma-classification (image)
    2. spooky-author-identification (text)
    3. tabular-playground-series-may-2022 (tabular)
    4. text-normalization-challenge-english-language (seq2seq)
    5. the-icml-2013-whale-challenge-right-whale-redux (audio)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# MLEbench lite competitions (5 total)
COMPETITIONS = [
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "the-icml-2013-whale-challenge-right-whale-redux"
]

# Mapping for data folder names if different from competition_id
DATA_FOLDER_MAP = {
    "the-icml-2013-whale-challenge-right-whale-redux": "right-whale-redux"
}

# Seeds for reproducibility (0, 1, 2 as per MLE-Bench standard)
SEEDS = [0, 1, 2]


def get_mlebench_data_path(competition_id: str) -> str:
    """
    Get the mlebench prepared public data path for a competition.
    
    MLEbench prepares data in its own directory structure which is required
    for proper grading (the test set and answers match).
    
    Args:
        competition_id: Competition identifier
        
    Returns:
        Path to mlebench's prepared public data directory
    """
    try:
        from mlebench.registry import registry
        competition = registry.get_competition(competition_id)
        return str(competition.public_dir)
    except Exception as e:
        print(f"[WARN] Could not get mlebench data path: {e}")
        return None


def validate_data_folders(data_root: str) -> Tuple[bool, List[str]]:
    """
    Validate that all competition data folders exist.
    
    Args:
        data_root: Root directory containing competition data folders
        
    Returns:
        Tuple of (all_exist, missing_folders)
    """
    missing = []
    for comp_id in COMPETITIONS:
        folder_name = DATA_FOLDER_MAP.get(comp_id, comp_id)
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.exists(folder_path):
            missing.append(folder_name)
    
    return len(missing) == 0, missing


def run_single_experiment(
    competition_id: str,
    seed: int,
    data_root: str,
    output_root: str,
    max_runtime_hours: float = 4.0,
    gemini_model: str = "gemini-2.5-pro",
    cloud_mode: bool = False,
    use_mlebench_data: bool = True
) -> Dict:
    """
    Run a single experiment (one competition, one seed).
    
    Args:
        competition_id: Competition identifier
        seed: Random seed
        data_root: Root data directory (fallback if mlebench data not available)
        output_root: Root output directory
        max_runtime_hours: Max runtime per experiment
        gemini_model: Gemini model to use
        cloud_mode: Whether to use cloud (Colab via local kernel) execution
        use_mlebench_data: Whether to use mlebench's prepared data directory
        
    Returns:
        Dictionary with run results
    """
    # Try to use mlebench's prepared data path for correct test set matching
    if use_mlebench_data:
        mlebench_path = get_mlebench_data_path(competition_id)
        if mlebench_path and os.path.exists(mlebench_path):
            dataset_path = mlebench_path
            print(f"[INFO] Using mlebench prepared data: {dataset_path}")
        else:
            folder_name = DATA_FOLDER_MAP.get(competition_id, competition_id)
            dataset_path = os.path.join(data_root, folder_name)
            print(f"[WARN] mlebench data not available, using: {dataset_path}")
    else:
        folder_name = DATA_FOLDER_MAP.get(competition_id, competition_id)
        dataset_path = os.path.join(data_root, folder_name)
    
    output_dir = os.path.join(output_root, competition_id, f"seed_{seed}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Running: {competition_id} with seed {seed}")
    print(f"Dataset: {dataset_path}")
    print(f"Output:  {output_dir}")
    print(f"Mode:    {'CLOUD (Colab)' if cloud_mode else 'NORMAL (Local/Docker)'}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Build command
    cmd = [
        sys.executable, "hybrid_agent.py",
        "--dataset_path", dataset_path,
        "--competition_id", competition_id,
        "--output_dir", output_dir,
        "--seed", str(seed),
        "--max_runtime_hours", str(max_runtime_hours),
        "--gemini_model", gemini_model
    ]
    
    # Add execution mode flag
    if cloud_mode:
        cmd.append("-c")
    else:
        cmd.append("-n")
    
    # Run the agent
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(max_runtime_hours * 3600) + 300  # Add 5 min buffer
        )
        
        runtime = time.time() - start_time
        success = result.returncode == 0
        
        # Check if submission.csv was created
        # Look for submission.csv in output_dir or any session subdirectory
        submission_path = None
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f == "submission.csv" or f.startswith("submission_"):
                    submission_path = os.path.join(root, f)
                    break
            if submission_path:
                break
        
        # Copy/symlink to expected location for mlebench grading
        final_submission = os.path.join(output_dir, "submission.csv")
        if submission_path and submission_path != final_submission:
            import shutil
            shutil.copy(submission_path, final_submission)
        
        run_result = {
            "competition_id": competition_id,
            "seed": seed,
            "success": success,
            "runtime_seconds": runtime,
            "submission_path": final_submission if os.path.exists(final_submission) else None,
            "exit_code": result.returncode,
            "stdout_tail": result.stdout[-2000:] if result.stdout else "",
            "stderr_tail": result.stderr[-2000:] if result.stderr else "",
            "timestamp": datetime.now().isoformat()
        }
        
        if success and os.path.exists(final_submission):
            print(f"✓ SUCCESS: Submission created at {final_submission}")
        else:
            print(f"✗ FAILED: Exit code {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr[-500:]}")
        
        return run_result
        
    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        print(f"✗ TIMEOUT after {runtime/3600:.2f} hours")
        return {
            "competition_id": competition_id,
            "seed": seed,
            "success": False,
            "runtime_seconds": runtime,
            "submission_path": None,
            "exit_code": -1,
            "error": "Timeout",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        runtime = time.time() - start_time
        print(f"✗ ERROR: {e}")
        return {
            "competition_id": competition_id,
            "seed": seed,
            "success": False,
            "runtime_seconds": runtime,
            "submission_path": None,
            "exit_code": -1,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def run_all_experiments(
    data_root: str,
    output_root: str,
    max_runtime_hours: float = 4.0,
    gemini_model: str = "gemini-2.5-pro",
    competitions: List[str] = None,
    seeds: List[int] = None,
    cloud_mode: bool = False
) -> List[Dict]:
    """
    Run all experiments (5 competitions × 3 seeds = 15 runs).
    
    Args:
        data_root: Root data directory
        output_root: Root output directory
        max_runtime_hours: Max runtime per experiment
        gemini_model: Gemini model to use
        competitions: List of competition IDs (default: all 5)
        seeds: List of seeds (default: [0, 1, 2])
        cloud_mode: Whether to use cloud (Colab via local kernel) execution
        
    Returns:
        List of run result dictionaries
    """
    if competitions is None:
        competitions = COMPETITIONS
    if seeds is None:
        seeds = SEEDS
    
    # Validate only the competitions we're actually running
    missing = []
    for comp_id in competitions:
        folder_name = DATA_FOLDER_MAP.get(comp_id, comp_id)
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.exists(folder_path):
            missing.append(folder_name)
    
    if missing:
        print(f"ERROR: Missing data folders: {missing}")
        print(f"Please ensure data is downloaded to {data_root}/{{competition_id}}/")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("MLE-Bench Style Evaluation")
    print(f"{'='*80}")
    print(f"Competitions: {len(competitions)}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {len(competitions) * len(seeds)}")
    print(f"Max runtime per run: {max_runtime_hours} hours")
    print(f"Execution mode: {'CLOUD (Colab)' if cloud_mode else 'NORMAL (Local/Docker)'}")
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"{'='*80}")
    
    all_results = []
    total_runs = len(competitions) * len(seeds)
    current_run = 0
    
    for comp_id in competitions:
        for seed in seeds:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] Starting run...")
            
            result = run_single_experiment(
                competition_id=comp_id,
                seed=seed,
                data_root=data_root,
                output_root=output_root,
                max_runtime_hours=max_runtime_hours,
                gemini_model=gemini_model,
                cloud_mode=cloud_mode
            )
            
            all_results.append(result)
            
            # Save intermediate results
            results_file = os.path.join(output_root, "run_results.json")
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
    
    return all_results


def print_summary(results: List[Dict]):
    """Print summary of all runs."""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    total = len(results)
    successful = sum(1 for r in results if r.get("success", False))
    with_submission = sum(1 for r in results if r.get("submission_path"))
    total_runtime = sum(r.get("runtime_seconds", 0) for r in results)
    
    print(f"\nOverall:")
    print(f"  Total runs: {total}")
    print(f"  Successful: {successful} ({100*successful/total:.1f}%)")
    print(f"  With submission: {with_submission} ({100*with_submission/total:.1f}%)")
    print(f"  Total runtime: {total_runtime/3600:.2f} hours")
    
    print(f"\nPer Competition:")
    for comp_id in COMPETITIONS:
        comp_results = [r for r in results if r.get("competition_id") == comp_id]
        if comp_results:
            comp_success = sum(1 for r in comp_results if r.get("success"))
            comp_subs = sum(1 for r in comp_results if r.get("submission_path"))
            print(f"  {comp_id}:")
            print(f"    Runs: {len(comp_results)}, Success: {comp_success}, Submissions: {comp_subs}")
    
    print(f"\n{'='*80}")
    print("Next step: Run grading with mlebench grade-sample")
    first_submission = results[0].get('submission_path') if results else None
    results_dir = os.path.dirname(first_submission) if first_submission else './mlebench_results'
    print(f"  python compute_mlebench_scores.py --results_dir {results_dir}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Run MLE-Bench style evaluation on 5 competitions × 3 seeds\n\nNote: Uses mlebench prepared data automatically. No need to specify data_root."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="(Optional) Fallback data root if mlebench data not available"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./mlebench_results",
        help="Root directory for evaluation outputs"
    )
    parser.add_argument(
        "--max_runtime_hours",
        type=float,
        default=4.0,
        help="Maximum runtime per experiment in hours (default: 4)"
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-2.5-pro",
        help="Gemini model to use"
    )
    parser.add_argument(
        "--competitions",
        type=str,
        nargs="+",
        default=None,
        help="Specific competitions to run (default: all 5)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Specific seeds to use (default: 0, 1, 2)"
    )
    
    # Execution mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-n", "--normal",
        action="store_true",
        default=True,
        help="Normal execution mode (local/Docker) [default]"
    )
    mode_group.add_argument(
        "-c", "--cloud",
        action="store_true",
        help="Cloud execution mode (local Colab kernel via notebook)"
    )
    
    # Watcher mode for auto-evaluation
    parser.add_argument(
        "-w", "--watch",
        action="store_true",
        help="Watch mode: wait for submission.csv then trigger evaluation and next seed"
    )
    
    args = parser.parse_args()
    
    # Determine execution mode
    cloud_mode = args.cloud
    
    # Create output root
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    
    # Run all experiments (or watch mode)
    if args.watch:
        # Watch mode: run single experiment and watch for submission
        from src.evaluation.submission_watcher import run_with_watcher
        reports = run_with_watcher(
            data_root=args.data_root,
            output_root=args.output_root,
            max_runtime_hours=args.max_runtime_hours,
            gemini_model=args.gemini_model,
            competitions=args.competitions,
            seeds=args.seeds,
            cloud_mode=cloud_mode
        )
        # Watch mode returns its own summary, just return success
        total_medals = sum(r.get('medals', {}).get('total', 0) for r in reports)
        return 0 if total_medals > 0 else 1
    else:
        results = run_all_experiments(
            data_root=args.data_root,
            output_root=args.output_root,
            max_runtime_hours=args.max_runtime_hours,
            gemini_model=args.gemini_model,
            competitions=args.competitions,
            seeds=args.seeds,
            cloud_mode=cloud_mode
        )
        
        # Print summary
        print_summary(results)
        
        return 0 if all(r.get("success") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

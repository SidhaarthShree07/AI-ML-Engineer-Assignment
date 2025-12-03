"""
Submission Watcher Module

Watches for submission.csv in session folders and automatically triggers:
1. Evaluation with medal metrics
2. Seed increment for next run
3. Report generation

This enables a fully automated evaluation loop with manual Colab execution.
"""

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


def get_mlebench_data_path(competition_id: str) -> Optional[str]:
    """
    Get the mlebench prepared public data path for a competition.
    
    Args:
        competition_id: Competition identifier
        
    Returns:
        Path to mlebench's prepared public data directory, or None if not available
    """
    try:
        from mlebench.registry import registry
        competition = registry.get_competition(competition_id)
        return str(competition.public_dir)
    except Exception as e:
        logger.warning(f"Could not get mlebench data path: {e}")
        return None


@dataclass
class SubmissionEvent:
    """Event when a submission file is detected."""
    submission_path: str
    session_dir: str
    competition_id: str
    seed: int
    timestamp: str


@dataclass
class EvaluationResult:
    """Result of evaluating a single submission."""
    competition_id: str
    seed: int
    submission_path: str
    has_submission: bool
    score: Optional[float] = None
    medal: Optional[str] = None  # "gold", "silver", "bronze", or None
    is_medal: bool = False
    error: Optional[str] = None


class SubmissionWatcher:
    """
    Watches for submission.csv files and triggers evaluation.
    
    Workflow:
    1. Agent generates notebook and waits for execution
    2. User runs notebook in Colab (connected to VS Code)
    3. Watcher detects submission.csv when it appears
    4. Evaluates submission and generates medal metrics
    5. Increments seed and triggers next run
    """
    
    def __init__(
        self,
        output_root: str,
        competition_id: str,
        seeds: List[int],
        poll_interval: float = 5.0,
        timeout_per_seed: float = 14400.0  # 4 hours
    ):
        """
        Initialize the submission watcher.
        
        Args:
            output_root: Root directory for outputs (e.g., ./mlebench_results)
            competition_id: Current competition ID
            seeds: List of seeds to run
            poll_interval: Seconds between file checks
            timeout_per_seed: Max wait time per seed in seconds
        """
        self.output_root = Path(output_root)
        self.competition_id = competition_id
        self.seeds = seeds
        self.poll_interval = poll_interval
        self.timeout_per_seed = timeout_per_seed
        
        # Track progress
        self.current_seed_index = 0
        self.results: List[EvaluationResult] = []
        
        # Create output directory
        self.output_root.mkdir(parents=True, exist_ok=True)
    
    def _get_session_dir(self, seed: int) -> Path:
        """Get session directory for a specific seed."""
        return self.output_root / self.competition_id / f"seed_{seed}"
    
    def _find_submission_file(self, session_dir: Path) -> Optional[Path]:
        """
        Find submission.csv in session directory.
        
        Checks:
        1. session_dir/submission.csv
        2. session_dir/*/submission.csv (in subdirectories)
        """
        if not session_dir.exists():
            return None
        
        # Check direct path
        direct_path = session_dir / "submission.csv"
        if direct_path.exists():
            return direct_path
        
        # Check subdirectories
        for subdir in session_dir.iterdir():
            if subdir.is_dir():
                sub_path = subdir / "submission.csv"
                if sub_path.exists():
                    return sub_path
        
        return None
    
    def _evaluate_submission(self, submission_path: Path, seed: int) -> EvaluationResult:
        """
        Evaluate a submission file and compute medal metrics.
        
        Args:
            submission_path: Path to submission.csv
            seed: Current seed
            
        Returns:
            EvaluationResult with score and medal info
        """
        print(f"\n[EVAL] Evaluating submission: {submission_path}")
        
        try:
            # Try to use MLEBench grader
            from src.evaluation.mlebench_grader import MLEBenchGrader
            
            grader = MLEBenchGrader()
            grade_result = grader.grade_submission(str(submission_path), self.competition_id)
            
            if grade_result:
                result = EvaluationResult(
                    competition_id=self.competition_id,
                    seed=seed,
                    submission_path=str(submission_path),
                    has_submission=True,
                    score=grade_result.score,
                    medal=grade_result.medal,
                    is_medal=grade_result.is_medal
                )
                
                medal_str = grade_result.medal if grade_result.medal else "No Medal"
                print(f"[EVAL] Score: {grade_result.score:.4f}, Medal: {medal_str}")
                return result
            else:
                print("[EVAL] [WARN] Grader returned no result")
                return EvaluationResult(
                    competition_id=self.competition_id,
                    seed=seed,
                    submission_path=str(submission_path),
                    has_submission=True,
                    error="Grader returned no result"
                )
                
        except ImportError:
            print("[EVAL] [WARN] MLEBench grader not available, marking as submitted")
            return EvaluationResult(
                competition_id=self.competition_id,
                seed=seed,
                submission_path=str(submission_path),
                has_submission=True,
                error="Grader not available"
            )
        except Exception as e:
            print(f"[EVAL] [ERROR] Evaluation failed: {e}")
            return EvaluationResult(
                competition_id=self.competition_id,
                seed=seed,
                submission_path=str(submission_path),
                has_submission=True,
                error=str(e)
            )
    
    def _generate_report(self) -> Dict:
        """
        Generate evaluation report with medal metrics.
        
        Returns:
            Dict with medal statistics
        """
        total_runs = len(self.results)
        submissions = [r for r in self.results if r.has_submission]
        medals = [r for r in self.results if r.is_medal]
        
        # Medal breakdown
        gold = sum(1 for r in self.results if r.medal == "gold")
        silver = sum(1 for r in self.results if r.medal == "silver")
        bronze = sum(1 for r in self.results if r.medal == "bronze")
        
        # Any Medal percentage
        any_medal_pct = 100 * len(medals) / total_runs if total_runs > 0 else 0
        
        report = {
            "competition_id": self.competition_id,
            "total_seeds": len(self.seeds),
            "total_runs": total_runs,
            "submissions": len(submissions),
            "medals": {
                "gold": gold,
                "silver": silver,
                "bronze": bronze,
                "total": len(medals)
            },
            "any_medal_percent": any_medal_pct,
            "results": [
                {
                    "seed": r.seed,
                    "has_submission": r.has_submission,
                    "score": r.score,
                    "medal": r.medal,
                    "is_medal": r.is_medal
                }
                for r in self.results
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _save_report(self, report: Dict) -> str:
        """Save evaluation report to file."""
        report_path = self.output_root / self.competition_id / "evaluation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"[REPORT] Saved to: {report_path}")
        return str(report_path)
    
    def _print_report(self, report: Dict):
        """Print evaluation report to console."""
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        print(f"\nCompetition: {report['competition_id']}")
        print(f"Total Seeds: {report['total_seeds']}")
        print(f"Submissions: {report['submissions']}")
        print(f"\nMedals:")
        print(f"  Gold:   {report['medals']['gold']}")
        print(f"  Silver: {report['medals']['silver']}")
        print(f"  Bronze: {report['medals']['bronze']}")
        print(f"  Total:  {report['medals']['total']}")
        print(f"\n*** Any Medal (%): {report['any_medal_percent']:.2f}% ***")
        print("\nPer-Seed Results:")
        for r in report['results']:
            medal_str = r['medal'] if r['medal'] else "None"
            score_str = f"{r['score']:.4f}" if r['score'] else "N/A"
            print(f"  Seed {r['seed']}: Score={score_str}, Medal={medal_str}")
        print("=" * 60 + "\n")
    
    def wait_for_submission(
        self,
        seed: int,
        on_found: Optional[Callable[[SubmissionEvent], None]] = None
    ) -> Optional[EvaluationResult]:
        """
        Wait for submission.csv to appear for a specific seed.
        
        Args:
            seed: Seed to watch for
            on_found: Optional callback when submission is found
            
        Returns:
            EvaluationResult if found, None if timeout
        """
        session_dir = self._get_session_dir(seed)
        session_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"[WATCH] 👀 WAITING FOR SUBMISSION (seed={seed})")
        print(f"{'='*60}")
        print(f"[WATCH] Session Dir: {session_dir}")
        print(f"[WATCH] Timeout: {self.timeout_per_seed/3600:.1f} hours")
        print(f"[WATCH] Notebook: D:\\Hexo.ai Project\\executor\\workspace\\Generated code.ipynb")
        print(f"{'='*60}")
        print(f"\n[WATCH] 📝 INSTRUCTIONS:")
        print(f"  1. Open the Generated code.ipynb in VS Code")
        print(f"  2. Select your Colab kernel")
        print(f"  3. Click 'Run All'")
        print(f"  4. Wait for training to complete")
        print(f"  5. Copy submission.csv to: {session_dir}")
        print(f"\n[WATCH] Polling every {self.poll_interval}s for submission.csv...\n")
        logger.info(f"[WATCH] Session: {session_dir}")
        logger.info(f"[WATCH] Timeout: {self.timeout_per_seed/3600:.1f} hours")
        logger.info(f"{'='*60}\n")
        
        start_time = time.time()
        last_log = 0
        
        while True:
            elapsed = time.time() - start_time
            
            # Timeout check
            if elapsed > self.timeout_per_seed:
                logger.warning(f"[WATCH] Timeout after {elapsed/3600:.2f} hours")
                return EvaluationResult(
                    competition_id=self.competition_id,
                    seed=seed,
                    submission_path="",
                    has_submission=False,
                    error="Timeout waiting for submission"
                )
            
            # Check for submission
            submission_path = self._find_submission_file(session_dir)
            
            if submission_path:
                print(f"\n[WATCH] [OK] SUBMISSION FOUND: {submission_path}")
                
                # Create event
                event = SubmissionEvent(
                    submission_path=str(submission_path),
                    session_dir=str(session_dir),
                    competition_id=self.competition_id,
                    seed=seed,
                    timestamp=datetime.now().isoformat()
                )
                
                # Callback
                if on_found:
                    on_found(event)
                
                # Evaluate
                result = self._evaluate_submission(submission_path, seed)
                return result
            
            # Progress log every 30 seconds
            if elapsed - last_log >= 30:
                print(f"[WATCH] ⏳ Still waiting... ({elapsed/60:.1f} min elapsed)")
                last_log = elapsed
            
            # Wait before next check
            time.sleep(self.poll_interval)
    
    def run_evaluation_loop(
        self,
        run_agent_callback: Optional[Callable[[int], None]] = None
    ) -> Dict:
        """
        Run the complete evaluation loop for all seeds.
        
        For each seed:
        1. Call run_agent_callback to generate notebook
        2. Wait for submission.csv to appear
        3. Evaluate and record result
        4. Move to next seed
        
        Args:
            run_agent_callback: Function to call to start agent for a seed
            
        Returns:
            Final evaluation report
        """
        print("\n" + "=" * 70)
        print("[LOOP] STARTING EVALUATION LOOP")
        print("=" * 70)
        print(f"[LOOP] Competition: {self.competition_id}")
        print(f"[LOOP] Seeds: {self.seeds}")
        print(f"[LOOP] Total Runs: {len(self.seeds)}")
        print("=" * 70 + "\n")
        
        for i, seed in enumerate(self.seeds):
            print("\n" + "+" * 70)
            print(f"[LOOP] SEED {seed} ({i+1}/{len(self.seeds)})")
            print("+" * 70)
            
            # Run agent to generate notebook
            if run_agent_callback:
                print(f"[LOOP] Calling agent to generate Gemini-enhanced notebook...")
                run_agent_callback(seed)
            
            # Wait for submission
            result = self.wait_for_submission(seed)
            
            if result:
                self.results.append(result)
                
                if result.is_medal:
                    print(f"\n[LOOP] MEDAL: {result.medal} (score: {result.score})")
                elif result.has_submission:
                    print(f"\n[LOOP] [NO MEDAL] (score: {result.score})")
                else:
                    print(f"\n[LOOP] [WARN] No submission for seed {seed}")
            
            # Generate intermediate report
            report = self._generate_report()
            self._save_report(report)
        
        # Final report
        final_report = self._generate_report()
        self._save_report(final_report)
        self._print_report(final_report)
        
        return final_report


def run_with_watcher(
    data_root: str,
    output_root: str,
    max_runtime_hours: float = 4.0,
    gemini_model: str = "gemini-2.5-pro",
    competitions: List[str] = None,
    seeds: List[int] = None,
    cloud_mode: bool = True
) -> List[Dict]:
    """
    Run experiments with submission watcher for auto-evaluation.
    
    This function:
    1. For each competition/seed, generates the notebook
    2. Watches for submission.csv to appear
    3. When found, evaluates and generates medal metrics
    4. Moves to next seed automatically
    
    Args:
        data_root: Root data directory
        output_root: Root output directory
        max_runtime_hours: Max runtime per experiment
        gemini_model: Gemini model to use
        competitions: List of competition IDs
        seeds: List of seeds
        cloud_mode: Should be True for watcher mode
        
    Returns:
        List of evaluation reports per competition
    """
    import subprocess
    import sys
    
    # Default competitions and seeds
    if competitions is None:
        competitions = ["tabular-playground-series-may-2022"]  # Start with one
    if seeds is None:
        seeds = [0, 1, 2]
    
    # Mapping for data folder names
    DATA_FOLDER_MAP = {
        "the-icml-2013-whale-challenge-right-whale-redux": "right-whale-redux"
    }
    
    all_reports = []
    
    for comp_id in competitions:
        print("\n" + "=" * 80)
        print(f"[WATCHER] 🏆 STARTING COMPETITION: {comp_id}")
        print("=" * 80)
        print(f"[WATCHER] Seeds to run: {seeds}")
        print(f"[WATCHER] Gemini Model: {gemini_model}")
        print(f"[WATCHER] Max Runtime per Seed: {max_runtime_hours} hours")
        print("=" * 80 + "\n")
        
        # Use mlebench prepared data path (required for correct grading)
        mlebench_path = get_mlebench_data_path(comp_id)
        if mlebench_path and os.path.exists(mlebench_path):
            dataset_path = mlebench_path
            print(f"[WATCHER] Using mlebench data: {dataset_path}")
        else:
            folder_name = DATA_FOLDER_MAP.get(comp_id, comp_id)
            dataset_path = os.path.join(data_root, folder_name) if data_root else f"./.data/{folder_name}"
            print(f"[WATCHER] WARNING: Using fallback data: {dataset_path}")
            print(f"[WATCHER] Note: mlebench grading may fail with non-mlebench data!")
        
        # Create watcher for this competition
        watcher = SubmissionWatcher(
            output_root=output_root,
            competition_id=comp_id,
            seeds=seeds,
            poll_interval=5.0,
            timeout_per_seed=max_runtime_hours * 3600
        )
        
        def run_agent_for_seed(seed: int):
            """Run the agent to generate notebook for a seed."""
            output_dir = os.path.join(output_root, comp_id)
            
            cmd = [
                sys.executable, "hybrid_agent.py",
                "--dataset_path", dataset_path,
                "--competition_id", comp_id,
                "--output_dir", output_dir,
                "--seed", str(seed),
                "--max_runtime_hours", str(max_runtime_hours),
                "--gemini_model", gemini_model,
                "-c",  # Cloud mode
                "--generate_only"  # Only generate notebook, don't wait for execution
            ]
            
            print("\n" + "=" * 70)
            print(f"[WATCHER] 🚀 GENERATING NOTEBOOK FOR SEED {seed}")
            print("=" * 70)
            print(f"[WATCHER] Command: {' '.join(cmd)}")
            print(f"[WATCHER] Gemini Model: {gemini_model}")
            print(f"[WATCHER] Dataset: {dataset_path}")
            print(f"[WATCHER] Output: {output_dir}")
            print(f"[WATCHER] ⏳ This may take 5-10 minutes (profiling + Gemini code generation)...")
            print("=" * 70 + "\n")
            
            try:
                # Run agent and capture output for logging
                # Increased timeout to 600s (10 min) for profiling + code generation
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 min for notebook generation
                )
                
                # Parse and display relevant output
                stdout_lines = result.stdout.split('\n') if result.stdout else []
                
                # Look for Gemini-related logs and print them
                gemini_logs = []
                important_logs = []
                for line in stdout_lines:
                    if '[GEMINI]' in line:
                        gemini_logs.append(line)
                    elif any(kw in line for kw in ['Strategy:', 'Modality:', 'code_generated', 'Notebook written', 'Cloud Executor']):
                        important_logs.append(line)
                
                if gemini_logs:
                    print("\n[WATCHER] 🤖 GEMINI CODE GENERATION:")
                    print("-" * 50)
                    for log in gemini_logs:
                        print(f"  {log.strip()}")
                    print("-" * 50)
                
                if important_logs:
                    print("\n[WATCHER] 📋 AGENT PROGRESS:")
                    for log in important_logs[:10]:  # Limit to 10 lines
                        print(f"  {log.strip()}")
                
                if result.returncode != 0:
                    print(f"\n[WATCHER] [WARN] Agent returned code {result.returncode}")
                    if result.stderr:
                        print(f"[WATCHER] stderr: {result.stderr[-500:]}")
                        
            except subprocess.TimeoutExpired:
                print(f"\n[WATCHER] ⏰ TIMEOUT: Agent took more than 10 minutes")
                print(f"[WATCHER] The notebook may still have been generated. Check:")
                print(f"[WATCHER]   D:\\Hexo.ai Project\\executor\\workspace\\Generated code.ipynb")
                print(f"[WATCHER] Continuing to watch for submission...")
            except Exception as e:
                print(f"\n[WATCHER] [ERROR] Error running agent: {e}")
                print(f"[WATCHER] Continuing to watch for submission...")
        
        # Run evaluation loop
        report = watcher.run_evaluation_loop(run_agent_callback=run_agent_for_seed)
        all_reports.append(report)
    
    # Generate combined report
    total_medals = sum(r['medals']['total'] for r in all_reports)
    total_runs = sum(r['total_runs'] for r in all_reports)
    overall_pct = 100 * total_medals / total_runs if total_runs > 0 else 0
    
    print("\n" + "=" * 80)
    print("OVERALL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nCompetitions: {len(all_reports)}")
    print(f"Total Runs: {total_runs}")
    print(f"Total Medals: {total_medals}")
    print(f"\n*** Overall Any Medal (%): {overall_pct:.2f}% ***")
    print("=" * 80)
    
    return all_reports

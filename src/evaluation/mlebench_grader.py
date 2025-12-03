"""
MLE-Bench Grader Integration

Integrates with the mlebench grading system (from git clone https://github.com/openai/mle-bench)
to grade submissions and compute medal results.

Usage:
    from src.evaluation.mlebench_grader import MLEBenchGrader
    
    grader = MLEBenchGrader()
    result = grader.grade_submission(
        submission_path="./mlebench_results/competition/seed_0/submission.csv",
        competition_id="siim-isic-melanoma-classification"
    )
"""

import json
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Try to import mlebench Python API
try:
    from mlebench.grade import grade_csv
    from mlebench.registry import registry as mlebench_registry
    MLEBENCH_AVAILABLE = True
except ImportError:
    MLEBENCH_AVAILABLE = False
    logger.warning("mlebench package not installed. Run: pip install -e ./mle-bench")


@dataclass
class GradeResult:
    """Result from grading a submission."""
    competition_id: str
    submission_path: str
    score: float
    medal: str  # "gold", "silver", "bronze", "none"
    is_medal: bool  # True if bronze or better
    threshold: float  # Medal threshold used
    raw_output: str
    error: Optional[str] = None


class MLEBenchGrader:
    """
    Integration with MLE-Bench grading system.
    
    MLE-Bench must be cloned from GitHub:
        git clone https://github.com/openai/mle-bench
        cd mle-bench
        pip install -e .
    """
    
    # Competition medal thresholds (Any Medal = Bronze threshold)
    # These are approximate thresholds for the 5 MLEbench lite competitions
    MEDAL_THRESHOLDS = {
        "siim-isic-melanoma-classification": {
            "metric": "auc",
            "bronze": 0.85,
            "silver": 0.90,
            "gold": 0.94
        },
        "spooky-author-identification": {
            "metric": "logloss",  # Lower is better
            "bronze": 0.50,  # Below this is good
            "silver": 0.40,
            "gold": 0.30,
            "lower_is_better": True
        },
        "tabular-playground-series-may-2022": {
            "metric": "auc",
            "bronze": 0.80,
            "silver": 0.85,
            "gold": 0.90
        },
        "text-normalization-challenge-english-language": {
            "metric": "accuracy",
            "bronze": 0.95,
            "silver": 0.97,
            "gold": 0.99
        },
        "the-icml-2013-whale-challenge-right-whale-redux": {
            "metric": "auc",
            "bronze": 0.88,
            "silver": 0.92,
            "gold": 0.96
        }
    }
    
    def __init__(self, mlebench_path: str = None):
        """
        Initialize grader.
        
        Args:
            mlebench_path: Path to mle-bench repo (optional, not needed if mlebench is installed)
        """
        self.mlebench_path = mlebench_path
        if mlebench_path:
            self.mlebench_path = Path(mlebench_path)
    
    def check_mlebench_installed(self) -> bool:
        """Check if mlebench Python API is available."""
        return MLEBENCH_AVAILABLE
    
    def grade_submission(
        self,
        submission_path: str,
        competition_id: str,
        use_cli: bool = True
    ) -> GradeResult:
        """
        Grade a single submission.
        
        Args:
            submission_path: Path to submission.csv
            competition_id: Competition identifier
            use_cli: Ignored - now always uses Python API if available
            
        Returns:
            GradeResult with score and medal status
        """
        if not os.path.exists(submission_path):
            return GradeResult(
                competition_id=competition_id,
                submission_path=submission_path,
                score=0.0,
                medal="none",
                is_medal=False,
                threshold=0.0,
                raw_output="",
                error=f"Submission file not found: {submission_path}"
            )
        
        if self.check_mlebench_installed():
            return self._grade_with_python_api(submission_path, competition_id)
        else:
            return self._grade_locally(submission_path, competition_id)
    
    def _grade_with_python_api(
        self,
        submission_path: str,
        competition_id: str
    ) -> GradeResult:
        """Grade using mlebench Python API directly."""
        logger.info(f"Grading {submission_path} with mlebench Python API")
        
        try:
            # Get competition from registry
            competition = mlebench_registry.get_competition(competition_id)
            
            # Grade the submission
            report = grade_csv(Path(submission_path), competition)
            
            # Determine medal
            if report.gold_medal:
                medal = "gold"
            elif report.silver_medal:
                medal = "silver"
            elif report.bronze_medal:
                medal = "bronze"
            else:
                medal = "none"
            
            score = report.score if report.score is not None else 0.0
            
            thresholds = self.MEDAL_THRESHOLDS.get(competition_id, {})
            bronze_threshold = thresholds.get("bronze", report.bronze_threshold or 0.0)
            
            return GradeResult(
                competition_id=competition_id,
                submission_path=submission_path,
                score=score,
                medal=medal,
                is_medal=report.any_medal,
                threshold=bronze_threshold,
                raw_output=f"score={score}, gold={report.gold_medal}, silver={report.silver_medal}, bronze={report.bronze_medal}",
                error=None
            )
            
        except Exception as e:
            logger.error(f"Error grading with mlebench API: {e}")
            return GradeResult(
                competition_id=competition_id,
                submission_path=submission_path,
                score=0.0,
                medal="none",
                is_medal=False,
                threshold=0.0,
                raw_output="",
                error=str(e)
            )
    
    def _grade_locally(
        self,
        submission_path: str,
        competition_id: str
    ) -> GradeResult:
        """
        Grade locally without mlebench Python API.
        
        This is a fallback that returns a placeholder result.
        For real grading, mlebench should be installed.
        """
        logger.warning("mlebench Python API not available, using placeholder grading")
        print("mlebench Python API not available, using placeholder grading")
        
        thresholds = self.MEDAL_THRESHOLDS.get(competition_id, {})
        bronze_threshold = thresholds.get("bronze", 0.0)
        
        # Return a placeholder result indicating manual grading is needed
        return GradeResult(
            competition_id=competition_id,
            submission_path=submission_path,
            score=-1.0,  # Placeholder
            medal="unknown",
            is_medal=False,
            threshold=bronze_threshold,
            raw_output="mlebench Python API not available - install with: pip install -e ./mle-bench",
            error="Install mlebench: pip install -e ./mle-bench"
        )
    
    def _parse_mlebench_output(self, output: str) -> tuple:
        """
        Parse mlebench output to extract score and medal.
        
        Returns:
            Tuple of (score: float, medal: str)
        """
        score = 0.0
        medal = "none"
        
        # Try to parse JSON output
        try:
            # Look for JSON in output
            import re
            json_match = re.search(r'\{[^{}]+\}', output)
            if json_match:
                data = json.loads(json_match.group())
                score = data.get("score", 0.0)
                medal = data.get("medal", "none").lower()
                return score, medal
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Try to parse text output
        output_lower = output.lower()
        
        if "gold" in output_lower:
            medal = "gold"
        elif "silver" in output_lower:
            medal = "silver"
        elif "bronze" in output_lower:
            medal = "bronze"
        
        # Try to extract score
        import re
        score_patterns = [
            r'score[:\s]+([0-9.]+)',
            r'auc[:\s]+([0-9.]+)',
            r'accuracy[:\s]+([0-9.]+)',
            r'([0-9.]+)\s*(?:auc|accuracy|score)'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, output_lower)
            if match:
                try:
                    score = float(match.group(1))
                    break
                except ValueError:
                    pass
        
        return score, medal
    
    def grade_all_submissions(
        self,
        results_dir: str
    ) -> List[GradeResult]:
        """
        Grade all submissions in a results directory.
        
        Expected structure:
            results_dir/
                competition_id/
                    seed_0/submission.csv
                    seed_1/submission.csv
                    seed_2/submission.csv
        
        Args:
            results_dir: Root directory with competition/seed_N/submission.csv
            
        Returns:
            List of GradeResult objects
        """
        all_results = []
        
        for competition_id in self.MEDAL_THRESHOLDS.keys():
            comp_dir = os.path.join(results_dir, competition_id)
            if not os.path.exists(comp_dir):
                logger.warning(f"Competition directory not found: {comp_dir}")
                continue
            
            for seed in [0, 1, 2]:
                submission_path = os.path.join(comp_dir, f"seed_{seed}", "submission.csv")
                result = self.grade_submission(submission_path, competition_id)
                all_results.append(result)
                
                logger.info(
                    f"Graded {competition_id}/seed_{seed}: "
                    f"score={result.score:.4f}, medal={result.medal}"
                )
        
        return all_results

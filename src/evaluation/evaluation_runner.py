"""Multi-seed evaluation system for HybridAutoMLE agent"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics

from src.models.data_models import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class SeedResult:
    """Results from a single seed execution"""
    seed: int
    success: bool
    metrics: Dict[str, float]
    submission_path: Optional[str]
    runtime_seconds: float
    error: Optional[str] = None


@dataclass
class EvaluationReport:
    """Complete evaluation report with statistics"""
    competition_id: str
    num_seeds: int
    successful_runs: int
    failed_runs: int
    mean_metrics: Dict[str, float]
    std_error_metrics: Dict[str, float]
    individual_results: List[SeedResult]
    medal_comparison: Dict[str, Any]
    total_runtime_seconds: float
    compute_metrics: Dict[str, float]


class EvaluationRunner:
    """
    Multi-seed evaluation system for autonomous ML agent.
    
    Runs the agent with multiple random seeds and aggregates results
    to provide robust performance metrics with statistical measures.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize evaluation runner.
        
        Args:
            output_dir: Base directory for evaluation outputs
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Competition medal thresholds (Any Medal Score)
        # These represent the minimum score to achieve any medal
        self.medal_thresholds = {
            "siim-isic-melanoma-classification": 0.85,
            "spooky-author-identification": 0.90,
            "tabular-playground-series-may-2022": 0.75,
            "text-normalization-challenge-english-language": 0.80,
            "the-icml-2013-whale-challenge-right-whale-redux": 0.88
        }
        
        logger.info(f"EvaluationRunner initialized with output_dir: {output_dir}")
    
    def run_with_seeds(
        self,
        agent_class,
        config: AgentConfig,
        seeds: List[int]
    ) -> List[SeedResult]:
        """
        Run agent with multiple random seeds.
        
        Args:
            agent_class: The HybridAutoMLEAgent class to instantiate
            config: Base agent configuration
            seeds: List of random seeds to use
        
        Returns:
            List of SeedResult objects, one per seed
        """
        logger.info(f"Starting multi-seed evaluation with {len(seeds)} seeds")
        logger.info(f"Seeds: {seeds}")
        
        results = []
        
        for seed in seeds:
            logger.info(f"=" * 80)
            logger.info(f"Running with seed: {seed}")
            logger.info(f"=" * 80)
            
            # Create seed-specific configuration
            seed_config = self._create_seed_config(config, seed)
            
            # Run agent with this seed
            seed_result = self._run_single_seed(agent_class, seed_config, seed)
            
            results.append(seed_result)
            
            # Log intermediate result
            if seed_result.success:
                logger.info(f"Seed {seed} completed successfully")
                logger.info(f"Metrics: {seed_result.metrics}")
            else:
                logger.warning(f"Seed {seed} failed: {seed_result.error}")
        
        logger.info(f"Multi-seed evaluation complete: {len(results)} runs")
        
        return results
    
    def _create_seed_config(self, base_config: AgentConfig, seed: int) -> AgentConfig:
        """
        Create seed-specific configuration.
        
        Args:
            base_config: Base configuration
            seed: Random seed
        
        Returns:
            New AgentConfig with seed-specific output directory
        """
        # Create seed-specific output directory
        seed_output_dir = os.path.join(self.output_dir, f"seed_{seed}")
        
        # Create new config with seed-specific settings
        seed_config = AgentConfig(
            dataset_path=base_config.dataset_path,
            competition_id=base_config.competition_id,
            output_dir=seed_output_dir,
            max_runtime_hours=base_config.max_runtime_hours,
            num_seeds=1,  # Single seed per run
            gemini_model=base_config.gemini_model,
            gemini_api_key=base_config.gemini_api_key,
            eval_mode=False,
            competitions=[],
            docker_image=base_config.docker_image,
            resource_constraints=base_config.resource_constraints
        )
        
        return seed_config
    
    def _run_single_seed(
        self,
        agent_class,
        config: AgentConfig,
        seed: int
    ) -> SeedResult:
        """
        Run agent with a single seed.
        
        Args:
            agent_class: The HybridAutoMLEAgent class
            config: Agent configuration
            seed: Random seed
        
        Returns:
            SeedResult with execution results
        """
        start_time = time.time()
        
        try:
            # Set random seed in environment
            os.environ['PYTHONHASHSEED'] = str(seed)
            
            # Create and run agent
            agent = agent_class(config)
            results = agent.run()
            
            runtime = time.time() - start_time
            
            # Extract metrics from results
            metrics = self._extract_metrics(results)
            
            # Get submission path
            submission_path = results.get('submission', {}).get('filepath')
            
            return SeedResult(
                seed=seed,
                success=True,
                metrics=metrics,
                submission_path=submission_path,
                runtime_seconds=runtime,
                error=None
            )
            
        except Exception as e:
            runtime = time.time() - start_time
            logger.error(f"Seed {seed} failed with error: {e}", exc_info=True)
            
            return SeedResult(
                seed=seed,
                success=False,
                metrics={},
                submission_path=None,
                runtime_seconds=runtime,
                error=str(e)
            )
    
    def _extract_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract metrics from agent results.
        
        Args:
            results: Agent execution results
        
        Returns:
            Dictionary of metric name to value
        """
        metrics = {}
        
        # Extract execution metrics
        if 'execution' in results:
            exec_data = results['execution']
            metrics['execution_time_seconds'] = exec_data.get('execution_time', 0.0)
            metrics['gpu_memory_used_gb'] = exec_data.get('gpu_memory_used_gb', 0.0)
        
        # Extract validation score if available
        # Note: In a real implementation, this would come from the training logs
        # For now, we'll use a placeholder
        metrics['validation_score'] = 0.0
        
        # Extract resource efficiency
        if 'total_runtime_seconds' in results:
            metrics['total_runtime_seconds'] = results['total_runtime_seconds']
        
        return metrics
    
    def calculate_statistics(
        self,
        results: List[SeedResult]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate mean and standard error for metrics across seeds.
        
        Args:
            results: List of SeedResult objects
        
        Returns:
            Dictionary with 'mean' and 'std_error' subdictionaries
        """
        logger.info("Calculating statistics across seeds")
        
        # Filter successful results
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            logger.warning("No successful results to calculate statistics")
            return {'mean': {}, 'std_error': {}}
        
        # Collect all metric names
        all_metric_names = set()
        for result in successful_results:
            all_metric_names.update(result.metrics.keys())
        
        # Calculate statistics for each metric
        mean_metrics = {}
        std_error_metrics = {}
        
        for metric_name in all_metric_names:
            # Collect values for this metric
            values = [
                r.metrics.get(metric_name, 0.0)
                for r in successful_results
                if metric_name in r.metrics
            ]
            
            if values:
                # Calculate mean
                mean_metrics[metric_name] = statistics.mean(values)
                
                # Calculate standard error of the mean
                if len(values) > 1:
                    std_dev = statistics.stdev(values)
                    std_error_metrics[metric_name] = std_dev / (len(values) ** 0.5)
                else:
                    std_error_metrics[metric_name] = 0.0
                
                logger.info(
                    f"Metric '{metric_name}': "
                    f"mean={mean_metrics[metric_name]:.4f}, "
                    f"SE={std_error_metrics[metric_name]:.4f}"
                )
        
        return {
            'mean': mean_metrics,
            'std_error': std_error_metrics
        }
    
    def compare_to_thresholds(
        self,
        competition_id: str,
        mean_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Compare performance against competition medal thresholds.
        
        Args:
            competition_id: Competition identifier
            mean_metrics: Mean metrics across seeds
        
        Returns:
            Dictionary with threshold comparison results
        """
        logger.info(f"Comparing to medal thresholds for {competition_id}")
        
        threshold = self.medal_thresholds.get(competition_id)
        
        if threshold is None:
            logger.warning(f"No medal threshold defined for {competition_id}")
            return {
                'has_threshold': False,
                'threshold': None,
                'achieved_medal': False,
                'score': None,
                'margin': None
            }
        
        # Get validation score (primary metric)
        score = mean_metrics.get('validation_score', 0.0)
        
        # Check if medal achieved
        achieved_medal = score >= threshold
        margin = score - threshold
        
        result = {
            'has_threshold': True,
            'threshold': threshold,
            'achieved_medal': achieved_medal,
            'score': score,
            'margin': margin
        }
        
        logger.info(
            f"Medal comparison: score={score:.4f}, "
            f"threshold={threshold:.4f}, "
            f"achieved={achieved_medal}, "
            f"margin={margin:+.4f}"
        )
        
        return result
    
    def generate_report(
        self,
        competition_id: str,
        results: List[SeedResult]
    ) -> EvaluationReport:
        """
        Generate comprehensive evaluation report.
        
        Args:
            competition_id: Competition identifier
            results: List of SeedResult objects
        
        Returns:
            EvaluationReport with all metrics and statistics
        """
        logger.info("Generating evaluation report")
        
        # Count successes and failures
        successful_runs = sum(1 for r in results if r.success)
        failed_runs = len(results) - successful_runs
        
        # Calculate statistics
        stats = self.calculate_statistics(results)
        mean_metrics = stats['mean']
        std_error_metrics = stats['std_error']
        
        # Compare to thresholds
        medal_comparison = self.compare_to_thresholds(competition_id, mean_metrics)
        
        # Calculate total runtime
        total_runtime = sum(r.runtime_seconds for r in results)
        
        # Calculate compute metrics
        compute_metrics = self._calculate_compute_metrics(results)
        
        report = EvaluationReport(
            competition_id=competition_id,
            num_seeds=len(results),
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            mean_metrics=mean_metrics,
            std_error_metrics=std_error_metrics,
            individual_results=results,
            medal_comparison=medal_comparison,
            total_runtime_seconds=total_runtime,
            compute_metrics=compute_metrics
        )
        
        # Save report to file
        self._save_report(report)
        
        logger.info("Evaluation report generated")
        
        return report
    
    def _calculate_compute_metrics(
        self,
        results: List[SeedResult]
    ) -> Dict[str, float]:
        """
        Calculate compute usage and resource efficiency metrics.
        
        Args:
            results: List of SeedResult objects
        
        Returns:
            Dictionary of compute metrics
        """
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {}
        
        # Calculate average runtime
        avg_runtime = statistics.mean([r.runtime_seconds for r in successful_results])
        
        # Calculate average GPU memory usage
        gpu_memory_values = [
            r.metrics.get('gpu_memory_used_gb', 0.0)
            for r in successful_results
        ]
        avg_gpu_memory = statistics.mean(gpu_memory_values) if gpu_memory_values else 0.0
        
        # Calculate resource efficiency (score per GPU-hour)
        # This is a placeholder - in real implementation would use actual scores
        validation_scores = [
            r.metrics.get('validation_score', 0.0)
            for r in successful_results
        ]
        avg_score = statistics.mean(validation_scores) if validation_scores else 0.0
        
        gpu_hours = avg_runtime / 3600.0
        resource_efficiency = avg_score / gpu_hours if gpu_hours > 0 else 0.0
        
        return {
            'avg_runtime_seconds': avg_runtime,
            'avg_runtime_hours': avg_runtime / 3600.0,
            'avg_gpu_memory_gb': avg_gpu_memory,
            'resource_efficiency': resource_efficiency
        }
    
    def _save_report(self, report: EvaluationReport):
        """
        Save evaluation report to JSON file.
        
        Args:
            report: EvaluationReport to save
        """
        report_path = os.path.join(
            self.output_dir,
            f"evaluation_report_{report.competition_id}.json"
        )
        
        # Convert to dictionary
        report_dict = asdict(report)
        
        # Save to file
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Evaluation report saved to: {report_path}")
    
    def print_summary(self, report: EvaluationReport):
        """
        Print human-readable summary of evaluation report.
        
        Args:
            report: EvaluationReport to summarize
        """
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Competition: {report.competition_id}")
        print(f"Number of Seeds: {report.num_seeds}")
        print(f"Successful Runs: {report.successful_runs}")
        print(f"Failed Runs: {report.failed_runs}")
        print()
        
        print("MEAN METRICS:")
        for metric, value in report.mean_metrics.items():
            std_error = report.std_error_metrics.get(metric, 0.0)
            print(f"  {metric}: {value:.4f} ± {std_error:.4f}")
        print()
        
        print("MEDAL COMPARISON:")
        if report.medal_comparison['has_threshold']:
            print(f"  Threshold: {report.medal_comparison['threshold']:.4f}")
            print(f"  Score: {report.medal_comparison['score']:.4f}")
            print(f"  Achieved Medal: {report.medal_comparison['achieved_medal']}")
            print(f"  Margin: {report.medal_comparison['margin']:+.4f}")
        else:
            print("  No threshold defined for this competition")
        print()
        
        print("COMPUTE METRICS:")
        for metric, value in report.compute_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()
        
        print(f"Total Runtime: {report.total_runtime_seconds / 3600:.2f} hours")
        print("=" * 80)

"""HybridAutoMLE: Main entry point for autonomous ML agent"""

import argparse
import sys
import os
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from src.models.data_models import AgentConfig, ResourceConstraints
from src.controller.controller import Controller
from src.executor.executor import Executor, DockerConfig
from src.state_manager.state_manager import StateManager
from src.submission.submission_generator import SubmissionGenerator
from src.utils.gemini_client import GeminiClient
from src.error_handler.error_handler import ErrorHandler
from src.self_improvement.code_enhancer import CodeEnhancer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridAutoMLEAgent:
    """
    Main autonomous ML agent that orchestrates the complete pipeline.
    
    The agent operates through five phases:
    1. Initialization: Setup components (Controller, Executor, StateManager)
    2. Profiling: Dataset analysis and verification
    3. Strategy Development: Strategy selection and code generation
    4. Execution: Training with monitoring
    5. Finalization: Inference and submission generation
    
    All phases are logged to StateManager for complete reasoning traces.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the agent with configuration.
        
        Args:
            config: Agent configuration with dataset path, competition ID, etc.
        """
        self.config = config
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components (will be set in initialization phase)
        self.controller: Optional[Controller] = None
        self.executor: Optional[Executor] = None
        self.state_manager: Optional[StateManager] = None
        self.error_handler: Optional[ErrorHandler] = None
        self.submission_generator: Optional[SubmissionGenerator] = None
        self.code_enhancer: Optional[CodeEnhancer] = None
        self.gemini_client: Optional[GeminiClient] = None
        
        # Track execution state
        self.dataset_profile = None
        self.strategy = None
        self.training_code = None
        self.execution_result = None
        
        logger.info(f"HybridAutoMLEAgent created with session_id: {self.session_id}")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete autonomous pipeline.
        
        This is the main entry point that orchestrates all phases without
        requiring any user input during execution.
        
        Returns:
            Dictionary with execution results and paths to artifacts
        """
        logger.info("=" * 80)
        logger.info("Starting HybridAutoMLEAgent autonomous execution")
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Competition: {self.config.competition_id}")
        logger.info(f"Dataset: {self.config.dataset_path}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Phase 1: Initialization
            self._phase_initialization()
            
            # Phase 2: Profiling
            self._phase_profiling()
            
            # Phase 3: Strategy Development
            self._phase_strategy_development()
            
            # Phase 4: Execution
            self._phase_execution()
            
            # Phase 5: Finalization
            results = self._phase_finalization()
            
            # Calculate total runtime
            total_runtime = time.time() - start_time
            results['total_runtime_seconds'] = total_runtime
            results['total_runtime_hours'] = total_runtime / 3600
            
            logger.info("=" * 80)
            logger.info("Agent execution completed successfully")
            logger.info(f"Total runtime: {total_runtime / 3600:.2f} hours")
            logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            
            # Log failure to state manager if available
            if self.state_manager:
                self.state_manager.log_action(
                    phase="failure",
                    action="agent_execution_failed",
                    input_data={"error": str(e)},
                    output_data={"success": False}
                )
                self.state_manager.save_trace()
            
            raise
    
    def _phase_initialization(self):
        """
        Phase 1: Initialize all components.
        
        Sets up:
        - StateManager for reasoning traces
        - GeminiClient for LLM-based reasoning
        - Controller for decision making
        - Executor for code execution
        - ErrorHandler for recovery
        - SubmissionGenerator for final output
        """
        logger.info("Phase 1: Initialization")
        
        # Create session output directory
        session_dir = os.path.join(self.config.output_dir, f"session_{self.session_id}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Initialize StateManager
        self.state_manager = StateManager(
            session_id=self.session_id,
            output_dir=session_dir
        )
        
        self.state_manager.log_action(
            phase="initialization",
            action="create_session",
            input_data={
                "session_id": self.session_id,
                "competition_id": self.config.competition_id,
                "dataset_path": self.config.dataset_path
            },
            output_data={"session_dir": session_dir}
        )
        
        # Initialize Gemini client
        gemini_api_key = self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.warning("No Gemini API key provided - running without LLM enhancement")
            self.gemini_client = None
        else:
            self.gemini_client = GeminiClient(
                api_key=gemini_api_key,
                model=self.config.gemini_model
            )
            logger.info(f"Gemini client initialized with model: {self.config.gemini_model}")
        
        # Initialize Controller
        self.controller = Controller(
            gemini_client=self.gemini_client,
            config=self.config
        )
        
        # Initialize CodeEnhancer for self-improvement
        self.code_enhancer = CodeEnhancer(
            gemini_client=self.gemini_client,
            max_iterations=3  # Max 3 enhancement iterations
        )
        
        # Initialize Executor with Docker
        # Limit CPU count to 32 max (Docker limit on most systems)
        cpu_count = min(float(self.config.resource_constraints.max_cpu_cores), 32.0)
        docker_config = DockerConfig(
            image="ml-sandbox:latest",
            gpu_enabled=True,
            memory_limit=f"{int(self.config.resource_constraints.max_ram_gb)}g",
            cpu_count=cpu_count,
            use_docker=True  # Set to False to disable Docker and run locally
        )
        self.executor = Executor(docker_config)
        
        # Initialize ErrorHandler
        self.error_handler = ErrorHandler(
            controller=self.controller,
            state_manager=self.state_manager
        )
        
        # Initialize SubmissionGenerator
        self.submission_generator = SubmissionGenerator(
            competition_id=self.config.competition_id,
            output_dir=session_dir,
            dataset_path=self.config.dataset_path
        )
        
        self.state_manager.log_action(
            phase="initialization",
            action="components_initialized",
            input_data={},
            output_data={
                "controller": "ready",
                "executor": "ready",
                "error_handler": "ready",
                "submission_generator": "ready"
            }
        )
        
        logger.info("Initialization complete")
    
    def _phase_profiling(self):
        """
        Phase 2: Dataset analysis and verification.
        
        Performs:
        - Hybrid modality detection
        - Statistical profiling
        - File path verification
        - Dataset profile creation
        """
        logger.info("Phase 2: Profiling")
        
        self.state_manager.log_action(
            phase="profiling",
            action="start_dataset_analysis",
            input_data={"dataset_path": self.config.dataset_path},
            output_data={}
        )
        
        # Analyze dataset using Controller
        self.dataset_profile = self.controller.analyze_dataset(self.config.dataset_path)
        
        # Log decision about detected modality
        self.state_manager.log_decision(
            decision_point="modality_detection",
            options=["tabular", "image", "text", "time_series", "multimodal"],
            selected=self.dataset_profile.modality,
            reasoning=f"Hybrid detection with {self.dataset_profile.confidence:.2f} confidence. "
                     f"Dataset has {self.dataset_profile.num_samples} samples, "
                     f"{self.dataset_profile.num_features} features, "
                     f"estimated GPU memory: {self.dataset_profile.estimated_gpu_memory_gb:.2f} GB"
        )
        
        self.state_manager.log_action(
            phase="profiling",
            action="dataset_analysis_complete",
            input_data={"dataset_path": self.config.dataset_path},
            output_data={
                "modality": self.dataset_profile.modality,
                "confidence": self.dataset_profile.confidence,
                "num_samples": self.dataset_profile.num_samples,
                "num_features": self.dataset_profile.num_features,
                "target_type": self.dataset_profile.target_type,
                "class_imbalance_ratio": self.dataset_profile.class_imbalance_ratio,
                "estimated_gpu_memory_gb": self.dataset_profile.estimated_gpu_memory_gb
            }
        )
        
        logger.info(f"Profiling complete: {self.dataset_profile.modality} modality detected")
    
    def _phase_strategy_development(self):
        """
        Phase 3: Strategy selection and code generation.
        
        Performs:
        - Strategy selection based on dataset profile
        - Training code generation using templates and Gemini
        - Code validation
        """
        logger.info("Phase 3: Strategy Development")
        
        # Select strategy
        self.strategy = self.controller.select_strategy(self.dataset_profile)
        
        # Log strategy selection decision
        self.state_manager.log_decision(
            decision_point="strategy_selection",
            options=[
                f"{self.strategy.primary_model}",
                f"{self.strategy.fallback_model}" if self.strategy.fallback_model else "no_fallback"
            ],
            selected=self.strategy.primary_model,
            reasoning=f"Selected {self.strategy.primary_model} for {self.dataset_profile.modality} modality. "
                     f"Batch size: {self.strategy.batch_size}, "
                     f"Max epochs: {self.strategy.max_epochs}, "
                     f"Loss: {self.strategy.loss_function}"
        )
        
        self.state_manager.log_action(
            phase="strategy_development",
            action="strategy_selected",
            input_data={"modality": self.dataset_profile.modality},
            output_data={
                "primary_model": self.strategy.primary_model,
                "fallback_model": self.strategy.fallback_model,
                "batch_size": self.strategy.batch_size,
                "max_epochs": self.strategy.max_epochs,
                "loss_function": self.strategy.loss_function
            }
        )
        
        # Generate training code with comprehensive dataset info
        from src.dataset.dataset_handler import DatasetHandler
        
        # Get dataset handler to extract column names and paths
        dataset_handler = DatasetHandler(self.config.dataset_path, self.config.competition_id)
        
        # Get train and test paths
        try:
            train_path = dataset_handler.get_train_data_path()
            test_path = dataset_handler.get_test_data_path()
        except Exception as e:
            logger.warning(f"Could not get data paths: {e}")
            train_path = os.path.join(self.config.dataset_path, "train.csv")
            test_path = os.path.join(self.config.dataset_path, "test.csv")
        
        # Get column names
        try:
            target_column = dataset_handler.get_target_column()
            id_column = dataset_handler.get_id_column()
        except Exception as e:
            logger.warning(f"Could not detect columns: {e}")
            target_column = "target"
            id_column = "id"
        
        dataset_info = {
            "dataset_path": self.config.dataset_path,
            "train_path": train_path,
            "test_path": test_path,
            "target_column": target_column,
            "id_column": id_column,
            "competition_id": self.config.competition_id,
            "output_dir": os.path.join(self.config.output_dir, f"session_{self.session_id}"),
            "num_samples": self.dataset_profile.num_samples,
            "num_features": self.dataset_profile.num_features,
            "time_budget": 3600  # 1 hour for FLAML
        }
        
        self.training_code = self.controller.generate_code(
            strategy=self.strategy,
            profile=self.dataset_profile,
            dataset_info=dataset_info
        )
        
        self.state_manager.log_action(
            phase="strategy_development",
            action="code_generated",
            input_data={"strategy": self.strategy.primary_model},
            output_data={
                "code_length": len(self.training_code),
                "code_preview": self.training_code[:200] + "..."
            }
        )
        
        logger.info("Strategy development complete")
    
    def _phase_execution(self):
        """
        Phase 4: Training with monitoring and self-improvement.
        
        Performs:
        - Code execution in sandboxed environment
        - Resource monitoring
        - Self-improvement loop: If errors or poor performance, use LLM to enhance code
        - Retry with enhanced code if needed
        - Move on if performance is acceptable or max iterations reached
        """
        logger.info("Phase 4: Execution with Self-Improvement")
        
        self.state_manager.log_action(
            phase="execution",
            action="start_training",
            input_data={"strategy": self.strategy.primary_model},
            output_data={}
        )
        
        # Calculate timeout - if max_runtime_hours is 0, no timeout
        if self.config.max_runtime_hours > 0:
            max_runtime_seconds = int(self.config.max_runtime_hours * 3600)
            timeout_seconds = max_runtime_seconds  # Use full time, not 80%
        else:
            timeout_seconds = None  # No timeout
        
        # Session directory for saving artifacts
        session_dir = os.path.join(self.config.output_dir, f"session_{self.session_id}")
        
        # Current code to execute (may be enhanced in self-improvement loop)
        current_code = self.training_code
        iteration = 0
        
        # Reset code enhancer for this session
        if self.code_enhancer:
            self.code_enhancer.reset()
        
        # Self-improvement loop
        while True:
            iteration += 1
            logger.info("=" * 80)
            logger.info(f"TRAINING ITERATION {iteration}")
            logger.info("=" * 80)
            
            # Save training code to session directory for debugging
            train_code_path = os.path.join(session_dir, f"train_v{iteration}.py")
            with open(train_code_path, 'w') as f:
                f.write(current_code)
            logger.info(f"Training code saved to: {train_code_path}")
            
            # Also save as latest train.py
            latest_train_path = os.path.join(session_dir, "train.py")
            with open(latest_train_path, 'w') as f:
                f.write(current_code)
            
            # Log execution details
            if timeout_seconds:
                logger.info(f"Max runtime: {self.config.max_runtime_hours:.2f} hours ({timeout_seconds}s)")
            else:
                logger.info("Max runtime: No limit")
            logger.info(f"Strategy: {self.strategy.primary_model}")
            
            # Execute training code with real-time output
            # CRITICAL: Controller never executes code - always delegate to Executor
            self.execution_result = self.executor.execute_code(
                code=current_code,
                timeout=timeout_seconds,
                show_progress=True
            )
            
            # Log execution result
            self.state_manager.log_action(
                phase="execution",
                action=f"training_iteration_{iteration}",
                input_data={"timeout": timeout_seconds, "iteration": iteration},
                output_data={
                    "success": self.execution_result.success,
                    "exit_code": self.execution_result.exit_code,
                    "execution_time": self.execution_result.execution_time,
                    "gpu_memory_used_gb": self.execution_result.resource_usage.gpu_memory_used_gb,
                    "gpu_utilization": self.execution_result.resource_usage.gpu_utilization_percent,
                    "stdout_length": len(self.execution_result.stdout),
                    "stderr_length": len(self.execution_result.stderr)
                }
            )
            
            # Self-improvement decision: Should we enhance and retry?
            if self.code_enhancer:
                decision = self.code_enhancer.analyze_and_decide(
                    execution_success=self.execution_result.success,
                    stdout=self.execution_result.stdout,
                    stderr=self.execution_result.stderr,
                    exit_code=self.execution_result.exit_code,
                    current_code=current_code,
                    strategy=self.strategy,
                    profile=self.dataset_profile
                )
                
                # Log the decision
                self.state_manager.log_action(
                    phase="self_improvement",
                    action="enhancement_decision",
                    input_data={
                        "iteration": iteration,
                        "execution_success": self.execution_result.success
                    },
                    output_data={
                        "should_retry": decision.should_retry,
                        "reason": decision.reason,
                        "improvements_made": decision.improvements_made,
                        "performance_acceptable": decision.performance_acceptable
                    }
                )
                
                if decision.should_retry and decision.enhanced_code:
                    logger.info(f"Self-improvement: Enhancing code and retrying")
                    logger.info(f"Improvements: {decision.improvements_made}")
                    current_code = decision.enhanced_code
                    self.training_code = current_code  # Update for finalization
                    continue  # Retry with enhanced code
                else:
                    logger.info(f"Self-improvement: {decision.reason}")
                    if decision.performance_acceptable:
                        logger.info("Performance is acceptable - proceeding to finalization")
                    else:
                        logger.warning("Could not achieve acceptable performance, but moving on")
                    break  # Exit loop
            else:
                # No code enhancer - just check for success
                if self.execution_result.success:
                    logger.info("Training execution successful - proceeding to finalization")
                    break
                else:
                    logger.warning("Training execution failed, but no LLM available for enhancement")
                    break
        
        # Final error handling if all iterations failed
        if not self.execution_result.success:
            logger.warning("Training execution failed after all iterations, attempting legacy recovery")
            
            # Use error handler for recovery (legacy approach)
            recovery_success = self.error_handler.handle_execution_error(
                error_log=self.execution_result.stderr,
                strategy=self.strategy,
                executor=self.executor,
                max_retries=2
            )
            
            if not recovery_success:
                logger.error("Recovery failed after maximum retries")
                # Don't raise - try to proceed with whatever we have
                logger.warning("Proceeding to finalization despite errors")
        
        logger.info("Execution phase complete")
    
    def _phase_finalization(self) -> Dict[str, Any]:
        """
        Phase 5: Inference and submission generation.
        
        Performs:
        - Model loading
        - Test set inference
        - Submission file generation
        - Validation
        - Reasoning trace saving
        
        Returns:
            Dictionary with results and artifact paths
        """
        logger.info("Phase 5: Finalization")
        
        self.state_manager.log_action(
            phase="finalization",
            action="start_submission_generation",
            input_data={},
            output_data={}
        )
        
        # Generate submission file
        # Note: In a real implementation, we would extract the model path
        # from the execution artifacts. For now, we use a placeholder.
        session_dir = os.path.join(self.config.output_dir, f"session_{self.session_id}")
        model_path = os.path.join(session_dir, "model.pth")
        
        # Use DatasetHandler to get correct test data path
        from src.dataset.dataset_handler import DatasetHandler
        dataset_handler = DatasetHandler(self.config.dataset_path, self.config.competition_id)
        try:
            test_data_path = dataset_handler.get_test_data_path()
        except Exception as e:
            logger.warning(f"Could not get test data path: {e}")
            test_data_path = os.path.join(self.config.dataset_path, "test.csv")
        
        try:
            submission_result = self.submission_generator.generate_and_save_submission(
                model_path=model_path,
                test_data_path=test_data_path,
                strategy=self.strategy,
                strategy_id=self.strategy.primary_model,
                seed=None,
                validate=True
            )
            
            self.state_manager.log_action(
                phase="finalization",
                action="submission_generated",
                input_data={"test_data_path": test_data_path},
                output_data={
                    "filepath": submission_result["filepath"],
                    "row_count": submission_result["row_count"],
                    "validation": submission_result["validation"]
                }
            )
            
            logger.info(f"Submission file generated: {submission_result['filepath']}")
            
        except Exception as e:
            logger.error(f"Submission generation failed: {e}")
            submission_result = {
                "filepath": None,
                "error": str(e)
            }
        
        # Save reasoning trace
        trace_path = self.state_manager.save_trace()
        logger.info(f"Reasoning trace saved: {trace_path}")
        
        # Cleanup executor
        self.executor.cleanup()
        
        # Compile results
        results = {
            "session_id": self.session_id,
            "competition_id": self.config.competition_id,
            "dataset_profile": {
                "modality": self.dataset_profile.modality,
                "confidence": self.dataset_profile.confidence,
                "num_samples": self.dataset_profile.num_samples,
                "num_features": self.dataset_profile.num_features
            },
            "strategy": {
                "primary_model": self.strategy.primary_model,
                "fallback_model": self.strategy.fallback_model
            },
            "execution": {
                "success": self.execution_result.success,
                "execution_time": self.execution_result.execution_time,
                "gpu_memory_used_gb": self.execution_result.resource_usage.gpu_memory_used_gb
            },
            "submission": submission_result,
            "reasoning_trace": trace_path,
            "output_dir": session_dir
        }
        
        logger.info("Finalization complete")
        
        return results


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="HybridAutoMLE: Autonomous Machine Learning Engineering Agent"
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--competition_id",
        type=str,
        required=True,
        help="Competition identifier"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    # Optional arguments
    parser.add_argument(
        "--max_runtime_hours",
        type=float,
        default=24.0,
        help="Maximum runtime in hours (default: 24, set to 0 for no limit)"
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=3,
        help="Number of random seeds for evaluation (default: 3)"
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini model to use (default: gemini-2.0-flash)"
    )
    parser.add_argument(
        "--eval_mode",
        action="store_true",
        help="Enable evaluation mode for multiple competitions"
    )
    parser.add_argument(
        "--competitions",
        type=str,
        nargs="+",
        help="List of competitions for eval mode"
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Validate arguments
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get Gemini API key from environment
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    # Create agent configuration
    config = AgentConfig(
        dataset_path=args.dataset_path,
        competition_id=args.competition_id,
        output_dir=args.output_dir,
        max_runtime_hours=args.max_runtime_hours,
        num_seeds=args.num_seeds,
        gemini_model=args.gemini_model,
        gemini_api_key=gemini_api_key,
        eval_mode=args.eval_mode,
        competitions=args.competitions or []
    )
    
    # Create and run agent
    agent = HybridAutoMLEAgent(config)
    
    try:
        results = agent.run()
        
        # Print summary
        print("\n" + "=" * 80)
        print("EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Session ID: {results['session_id']}")
        print(f"Competition: {results['competition_id']}")
        print(f"Modality: {results['dataset_profile']['modality']}")
        print(f"Strategy: {results['strategy']['primary_model']}")
        print(f"Execution Time: {results['execution']['execution_time']:.2f} seconds")
        print(f"Total Runtime: {results['total_runtime_hours']:.2f} hours")
        print(f"Submission File: {results['submission'].get('filepath', 'N/A')}")
        print(f"Reasoning Trace: {results['reasoning_trace']}")
        print(f"Output Directory: {results['output_dir']}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

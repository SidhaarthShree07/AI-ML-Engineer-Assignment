"""
Cloud Executor Module

Executes code on local Colab kernel connected to VS Code.
Generates Jupyter notebooks for local execution with polling for completion.
"""

import json
import logging
import os
import re
import time
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from src.models.data_models import ExecutionResult, ResourceMetrics

logger = logging.getLogger(__name__)


@dataclass
class CloudConfig:
    """Configuration for cloud execution."""
    notebook_path: str = ""
    workspace_dir: str = ""
    poll_interval: float = 5.0  # seconds between polls
    max_wait_time: float = 14400.0  # 4 hours max wait
    local_output_dir: Optional[str] = None  # Local dir for output
    use_papermill: bool = False  # True for local kernel, False for Colab


class CloudExecutor:
    """
    Cloud-based code executor using Jupyter notebooks with local Colab kernel.
    
    Supports two execution modes:
    1. Papermill mode: For local Jupyter kernels (automated execution)
    2. Polling mode: For Colab kernels connected to VS Code (manual Run All + polling)
    """
    
    def __init__(self, config: CloudConfig):
        """Initialize the cloud executor."""
        self.config = config
        self.workspace_dir = Path(config.workspace_dir) if config.workspace_dir else Path.cwd()
        self.notebook_path = config.notebook_path or str(self.workspace_dir / "cloud_training.ipynb")
        self.local_output_dir = config.local_output_dir
        self.use_papermill = config.use_papermill
        
        # Create workspace if needed
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[INIT] CloudExecutor initialized")
        logger.info(f"  Notebook: {self.notebook_path}")
        logger.info(f"  Workspace: {self.workspace_dir}")
        logger.info(f"  Mode: {'Papermill (local kernel)' if self.use_papermill else 'Polling (Colab kernel)'}")

    def _build_notebook_cells(
        self,
        code: str,
        submission_filename: str = "submission.csv"
    ) -> List[Dict[str, Any]]:
        """Build notebook cells for execution with local data paths."""
        
        cells = []
        
        # Cell 1: Title/Header
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Cloud Training Notebook\n",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                "\n",
                "**Instructions:**\n",
                "1. Ensure GPU runtime is enabled (Runtime > Change runtime type > GPU)\n",
                "2. Click 'Run All' to execute all cells\n",
                "3. Wait for completion marker at the end\n",
                "4. submission.csv will be saved to /content/submission.csv\n"
            ]
        })
        
        # Cell 2: Install required packages
        install_code = '''# Install required packages
import subprocess
import sys

packages = [
    'pandas',
    'numpy',
    'scikit-learn',
    'lightgbm',
    'xgboost',
    'catboost',
    'flaml',
    'scipy',
    'matplotlib',
    'seaborn',
    'tqdm',
]

for pkg in packages:
    try:
        __import__(pkg.replace('-', '_').split('[')[0])
        print(f"[OK] {pkg} already installed")
    except ImportError:
        print(f"[INSTALLING] {pkg}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
        print(f"[OK] {pkg} installed")

print("\\n[SUCCESS] All packages ready!")
'''
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": install_code.split('\n'),
            "execution_count": None,
            "outputs": []
        })
        
        # Cell 3: GPU Check
        gpu_check_code = '''# Check GPU availability
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[OK] GPU Available: {gpu_name}")
    print(f"[OK] GPU Memory: {gpu_memory:.1f} GB")
    DEVICE = "cuda"
else:
    print("[WARNING] No GPU detected! Training will be slow.")
    print("[TIP] Go to Runtime > Change runtime type > GPU")
    DEVICE = "cpu"

print(f"\\n[INFO] Using device: {DEVICE}")
'''
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": gpu_check_code.split('\n'),
            "execution_count": None,
            "outputs": []
        })
        
        # Cell 4: Main Training Code (paths are already in the template)
        training_code = f'''# Training Code
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import warnings
warnings.filterwarnings('ignore')

print("[START] Training started...")
print("=" * 50)

{code}

print("=" * 50)
print("[COMPLETE] Training finished!")
'''
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": training_code.split('\n'),
            "execution_count": None,
            "outputs": []
        })
        
        # Cell 5: Save submission locally
        save_submission_code = f'''# Save submission file
import os
import shutil

SUBMISSION_FILENAME = "{submission_filename}"
OUTPUT_DIR = "."

# Find submission file in current directory
submission_file = None
if os.path.exists('submission.csv'):
    submission_file = 'submission.csv'
else:
    for f in os.listdir('.'):
        if f.endswith('.csv') and 'submission' in f.lower():
            submission_file = f
            break

if submission_file:
    print(f"[OK] Submission found: {{submission_file}}")
    
    # Print preview
    import pandas as pd
    df = pd.read_csv(submission_file)
    print(f"[INFO] Submission shape: {{df.shape}}")
    print(f"[INFO] Preview:")
    print(df.head())
else:
    print("[WARNING] No submission file found!")
    print("[INFO] Looking in current directory:")
    for f in os.listdir('.'):
        print(f"  - {{f}}")
'''
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": save_submission_code.split('\n'),
            "execution_count": None,
            "outputs": []
        })
        
        # Cell 5: Completion Marker
        completion_code = '''# Execution Complete Marker
import datetime

print("\\n" + "=" * 60)
print("---END_OF_EXECUTION---")
print("=" * 60)
print(f"Completed at: {datetime.datetime.now()}")
print("\\n[SUCCESS] All cells executed successfully!")
print("[INFO] submission.csv should be in current directory")
'''
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": completion_code.split('\n'),
            "execution_count": None,
            "outputs": []
        })
        
        return cells

    def _write_notebook(self, cells: List[Dict[str, Any]], path: str) -> None:
        """Write cells to a notebook file."""
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 4,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0"
                },
                "colab": {
                    "provenance": [],
                    "gpuType": "T4"
                },
                "accelerator": "GPU"
            },
            "cells": cells
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[OK] Notebook written: {path}")

    def _read_notebook(self, path: str) -> Optional[Dict[str, Any]]:
        """Read a notebook file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[ERROR] Failed to read notebook: {e}")
            return None

    def _check_execution_status(self, notebook_path: str) -> Tuple[bool, bool, str, bool]:
        """
        Check if notebook execution is complete.
        
        Returns:
            (is_complete, is_success, output, has_any_output)
        """
        notebook = self._read_notebook(notebook_path)
        if not notebook:
            return False, False, "Failed to read notebook", False
        
        cells = notebook.get('cells', [])
        all_output = []
        has_any_output = False
        found_end_marker = False
        found_error = False
        
        for cell in cells:
            if cell.get('cell_type') != 'code':
                continue
            
            outputs = cell.get('outputs', [])
            if outputs:
                has_any_output = True
            
            for output in outputs:
                output_type = output.get('output_type', '')
                
                # Check for errors
                if output_type == 'error':
                    found_error = True
                    ename = output.get('ename', 'Error')
                    evalue = output.get('evalue', '')
                    traceback = output.get('traceback', [])
                    error_msg = f"[ERROR] {ename}: {evalue}"
                    all_output.append(error_msg)
                    if traceback:
                        all_output.append('\n'.join(traceback[:5]))  # First 5 lines
                
                # Check for stream output (stdout/stderr)
                elif output_type == 'stream':
                    text = output.get('text', '')
                    if isinstance(text, list):
                        text = ''.join(text)
                    all_output.append(text)
                    
                    if '---END_OF_EXECUTION---' in text:
                        found_end_marker = True
                
                # Check for execute_result
                elif output_type == 'execute_result':
                    data = output.get('data', {})
                    if 'text/plain' in data:
                        text = data['text/plain']
                        if isinstance(text, list):
                            text = ''.join(text)
                        all_output.append(text)
        
        combined_output = '\n'.join(all_output)
        
        if found_end_marker:
            return True, not found_error, combined_output, has_any_output
        
        if found_error:
            return True, False, combined_output, has_any_output
        
        return False, False, combined_output, has_any_output

    def _execute_with_papermill(
        self,
        input_notebook: str,
        output_notebook: str,
        timeout: int = 14400
    ) -> Tuple[bool, str]:
        """
        Execute notebook using papermill (for local kernels).
        
        Returns:
            (success, output)
        """
        try:
            import papermill as pm
            
            logger.info(f"[PAPERMILL] Executing notebook: {input_notebook}")
            logger.info(f"[PAPERMILL] Output will be saved to: {output_notebook}")
            
            # Execute notebook
            pm.execute_notebook(
                input_notebook,
                output_notebook,
                kernel_name='python3',
                progress_bar=True,
                request_save_on_cell_execute=True,
                execution_timeout=timeout
            )
            
            # Read output notebook for results
            is_complete, is_success, output, _ = self._check_execution_status(output_notebook)
            
            return is_success, output
            
        except Exception as e:
            error_msg = f"[ERROR] Papermill execution failed: {e}"
            logger.error(error_msg)
            return False, error_msg

    def _execute_with_polling(
        self,
        notebook_path: str,
        poll_interval: float = 5.0,
        max_wait_time: float = 14400.0
    ) -> Tuple[bool, str]:
        """
        Execute notebook by polling for completion (for Colab kernels).
        User must manually click 'Run All' in VS Code.
        
        Returns:
            (success, output)
        """
        logger.info(f"\n{'='*60}")
        logger.info("[POLLING MODE] Local Colab kernel")
        logger.info(f"{'='*60}")
        logger.info(f"Notebook: {notebook_path}")
        logger.info(f"Poll interval: {poll_interval}s")
        logger.info(f"Max wait: {max_wait_time}s ({max_wait_time/3600:.1f} hours)")
        logger.info("")
        logger.info("[ACTION REQUIRED]")
        logger.info("  1. Open the notebook in VS Code")
        logger.info("  2. Select the Colab kernel (Python 3 ipykernel Colab)")
        logger.info("  3. Click 'Run All' to execute all cells")
        logger.info("")
        logger.info("[WAITING] Polling for completion...")
        logger.info(f"{'='*60}\n")
        
        # Try to open the notebook in VS Code
        try:
            subprocess.Popen(
                ['code', notebook_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True
            )
            logger.info(f"[OK] Opened notebook in VS Code: {notebook_path}")
        except Exception as e:
            logger.warning(f"[WARNING] Could not auto-open notebook: {e}")
            logger.info(f"[INFO] Please manually open: {notebook_path}")
        
        # Polling loop
        start_time = time.time()
        last_status = ""
        poll_count = 0
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_wait_time:
                return False, f"[TIMEOUT] Max wait time ({max_wait_time}s) exceeded"
            
            # Check status
            is_complete, is_success, output, has_output = self._check_execution_status(notebook_path)
            
            # Show progress
            poll_count += 1
            status = f"[POLL #{poll_count}] Elapsed: {elapsed:.0f}s | Output: {'Yes' if has_output else 'No'} | Complete: {is_complete}"
            
            if status != last_status:
                logger.info(status)
                last_status = status
            
            if is_complete:
                if is_success:
                    logger.info(f"\n[SUCCESS] Notebook execution completed in {elapsed:.0f}s")
                else:
                    logger.error(f"\n[FAILED] Notebook execution failed after {elapsed:.0f}s")
                return is_success, output
            
            # Wait before next poll
            time.sleep(poll_interval)

    def execute_code(
        self,
        code: str,
        timeout: int = 14400,
        submission_filename: str = "submission.csv",
        show_progress: bool = True,  # Ignored for cloud, but needed for API compatibility
        generate_only: bool = False  # If True, just generate notebook, don't wait for execution
    ) -> ExecutionResult:
        """
        Execute code in cloud environment.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            submission_filename: Name for the submission file
            show_progress: Ignored (for API compatibility with Executor)
            generate_only: If True, only generate notebook without waiting for execution
            
        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("[CLOUD EXECUTOR] Starting execution")
        print("=" * 60)
        print(f"  Mode: {'Generate Only' if generate_only else ('Papermill' if self.use_papermill else 'Polling (Colab)')}")
        print(f"  Notebook: {self.notebook_path}")
        
        # Build and write notebook
        cells = self._build_notebook_cells(code, submission_filename)
        self._write_notebook(cells, self.notebook_path)
        
        print(f"\n[CLOUD EXECUTOR] [OK] Notebook generated: {self.notebook_path}")
        
        # If generate_only, return immediately after writing notebook
        if generate_only:
            elapsed = time.time() - start_time
            print(f"[CLOUD EXECUTOR] Generate-only mode - returning immediately")
            print(f"[CLOUD EXECUTOR] User should run notebook manually and place submission.csv in session folder")
            return ExecutionResult(
                success=True,
                stdout="Notebook generated successfully (generate_only mode)",
                stderr="",
                exit_code=0,
                execution_time=elapsed,
                resource_usage=ResourceMetrics(
                    gpu_memory_used_gb=0,
                    gpu_memory_total_gb=15.0,
                    gpu_utilization_percent=0,
                    ram_used_gb=0,
                    ram_total_gb=12.0,
                    cpu_percent=0,
                    runtime_seconds=elapsed
                ),
                artifacts=[self.notebook_path]
            )
        
        # Execute based on mode
        if self.use_papermill:
            # Papermill mode for local kernel
            output_notebook = self.notebook_path.replace('.ipynb', '_executed.ipynb')
            success, output = self._execute_with_papermill(
                self.notebook_path,
                output_notebook,
                timeout
            )
        else:
            # Polling mode for Colab kernel
            success, output = self._execute_with_polling(
                self.notebook_path,
                self.config.poll_interval,
                self.config.max_wait_time
            )
        
        elapsed = time.time() - start_time
        
        # Build result with correct ExecutionResult format
        return ExecutionResult(
            success=success,
            stdout=output if success else "",
            stderr="" if success else output,
            exit_code=0 if success else 1,
            execution_time=elapsed,
            resource_usage=ResourceMetrics(
                gpu_memory_used_gb=0,  # Not tracked in cloud mode
                gpu_memory_total_gb=15.0,  # Colab T4 has ~15GB
                gpu_utilization_percent=0,
                ram_used_gb=0,
                ram_total_gb=12.0,  # Colab has ~12GB RAM
                cpu_percent=0,
                runtime_seconds=elapsed
            ),
            artifacts=[]
        )

    def cleanup(self):
        """Cleanup any temporary files."""
        # Keep notebooks for debugging
        logger.info("[CLEANUP] CloudExecutor cleanup (notebooks preserved)")


def create_cloud_executor(
    workspace_dir: str = "",
    notebook_path: str = "",
    local_output_dir: Optional[str] = None,
    use_papermill: bool = False,
    poll_interval: float = 5.0,
    max_wait_time: float = 14400.0
) -> CloudExecutor:
    """
    Factory function to create CloudExecutor with configuration.
    
    Args:
        workspace_dir: Directory for notebook files
        notebook_path: Path for generated notebook
        local_output_dir: Local directory for output
        use_papermill: True for local kernel (papermill), False for Colab (polling)
        poll_interval: Seconds between status checks (polling mode)
        max_wait_time: Maximum wait time in seconds
        
    Returns:
        Configured CloudExecutor instance
    """
    config = CloudConfig(
        notebook_path=notebook_path,
        workspace_dir=workspace_dir,
        poll_interval=poll_interval,
        max_wait_time=max_wait_time,
        local_output_dir=local_output_dir,
        use_papermill=use_papermill
    )
    
    return CloudExecutor(config)

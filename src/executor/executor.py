"""Executor: Safe code execution environment with Docker isolation"""

import subprocess
import tempfile
import time
import os
import sys
import psutil
import shutil
import logging
from pathlib import Path
from typing import Optional
from src.models import ExecutionResult, ResourceMetrics

logger = logging.getLogger(__name__)


class DockerConfig:
    """Docker configuration for sandboxed execution"""
    
    def __init__(self, 
                 image: str = "ml-sandbox:latest", 
                 gpu_enabled: bool = True,
                 memory_limit: str = "16g",
                 cpu_count: float = 4.0,
                 use_docker: bool = True):
        self.image = image
        self.gpu_enabled = gpu_enabled
        self.memory_limit = memory_limit
        self.cpu_count = cpu_count
        self.use_docker = use_docker
        self.docker_built = False


class Executor:
    """Safe code execution environment with Docker isolation"""
    
    def __init__(self, docker_config: DockerConfig):
        self.docker_config = docker_config
        self.workspace_dir = Path("executor/workspace")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        if self.docker_config.use_docker:
            self._ensure_docker_image()
    
    def _ensure_docker_image(self):
        """Build Docker image if it doesn't exist"""
        if self.docker_config.docker_built:
            return
        
        logger.info("Checking Docker image...")
        try:
            result = subprocess.run(
                ["docker", "images", "-q", self.docker_config.image],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.stdout.strip():
                logger.info(f"Docker image '{self.docker_config.image}' already exists")
                self.docker_config.docker_built = True
                return
        except Exception as e:
            logger.warning(f"Could not check Docker image: {e}")
        
        # Build image
        logger.info(f"Building Docker image '{self.docker_config.image}'...")
        logger.info("This may take a few minutes on first run...")
        try:
            if Path("requirements.txt").exists():
                shutil.copy("requirements.txt", "executor/requirements.txt")
            result = subprocess.run(
                ["docker", "build", "-t", self.docker_config.image, "executor/"],
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode == 0:
                logger.info("Docker image built successfully!")
                self.docker_config.docker_built = True
            else:
                logger.error(f"Docker build failed: {result.stderr}")
                logger.warning("Falling back to local execution")
                self.docker_config.use_docker = False
        except subprocess.TimeoutExpired:
            logger.error("Docker build timed out")
            self.docker_config.use_docker = False
        except Exception as e:
            logger.error(f"Docker build error: {e}")
            self.docker_config.use_docker = False
    
    def execute_code(self, code: str, timeout: Optional[int] = None, show_progress: bool = True, generate_only: bool = False) -> ExecutionResult:
        # generate_only is ignored for local/Docker executor (only applies to cloud)
        start_time = time.time()
        if self.docker_config.use_docker:
            return self._execute_in_docker(code, timeout, show_progress, start_time)
        else:
            return self._execute_locally(code, timeout, show_progress, start_time)
    
    def _execute_in_docker(self, code: str, timeout: Optional[int], show_progress: bool, start_time: float) -> ExecutionResult:
        code_file = self.workspace_dir / "generated_code.py"
        code_file.write_text(code)
        
        if show_progress:
            if timeout:
                logger.info(f"Starting Docker execution (timeout: {timeout}s = {timeout/60:.1f} minutes)")
            else:
                logger.info("Starting Docker execution (no timeout limit)")
            logger.info(f"Code saved to: {code_file}")
            logger.info("=" * 80)
            logger.info("TRAINING OUTPUT:")
            logger.info("=" * 80)
        
        docker_cmd = [
            "docker", "run",
            "--rm",
            "-v", f"{self.workspace_dir.absolute()}:/app/workspace",
            "-v", f"{Path.cwd().absolute()}/data:/app/data:ro",
            "-v", f"{Path.cwd().absolute()}/data_test:/app/data_test:ro",
            "-w", "/app",
            "-m", self.docker_config.memory_limit,
            f"--cpus={self.docker_config.cpu_count}",
        ]
        
        if self.docker_config.gpu_enabled:
            try:
                result = subprocess.run(
                    ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.8.0-base-ubuntu22.04", "nvidia-smi"],
                    capture_output=True,
                    timeout=10
                )
                if result.returncode == 0:
                    docker_cmd.extend(["--gpus", "all"])
                    logger.info("GPU support enabled in Docker")
            except:
                logger.info("GPU not available in Docker, using CPU only")
        
        # Use unbuffered Python to avoid log freeze
        docker_cmd.extend([
            self.docker_config.image,
            "python", "-u", "/app/workspace/generated_code.py"
        ])
        
        try:
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            stdout_lines = []
            
            while True:
                return_code = process.poll()
                if return_code is not None:
                    remaining_stdout = process.communicate()[0]
                    if remaining_stdout:
                        for line in remaining_stdout.splitlines():
                            stdout_lines.append(line + '\n')
                            if show_progress:
                                print(line)
                    break
                
                if timeout and (time.time() - start_time) > timeout:
                    process.kill()
                    process.wait()
                    raise subprocess.TimeoutExpired(docker_cmd, timeout)
                
                line = process.stdout.readline()
                if line:
                    stdout_lines.append(line)
                    if show_progress:
                        print(line.rstrip())
                time.sleep(0.05)
            
            execution_time = time.time() - start_time
            resource_usage = self.get_resource_usage()
            resource_usage.runtime_seconds = execution_time
            
            if show_progress:
                logger.info("=" * 80)
                logger.info(f"Docker execution completed in {execution_time/60:.1f} minutes")
            
            return ExecutionResult(
                success=(return_code == 0),
                stdout=''.join(stdout_lines),
                stderr="",
                exit_code=return_code,
                execution_time=execution_time,
                resource_usage=resource_usage,
                artifacts=[]
            )
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            resource_usage = self.get_resource_usage()
            resource_usage.runtime_seconds = execution_time
            return ExecutionResult(
                success=False,
                stdout=''.join(stdout_lines),
                stderr=f"Timeout after {timeout} seconds",
                exit_code=-1,
                execution_time=execution_time,
                resource_usage=resource_usage,
                artifacts=[]
            )
        except Exception as e:
            execution_time = time.time() - start_time
            resource_usage = self.get_resource_usage()
            resource_usage.runtime_seconds = execution_time
            return ExecutionResult(
                success=False,
                stdout=''.join(stdout_lines),
                stderr=f"Docker execution error: {str(e)}",
                exit_code=-1,
                execution_time=execution_time,
                resource_usage=resource_usage,
                artifacts=[]
            )
    
    def _execute_locally(self, code: str, timeout: Optional[int], show_progress: bool, start_time: float) -> ExecutionResult:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name
        
        try:
            if show_progress:
                logger.info(f"Starting local execution: {temp_file}")
                logger.info("=" * 80)
            
            process = subprocess.Popen(
                [sys.executable, "-u", temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            stdout_lines = []
            
            while True:
                return_code = process.poll()
                if return_code is not None:
                    remaining_stdout = process.communicate()[0]
                    if remaining_stdout:
                        for line in remaining_stdout.splitlines():
                            stdout_lines.append(line + '\n')
                            if show_progress:
                                print(line)
                    break
                
                if timeout and (time.time() - start_time) > timeout:
                    process.kill()
                    process.wait()
                    raise subprocess.TimeoutExpired([sys.executable, temp_file], timeout)
                
                line = process.stdout.readline()
                if line:
                    stdout_lines.append(line)
                    if show_progress:
                        print(line.rstrip())
                time.sleep(0.05)
            
            execution_time = time.time() - start_time
            resource_usage = self.get_resource_usage()
            resource_usage.runtime_seconds = execution_time
            
            return ExecutionResult(
                success=(return_code == 0),
                stdout=''.join(stdout_lines),
                stderr="",
                exit_code=return_code,
                execution_time=execution_time,
                resource_usage=resource_usage,
                artifacts=[]
            )
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def get_resource_usage(self) -> ResourceMetrics:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        ram_used_gb = memory.used / (1024 ** 3)
        ram_total_gb = memory.total / (1024 ** 3)
        
        gpu_memory_used_gb = 0.0
        gpu_memory_total_gb = 24.0
        gpu_utilization_percent = 0.0
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split('\n')[0].split(',')
                if len(parts) >= 3:
                    gpu_memory_used_gb = float(parts[0].strip()) / 1024.0
                    gpu_memory_total_gb = float(parts[1].strip()) / 1024.0
                    gpu_utilization_percent = float(parts[2].strip())
        except:
            pass
        
        return ResourceMetrics(
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
            gpu_utilization_percent=gpu_utilization_percent,
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
            cpu_percent=cpu_percent,
            runtime_seconds=0.0
        )
    
    def cleanup(self):
        if self.workspace_dir.exists():
            try:
                for file in self.workspace_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
            except:
                pass

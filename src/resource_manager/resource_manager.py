"""Resource Manager for monitoring and OOM recovery"""

import subprocess
import psutil
import time
from typing import Optional, List, Dict, Any
from dataclasses import replace

from src.models.data_models import (
    ResourceMetrics,
    Strategy,
    ResourceConstraints,
    RecoveryPlan
)


class ResourceStatus:
    """Resource status information"""
    def __init__(
        self,
        gpu_memory_percent: float,
        ram_percent: float,
        runtime_percent: float,
        needs_intervention: bool,
        intervention_type: Optional[str] = None
    ):
        self.gpu_memory_percent = gpu_memory_percent
        self.ram_percent = ram_percent
        self.runtime_percent = runtime_percent
        self.needs_intervention = needs_intervention
        self.intervention_type = intervention_type


class ResourceManager:
    """Manages resource monitoring and OOM recovery"""
    
    def __init__(
        self,
        max_vram_gb: float = 24.0,
        max_ram_gb: float = 440.0,
        max_runtime_sec: int = 86400  # 24 hours
    ):
        """
        Initialize ResourceManager with resource limits
        
        Args:
            max_vram_gb: Maximum GPU VRAM in GB (default: 24GB)
            max_ram_gb: Maximum system RAM in GB (default: 440GB)
            max_runtime_sec: Maximum runtime in seconds (default: 24 hours)
        """
        self.max_vram_gb = max_vram_gb
        self.max_ram_gb = max_ram_gb
        self.max_runtime_sec = max_runtime_sec
        self.start_time = time.time()
        
        # Thresholds
        self.gpu_threshold = 0.90  # 90%
        self.ram_threshold = 0.90  # 90%
        self.runtime_threshold = 0.80  # 80%
    
    def monitor_execution(self, process: Optional[psutil.Process] = None) -> ResourceStatus:
        """
        Monitor process resource usage
        
        Args:
            process: Process to monitor (if None, monitors system-wide)
            
        Returns:
            ResourceStatus with current usage and intervention needs
        """
        # Get GPU memory usage
        gpu_memory_used = self.get_gpu_memory_usage()
        gpu_memory_percent = gpu_memory_used / self.max_vram_gb if self.max_vram_gb > 0 else 0.0
        
        # Get RAM usage
        ram_info = psutil.virtual_memory()
        ram_used_gb = ram_info.used / (1024 ** 3)
        ram_percent = ram_used_gb / self.max_ram_gb if self.max_ram_gb > 0 else 0.0
        
        # Get runtime percentage
        elapsed = time.time() - self.start_time
        runtime_percent = elapsed / self.max_runtime_sec if self.max_runtime_sec > 0 else 0.0
        
        # Determine if intervention is needed
        needs_intervention = False
        intervention_type = None
        
        if gpu_memory_percent >= self.gpu_threshold:
            needs_intervention = True
            intervention_type = "GPU_OOM_PREVENTION"
        elif ram_percent >= self.ram_threshold:
            needs_intervention = True
            intervention_type = "RAM_CLEANUP"
        elif runtime_percent >= self.runtime_threshold:
            needs_intervention = True
            intervention_type = "TIMEOUT_APPROACHING"
        
        return ResourceStatus(
            gpu_memory_percent=gpu_memory_percent,
            ram_percent=ram_percent,
            runtime_percent=runtime_percent,
            needs_intervention=needs_intervention,
            intervention_type=intervention_type
        )
    
    def get_gpu_memory_usage(self) -> float:
        """
        Get current GPU memory usage in GB using nvidia-smi
        
        Returns:
            GPU memory used in GB (0.0 if no GPU or error)
        """
        try:
            # Run nvidia-smi to get GPU memory usage
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse output (in MB)
                memory_mb = float(result.stdout.strip().split('\n')[0])
                return memory_mb / 1024.0  # Convert to GB
            else:
                return 0.0
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
            # nvidia-smi not available or error occurred
            return 0.0
    
    def trigger_oom_prevention(self, strategy: Strategy) -> Strategy:
        """
        Modify strategy to prevent OOM
        
        Args:
            strategy: Current strategy
            
        Returns:
            Modified strategy with OOM prevention measures
        """
        # Create a copy of the strategy
        new_strategy = replace(strategy)
        
        # Stage 1: Reduce batch size and enable gradient accumulation
        if new_strategy.batch_size > 1:
            new_strategy.batch_size = max(1, new_strategy.batch_size // 2)
            new_strategy.gradient_accumulation_steps = new_strategy.gradient_accumulation_steps * 2
        
        # Stage 2: Enable mixed precision
        if not new_strategy.mixed_precision:
            new_strategy.mixed_precision = True
        
        # Stage 3: Downsize model if possible
        if new_strategy.model_size == "large":
            new_strategy.model_size = "medium"
        elif new_strategy.model_size == "medium":
            new_strategy.model_size = "small"
        
        return new_strategy
    
    def handle_oom_error(self, error_log: str, strategy: Strategy) -> List[RecoveryPlan]:
        """
        Generate recovery actions for OOM error
        
        Args:
            error_log: Error log containing OOM information
            strategy: Current strategy
            
        Returns:
            List of recovery actions ordered by priority
        """
        recovery_actions = []
        
        # Determine error type
        is_cuda_oom = "CUDA out of memory" in error_log or "OutOfMemoryError" in error_log
        is_ram_oom = "MemoryError" in error_log or "Cannot allocate memory" in error_log
        
        if is_cuda_oom:
            # CUDA OOM recovery protocol
            # Priority 1: Halve batch size and enable gradient accumulation
            if strategy.batch_size > 1:
                recovery_actions.append(RecoveryPlan(
                    action="reduce_batch_size",
                    params={
                        "new_batch_size": strategy.batch_size // 2,
                        "gradient_accumulation_steps": strategy.gradient_accumulation_steps * 2
                    },
                    priority=1
                ))
            
            # Priority 2: Enable mixed precision
            if not strategy.mixed_precision:
                recovery_actions.append(RecoveryPlan(
                    action="enable_mixed_precision",
                    params={"mixed_precision": True},
                    priority=2
                ))
            
            # Priority 3: Downsize model architecture
            if strategy.model_size != "small":
                new_size = "medium" if strategy.model_size == "large" else "small"
                recovery_actions.append(RecoveryPlan(
                    action="downsize_model",
                    params={"new_model_size": new_size},
                    priority=3
                ))
        
        elif is_ram_oom:
            # RAM OOM recovery protocol
            # Priority 1: Reduce batch size
            if strategy.batch_size > 1:
                recovery_actions.append(RecoveryPlan(
                    action="reduce_batch_size",
                    params={
                        "new_batch_size": strategy.batch_size // 2,
                        "gradient_accumulation_steps": strategy.gradient_accumulation_steps * 2
                    },
                    priority=1
                ))
            
            # Priority 2: Apply feature selection (for tabular)
            if strategy.modality == "tabular":
                recovery_actions.append(RecoveryPlan(
                    action="apply_feature_selection",
                    params={"max_features": 50},
                    priority=2
                ))
        
        # Priority 4: Activate fallback strategy
        if strategy.fallback_model:
            recovery_actions.append(RecoveryPlan(
                action="activate_fallback",
                params={"fallback_model": strategy.fallback_model},
                priority=4
            ))
        
        # Sort by priority
        recovery_actions.sort(key=lambda x: x.priority)
        
        return recovery_actions
    
    def get_resource_metrics(self) -> ResourceMetrics:
        """
        Get current resource metrics
        
        Returns:
            ResourceMetrics with current usage
        """
        # GPU metrics
        gpu_memory_used = self.get_gpu_memory_usage()
        gpu_memory_total = self.max_vram_gb
        
        # Try to get GPU utilization
        gpu_utilization = 0.0
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_utilization = float(result.stdout.strip().split('\n')[0])
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
            pass
        
        # RAM metrics
        ram_info = psutil.virtual_memory()
        ram_used_gb = ram_info.used / (1024 ** 3)
        ram_total_gb = ram_info.total / (1024 ** 3)
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Runtime
        runtime_seconds = time.time() - self.start_time
        
        return ResourceMetrics(
            gpu_memory_used_gb=gpu_memory_used,
            gpu_memory_total_gb=gpu_memory_total,
            gpu_utilization_percent=gpu_utilization,
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
            cpu_percent=cpu_percent,
            runtime_seconds=runtime_seconds
        )

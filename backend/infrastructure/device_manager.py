"""
Device Manager for GPU acceleration and hardware optimization.

This module provides comprehensive device management capabilities including
automatic device detection (CUDA, MPS, CPU), GPU memory management,
model configuration optimization, and performance monitoring.

"""

import os
import sys
import logging
import psutil
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from backend.core.models.data_models import Configuration, GPUStatus
from backend.core.models.errors import ModelLoadingError

# Configure logging
logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """Enumeration of supported device types."""
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon GPU
    CPU = "cpu"
    AUTO = "auto"


@dataclass
class DeviceCapabilities:
    """Detailed device capabilities and specifications."""
    device_type: DeviceType
    device_name: str
    memory_total_mb: int = 0
    memory_available_mb: int = 0
    compute_capability: Tuple[int, int] = (0, 0)
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_int8: bool = False
    supports_int4: bool = False
    max_batch_size: int = 1
    recommended_dtype: str = "float32"
    performance_score: float = 0.0  # Relative performance score


@dataclass
class ModelOptimizationConfig:
    """Configuration for model optimization based on device capabilities."""
    device_map: Union[str, Dict[str, Any]]
    torch_dtype: str
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = False
    low_cpu_mem_usage: bool = False
    max_memory: Optional[Dict[str, str]] = None
    offload_folder: Optional[str] = None
    batch_size: int = 1
    gradient_checkpointing: bool = False


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    device_utilization: float = 0.0
    memory_utilization: float = 0.0
    temperature: float = 0.0
    power_usage: float = 0.0
    inference_time_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    timestamp: float = field(default_factory=time.time)


class DeviceDetector:
    """
    Detects and evaluates available computing devices.
    
    This class provides methods to discover available devices (CUDA, MPS, CPU),
    assess their capabilities, and rank them by performance potential.
    """
    
    def __init__(self):
        self._torch_available = self._check_torch_availability()
        self._detected_devices: List[DeviceCapabilities] = []
        self._detection_complete = False
    
    def _check_torch_availability(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            logger.warning("PyTorch not available. GPU acceleration disabled.")
            return False
    
    def detect_devices(self) -> List[DeviceCapabilities]:
        """
        Detect all available computing devices and their capabilities.
        
        Returns:
            List of DeviceCapabilities sorted by performance potential
        """
        if self._detection_complete:
            return self._detected_devices
        
        self._detected_devices = []
        
        if not self._torch_available:
            # Fallback to CPU only
            cpu_device = self._detect_cpu_capabilities()
            self._detected_devices.append(cpu_device)
        else:
            # Detect GPU devices first (higher priority)
            cuda_devices = self._detect_cuda_devices()
            mps_devices = self._detect_mps_devices()
            cpu_device = self._detect_cpu_capabilities()
            
            self._detected_devices.extend(cuda_devices)
            self._detected_devices.extend(mps_devices)
            self._detected_devices.append(cpu_device)
        
        # Sort by performance score (highest first)
        self._detected_devices.sort(key=lambda d: d.performance_score, reverse=True)
        
        self._detection_complete = True
        logger.info(f"Detected {len(self._detected_devices)} computing devices")
        
        return self._detected_devices

    def _detect_cuda_devices(self) -> List[DeviceCapabilities]:
        """Detect CUDA-capable devices."""
        cuda_devices = []
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                logger.info("CUDA not available")
                return cuda_devices
            
            device_count = torch.cuda.device_count()
            logger.info(f"Found {device_count} CUDA device(s)")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                
                # Get memory information
                total_memory = props.total_memory // (1024 * 1024)  # Convert to MB
                try:
                    torch.cuda.set_device(i)
                    available_memory = (props.total_memory - torch.cuda.memory_allocated(i)) // (1024 * 1024)
                except Exception:
                    available_memory = int(total_memory * 0.9)  # Estimate 90% available
                
                # Determine capabilities
                compute_cap = (props.major, props.minor)
                supports_fp16 = compute_cap >= (5, 3)  # Maxwell and newer
                supports_bf16 = compute_cap >= (8, 0)  # Ampere and newer
                supports_int8 = compute_cap >= (6, 1)  # Pascal and newer
                supports_int4 = compute_cap >= (7, 5)  # Turing and newer
                
                # Calculate performance score based on memory and compute capability
                performance_score = (
                    total_memory * 0.4 +  # Memory weight
                    (compute_cap[0] * 10 + compute_cap[1]) * 100 +  # Compute capability weight
                    props.multi_processor_count * 10  # SM count weight
                )
                
                # Determine recommended dtype
                if supports_bf16:
                    recommended_dtype = "bfloat16"
                elif supports_fp16:
                    recommended_dtype = "float16"
                else:
                    recommended_dtype = "float32"
                
                # Estimate max batch size based on memory
                max_batch_size = max(1, min(32, total_memory // 1000))
                
                device_cap = DeviceCapabilities(
                    device_type=DeviceType.CUDA,
                    device_name=f"{props.name} (CUDA {i})",
                    memory_total_mb=total_memory,
                    memory_available_mb=available_memory,
                    compute_capability=compute_cap,
                    supports_fp16=supports_fp16,
                    supports_bf16=supports_bf16,
                    supports_int8=supports_int8,
                    supports_int4=supports_int4,
                    max_batch_size=max_batch_size,
                    recommended_dtype=recommended_dtype,
                    performance_score=performance_score
                )
                
                cuda_devices.append(device_cap)
                logger.info(f"CUDA Device {i}: {props.name}, {total_memory}MB, Compute {compute_cap}")
        
        except Exception as e:
            logger.warning(f"Error detecting CUDA devices: {e}")
        
        return cuda_devices

    def _detect_mps_devices(self) -> List[DeviceCapabilities]:
        """Detect Apple Silicon MPS devices."""
        mps_devices = []
        
        try:
            import torch
            
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                logger.info("MPS not available")
                return mps_devices
            
            # Get system memory as proxy for unified memory
            system_memory = psutil.virtual_memory().total // (1024 * 1024)
            available_memory = psutil.virtual_memory().available // (1024 * 1024)
            
            # Apple Silicon typically uses unified memory
            # Estimate GPU portion (usually 60-80% can be used for ML)
            gpu_memory_estimate = int(system_memory * 0.7)
            gpu_available_estimate = int(available_memory * 0.7)
            
            # Determine Apple Silicon generation (rough estimation)
            # This is a simplified approach - in practice, you'd use more sophisticated detection
            performance_score = gpu_memory_estimate * 0.3 + 5000  # Base score for Apple Silicon
            
            device_cap = DeviceCapabilities(
                device_type=DeviceType.MPS,
                device_name="Apple Silicon GPU (MPS)",
                memory_total_mb=gpu_memory_estimate,
                memory_available_mb=gpu_available_estimate,
                compute_capability=(8, 0),  # Assume modern capabilities
                supports_fp16=True,
                supports_bf16=True,  # Apple Silicon supports bfloat16
                supports_int8=True,
                supports_int4=False,  # Limited int4 support
                max_batch_size=max(1, min(16, gpu_memory_estimate // 1000)),
                recommended_dtype="float16",
                performance_score=performance_score
            )
            
            mps_devices.append(device_cap)
            logger.info(f"MPS Device: Apple Silicon GPU, ~{gpu_memory_estimate}MB unified memory")
        
        except Exception as e:
            logger.warning(f"Error detecting MPS devices: {e}")
        
        return mps_devices
    
    def _detect_cpu_capabilities(self) -> DeviceCapabilities:
        """Detect CPU capabilities."""
        # Get CPU information
        cpu_count = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory()
        total_memory = memory.total // (1024 * 1024)
        available_memory = memory.available // (1024 * 1024)
        
        # CPU performance score (much lower than GPU)
        performance_score = cpu_count * 50 + total_memory * 0.1
        
        # Estimate max batch size for CPU (conservative)
        max_batch_size = max(1, min(4, available_memory // 2000))
        
        return DeviceCapabilities(
            device_type=DeviceType.CPU,
            device_name=f"CPU ({cpu_count} cores)",
            memory_total_mb=total_memory,
            memory_available_mb=available_memory,
            compute_capability=(0, 0),
            supports_fp16=False,  # CPU typically uses float32
            supports_bf16=False,
            supports_int8=True,   # CPU can handle int8
            supports_int4=True,   # CPU can handle int4
            max_batch_size=max_batch_size,
            recommended_dtype="float32",
            performance_score=performance_score
        )
    
    def get_best_device(self, memory_requirement_mb: int = 0) -> Optional[DeviceCapabilities]:
        """
        Get the best available device that meets memory requirements.
        
        Args:
            memory_requirement_mb: Minimum memory requirement in MB
            
        Returns:
            Best suitable device or None if no device meets requirements
        """
        devices = self.detect_devices()
        
        for device in devices:
            if device.memory_available_mb >= memory_requirement_mb:
                return device
        
        # If no device meets requirements, return the best available
        return devices[0] if devices else None


class MemoryManager:
    """
    Manages GPU memory allocation and optimization.
    
    This class provides methods for monitoring memory usage, optimizing
    memory allocation, and implementing memory-efficient strategies.
    """
    
    def __init__(self, device_capabilities: DeviceCapabilities):
        self.device_capabilities = device_capabilities
        self._torch_available = self._check_torch_availability()
    
    def _check_torch_availability(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def get_memory_info(self) -> Dict[str, int]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory information in MB
        """
        memory_info = {
            "total_mb": self.device_capabilities.memory_total_mb,
            "available_mb": self.device_capabilities.memory_available_mb,
            "used_mb": 0,
            "cached_mb": 0,
            "reserved_mb": 0
        }
        
        if not self._torch_available:
            return memory_info
        
        try:
            import torch
            
            if self.device_capabilities.device_type == DeviceType.CUDA and torch.cuda.is_available():
                # Get CUDA memory info
                allocated = torch.cuda.memory_allocated() // (1024 * 1024)
                cached = torch.cuda.memory_reserved() // (1024 * 1024)
                
                memory_info.update({
                    "used_mb": allocated,
                    "cached_mb": cached,
                    "reserved_mb": cached,
                    "available_mb": self.device_capabilities.memory_total_mb - cached
                })
            
            elif self.device_capabilities.device_type == DeviceType.MPS:
                # For MPS, use system memory as approximation
                system_memory = psutil.virtual_memory()
                used_system = (system_memory.total - system_memory.available) // (1024 * 1024)
                
                # Estimate MPS usage (rough approximation)
                estimated_mps_usage = min(used_system // 2, self.device_capabilities.memory_total_mb)
                
                memory_info.update({
                    "used_mb": estimated_mps_usage,
                    "available_mb": self.device_capabilities.memory_total_mb - estimated_mps_usage
                })
            
            else:  # CPU
                system_memory = psutil.virtual_memory()
                memory_info.update({
                    "used_mb": (system_memory.total - system_memory.available) // (1024 * 1024),
                    "available_mb": system_memory.available // (1024 * 1024)
                })
        
        except Exception as e:
            logger.warning(f"Error getting memory info: {e}")
        
        return memory_info

    def estimate_model_memory_usage(self, model_name: str, model_config: Dict[str, Any]) -> int:
        """
        Estimate memory usage for a specific model configuration.
        
        Args:
            model_name: Name of the model
            model_config: Model configuration parameters
            
        Returns:
            Estimated memory usage in MB
        """
        # Base memory estimates for common model types (rough approximations)
        base_estimates = {
            "whisper-tiny": 150,
            "whisper-small": 300,
            "whisper-medium": 800,
            "whisper-large": 1500,
            "qwen-1.5b": 1500,
            "smollm": 1200,
        }
        
        # Find matching base estimate
        base_memory = 500  # Default estimate
        for key, memory in base_estimates.items():
            if key in model_name.lower():
                base_memory = memory
                break
        
        # Adjust based on configuration
        dtype_multipliers = {
            "float32": 1.0,
            "float16": 0.5,
            "bfloat16": 0.5,
            "int8": 0.25,
            "int4": 0.125
        }
        
        dtype = model_config.get("torch_dtype", "float32")
        dtype_multiplier = dtype_multipliers.get(dtype, 1.0)
        
        # Account for quantization
        if model_config.get("load_in_8bit", False):
            dtype_multiplier = min(dtype_multiplier, 0.25)
        elif model_config.get("load_in_4bit", False):
            dtype_multiplier = min(dtype_multiplier, 0.125)
        
        # Account for batch size
        batch_size = model_config.get("batch_size", 1)
        batch_multiplier = 1.0 + (batch_size - 1) * 0.3  # Non-linear scaling
        
        # Calculate final estimate with overhead
        estimated_memory = int(base_memory * dtype_multiplier * batch_multiplier * 1.2)  # 20% overhead
        
        return estimated_memory
    
    def optimize_memory_allocation(self, required_memory_mb: int) -> Dict[str, Any]:
        """
        Optimize memory allocation strategy based on requirements.
        
        Args:
            required_memory_mb: Required memory in MB
            
        Returns:
            Dictionary with optimization recommendations
        """
        memory_info = self.get_memory_info()
        available_memory = memory_info["available_mb"]
        
        optimization = {
            "can_fit": available_memory >= required_memory_mb,
            "memory_pressure": required_memory_mb / available_memory if available_memory > 0 else float('inf'),
            "recommendations": []
        }
        
        if optimization["memory_pressure"] > 1.0:
            # Memory pressure - need optimizations
            optimization["recommendations"].extend([
                "enable_quantization",
                "reduce_batch_size",
                "enable_gradient_checkpointing"
            ])
            
            if optimization["memory_pressure"] > 1.5:
                optimization["recommendations"].extend([
                    "use_cpu_offload",
                    "enable_model_sharding"
                ])
        
        elif optimization["memory_pressure"] > 0.8:
            # Moderate memory usage
            optimization["recommendations"].append("monitor_memory_usage")
        
        return optimization
    
    def clear_cache(self):
        """Clear GPU memory cache if available."""
        if not self._torch_available:
            return
        
        try:
            import torch
            
            if self.device_capabilities.device_type == DeviceType.CUDA and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
            
            elif self.device_capabilities.device_type == DeviceType.MPS:
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                    logger.info("MPS cache cleared")
        
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")


class ModelOptimizer:
    """
    Optimizes model configuration based on device capabilities.
    
    This class provides methods to generate optimal model configurations
    for different hardware setups and memory constraints.
    """
    
    def __init__(self, device_capabilities: DeviceCapabilities, memory_manager: MemoryManager):
        self.device_capabilities = device_capabilities
        self.memory_manager = memory_manager
    
    def optimize_for_device(self, model_name: str, base_config: Dict[str, Any]) -> ModelOptimizationConfig:
        """
        Generate optimized model configuration for the current device.
        
        Args:
            model_name: Name of the model to optimize
            base_config: Base model configuration
            
        Returns:
            Optimized ModelOptimizationConfig
        """
        # Start with base configuration
        config = ModelOptimizationConfig(
            device_map="auto",
            torch_dtype=self.device_capabilities.recommended_dtype,
            batch_size=min(base_config.get("batch_size", 1), self.device_capabilities.max_batch_size)
        )
        
        # Device-specific optimizations
        if self.device_capabilities.device_type == DeviceType.CUDA:
            config = self._optimize_for_cuda(model_name, base_config, config)
        elif self.device_capabilities.device_type == DeviceType.MPS:
            config = self._optimize_for_mps(model_name, base_config, config)
        else:  # CPU
            config = self._optimize_for_cpu(model_name, base_config, config)
        
        # Memory-based optimizations
        estimated_memory = self.memory_manager.estimate_model_memory_usage(model_name, config.__dict__)
        memory_optimization = self.memory_manager.optimize_memory_allocation(estimated_memory)
        
        if not memory_optimization["can_fit"]:
            config = self._apply_memory_optimizations(config, memory_optimization["recommendations"])
        
        return config
    
    def _optimize_for_cuda(self, model_name: str, base_config: Dict[str, Any], config: ModelOptimizationConfig) -> ModelOptimizationConfig:
        """Optimize configuration for CUDA devices."""
        # Use GPU device mapping
        config.device_map = "auto"
        
        # Enable quantization based on memory
        if self.device_capabilities.memory_total_mb < 4000:
            config.load_in_4bit = True
            config.torch_dtype = "float16"
        elif self.device_capabilities.memory_total_mb < 8000:
            config.load_in_8bit = True
            config.torch_dtype = "float16"
        else:
            # High-memory GPU - use best precision
            if self.device_capabilities.supports_bf16:
                config.torch_dtype = "bfloat16"
            elif self.device_capabilities.supports_fp16:
                config.torch_dtype = "float16"
        
        # Enable flash attention for compatible GPUs
        if (self.device_capabilities.compute_capability >= (8, 0) and 
            self.device_capabilities.memory_total_mb >= 6000):
            config.use_flash_attention = True
        
        # Set memory limits
        memory_fraction = base_config.get("memory_fraction", 0.8)
        max_memory_mb = int(self.device_capabilities.memory_total_mb * memory_fraction)
        config.max_memory = {"0": f"{max_memory_mb}MB"}
        
        return config

    def _optimize_for_mps(self, model_name: str, base_config: Dict[str, Any], config: ModelOptimizationConfig) -> ModelOptimizationConfig:
        """Optimize configuration for Apple Silicon MPS."""
        config.device_map = "mps"
        config.torch_dtype = "float16"  # MPS works well with float16
        
        # Conservative batch size for MPS
        config.batch_size = min(config.batch_size, 4)
        
        # Enable low CPU memory usage
        config.low_cpu_mem_usage = True
        
        # MPS doesn't support all quantization methods
        config.load_in_8bit = False
        config.load_in_4bit = False
        
        return config
    
    def _optimize_for_cpu(self, model_name: str, base_config: Dict[str, Any], config: ModelOptimizationConfig) -> ModelOptimizationConfig:
        """Optimize configuration for CPU."""
        config.device_map = "cpu"
        config.torch_dtype = "float32"  # CPU typically uses float32
        
        # Very conservative batch size for CPU
        config.batch_size = min(config.batch_size, 2)
        
        # Enable CPU optimizations
        config.low_cpu_mem_usage = True
        
        # CPU can benefit from quantization
        if self.device_capabilities.memory_total_mb < 16000:
            config.load_in_8bit = True
        
        return config
    
    def _apply_memory_optimizations(self, config: ModelOptimizationConfig, recommendations: List[str]) -> ModelOptimizationConfig:
        """Apply memory optimization recommendations."""
        for recommendation in recommendations:
            if recommendation == "enable_quantization":
                if not config.load_in_8bit and not config.load_in_4bit:
                    config.load_in_8bit = True
            
            elif recommendation == "reduce_batch_size":
                config.batch_size = max(1, config.batch_size // 2)
            
            elif recommendation == "enable_gradient_checkpointing":
                config.gradient_checkpointing = True
            
            elif recommendation == "use_cpu_offload":
                config.offload_folder = "./temp/model_offload"
            
            elif recommendation == "enable_model_sharding":
                config.device_map = "auto"
        
        return config


class PerformanceMonitor:
    """
    Monitors device performance and resource usage.
    
    This class provides methods to track device utilization, memory usage,
    and performance metrics during model inference.
    """
    
    def __init__(self, device_capabilities: DeviceCapabilities):
        self.device_capabilities = device_capabilities
        self._torch_available = self._check_torch_availability()
        self._metrics_history: List[PerformanceMetrics] = []
    
    def _check_torch_availability(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def collect_metrics(self) -> PerformanceMetrics:
        """
        Collect current performance metrics.
        
        Returns:
            Current PerformanceMetrics
        """
        metrics = PerformanceMetrics()
        
        try:
            if self.device_capabilities.device_type == DeviceType.CUDA and self._torch_available:
                metrics = self._collect_cuda_metrics()
            elif self.device_capabilities.device_type == DeviceType.MPS:
                metrics = self._collect_mps_metrics()
            else:  # CPU
                metrics = self._collect_cpu_metrics()
        
        except Exception as e:
            logger.warning(f"Error collecting performance metrics: {e}")
        
        self._metrics_history.append(metrics)
        
        # Keep only last 100 metrics to prevent memory growth
        if len(self._metrics_history) > 100:
            self._metrics_history = self._metrics_history[-100:]
        
        return metrics
    
    def _collect_cuda_metrics(self) -> PerformanceMetrics:
        """Collect CUDA-specific metrics."""
        import torch
        
        metrics = PerformanceMetrics()
        
        if torch.cuda.is_available():
            # Memory utilization
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            metrics.memory_utilization = (allocated_memory / total_memory) * 100
            
            # Try to get GPU utilization (requires nvidia-ml-py)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.device_utilization = utilization.gpu
                
                # Temperature
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics.temperature = temperature
                
                # Power usage
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                metrics.power_usage = power
                
            except ImportError:
                logger.debug("nvidia-ml-py not available, limited GPU metrics")
            except Exception as e:
                logger.debug(f"Error collecting detailed GPU metrics: {e}")
        
        return metrics

    def _collect_mps_metrics(self) -> PerformanceMetrics:
        """Collect MPS-specific metrics."""
        metrics = PerformanceMetrics()
        
        # For MPS, use system-level metrics as approximation
        memory = psutil.virtual_memory()
        metrics.memory_utilization = memory.percent
        
        # CPU utilization as proxy for MPS utilization
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.device_utilization = cpu_percent
        
        return metrics
    
    def _collect_cpu_metrics(self) -> PerformanceMetrics:
        """Collect CPU-specific metrics."""
        metrics = PerformanceMetrics()
        
        # CPU utilization
        metrics.device_utilization = psutil.cpu_percent(interval=0.1)
        
        # Memory utilization
        memory = psutil.virtual_memory()
        metrics.memory_utilization = memory.percent
        
        # CPU temperature (if available)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get first available temperature sensor
                for sensor_name, sensor_list in temps.items():
                    if sensor_list:
                        metrics.temperature = sensor_list[0].current
                        break
        except (AttributeError, OSError):
            pass  # Temperature sensors not available on all systems
        
        return metrics
    
    def get_average_metrics(self, window_size: int = 10) -> Optional[PerformanceMetrics]:
        """
        Get average metrics over a specified window.
        
        Args:
            window_size: Number of recent metrics to average
            
        Returns:
            Averaged PerformanceMetrics or None if insufficient data
        """
        if len(self._metrics_history) < window_size:
            return None
        
        recent_metrics = self._metrics_history[-window_size:]
        
        avg_metrics = PerformanceMetrics(
            device_utilization=sum(m.device_utilization for m in recent_metrics) / len(recent_metrics),
            memory_utilization=sum(m.memory_utilization for m in recent_metrics) / len(recent_metrics),
            temperature=sum(m.temperature for m in recent_metrics) / len(recent_metrics),
            power_usage=sum(m.power_usage for m in recent_metrics) / len(recent_metrics),
            inference_time_ms=sum(m.inference_time_ms for m in recent_metrics) / len(recent_metrics),
            throughput_tokens_per_sec=sum(m.throughput_tokens_per_sec for m in recent_metrics) / len(recent_metrics)
        )
        
        return avg_metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary with performance summary
        """
        if not self._metrics_history:
            return {"status": "no_data", "message": "No performance data collected yet"}
        
        current_metrics = self._metrics_history[-1]
        avg_metrics = self.get_average_metrics()
        
        summary = {
            "device_name": self.device_capabilities.device_name,
            "device_type": self.device_capabilities.device_type.value,
            "current": {
                "device_utilization": current_metrics.device_utilization,
                "memory_utilization": current_metrics.memory_utilization,
                "temperature": current_metrics.temperature,
                "power_usage": current_metrics.power_usage
            },
            "metrics_collected": len(self._metrics_history)
        }
        
        if avg_metrics:
            summary["average"] = {
                "device_utilization": avg_metrics.device_utilization,
                "memory_utilization": avg_metrics.memory_utilization,
                "temperature": avg_metrics.temperature,
                "power_usage": avg_metrics.power_usage
            }
        
        # Performance assessment
        if current_metrics.device_utilization > 90:
            summary["status"] = "high_load"
            summary["recommendation"] = "Device under high load, consider reducing batch size"
        elif current_metrics.memory_utilization > 90:
            summary["status"] = "memory_pressure"
            summary["recommendation"] = "High memory usage, consider enabling quantization"
        elif current_metrics.temperature > 80:
            summary["status"] = "thermal_throttling"
            summary["recommendation"] = "High temperature detected, check cooling"
        else:
            summary["status"] = "optimal"
            summary["recommendation"] = "Performance is within normal parameters"
        
        return summary


class DeviceManager:
    """
    Main Device Manager class that orchestrates all device management operations.
    
    This class provides the primary interface for device detection, configuration
    optimization, memory management, and performance monitoring.
    """
    
    def __init__(self, config: Configuration):
        self.config = config
        self.detector = DeviceDetector()
        self.selected_device: Optional[DeviceCapabilities] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.model_optimizer: Optional[ModelOptimizer] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
        # Initialize device selection
        self._initialize_device()
    
    def _initialize_device(self):
        """Initialize device selection and related managers."""
        try:
            # Detect available devices
            available_devices = self.detector.detect_devices()
            
            if not available_devices:
                raise ModelLoadingError("No computing devices detected")
            
            # Select device based on configuration
            if self.config.device == "auto":
                self.selected_device = available_devices[0]  # Best device
            else:
                # Find specific device type
                for device in available_devices:
                    if device.device_type.value == self.config.device:
                        self.selected_device = device
                        break
                
                if not self.selected_device:
                    logger.warning(f"Requested device '{self.config.device}' not available, using best available")
                    self.selected_device = available_devices[0]
            
            # Initialize managers
            self.memory_manager = MemoryManager(self.selected_device)
            self.model_optimizer = ModelOptimizer(self.selected_device, self.memory_manager)
            self.performance_monitor = PerformanceMonitor(self.selected_device)
            
            logger.info(f"Device Manager initialized with: {self.selected_device.device_name}")
        
        except Exception as e:
            logger.error(f"Error initializing Device Manager: {e}")
            raise ModelLoadingError(f"Device initialization failed: {str(e)}")
    
    def get_optimal_device(self) -> str:
        """
        Get the optimal device string for PyTorch.
        
        Returns:
            Device string (e.g., "cuda:0", "mps", "cpu")
        """
        if not self.selected_device:
            return "cpu"
        
        if self.selected_device.device_type == DeviceType.CUDA:
            return "cuda:0"
        elif self.selected_device.device_type == DeviceType.MPS:
            return "mps"
        else:
            return "cpu"

    def get_model_config(self, model_name: str, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get optimized model configuration for the selected device.
        
        Args:
            model_name: Name of the model to configure
            base_config: Base configuration parameters
            
        Returns:
            Optimized model configuration dictionary
        """
        if not self.model_optimizer:
            raise ModelLoadingError("Device Manager not properly initialized")
        
        base_config = base_config or {}
        
        # Add configuration parameters
        base_config.update({
            "batch_size": self.config.batch_size,
            "memory_fraction": self.config.memory_fraction,
            "torch_dtype": self.config.torch_dtype
        })
        
        # Get optimized configuration
        optimized_config = self.model_optimizer.optimize_for_device(model_name, base_config)
        
        # Convert to dictionary format
        config_dict = {
            "device_map": optimized_config.device_map,
            "torch_dtype": optimized_config.torch_dtype,
            "low_cpu_mem_usage": optimized_config.low_cpu_mem_usage,
            "batch_size": optimized_config.batch_size,
        }
        
        # Add quantization settings
        if optimized_config.load_in_8bit:
            config_dict["load_in_8bit"] = True
        if optimized_config.load_in_4bit:
            config_dict["load_in_4bit"] = True
        
        # Add advanced settings
        if optimized_config.use_flash_attention:
            config_dict["use_flash_attention"] = True
        if optimized_config.max_memory:
            config_dict["max_memory"] = optimized_config.max_memory
        if optimized_config.offload_folder:
            config_dict["offload_folder"] = optimized_config.offload_folder
        
        return config_dict
    
    def check_gpu_availability(self) -> GPUStatus:
        """
        Check GPU availability and return status.
        
        Returns:
            GPUStatus object with current GPU information
        """
        if not self.selected_device:
            return GPUStatus(available=False)
        
        gpu_available = self.selected_device.device_type in [DeviceType.CUDA, DeviceType.MPS]
        
        return GPUStatus(
            available=gpu_available,
            device_name=self.selected_device.device_name,
            memory_total=self.selected_device.memory_total_mb,
            memory_free=self.selected_device.memory_available_mb,
            compute_capability=self.selected_device.compute_capability,
            recommended_settings=self.get_model_config("default")
        )
    
    def optimize_for_device(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize model configuration for the current device.
        
        Args:
            model_config: Base model configuration
            
        Returns:
            Optimized configuration dictionary
        """
        if not self.selected_device:
            return model_config
        
        model_name = model_config.get("model_name", "unknown")
        return self.get_model_config(model_name, model_config)

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get current memory information.
        
        Returns:
            Dictionary with memory information
        """
        if not self.memory_manager:
            return {"error": "Memory manager not initialized"}
        
        memory_info = self.memory_manager.get_memory_info()
        
        # Add device information
        memory_info.update({
            "device_name": self.selected_device.device_name if self.selected_device else "Unknown",
            "device_type": self.selected_device.device_type.value if self.selected_device else "unknown"
        })
        
        return memory_info
    
    def clear_memory_cache(self):
        """Clear device memory cache."""
        if self.memory_manager:
            self.memory_manager.clear_cache()
    
    def start_performance_monitoring(self) -> PerformanceMetrics:
        """
        Start performance monitoring and return initial metrics.
        
        Returns:
            Initial PerformanceMetrics
        """
        if not self.performance_monitor:
            raise ModelLoadingError("Performance monitor not initialized")
        
        return self.performance_monitor.collect_metrics()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Performance summary dictionary
        """
        if not self.performance_monitor:
            return {"error": "Performance monitor not initialized"}
        
        return self.performance_monitor.get_performance_summary()
    
    def get_device_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive device and system summary.
        
        Returns:
            Complete device summary
        """
        summary = {
            "selected_device": {
                "name": self.selected_device.device_name if self.selected_device else "None",
                "type": self.selected_device.device_type.value if self.selected_device else "unknown",
                "memory_total_mb": self.selected_device.memory_total_mb if self.selected_device else 0,
                "memory_available_mb": self.selected_device.memory_available_mb if self.selected_device else 0,
                "performance_score": self.selected_device.performance_score if self.selected_device else 0
            },
            "available_devices": []
        }
        
        # Add all available devices
        for device in self.detector.detect_devices():
            summary["available_devices"].append({
                "name": device.device_name,
                "type": device.device_type.value,
                "memory_total_mb": device.memory_total_mb,
                "performance_score": device.performance_score,
                "capabilities": {
                    "supports_fp16": device.supports_fp16,
                    "supports_bf16": device.supports_bf16,
                    "supports_int8": device.supports_int8,
                    "supports_int4": device.supports_int4
                }
            })
        
        # Add memory information
        if self.memory_manager:
            summary["memory_info"] = self.get_memory_info()
        
        # Add performance information
        if self.performance_monitor:
            summary["performance"] = self.get_performance_summary()
        
        return summary

    def validate_model_requirements(self, model_name: str, model_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate if the current device can handle the specified model.
        
        Args:
            model_name: Name of the model
            model_config: Model configuration
            
        Returns:
            Tuple of (can_handle, warnings_list)
        """
        if not self.memory_manager or not self.selected_device:
            return False, ["Device manager not properly initialized"]
        
        # Estimate memory requirements
        estimated_memory = self.memory_manager.estimate_model_memory_usage(model_name, model_config)
        
        # Check if device can handle the memory requirements
        memory_info = self.memory_manager.get_memory_info()
        available_memory = memory_info["available_mb"]
        
        warnings = []
        can_handle = True
        
        if estimated_memory > available_memory:
            can_handle = False
            warnings.append(
                f"Insufficient memory for {model_name}. "
                f"Required: ~{estimated_memory}MB, Available: {available_memory}MB"
            )
        elif estimated_memory > available_memory * 0.8:
            warnings.append(
                f"High memory usage expected for {model_name}. "
                f"Required: ~{estimated_memory}MB, Available: {available_memory}MB"
            )
        
        # Check device capabilities
        if model_config.get("torch_dtype") == "float16" and not self.selected_device.supports_fp16:
            warnings.append("Device may not support float16 precision efficiently")
        
        if model_config.get("torch_dtype") == "bfloat16" and not self.selected_device.supports_bf16:
            warnings.append("Device may not support bfloat16 precision")
        
        return can_handle, warnings

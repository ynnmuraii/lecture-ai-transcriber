"""
Unit tests for the Device Manager module.

These tests verify device detection and selection logic, GPU memory management
and fallback behavior, and model configuration optimization functionality.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, PropertyMock
from typing import List, Dict, Any

from backend.infrastructure.device_manager import (
    DeviceDetector, MemoryManager, ModelOptimizer, PerformanceMonitor,
    DeviceManager, DeviceType, DeviceCapabilities, ModelOptimizationConfig,
    PerformanceMetrics
)
from backend.core.models.data_models import Configuration, GPUStatus
from backend.core.models.errors import ModelLoadingError


class TestDeviceDetector:
    """Test the DeviceDetector class for device detection and selection logic."""
    
    def test_initialization(self):
        """Test DeviceDetector initialization."""
        detector = DeviceDetector()
        
        assert hasattr(detector, '_torch_available')
        assert hasattr(detector, '_detected_devices')
        assert hasattr(detector, '_detection_complete')
        assert detector._detection_complete is False
    
    @patch('backend.infrastructure.device_manager.DeviceDetector._check_torch_availability')
    def test_initialization_without_torch(self, mock_torch_check):
        """Test initialization when PyTorch is not available."""
        mock_torch_check.return_value = False
        
        detector = DeviceDetector()
        
        assert detector._torch_available is False
    
    @patch('backend.infrastructure.device_manager.psutil.cpu_count')
    @patch('backend.infrastructure.device_manager.psutil.virtual_memory')
    def test_detect_cpu_capabilities(self, mock_memory, mock_cpu_count):
        """Test CPU capabilities detection."""
        # Mock system information
        mock_cpu_count.return_value = 8
        mock_memory.return_value = MagicMock(
            total=16 * 1024 * 1024 * 1024,  # 16GB
            available=8 * 1024 * 1024 * 1024  # 8GB available
        )
        
        detector = DeviceDetector()
        cpu_device = detector._detect_cpu_capabilities()
        
        assert cpu_device.device_type == DeviceType.CPU
        assert cpu_device.device_name == "CPU (8 cores)"
        assert cpu_device.memory_total_mb == 16 * 1024
        assert cpu_device.memory_available_mb == 8 * 1024
        assert cpu_device.supports_fp16 is False
        assert cpu_device.supports_int8 is True
        assert cpu_device.recommended_dtype == "float32"
        assert cpu_device.performance_score > 0
    
    @patch('backend.infrastructure.device_manager.DeviceDetector._check_torch_availability')
    @patch('backend.infrastructure.device_manager.DeviceDetector._detect_cpu_capabilities')
    def test_detect_devices_no_torch(self, mock_cpu_detect, mock_torch_check):
        """Test device detection when PyTorch is not available."""
        mock_torch_check.return_value = False
        mock_cpu_device = DeviceCapabilities(
            device_type=DeviceType.CPU,
            device_name="CPU (4 cores)",
            memory_total_mb=8192,
            performance_score=100
        )
        mock_cpu_detect.return_value = mock_cpu_device
        
        detector = DeviceDetector()
        devices = detector.detect_devices()
        
        assert len(devices) == 1
        assert devices[0].device_type == DeviceType.CPU
        assert detector._detection_complete is True
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    @patch('backend.infrastructure.device_manager.DeviceDetector._check_torch_availability')
    def test_detect_cuda_devices(self, mock_torch_check, mock_props, mock_count, mock_available):
        """Test CUDA device detection."""
        mock_torch_check.return_value = True
        mock_available.return_value = True
        mock_count.return_value = 1
        
        # Mock GPU properties
        mock_gpu_props = MagicMock()
        mock_gpu_props.name = "NVIDIA RTX 4090"
        mock_gpu_props.total_memory = 24 * 1024 * 1024 * 1024  # 24GB
        mock_gpu_props.major = 8
        mock_gpu_props.minor = 9
        mock_gpu_props.multi_processor_count = 128
        mock_props.return_value = mock_gpu_props
        
        detector = DeviceDetector()
        cuda_devices = detector._detect_cuda_devices()
        
        assert len(cuda_devices) == 1
        device = cuda_devices[0]
        assert device.device_type == DeviceType.CUDA
        assert "RTX 4090" in device.device_name
        assert device.memory_total_mb == 24 * 1024
        assert device.compute_capability == (8, 9)
        assert device.supports_fp16 is True
        assert device.supports_bf16 is True
        assert device.recommended_dtype == "bfloat16"
    
    @patch('torch.backends.mps.is_available')
    @patch('backend.infrastructure.device_manager.psutil.virtual_memory')
    @patch('backend.infrastructure.device_manager.DeviceDetector._check_torch_availability')
    def test_detect_mps_devices(self, mock_torch_check, mock_memory, mock_mps_available):
        """Test Apple Silicon MPS device detection."""
        mock_torch_check.return_value = True
        mock_mps_available.return_value = True
        mock_memory.return_value = MagicMock(
            total=32 * 1024 * 1024 * 1024,  # 32GB unified memory
            available=16 * 1024 * 1024 * 1024  # 16GB available
        )
        
        detector = DeviceDetector()
        mps_devices = detector._detect_mps_devices()
        
        assert len(mps_devices) == 1
        device = mps_devices[0]
        assert device.device_type == DeviceType.MPS
        assert "Apple Silicon GPU" in device.device_name
        assert device.supports_fp16 is True
        assert device.supports_bf16 is True
        assert device.recommended_dtype == "float16"
    
    def test_get_best_device_with_memory_requirement(self):
        """Test getting best device with memory requirements."""
        detector = DeviceDetector()
        
        # Mock detected devices
        devices = [
            DeviceCapabilities(
                device_type=DeviceType.CUDA,
                device_name="GPU 1",
                memory_available_mb=8000,
                performance_score=1000
            ),
            DeviceCapabilities(
                device_type=DeviceType.CPU,
                device_name="CPU",
                memory_available_mb=16000,
                performance_score=100
            )
        ]
        detector._detected_devices = devices
        detector._detection_complete = True
        
        # Test with low memory requirement - should get GPU
        best_device = detector.get_best_device(memory_requirement_mb=4000)
        assert best_device.device_type == DeviceType.CUDA
        
        # Test with high memory requirement - should get CPU
        best_device = detector.get_best_device(memory_requirement_mb=12000)
        assert best_device.device_type == DeviceType.CPU
        
        # Test with impossible requirement - should get best available
        best_device = detector.get_best_device(memory_requirement_mb=50000)
        assert best_device.device_type == DeviceType.CUDA  # Highest performance score


class TestMemoryManager:
    """Test the MemoryManager class for GPU memory management and fallback behavior."""
    
    def test_initialization(self):
        """Test MemoryManager initialization."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name="Test GPU",
            memory_total_mb=8000
        )
        
        memory_manager = MemoryManager(device_caps)
        
        assert memory_manager.device_capabilities == device_caps
        assert hasattr(memory_manager, '_torch_available')
    
    def test_get_memory_info_cpu(self):
        """Test memory info retrieval for CPU."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CPU,
            device_name="CPU",
            memory_total_mb=16000,
            memory_available_mb=8000
        )
        
        with patch('backend.infrastructure.device_manager.psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = MagicMock(
                total=16 * 1024 * 1024 * 1024,
                available=8 * 1024 * 1024 * 1024
            )
            
            memory_manager = MemoryManager(device_caps)
            memory_info = memory_manager.get_memory_info()
            
            assert memory_info['total_mb'] == 16000
            assert 'used_mb' in memory_info
            assert 'available_mb' in memory_info
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_get_memory_info_cuda(self, mock_reserved, mock_allocated, mock_available):
        """Test memory info retrieval for CUDA."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name="CUDA GPU",
            memory_total_mb=8000
        )
        
        mock_available.return_value = True
        mock_allocated.return_value = 2 * 1024 * 1024 * 1024  # 2GB
        mock_reserved.return_value = 3 * 1024 * 1024 * 1024   # 3GB
        
        memory_manager = MemoryManager(device_caps)
        memory_manager._torch_available = True
        
        memory_info = memory_manager.get_memory_info()
        
        assert memory_info['used_mb'] == 2 * 1024
        assert memory_info['cached_mb'] == 3 * 1024
        assert memory_info['reserved_mb'] == 3 * 1024
    
    def test_estimate_model_memory_usage(self):
        """Test model memory usage estimation."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name="Test GPU",
            memory_total_mb=8000
        )
        
        memory_manager = MemoryManager(device_caps)
        
        # Test Whisper model estimation
        whisper_config = {
            "torch_dtype": "float16",
            "batch_size": 1
        }
        memory_usage = memory_manager.estimate_model_memory_usage("whisper-medium", whisper_config)
        
        assert memory_usage > 0
        assert isinstance(memory_usage, int)
        
        # Test with quantization
        quantized_config = {
            "torch_dtype": "float16",
            "load_in_8bit": True,
            "batch_size": 1
        }
        quantized_usage = memory_manager.estimate_model_memory_usage("whisper-medium", quantized_config)
        
        assert quantized_usage < memory_usage  # Quantized should use less memory
    
    def test_optimize_memory_allocation_sufficient_memory(self):
        """Test memory optimization when sufficient memory is available."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name="Test GPU",
            memory_total_mb=8000,
            memory_available_mb=6000
        )
        
        memory_manager = MemoryManager(device_caps)
        
        # Mock memory info
        with patch.object(memory_manager, 'get_memory_info') as mock_memory_info:
            mock_memory_info.return_value = {"available_mb": 6000}
            
            optimization = memory_manager.optimize_memory_allocation(required_memory_mb=2000)
            
            assert optimization["can_fit"] is True
            assert optimization["memory_pressure"] < 1.0
            assert len(optimization["recommendations"]) == 0 or "monitor_memory_usage" in optimization["recommendations"]
    
    def test_optimize_memory_allocation_insufficient_memory(self):
        """Test memory optimization when memory is insufficient."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name="Test GPU",
            memory_total_mb=4000,
            memory_available_mb=2000
        )
        
        memory_manager = MemoryManager(device_caps)
        
        # Mock memory info
        with patch.object(memory_manager, 'get_memory_info') as mock_memory_info:
            mock_memory_info.return_value = {"available_mb": 2000}
            
            optimization = memory_manager.optimize_memory_allocation(required_memory_mb=4000)
            
            assert optimization["can_fit"] is False
            assert optimization["memory_pressure"] > 1.0
            assert "enable_quantization" in optimization["recommendations"]
            assert "reduce_batch_size" in optimization["recommendations"]
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_clear_cache_cuda(self, mock_empty_cache, mock_available):
        """Test clearing CUDA cache."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name="CUDA GPU",
            memory_total_mb=8000
        )
        
        mock_available.return_value = True
        
        memory_manager = MemoryManager(device_caps)
        memory_manager._torch_available = True
        
        memory_manager.clear_cache()
        
        mock_empty_cache.assert_called_once()


class TestModelOptimizer:
    """Test the ModelOptimizer class for model configuration optimization."""
    
    def test_initialization(self):
        """Test ModelOptimizer initialization."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name="Test GPU",
            memory_total_mb=8000
        )
        memory_manager = MemoryManager(device_caps)
        
        optimizer = ModelOptimizer(device_caps, memory_manager)
        
        assert optimizer.device_capabilities == device_caps
        assert optimizer.memory_manager == memory_manager
    
    def test_optimize_for_cuda_high_memory(self):
        """Test optimization for high-memory CUDA device."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name="RTX 4090",
            memory_total_mb=24000,
            compute_capability=(8, 9),
            supports_bf16=True,
            supports_fp16=True,
            max_batch_size=8
        )
        memory_manager = MemoryManager(device_caps)
        optimizer = ModelOptimizer(device_caps, memory_manager)
        
        base_config = {"batch_size": 4, "memory_fraction": 0.8}
        
        # Mock memory estimation to return reasonable value
        with patch.object(memory_manager, 'estimate_model_memory_usage', return_value=2000):
            with patch.object(memory_manager, 'optimize_memory_allocation', 
                            return_value={"can_fit": True, "recommendations": []}):
                
                config = optimizer.optimize_for_device("whisper-large", base_config)
                
                assert config.device_map == "auto"
                assert config.torch_dtype == "bfloat16"  # Should use best precision
                assert config.use_flash_attention is True  # High-end GPU should enable flash attention
                assert config.batch_size <= device_caps.max_batch_size
    
    def test_optimize_for_cuda_low_memory(self):
        """Test optimization for low-memory CUDA device."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name="GTX 1060",
            memory_total_mb=3000,  # Low memory
            compute_capability=(6, 1),
            supports_fp16=True,
            supports_bf16=False,
            max_batch_size=2
        )
        memory_manager = MemoryManager(device_caps)
        optimizer = ModelOptimizer(device_caps, memory_manager)
        
        base_config = {"batch_size": 4, "memory_fraction": 0.8}
        
        # Mock memory estimation to return reasonable value
        with patch.object(memory_manager, 'estimate_model_memory_usage', return_value=1500):
            with patch.object(memory_manager, 'optimize_memory_allocation', 
                            return_value={"can_fit": True, "recommendations": []}):
                
                config = optimizer.optimize_for_device("whisper-medium", base_config)
                
                assert config.device_map == "auto"
                assert config.load_in_4bit is True  # Should enable aggressive quantization
                assert config.torch_dtype == "float16"
                assert config.use_flash_attention is False  # Low-end GPU shouldn't use flash attention
    
    def test_optimize_for_mps(self):
        """Test optimization for Apple Silicon MPS."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.MPS,
            device_name="Apple Silicon GPU",
            memory_total_mb=16000,
            supports_fp16=True,
            supports_bf16=True,
            max_batch_size=4
        )
        memory_manager = MemoryManager(device_caps)
        optimizer = ModelOptimizer(device_caps, memory_manager)
        
        base_config = {"batch_size": 8}
        
        # Mock memory estimation
        with patch.object(memory_manager, 'estimate_model_memory_usage', return_value=2000):
            with patch.object(memory_manager, 'optimize_memory_allocation', 
                            return_value={"can_fit": True, "recommendations": []}):
                
                config = optimizer.optimize_for_device("whisper-base", base_config)
                
                assert config.device_map == "mps"
                assert config.torch_dtype == "float16"
                assert config.batch_size <= 4  # Conservative batch size for MPS
                assert config.low_cpu_mem_usage is True
                assert config.load_in_8bit is False  # MPS doesn't support quantization
    
    def test_optimize_for_cpu(self):
        """Test optimization for CPU."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CPU,
            device_name="CPU (8 cores)",
            memory_total_mb=16000,
            max_batch_size=2
        )
        memory_manager = MemoryManager(device_caps)
        optimizer = ModelOptimizer(device_caps, memory_manager)
        
        base_config = {"batch_size": 4}
        
        # Mock memory estimation
        with patch.object(memory_manager, 'estimate_model_memory_usage', return_value=1000):
            with patch.object(memory_manager, 'optimize_memory_allocation', 
                            return_value={"can_fit": True, "recommendations": []}):
                
                config = optimizer.optimize_for_device("whisper-small", base_config)
                
                assert config.device_map == "cpu"
                assert config.torch_dtype == "float32"
                assert config.batch_size <= 2  # Very conservative for CPU
                assert config.low_cpu_mem_usage is True
    
    def test_apply_memory_optimizations(self):
        """Test application of memory optimization recommendations."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name="Test GPU",
            memory_total_mb=8000
        )
        memory_manager = MemoryManager(device_caps)
        optimizer = ModelOptimizer(device_caps, memory_manager)
        
        config = ModelOptimizationConfig(
            device_map="auto",
            torch_dtype="float16",
            batch_size=4
        )
        
        recommendations = [
            "enable_quantization",
            "reduce_batch_size",
            "enable_gradient_checkpointing",
            "use_cpu_offload"
        ]
        
        optimized_config = optimizer._apply_memory_optimizations(config, recommendations)
        
        assert optimized_config.load_in_8bit is True
        assert optimized_config.batch_size == 2  # Reduced from 4
        assert optimized_config.gradient_checkpointing is True
        assert optimized_config.offload_folder == "./temp/model_offload"


class TestPerformanceMonitor:
    """Test the PerformanceMonitor class."""
    
    def test_initialization(self):
        """Test PerformanceMonitor initialization."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name="Test GPU",
            memory_total_mb=8000
        )
        
        monitor = PerformanceMonitor(device_caps)
        
        assert monitor.device_capabilities == device_caps
        assert len(monitor._metrics_history) == 0
    
    @patch('backend.infrastructure.device_manager.psutil.cpu_percent')
    @patch('backend.infrastructure.device_manager.psutil.virtual_memory')
    def test_collect_cpu_metrics(self, mock_memory, mock_cpu_percent):
        """Test collecting CPU performance metrics."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CPU,
            device_name="CPU",
            memory_total_mb=16000
        )
        
        mock_cpu_percent.return_value = 45.0
        mock_memory.return_value = MagicMock(percent=60.0)
        
        monitor = PerformanceMonitor(device_caps)
        metrics = monitor.collect_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.device_utilization == 45.0
        assert metrics.memory_utilization == 60.0
        assert len(monitor._metrics_history) == 1
    
    def test_get_average_metrics(self):
        """Test getting average metrics over a window."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CPU,
            device_name="CPU",
            memory_total_mb=16000
        )
        
        monitor = PerformanceMonitor(device_caps)
        
        # Add some test metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                device_utilization=float(i * 10),
                memory_utilization=float(i * 5),
                temperature=float(30 + i)
            )
            monitor._metrics_history.append(metrics)
        
        avg_metrics = monitor.get_average_metrics(window_size=3)
        
        assert avg_metrics is not None
        assert avg_metrics.device_utilization == 30.0  # Average of [20, 30, 40]
        assert avg_metrics.memory_utilization == 15.0  # Average of [10, 15, 20]
        assert avg_metrics.temperature == 33.0  # Average of [32, 33, 34]
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        device_caps = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name="Test GPU",
            memory_total_mb=8000
        )
        
        monitor = PerformanceMonitor(device_caps)
        
        # Test with no data
        summary = monitor.get_performance_summary()
        assert summary["status"] == "no_data"
        
        # Add test metrics
        metrics = PerformanceMetrics(
            device_utilization=95.0,  # High load
            memory_utilization=50.0,
            temperature=70.0
        )
        monitor._metrics_history.append(metrics)
        
        summary = monitor.get_performance_summary()
        
        assert summary["device_name"] == "Test GPU"
        assert summary["device_type"] == "cuda"
        assert summary["status"] == "high_load"
        assert "high load" in summary["recommendation"].lower()


class TestDeviceManager:
    """Test the main DeviceManager class."""
    
    def test_initialization_success(self):
        """Test successful DeviceManager initialization."""
        config = Configuration()
        
        with patch('backend.infrastructure.device_manager.DeviceDetector.detect_devices') as mock_detect:
            mock_device = DeviceCapabilities(
                device_type=DeviceType.CUDA,
                device_name="Test GPU",
                memory_total_mb=8000,
                performance_score=1000
            )
            mock_detect.return_value = [mock_device]
            
            manager = DeviceManager(config)
            
            assert manager.selected_device == mock_device
            assert manager.memory_manager is not None
            assert manager.model_optimizer is not None
            assert manager.performance_monitor is not None
    
    def test_initialization_no_devices(self):
        """Test DeviceManager initialization when no devices are detected."""
        config = Configuration()
        
        with patch('backend.infrastructure.device_manager.DeviceDetector.detect_devices') as mock_detect:
            mock_detect.return_value = []
            
            with pytest.raises(ModelLoadingError) as exc_info:
                DeviceManager(config)
            
            assert "No computing devices detected" in str(exc_info.value)
    
    def test_get_optimal_device_cuda(self):
        """Test getting optimal device string for CUDA."""
        config = Configuration()
        
        with patch('backend.infrastructure.device_manager.DeviceDetector.detect_devices') as mock_detect:
            mock_device = DeviceCapabilities(
                device_type=DeviceType.CUDA,
                device_name="CUDA GPU",
                memory_total_mb=8000
            )
            mock_detect.return_value = [mock_device]
            
            manager = DeviceManager(config)
            device_string = manager.get_optimal_device()
            
            assert device_string == "cuda:0"
    
    def test_get_optimal_device_mps(self):
        """Test getting optimal device string for MPS."""
        config = Configuration()
        
        with patch('backend.infrastructure.device_manager.DeviceDetector.detect_devices') as mock_detect:
            mock_device = DeviceCapabilities(
                device_type=DeviceType.MPS,
                device_name="Apple Silicon GPU",
                memory_total_mb=16000
            )
            mock_detect.return_value = [mock_device]
            
            manager = DeviceManager(config)
            device_string = manager.get_optimal_device()
            
            assert device_string == "mps"
    
    def test_get_optimal_device_cpu(self):
        """Test getting optimal device string for CPU."""
        config = Configuration()
        
        with patch('backend.infrastructure.device_manager.DeviceDetector.detect_devices') as mock_detect:
            mock_device = DeviceCapabilities(
                device_type=DeviceType.CPU,
                device_name="CPU",
                memory_total_mb=16000
            )
            mock_detect.return_value = [mock_device]
            
            manager = DeviceManager(config)
            device_string = manager.get_optimal_device()
            
            assert device_string == "cpu"
    
    def test_get_model_config(self):
        """Test getting optimized model configuration."""
        config = Configuration(batch_size=2, memory_fraction=0.8)
        
        with patch('backend.infrastructure.device_manager.DeviceDetector.detect_devices') as mock_detect:
            mock_device = DeviceCapabilities(
                device_type=DeviceType.CUDA,
                device_name="Test GPU",
                memory_total_mb=8000,
                recommended_dtype="float16"
            )
            mock_detect.return_value = [mock_device]
            
            manager = DeviceManager(config)
            
            # Mock the optimizer
            mock_config = ModelOptimizationConfig(
                device_map="auto",
                torch_dtype="float16",
                batch_size=2,
                load_in_8bit=True
            )
            
            with patch.object(manager.model_optimizer, 'optimize_for_device', return_value=mock_config):
                model_config = manager.get_model_config("whisper-base")
                
                assert model_config["device_map"] == "auto"
                assert model_config["torch_dtype"] == "float16"
                assert model_config["batch_size"] == 2
                assert model_config["load_in_8bit"] is True
    
    def test_check_gpu_availability_gpu_present(self):
        """Test GPU availability check when GPU is present."""
        config = Configuration()
        
        with patch('backend.infrastructure.device_manager.DeviceDetector.detect_devices') as mock_detect:
            mock_device = DeviceCapabilities(
                device_type=DeviceType.CUDA,
                device_name="RTX 4090",
                memory_total_mb=24000,
                memory_available_mb=20000,
                compute_capability=(8, 9)
            )
            mock_detect.return_value = [mock_device]
            
            manager = DeviceManager(config)
            
            with patch.object(manager, 'get_model_config', return_value={"test": "config"}):
                gpu_status = manager.check_gpu_availability()
                
                assert gpu_status.available is True
                assert gpu_status.device_name == "RTX 4090"
                assert gpu_status.memory_total == 24000
                assert gpu_status.memory_free == 20000
                assert gpu_status.compute_capability == (8, 9)
    
    def test_check_gpu_availability_no_gpu(self):
        """Test GPU availability check when no GPU is present."""
        config = Configuration()
        
        with patch('backend.infrastructure.device_manager.DeviceDetector.detect_devices') as mock_detect:
            mock_device = DeviceCapabilities(
                device_type=DeviceType.CPU,
                device_name="CPU",
                memory_total_mb=16000
            )
            mock_detect.return_value = [mock_device]
            
            manager = DeviceManager(config)
            gpu_status = manager.check_gpu_availability()
            
            assert gpu_status.available is False
    
    def test_validate_model_requirements_sufficient_resources(self):
        """Test model requirements validation with sufficient resources."""
        config = Configuration()
        
        with patch('backend.infrastructure.device_manager.DeviceDetector.detect_devices') as mock_detect:
            mock_device = DeviceCapabilities(
                device_type=DeviceType.CUDA,
                device_name="Test GPU",
                memory_total_mb=8000,
                supports_fp16=True
            )
            mock_detect.return_value = [mock_device]
            
            manager = DeviceManager(config)
            
            # Mock memory manager methods
            with patch.object(manager.memory_manager, 'estimate_model_memory_usage', return_value=2000):
                with patch.object(manager.memory_manager, 'get_memory_info', 
                                return_value={"available_mb": 6000}):
                    
                    can_handle, warnings = manager.validate_model_requirements(
                        "whisper-base", {"torch_dtype": "float16"}
                    )
                    
                    assert can_handle is True
                    assert len(warnings) == 0 or all("high memory usage" not in w.lower() for w in warnings)
    
    def test_validate_model_requirements_insufficient_resources(self):
        """Test model requirements validation with insufficient resources."""
        config = Configuration()
        
        with patch('backend.infrastructure.device_manager.DeviceDetector.detect_devices') as mock_detect:
            mock_device = DeviceCapabilities(
                device_type=DeviceType.CUDA,
                device_name="Test GPU",
                memory_total_mb=4000,
                supports_fp16=True
            )
            mock_detect.return_value = [mock_device]
            
            manager = DeviceManager(config)
            
            # Mock memory manager methods
            with patch.object(manager.memory_manager, 'estimate_model_memory_usage', return_value=6000):
                with patch.object(manager.memory_manager, 'get_memory_info', 
                                return_value={"available_mb": 2000}):
                    
                    can_handle, warnings = manager.validate_model_requirements(
                        "whisper-large", {"torch_dtype": "float16"}
                    )
                    
                    assert can_handle is False
                    assert len(warnings) > 0
                    assert any("insufficient memory" in w.lower() for w in warnings)
    
    def test_get_device_summary(self):
        """Test getting comprehensive device summary."""
        config = Configuration()
        
        with patch('backend.infrastructure.device_manager.DeviceDetector.detect_devices') as mock_detect:
            mock_device = DeviceCapabilities(
                device_type=DeviceType.CUDA,
                device_name="Test GPU",
                memory_total_mb=8000,
                memory_available_mb=6000,
                performance_score=1000,
                supports_fp16=True,
                supports_bf16=True
            )
            mock_detect.return_value = [mock_device]
            
            manager = DeviceManager(config)
            
            # Mock manager methods
            with patch.object(manager, 'get_memory_info', return_value={"test": "memory"}):
                with patch.object(manager, 'get_performance_summary', return_value={"test": "performance"}):
                    
                    summary = manager.get_device_summary()
                    
                    assert summary["selected_device"]["name"] == "Test GPU"
                    assert summary["selected_device"]["type"] == "cuda"
                    assert summary["selected_device"]["memory_total_mb"] == 8000
                    assert len(summary["available_devices"]) == 1
                    assert summary["memory_info"]["test"] == "memory"
                    assert summary["performance"]["test"] == "performance"


if __name__ == '__main__':
    pytest.main([__file__])

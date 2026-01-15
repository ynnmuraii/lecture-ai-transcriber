"""
Unit tests for the Configuration Manager module.

These tests verify the core functionality of configuration loading,
validation, resource checking, and model fallback management.
"""

import pytest
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_manager import (
    ConfigurationManager, ResourceChecker, ModelFallbackManager,
    LLMProviderConfig, load_config
)
from models import ConfigurationError, Configuration, GPUStatus


class TestConfigurationManager:
    """Test the main ConfigurationManager class."""
    
    def test_load_valid_configuration(self):
        """Test loading a valid configuration file."""
        manager = ConfigurationManager('config/config.yaml')
        config = manager.load_configuration()
        
        assert isinstance(config, Configuration)
        assert config.whisper_model is not None
        assert config.text_generation_model is not None
        assert len(config.filler_words) > 0
        assert len(config.formula_patterns) > 0
    
    def test_load_nonexistent_file(self):
        """Test error handling for non-existent configuration file."""
        manager = ConfigurationManager('nonexistent.yaml')
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_configuration()
        
        assert "Configuration file not found" in str(exc_info.value)
        assert "nonexistent.yaml" in str(exc_info.value)
    
    def test_load_invalid_yaml(self):
        """Test error handling for invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: [unclosed')
            temp_file = f.name
        
        try:
            manager = ConfigurationManager(temp_file)
            
            with pytest.raises(ConfigurationError) as exc_info:
                manager.load_configuration()
            
            assert "Invalid YAML" in str(exc_info.value)
        finally:
            os.unlink(temp_file)
    
    def test_get_whisper_config(self):
        """Test extraction of Whisper configuration."""
        manager = ConfigurationManager('config/config.yaml')
        config = manager.load_configuration()
        
        whisper_config = manager.get_whisper_config()
        
        assert 'model_name' in whisper_config
        assert 'device' in whisper_config
        assert 'torch_dtype' in whisper_config
        assert whisper_config['model_name'] == config.whisper_model
    
    def test_get_llm_config(self):
        """Test extraction of LLM configuration."""
        manager = ConfigurationManager('config/config.yaml')
        config = manager.load_configuration()
        
        llm_config = manager.get_llm_config()
        
        assert 'model_name' in llm_config
        assert llm_config['model_name'] == config.text_generation_model
    
    def test_get_processing_config(self):
        """Test extraction of processing configuration."""
        manager = ConfigurationManager('config/config.yaml')
        config = manager.load_configuration()
        
        processing_config = manager.get_processing_config()
        
        assert 'filler_words' in processing_config
        assert 'cleaning_intensity' in processing_config
        assert 'formula_patterns' in processing_config
        assert len(processing_config['filler_words']) > 0
    
    def test_validate_environment(self):
        """Test environment validation."""
        manager = ConfigurationManager('config/config.yaml')
        manager.load_configuration()
        
        env_status = manager.validate_environment()
        
        assert hasattr(env_status, 'venv_active')
        assert hasattr(env_status, 'dependencies_installed')
        assert hasattr(env_status, 'python_version')
        assert env_status.python_version != ""
    
    def test_get_resource_summary(self):
        """Test resource summary generation."""
        manager = ConfigurationManager('config/config.yaml')
        manager.load_configuration()
        
        resource_summary = manager.get_resource_summary()
        
        assert 'system_memory_total_mb' in resource_summary
        assert 'cpu_count' in resource_summary
        assert 'gpu_available' in resource_summary
        assert resource_summary['system_memory_total_mb'] > 0


class TestResourceChecker:
    """Test the ResourceChecker class."""
    
    def test_initialization(self):
        """Test ResourceChecker initialization."""
        checker = ResourceChecker()
        
        assert checker.system_memory_mb > 0
        assert isinstance(checker.gpu_status, GPUStatus)
    
    def test_check_model_requirements_sufficient_memory(self):
        """Test model requirements checking with sufficient memory."""
        checker = ResourceChecker()
        model_config = {'memory_requirement_mb': 100}  # Very small requirement
        
        can_run, warnings = checker.check_model_requirements('test-model', model_config)
        
        assert can_run is True
        # May have warnings about low memory, but should be able to run
    
    def test_check_model_requirements_insufficient_memory(self):
        """Test model requirements checking with insufficient memory."""
        checker = ResourceChecker()
        # Set unrealistically high memory requirement
        model_config = {'memory_requirement_mb': 999999999}
        
        can_run, warnings = checker.check_model_requirements('test-model', model_config)
        
        assert can_run is False
        assert len(warnings) > 0
        assert "Insufficient system memory" in warnings[0]
    
    def test_get_resource_summary(self):
        """Test resource summary generation."""
        checker = ResourceChecker()
        
        summary = checker.get_resource_summary()
        
        assert 'system_memory_total_mb' in summary
        assert 'system_memory_available_mb' in summary
        assert 'cpu_count' in summary
        assert 'gpu_available' in summary
        assert summary['system_memory_total_mb'] > 0


class TestModelFallbackManager:
    """Test the ModelFallbackManager class."""
    
    def test_initialization(self):
        """Test ModelFallbackManager initialization."""
        resource_checker = ResourceChecker()
        fallback_manager = ModelFallbackManager(resource_checker)
        
        assert fallback_manager.resource_checker is resource_checker
        assert len(fallback_manager.WHISPER_MODEL_HIERARCHY) > 0
    
    def test_find_suitable_whisper_model_can_run_requested(self):
        """Test finding suitable model when requested model can run."""
        resource_checker = ResourceChecker()
        fallback_manager = ModelFallbackManager(resource_checker)
        
        model_configs = {
            'openai/whisper-tiny': {'memory_requirement_mb': 100}  # Very small
        }
        
        selected_model, warnings = fallback_manager.find_suitable_whisper_model(
            'openai/whisper-tiny', model_configs
        )
        
        assert selected_model == 'openai/whisper-tiny'
    
    def test_find_suitable_whisper_model_fallback(self):
        """Test finding suitable model with fallback."""
        resource_checker = ResourceChecker()
        fallback_manager = ModelFallbackManager(resource_checker)
        
        model_configs = {
            'openai/whisper-large-v3': {'memory_requirement_mb': 999999999},  # Too large
            'openai/whisper-tiny': {'memory_requirement_mb': 100}  # Small enough
        }
        
        selected_model, warnings = fallback_manager.find_suitable_whisper_model(
            'openai/whisper-large-v3', model_configs
        )
        
        assert selected_model == 'openai/whisper-tiny'
        assert len(warnings) > 0
        assert any("fallback" in warning.lower() for warning in warnings)
    
    def test_suggest_llm_alternatives(self):
        """Test LLM alternative suggestions."""
        resource_checker = ResourceChecker()
        fallback_manager = ModelFallbackManager(resource_checker)
        
        alternatives = fallback_manager.suggest_llm_alternatives('some-large-model')
        
        assert len(alternatives) > 0
        assert all(isinstance(alt, str) for alt in alternatives)


class TestLLMProviderConfig:
    """Test the LLMProviderConfig class."""
    
    def test_initialization(self):
        """Test LLMProviderConfig initialization."""
        config = Configuration()
        llm_config = LLMProviderConfig(config)
        
        assert llm_config.config is config
    
    def test_setup_huggingface_config_cpu(self):
        """Test HuggingFace config setup for CPU."""
        config = Configuration()
        llm_config = LLMProviderConfig(config)
        
        gpu_status = GPUStatus(available=False)
        hf_config = llm_config.setup_huggingface_config('test-model', gpu_status)
        
        assert hf_config['model_name'] == 'test-model'
        assert hf_config['device_map'] == 'cpu'
        assert hf_config['torch_dtype'] == 'float32'
    
    def test_setup_huggingface_config_gpu(self):
        """Test HuggingFace config setup for GPU."""
        config = Configuration()
        llm_config = LLMProviderConfig(config)
        
        gpu_status = GPUStatus(
            available=True,
            memory_total=8000,
            compute_capability=(8, 0)
        )
        hf_config = llm_config.setup_huggingface_config('test-model', gpu_status)
        
        assert hf_config['model_name'] == 'test-model'
        assert hf_config['device_map'] == 'auto'
        assert hf_config['torch_dtype'] == 'float16'


class TestConvenienceFunction:
    """Test the convenience load_config function."""
    
    def test_load_config_function(self):
        """Test the load_config convenience function."""
        config = load_config('config/config.yaml')
        
        assert isinstance(config, Configuration)
        assert config.whisper_model is not None
        assert config.text_generation_model is not None
    
    def test_load_config_function_nonexistent_file(self):
        """Test load_config with non-existent file."""
        with pytest.raises(ConfigurationError):
            load_config('nonexistent.yaml')


if __name__ == '__main__':
    pytest.main([__file__])
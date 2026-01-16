"""
Configuration Manager for the Lecture Transcriber system.

This module handles configuration parsing, validation, resource checking,
and model fallback management. It provides a unified interface for all
configuration-related operations while ensuring system requirements are met.

"""

import os
import sys
import yaml
import psutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from backend.core.models.data_models import (
    Configuration, WhisperModelSize, LLMProvider, GPUStatus, EnvironmentStatus
)
from backend.core.models.errors import ConfigurationError, ModelLoadingError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ResourceRequirements:
    """Resource requirements for a specific model."""
    memory_mb: int
    gpu_memory_mb: int = 0
    recommended_for: str = ""
    language_support: str = ""


class ResourceChecker:
    """
    Checks system resources and validates model requirements.
    
    This class provides methods to assess whether the current system
    has sufficient resources to run specific models, and provides
    warnings and recommendations when resources are limited.
    """
    
    def __init__(self):
        self.system_memory_mb = self._get_system_memory()
        self.gpu_status = self._check_gpu_status()
    
    def _get_system_memory(self) -> int:
        """Get total system memory in MB."""
        try:
            return int(psutil.virtual_memory().total / (1024 * 1024))
        except Exception as e:
            logger.warning(f"Could not determine system memory: {e}")
            return 8192  # Default assumption: 8GB
    
    def _check_gpu_status(self) -> GPUStatus:
        """Check GPU availability and status."""
        gpu_status = GPUStatus(available=False)
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_status.available = True
                gpu_status.device_name = torch.cuda.get_device_name(0)
                gpu_status.memory_total = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
                gpu_status.memory_free = int((torch.cuda.get_device_properties(0).total_memory - 
                                            torch.cuda.memory_allocated(0)) / (1024 * 1024))
                gpu_status.compute_capability = torch.cuda.get_device_capability(0)
                
                # Set recommended settings based on GPU capabilities
                if gpu_status.memory_total >= 8000:
                    gpu_status.recommended_settings = {
                        "torch_dtype": "float16",
                        "device_map": "auto",
                        "use_flash_attention": True
                    }
                elif gpu_status.memory_total >= 4000:
                    gpu_status.recommended_settings = {
                        "torch_dtype": "float16",
                        "device_map": "auto",
                        "load_in_8bit": True
                    }
                else:
                    gpu_status.recommended_settings = {
                        "torch_dtype": "float16",
                        "device_map": "auto",
                        "load_in_4bit": True
                    }
            
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Apple Silicon GPU
                gpu_status.available = True
                gpu_status.device_name = "Apple Silicon GPU"
                gpu_status.recommended_settings = {
                    "torch_dtype": "float16",
                    "device_map": "mps"
                }
                
        except ImportError:
            logger.info("PyTorch not available, GPU acceleration disabled")
        except Exception as e:
            logger.warning(f"Error checking GPU status: {e}")
        
        return gpu_status
    
    def check_model_requirements(self, model_name: str, model_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if system meets requirements for a specific model.
        
        Args:
            model_name: Name of the model to check
            model_config: Model configuration from config file
            
        Returns:
            Tuple of (can_run, warnings_list)
        """
        warnings = []
        can_run = True
        
        # Get memory requirements
        memory_req = model_config.get('memory_requirement_mb', 1000)
        
        # Check system memory
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        if available_memory < memory_req:
            can_run = False
            warnings.append(
                f"Insufficient system memory for {model_name}. "
                f"Required: {memory_req}MB, Available: {int(available_memory)}MB"
            )
        elif available_memory < memory_req * 1.5:
            warnings.append(
                f"Low system memory for {model_name}. "
                f"Required: {memory_req}MB, Available: {int(available_memory)}MB. "
                f"Consider using a smaller model."
            )
        
        # Check GPU memory if GPU is available
        if self.gpu_status.available and memory_req > 2000:
            if self.gpu_status.memory_free < memory_req * 0.8:
                warnings.append(
                    f"Limited GPU memory for {model_name}. "
                    f"GPU memory: {self.gpu_status.memory_free}MB free. "
                    f"Model may run slower or fall back to CPU."
                )
        
        return can_run, warnings
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get a summary of system resources."""
        memory = psutil.virtual_memory()
        
        summary = {
            "system_memory_total_mb": int(memory.total / (1024 * 1024)),
            "system_memory_available_mb": int(memory.available / (1024 * 1024)),
            "system_memory_percent_used": memory.percent,
            "cpu_count": psutil.cpu_count(),
            "gpu_available": self.gpu_status.available,
        }
        
        if self.gpu_status.available:
            summary.update({
                "gpu_name": self.gpu_status.device_name,
                "gpu_memory_total_mb": self.gpu_status.memory_total,
                "gpu_memory_free_mb": self.gpu_status.memory_free,
                "gpu_compute_capability": self.gpu_status.compute_capability
            })
        
        return summary


class ModelFallbackManager:
    """
    Manages automatic model size reduction when resources are insufficient.
    
    This class provides intelligent fallback strategies when the requested
    model cannot be loaded due to resource constraints.
    """
    
    # Model hierarchy from largest to smallest
    WHISPER_MODEL_HIERARCHY = [
        WhisperModelSize.LARGE_V3_RUSSIAN,
        WhisperModelSize.LARGE_V3,
        WhisperModelSize.LARGE_V3_TURBO,
        WhisperModelSize.MEDIUM,
        WhisperModelSize.SMALL,
        WhisperModelSize.TINY
    ]
    
    def __init__(self, resource_checker: ResourceChecker):
        self.resource_checker = resource_checker
    
    def find_suitable_whisper_model(self, requested_model: str, model_configs: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Find a suitable Whisper model based on system resources.
        
        Args:
            requested_model: Originally requested model
            model_configs: Model configurations from config file
            
        Returns:
            Tuple of (selected_model, warnings_list)
        """
        warnings = []
        
        # Check if requested model can run
        if requested_model in model_configs:
            can_run, model_warnings = self.resource_checker.check_model_requirements(
                requested_model, model_configs[requested_model]
            )
            if can_run:
                return requested_model, model_warnings
            else:
                warnings.extend(model_warnings)
                warnings.append(f"Attempting to find suitable alternative to {requested_model}")
        
        # Find the position of requested model in hierarchy
        try:
            requested_index = self.WHISPER_MODEL_HIERARCHY.index(WhisperModelSize(requested_model))
        except (ValueError, TypeError):
            # If requested model is not in hierarchy, start from the largest
            requested_index = 0
        
        # Try smaller models
        for i in range(requested_index + 1, len(self.WHISPER_MODEL_HIERARCHY)):
            candidate_model = self.WHISPER_MODEL_HIERARCHY[i].value
            
            if candidate_model in model_configs:
                can_run, model_warnings = self.resource_checker.check_model_requirements(
                    candidate_model, model_configs[candidate_model]
                )
                
                if can_run:
                    warnings.append(f"Selected {candidate_model} as fallback for {requested_model}")
                    warnings.extend(model_warnings)
                    return candidate_model, warnings
        
        # If no model can run, return the smallest one with warnings
        smallest_model = self.WHISPER_MODEL_HIERARCHY[-1].value
        warnings.append(
            f"System resources are very limited. Using smallest model: {smallest_model}. "
            f"Performance may be significantly impacted."
        )
        
        return smallest_model, warnings
    
    def suggest_llm_alternatives(self, requested_model: str) -> List[str]:
        """
        Suggest alternative LLM models if the requested one is unavailable.
        
        Args:
            requested_model: Originally requested LLM model
            
        Returns:
            List of alternative model suggestions
        """
        alternatives = []
        
        # Memory-efficient alternatives based on requested model
        if "large" in requested_model.lower() or "7b" in requested_model.lower():
            alternatives.extend([
                "microsoft/Phi-4-mini-instruct",  # Smaller but capable
                "HuggingFaceTB/SmolLM3-3B",      # Very efficient
            ])
        elif "medium" in requested_model.lower() or "3b" in requested_model.lower():
            alternatives.extend([
                "HuggingFaceTB/SmolLM3-3B",
                "microsoft/Phi-4-mini-instruct",
            ])
        else:
            # For small models, suggest even more efficient ones
            alternatives.extend([
                "HuggingFaceTB/SmolLM3-3B",
            ])
        
        return alternatives


class LLMProviderConfig:
    """
    Manages LLM provider configuration and setup.
    
    This class handles the configuration of different LLM providers,
    particularly focusing on Hugging Face setup and optimization.
    """
    
    def __init__(self, config: Configuration):
        self.config = config
    
    def setup_huggingface_config(self, model_name: str, gpu_status: GPUStatus) -> Dict[str, Any]:
        """
        Set up Hugging Face model configuration based on system capabilities.
        
        Args:
            model_name: Name of the HF model to configure
            gpu_status: Current GPU status
            
        Returns:
            Dictionary of model configuration parameters
        """
        hf_config = {
            "model_name": model_name,
            "trust_remote_code": False,  # Security best practice
            "use_cache": True,
        }
        
        # Device and precision configuration
        if gpu_status.available:
            hf_config.update({
                "device_map": "auto",
                "torch_dtype": "float16" if gpu_status.memory_total > 4000 else "float16",
            })
            
            # Memory optimization based on GPU memory
            if gpu_status.memory_total < 4000:
                hf_config["load_in_4bit"] = True
            elif gpu_status.memory_total < 8000:
                hf_config["load_in_8bit"] = True
            
            # Flash attention for compatible GPUs
            if gpu_status.compute_capability >= (8, 0):  # Ampere and newer
                hf_config["use_flash_attention"] = True
        else:
            # CPU configuration
            hf_config.update({
                "device_map": "cpu",
                "torch_dtype": "float32",
                "low_cpu_mem_usage": True,
            })
        
        return hf_config
    
    def validate_model_availability(self, model_name: str) -> Tuple[bool, str]:
        """
        Validate that a model is available from Hugging Face.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            Tuple of (is_available, error_message)
        """
        try:
            from transformers import AutoConfig
            
            # Try to load model config to verify availability
            AutoConfig.from_pretrained(model_name)
            return True, ""
            
        except Exception as e:
            error_msg = f"Model {model_name} is not available: {str(e)}"
            return False, error_msg


class ConfigurationManager:
    """
    Main configuration manager that orchestrates all configuration operations.
    
    This class provides the primary interface for loading, validating, and
    managing configuration throughout the application lifecycle.
    """
    
    def __init__(self, config_path: str = "backend/config/config.yaml"):
        self.config_path = config_path
        self.config: Optional[Configuration] = None
        self.resource_checker = ResourceChecker()
        self.fallback_manager = ModelFallbackManager(self.resource_checker)
        self.llm_provider_config = None
        self._raw_config: Dict[str, Any] = {}
    
    def load_configuration(self) -> Configuration:
        """
        Load and validate configuration from YAML file.
        
        Returns:
            Validated Configuration object
            
        Raises:
            ConfigurationError: If configuration is invalid or cannot be loaded
        """
        try:
            # Load YAML configuration
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._raw_config = yaml.safe_load(f)
            
            # Extract and validate configuration sections
            config_dict = self._extract_configuration_dict()
            
            # Create and validate Configuration object
            self.config = Configuration(**config_dict)
            
            # Set up LLM provider configuration
            self.llm_provider_config = LLMProviderConfig(self.config)
            
            # Perform resource validation and model fallback
            self._validate_and_adjust_models()
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return self.config
            
        except FileNotFoundError:
            raise ConfigurationError(
                f"Configuration file not found: {self.config_path}",
                field_name="config_path",
                invalid_value=self.config_path
            )
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {str(e)}",
                field_name="yaml_syntax"
            )
        except Exception as e:
            raise ConfigurationError(
                f"Error loading configuration: {str(e)}",
                field_name="general"
            )
    
    def _extract_configuration_dict(self) -> Dict[str, Any]:
        """Extract configuration parameters from raw YAML data."""
        config_dict = {}
        
        # Whisper configuration
        whisper_config = self._raw_config.get('whisper', {})
        config_dict['whisper_model'] = whisper_config.get('default_model', WhisperModelSize.MEDIUM)
        
        # LLM configuration
        llm_config = self._raw_config.get('llm', {})
        hf_config = llm_config.get('huggingface', {})
        config_dict['text_generation_model'] = hf_config.get('model_name', LLMProvider.MICROSOFT_PHI4)
        
        # Output configuration
        output_config = self._raw_config.get('output', {})
        config_dict['output_format'] = 'markdown' if output_config.get('formats', {}).get('markdown', True) else 'json'
        config_dict['output_directory'] = './output'
        
        # Preprocessing configuration
        preprocessing_config = self._raw_config.get('preprocessing', {})
        config_dict['filler_words'] = preprocessing_config.get('filler_words', [
            "эм", "ээ", "ну", "типа", "короче", "как бы"
        ])
        config_dict['cleaning_intensity'] = preprocessing_config.get('cleaning_intensity', 2)
        
        # Formula configuration
        formulas_config = self._raw_config.get('formulas', {})
        formula_patterns = {}
        formula_patterns.update(formulas_config.get('greek_letters', {}))
        formula_patterns.update(formulas_config.get('operations', {}))
        config_dict['formula_patterns'] = formula_patterns
        config_dict['formula_confidence_threshold'] = formulas_config.get('confidence_threshold', 0.7)
        
        # Environment configuration
        env_config = self._raw_config.get('environment', {})
        config_dict['venv_path'] = env_config.get('venv_path', './lecture_transcriber_env')
        config_dict['model_cache_path'] = env_config.get('model_cache_path', './lecture_transcriber_env/model_cache')
        
        # Resource configuration
        resources_config = self._raw_config.get('resources', {})
        config_dict['device'] = resources_config.get('device', 'auto')
        config_dict['batch_size'] = 1
        
        gpu_config = resources_config.get('gpu', {})
        config_dict['gpu_enabled'] = gpu_config.get('enabled', True)
        config_dict['memory_fraction'] = gpu_config.get('memory_fraction', 0.8)
        
        # Additional settings
        config_dict['torch_dtype'] = 'auto'
        config_dict['temp_directory'] = resources_config.get('temp_dir', './temp')
        config_dict['preserve_timestamps'] = True
        config_dict['enable_formula_formatting'] = True
        config_dict['enable_segment_merging'] = True
        
        # Logging configuration
        logging_config = self._raw_config.get('logging', {})
        config_dict['log_level'] = logging_config.get('level', 'INFO')
        config_dict['verbose_output'] = False
        
        # Development configuration
        dev_config = self._raw_config.get('development', {})
        config_dict['save_intermediate_files'] = dev_config.get('save_intermediate_files', False)
        
        return config_dict
    
    def _validate_and_adjust_models(self):
        """Validate model selections and apply fallbacks if necessary."""
        if not self.config:
            return
        
        warnings = []
        
        # Check Whisper model
        whisper_models = self._raw_config.get('whisper', {}).get('models', {})
        if whisper_models:
            suitable_model, model_warnings = self.fallback_manager.find_suitable_whisper_model(
                self.config.whisper_model, whisper_models
            )
            if suitable_model != self.config.whisper_model:
                self.config.whisper_model = suitable_model
                warnings.extend(model_warnings)
        
        # Check LLM model availability
        if self.llm_provider_config:
            is_available, error_msg = self.llm_provider_config.validate_model_availability(
                self.config.text_generation_model
            )
            if not is_available:
                alternatives = self.fallback_manager.suggest_llm_alternatives(
                    self.config.text_generation_model
                )
                if alternatives:
                    self.config.text_generation_model = alternatives[0]
                    warnings.append(f"LLM model fallback: using {alternatives[0]} instead")
                    warnings.append(error_msg)
        
        # Log all warnings
        for warning in warnings:
            logger.warning(warning)
    
    def get_whisper_config(self) -> Dict[str, Any]:
        """Get optimized Whisper model configuration."""
        if not self.config:
            raise ConfigurationError("Configuration not loaded")
        
        whisper_config = {
            "model_name": self.config.whisper_model,
            "device": self.config.device,
            "torch_dtype": self.config.torch_dtype,
        }
        
        # Add GPU-specific optimizations
        if self.config.gpu_enabled and self.resource_checker.gpu_status.available:
            whisper_config.update(self.resource_checker.gpu_status.recommended_settings)
        
        return whisper_config
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get optimized LLM configuration."""
        if not self.config or not self.llm_provider_config:
            raise ConfigurationError("Configuration not loaded")
        
        return self.llm_provider_config.setup_huggingface_config(
            self.config.text_generation_model,
            self.resource_checker.gpu_status
        )
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get text processing configuration."""
        if not self.config:
            raise ConfigurationError("Configuration not loaded")
        
        return {
            "filler_words": self.config.filler_words,
            "cleaning_intensity": self.config.cleaning_intensity,
            "formula_patterns": self.config.formula_patterns,
            "formula_confidence_threshold": self.config.formula_confidence_threshold,
            "preserve_timestamps": self.config.preserve_timestamps,
            "enable_formula_formatting": self.config.enable_formula_formatting,
            "enable_segment_merging": self.config.enable_segment_merging,
        }
    
    def validate_environment(self) -> EnvironmentStatus:
        """Validate virtual environment status and isolation."""
        venv_path = self.config.venv_path if self.config else "./lecture_transcriber_env"
        
        # Check if we're in a virtual environment
        venv_active = (
            hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        )
        
        # Check if venv directory exists
        venv_exists = os.path.exists(venv_path)
        
        # Check if dependencies are installed (basic check)
        dependencies_installed = True
        try:
            import torch
            import transformers
            import yaml
        except ImportError:
            dependencies_installed = False
        
        # Check model isolation (models should be in venv)
        models_isolated = True
        if self.config and self.config.model_cache_path:
            if not self.config.model_cache_path.startswith(venv_path):
                models_isolated = False
        
        return EnvironmentStatus(
            venv_active=venv_active,
            dependencies_installed=dependencies_installed,
            models_isolated=models_isolated,
            system_clean=venv_active and models_isolated,
            venv_path=venv_path,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive system resource summary."""
        base_summary = self.resource_checker.get_resource_summary()
        
        # Add configuration-specific information
        if self.config:
            base_summary.update({
                "configured_device": self.config.device,
                "gpu_enabled": self.config.gpu_enabled,
                "memory_fraction": self.config.memory_fraction,
                "selected_whisper_model": self.config.whisper_model,
                "selected_llm_model": self.config.text_generation_model,
            })
        
        return base_summary
    
    def save_configuration(self, output_path: Optional[str] = None) -> str:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Optional path to save configuration
            
        Returns:
            Path where configuration was saved
        """
        if not self.config:
            raise ConfigurationError("No configuration to save")
        
        save_path = output_path or self.config_path
        
        # Convert Configuration object back to YAML structure
        config_dict = {
            "whisper": {
                "default_model": self.config.whisper_model,
                "gpu_acceleration": {
                    "enabled": self.config.gpu_enabled,
                    "torch_dtype": self.config.torch_dtype,
                    "device_map": "auto"
                }
            },
            "llm": {
                "huggingface": {
                    "model_name": self.config.text_generation_model
                }
            },
            "preprocessing": {
                "filler_words": self.config.filler_words,
                "cleaning_intensity": self.config.cleaning_intensity
            },
            "formulas": {
                "confidence_threshold": self.config.formula_confidence_threshold
            },
            "environment": {
                "venv_path": self.config.venv_path,
                "model_cache_path": self.config.model_cache_path
            },
            "resources": {
                "device": self.config.device,
                "gpu": {
                    "enabled": self.config.gpu_enabled,
                    "memory_fraction": self.config.memory_fraction
                }
            },
            "logging": {
                "level": self.config.log_level
            }
        }
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"Configuration saved to {save_path}")
            return save_path
            
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {str(e)}")


# Convenience function for quick configuration loading
def load_config(config_path: str = "backend/config/config.yaml") -> Configuration:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated Configuration object
    """
    manager = ConfigurationManager(config_path)
    return manager.load_configuration()

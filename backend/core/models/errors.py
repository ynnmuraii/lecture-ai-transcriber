"""
Custom error classes for the Lecture Transcriber system.

This module contains all custom exception classes used throughout
the application for structured error handling and reporting.
"""

from typing import Dict, Any


class TranscriptionError(Exception):
    """
    Base exception for transcription-related errors.
    
    Provides structured error information including error type,
    recovery suggestions, and context for debugging.
    """
    
    def __init__(self, message: str, error_type: str = "general", recoverable: bool = True, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.recoverable = recoverable
        self.context = context or {}
    
    def __str__(self):
        return f"TranscriptionError ({self.error_type}): {self.message}"


class AudioExtractionError(Exception):
    """
    Exception for audio extraction and processing errors.
    
    Includes specific information about the problematic file and
    suggested actions for resolution.
    """
    
    def __init__(self, message: str, file_path: str = "", suggested_action: str = "", error_code: int = 0):
        super().__init__(message)
        self.message = message
        self.file_path = file_path
        self.suggested_action = suggested_action
        self.error_code = error_code
    
    def __str__(self):
        base_msg = f"AudioExtractionError: {self.message}"
        if self.file_path:
            base_msg += f" (File: {self.file_path})"
        if self.suggested_action:
            base_msg += f" - Suggestion: {self.suggested_action}"
        return base_msg


class ConfigurationError(Exception):
    """
    Exception for configuration validation and loading errors.
    
    Helps identify configuration issues with specific field information
    and validation guidance.
    """
    
    def __init__(self, message: str, field_name: str = "", invalid_value: Any = None):
        super().__init__(message)
        self.message = message
        self.field_name = field_name
        self.invalid_value = invalid_value
    
    def __str__(self):
        base_msg = f"ConfigurationError: {self.message}"
        if self.field_name:
            base_msg += f" (Field: {self.field_name})"
        if self.invalid_value is not None:
            base_msg += f" (Value: {self.invalid_value})"
        return base_msg


class ModelLoadingError(Exception):
    """
    Exception for ML model loading and initialization errors.
    
    Provides information about model availability, resource requirements,
    and fallback options.
    """
    
    def __init__(self, message: str, model_name: str = "", required_memory: int = 0, available_memory: int = 0):
        super().__init__(message)
        self.message = message
        self.model_name = model_name
        self.required_memory = required_memory
        self.available_memory = available_memory
    
    def __str__(self):
        base_msg = f"ModelLoadingError: {self.message}"
        if self.model_name:
            base_msg += f" (Model: {self.model_name})"
        if self.required_memory > 0:
            base_msg += f" (Required: {self.required_memory}MB, Available: {self.available_memory}MB)"
        return base_msg

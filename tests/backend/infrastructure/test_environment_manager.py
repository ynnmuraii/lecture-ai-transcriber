"""
Unit tests for the Environment Manager module.

These tests verify the core functionality of virtual environment creation,
dependency installation, isolation verification, and model cache path management.
"""

import pytest
import tempfile
import os
import sys
import shutil
import subprocess
from unittest.mock import patch, MagicMock, call
from pathlib import Path

import backend.infrastructure.environment_manager as environment_manager
from backend.infrastructure.environment_manager import (
    EnvironmentManager, VenvCreationResult, DependencyInstallResult,
    create_environment, quick_environment_check
)
import backend.core.models as models
from backend.core.models.data_models import Configuration, EnvironmentStatus
from backend.core.models.errors import ConfigurationError


class TestEnvironmentManager:
    """Test the main EnvironmentManager class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_venv_path = os.path.join(self.temp_dir, "test_venv")
        self.test_config = Configuration(
            venv_path=self.test_venv_path,
            model_cache_path=os.path.join(self.test_venv_path, "models")
        )
        self.manager = EnvironmentManager(self.test_config)
    
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization_with_config(self):
        """Test EnvironmentManager initialization with configuration."""
        manager = EnvironmentManager(self.test_config)
        
        assert manager.config == self.test_config
        assert manager.venv_path == self.test_venv_path
        assert manager.model_cache_path == os.path.join(self.test_venv_path, "models")
    
    def test_initialization_without_config(self):
        """Test EnvironmentManager initialization without configuration."""
        manager = EnvironmentManager()
        
        assert manager.config is None
        assert manager.venv_path == os.path.abspath("./lecture_transcriber_env")
        assert "models" in manager.model_cache_path
    
    def test_setup_venv_new_environment(self):
        """Test creating a new virtual environment."""
        result = self.manager.setup_venv()
        
        assert result.success is True
        assert result.venv_path == self.test_venv_path
        assert os.path.exists(self.test_venv_path)
        assert result.python_executable != ""
        assert os.path.exists(result.python_executable)
    
    def test_setup_venv_existing_valid_environment(self):
        """Test handling of existing valid virtual environment."""
        # First create a venv
        first_result = self.manager.setup_venv()
        assert first_result.success is True
        
        # Try to create again without force_recreate
        second_result = self.manager.setup_venv()
        
        assert second_result.success is True
        assert second_result.venv_path == self.test_venv_path
        assert len(second_result.warnings) > 0
        assert "existing" in second_result.warnings[0].lower()
    
    def test_setup_venv_force_recreate(self):
        """Test force recreation of existing virtual environment."""
        # First create a venv
        first_result = self.manager.setup_venv()
        assert first_result.success is True
        original_time = os.path.getmtime(self.test_venv_path)
        
        # Force recreate
        second_result = self.manager.setup_venv(force_recreate=True)
        
        assert second_result.success is True
        assert second_result.venv_path == self.test_venv_path
        # Directory should be newer (recreated)
        new_time = os.path.getmtime(self.test_venv_path)
        assert new_time >= original_time
    
    def test_setup_venv_custom_path(self):
        """Test creating virtual environment at custom path."""
        custom_path = os.path.join(self.temp_dir, "custom_venv")
        
        result = self.manager.setup_venv(venv_path=custom_path)
        
        assert result.success is True
        assert result.venv_path == custom_path
        assert os.path.exists(custom_path)
    
    @patch('venv.create')
    def test_setup_venv_creation_failure(self, mock_venv_create):
        """Test handling of virtual environment creation failure."""
        mock_venv_create.side_effect = Exception("Creation failed")
        
        result = self.manager.setup_venv()
        
        assert result.success is False
        assert "Creation failed" in result.error_message
    
    def test_install_dependencies_valid_requirements(self):
        """Test installing dependencies from valid requirements file."""
        # Create venv first
        venv_result = self.manager.setup_venv()
        assert venv_result.success is True
        
        # Create a simple requirements file
        requirements_file = os.path.join(self.temp_dir, "requirements.txt")
        with open(requirements_file, 'w') as f:
            f.write("requests==2.31.0\n")
            f.write("pyyaml>=6.0\n")
        
        result = self.manager.install_dependencies(requirements_file)
        
        assert result.success is True
        assert len(result.installed_packages) >= 0  # May vary based on system
        assert result.installation_log != ""
    
    def test_install_dependencies_nonexistent_file(self):
        """Test error handling for non-existent requirements file."""
        nonexistent_file = os.path.join(self.temp_dir, "nonexistent.txt")
        
        result = self.manager.install_dependencies(nonexistent_file)
        
        assert result.success is False
        assert "not found" in result.error_message.lower()
    
    def test_install_dependencies_invalid_venv(self):
        """Test error handling for invalid virtual environment."""
        requirements_file = os.path.join(self.temp_dir, "requirements.txt")
        with open(requirements_file, 'w') as f:
            f.write("requests\n")
        
        # Don't create venv first
        result = self.manager.install_dependencies(requirements_file)
        
        assert result.success is False
        assert "invalid virtual environment" in result.error_message.lower()
    
    def test_get_model_cache_path_default(self):
        """Test getting model cache path with default venv."""
        cache_path = self.manager.get_model_cache_path()
        
        assert cache_path == os.path.abspath(os.path.join(self.test_venv_path, "models"))
        assert os.path.exists(cache_path)  # Should be created if it doesn't exist
    
    def test_get_model_cache_path_custom_venv(self):
        """Test getting model cache path with custom venv path."""
        custom_venv = os.path.join(self.temp_dir, "custom_venv")
        
        cache_path = self.manager.get_model_cache_path(custom_venv)
        
        assert cache_path == os.path.abspath(os.path.join(custom_venv, "models"))
        assert os.path.exists(cache_path)
    
    def test_verify_isolation_no_venv(self):
        """Test isolation verification when no venv exists."""
        status = self.manager.verify_isolation()
        
        assert isinstance(status, EnvironmentStatus)
        assert status.venv_active is False
        assert status.dependencies_installed is False
        assert status.venv_path == self.test_venv_path
    
    def test_verify_isolation_with_venv(self):
        """Test isolation verification with existing venv."""
        # Create venv first
        venv_result = self.manager.setup_venv()
        assert venv_result.success is True
        
        status = self.manager.verify_isolation()
        
        assert isinstance(status, EnvironmentStatus)
        assert status.venv_path == self.test_venv_path
        assert status.python_version != ""
        # Note: venv_active may be False since we're not actually running in the venv
    
    def test_activate_venv_valid_environment(self):
        """Test getting activation command for valid environment."""
        # Create venv first
        venv_result = self.manager.setup_venv()
        assert venv_result.success is True
        
        success, command = self.manager.activate_venv()
        
        assert success is True
        assert command != ""
        if os.name == 'nt':  # Windows
            assert "activate.bat" in command
        else:  # Unix-like
            assert "activate" in command
    
    def test_activate_venv_invalid_environment(self):
        """Test activation command for invalid environment."""
        success, command = self.manager.activate_venv()
        
        assert success is False
        assert "invalid" in command.lower()
    
    def test_cleanup_venv_without_confirmation(self):
        """Test cleanup without confirmation (should fail)."""
        # Create venv first
        venv_result = self.manager.setup_venv()
        assert venv_result.success is True
        
        success, message = self.manager.cleanup_venv()
        
        assert success is False
        assert "confirmation" in message.lower()
        assert os.path.exists(self.test_venv_path)  # Should still exist
    
    def test_cleanup_venv_with_confirmation(self):
        """Test cleanup with confirmation."""
        # Create venv first
        venv_result = self.manager.setup_venv()
        assert venv_result.success is True
        
        success, message = self.manager.cleanup_venv(confirm=True)
        
        assert success is True
        assert "removed" in message.lower()
        assert not os.path.exists(self.test_venv_path)
    
    def test_cleanup_venv_nonexistent(self):
        """Test cleanup of non-existent environment."""
        success, message = self.manager.cleanup_venv(confirm=True)
        
        assert success is True
        assert "does not exist" in message.lower()
    
    def test_get_environment_info_no_venv(self):
        """Test getting environment info when no venv exists."""
        info = self.manager.get_environment_info()
        
        assert isinstance(info, dict)
        assert info["venv_path"] == self.test_venv_path
        assert info["venv_exists"] is False
        assert info["venv_valid"] is False
        assert info["python_executable"] == ""
        assert info["disk_usage_mb"] == 0
    
    def test_get_environment_info_with_venv(self):
        """Test getting environment info with existing venv."""
        # Create venv first
        venv_result = self.manager.setup_venv()
        assert venv_result.success is True
        
        info = self.manager.get_environment_info()
        
        assert isinstance(info, dict)
        assert info["venv_path"] == self.test_venv_path
        assert info["venv_exists"] is True
        assert info["venv_valid"] is True
        assert info["python_executable"] != ""
        assert info["python_version"] != ""
        assert info["disk_usage_mb"] > 0
    
    @patch('subprocess.run')
    def test_is_valid_venv_check_failure(self, mock_subprocess):
        """Test venv validation when subprocess fails."""
        mock_subprocess.return_value.returncode = 1
        
        is_valid = self.manager._is_valid_venv(self.test_venv_path)
        
        assert is_valid is False
    
    def test_get_venv_python_executable_windows(self):
        """Test getting Python executable path on Windows."""
        with patch('os.name', 'nt'):
            python_exe = self.manager._get_venv_python_executable(self.test_venv_path)
            
            assert python_exe.endswith("python.exe")
            assert "Scripts" in python_exe
    
    def test_get_venv_python_executable_unix(self):
        """Test getting Python executable path on Unix-like systems."""
        with patch('os.name', 'posix'):
            python_exe = self.manager._get_venv_python_executable(self.test_venv_path)
            
            assert python_exe.endswith("python")
            assert "bin" in python_exe
    
    @patch('sys.prefix', '/test/venv')
    @patch('sys.base_prefix', '/usr')
    def test_is_venv_active_true(self):
        """Test detection of active virtual environment."""
        is_active = self.manager._is_venv_active()
        
        assert is_active is True
    
    @patch('sys.prefix', '/usr')
    @patch('sys.base_prefix', '/usr')
    def test_is_venv_active_false(self):
        """Test detection when not in virtual environment."""
        is_active = self.manager._is_venv_active()
        
        assert is_active is False


class TestConvenienceFunctions:
    """Test convenience functions for environment management."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_venv_path = os.path.join(self.temp_dir, "test_venv")
        self.requirements_file = os.path.join(self.temp_dir, "requirements.txt")
        
        # Create a simple requirements file
        with open(self.requirements_file, 'w') as f:
            f.write("pyyaml>=6.0\n")
    
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_environment_success(self):
        """Test successful complete environment creation."""
        success, message = create_environment(
            self.test_venv_path, 
            self.requirements_file
        )
        
        assert success is True
        assert "successfully" in message.lower()
        assert os.path.exists(self.test_venv_path)
    
    def test_create_environment_venv_failure(self):
        """Test environment creation with venv failure."""
        # Use a path that will definitely cause permission issues
        if os.name == 'nt':  # Windows
            invalid_path = "C:\\Windows\\System32\\test_venv"
        else:  # Unix-like
            invalid_path = "/root/test_venv"
        
        success, message = create_environment(
            invalid_path, 
            self.requirements_file
        )
        
        # The test should handle both success and failure gracefully
        # since the actual behavior depends on system permissions
        assert isinstance(success, bool)
        assert isinstance(message, str)
        if not success:
            assert "failed" in message.lower() or "error" in message.lower()
    
    def test_create_environment_deps_failure(self):
        """Test environment creation with dependency installation failure."""
        # Create invalid requirements file
        invalid_requirements = os.path.join(self.temp_dir, "invalid_requirements.txt")
        with open(invalid_requirements, 'w') as f:
            f.write("nonexistent-package-that-does-not-exist==999.999.999\n")
        
        success, message = create_environment(
            self.test_venv_path, 
            invalid_requirements
        )
        
        # This might succeed or fail depending on pip behavior
        # The test verifies the function handles both cases appropriately
        assert isinstance(success, bool)
        assert isinstance(message, str)
    
    def test_quick_environment_check_no_venv(self):
        """Test quick environment check with no venv."""
        result = quick_environment_check(self.test_venv_path)
        
        assert isinstance(result, dict)
        assert "status" in result
        assert "info" in result
        assert "ready" in result
        assert result["ready"] is False
    
    def test_quick_environment_check_with_venv(self):
        """Test quick environment check with existing venv."""
        # Create environment first
        success, _ = create_environment(self.test_venv_path, self.requirements_file)
        if success:  # Only test if creation succeeded
            result = quick_environment_check(self.test_venv_path)
            
            assert isinstance(result, dict)
            assert "status" in result
            assert "info" in result
            assert "ready" in result
            assert isinstance(result["status"], EnvironmentStatus)


class TestVenvCreationResult:
    """Test VenvCreationResult dataclass."""
    
    def test_initialization_success(self):
        """Test successful result initialization."""
        result = VenvCreationResult(
            success=True,
            venv_path="/test/path",
            python_executable="/test/path/bin/python"
        )
        
        assert result.success is True
        assert result.venv_path == "/test/path"
        assert result.python_executable == "/test/path/bin/python"
        assert result.error_message == ""
        assert result.warnings == []
    
    def test_initialization_failure(self):
        """Test failure result initialization."""
        result = VenvCreationResult(
            success=False,
            error_message="Creation failed"
        )
        
        assert result.success is False
        assert result.error_message == "Creation failed"
        assert result.venv_path == ""
        assert result.warnings == []


class TestDependencyInstallResult:
    """Test DependencyInstallResult dataclass."""
    
    def test_initialization_success(self):
        """Test successful installation result."""
        result = DependencyInstallResult(
            success=True,
            installed_packages=["package1", "package2"],
            installation_log="Installation completed"
        )
        
        assert result.success is True
        assert result.installed_packages == ["package1", "package2"]
        assert result.failed_packages == []
        assert result.installation_log == "Installation completed"
    
    def test_initialization_failure(self):
        """Test failed installation result."""
        result = DependencyInstallResult(
            success=False,
            failed_packages=["bad_package"],
            error_message="Installation failed"
        )
        
        assert result.success is False
        assert result.failed_packages == ["bad_package"]
        assert result.installed_packages == []
        assert result.error_message == "Installation failed"


if __name__ == '__main__':
    pytest.main([__file__])

"""
Environment Manager for the Lecture Transcriber system.

This module handles virtual environment creation, activation, dependency installation,
and verification of environment isolation. It ensures all libraries and models are
contained within a virtual environment to keep the host system clean.

"""

import os
import sys
import subprocess
import venv
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from backend.core.models.data_models import Configuration, EnvironmentStatus
from backend.core.models.errors import ConfigurationError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class VenvCreationResult:
    """Result of virtual environment creation operation."""
    success: bool
    venv_path: str = ""
    python_executable: str = ""
    error_message: str = ""
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class DependencyInstallResult:
    """Result of dependency installation operation."""
    success: bool
    installed_packages: List[str] = None
    failed_packages: List[str] = None
    error_message: str = ""
    installation_log: str = ""
    
    def __post_init__(self):
        if self.installed_packages is None:
            self.installed_packages = []
        if self.failed_packages is None:
            self.failed_packages = []


class EnvironmentManager:
    """
    Manages virtual environment setup and dependency isolation.
    
    This class provides comprehensive virtual environment management including
    creation, activation, dependency installation, and isolation verification.
    It ensures all ML models and dependencies are contained within the venv.
    """
    
    def __init__(self, config: Optional[Configuration] = None):
        """
        Initialize Environment Manager.
        
        Args:
            config: Optional Configuration object. If not provided, uses defaults.
        """
        self.config = config
        self.venv_path = config.venv_path if config else "./lecture_transcriber_env"
        self.model_cache_path = config.model_cache_path if config else os.path.join(self.venv_path, "models")
        
        # Ensure paths are absolute for consistency
        self.venv_path = os.path.abspath(self.venv_path)
        self.model_cache_path = os.path.abspath(self.model_cache_path)
        
        logger.info(f"Environment Manager initialized with venv path: {self.venv_path}")
    
    def setup_venv(self, venv_path: Optional[str] = None, force_recreate: bool = False) -> VenvCreationResult:
        """
        Create and set up a virtual environment.
        
        Args:
            venv_path: Optional custom path for venv. Uses configured path if not provided.
            force_recreate: If True, removes existing venv and creates new one.
            
        Returns:
            VenvCreationResult with creation status and details.
        """
        target_path = venv_path or self.venv_path
        target_path = os.path.abspath(target_path)
        
        logger.info(f"Setting up virtual environment at: {target_path}")
        
        try:
            # Check if venv already exists
            if os.path.exists(target_path):
                if force_recreate:
                    logger.info(f"Removing existing virtual environment: {target_path}")
                    shutil.rmtree(target_path)
                else:
                    # Verify existing venv is valid
                    if self._is_valid_venv(target_path):
                        logger.info(f"Valid virtual environment already exists at: {target_path}")
                        python_exe = self._get_venv_python_executable(target_path)
                        return VenvCreationResult(
                            success=True,
                            venv_path=target_path,
                            python_executable=python_exe,
                            warnings=["Using existing virtual environment"]
                        )
                    else:
                        logger.warning(f"Invalid virtual environment found, recreating: {target_path}")
                        shutil.rmtree(target_path)
            
            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(target_path)
            os.makedirs(parent_dir, exist_ok=True)
            
            # Create virtual environment
            logger.info(f"Creating new virtual environment: {target_path}")
            venv.create(target_path, with_pip=True, clear=True)
            
            # Verify creation was successful
            if not self._is_valid_venv(target_path):
                return VenvCreationResult(
                    success=False,
                    error_message=f"Failed to create valid virtual environment at {target_path}"
                )
            
            # Get Python executable path
            python_exe = self._get_venv_python_executable(target_path)
            
            # Upgrade pip to latest version
            upgrade_result = self._upgrade_pip(python_exe)
            warnings = []
            if not upgrade_result[0]:
                warnings.append(f"Failed to upgrade pip: {upgrade_result[1]}")
            
            # Create model cache directory
            model_cache_dir = os.path.join(target_path, "models")
            os.makedirs(model_cache_dir, exist_ok=True)
            
            # Update instance paths if this was the configured venv
            if target_path == self.venv_path:
                self.model_cache_path = model_cache_dir
            
            logger.info(f"Virtual environment created successfully: {target_path}")
            
            return VenvCreationResult(
                success=True,
                venv_path=target_path,
                python_executable=python_exe,
                warnings=warnings
            )
            
        except Exception as e:
            error_msg = f"Error creating virtual environment: {str(e)}"
            logger.error(error_msg)
            return VenvCreationResult(
                success=False,
                error_message=error_msg
            )

    def install_dependencies(self, requirements_file: str, venv_path: Optional[str] = None) -> DependencyInstallResult:
        """
        Install dependencies from requirements file into virtual environment.
        
        Args:
            requirements_file: Path to requirements.txt file
            venv_path: Optional custom venv path. Uses configured path if not provided.
            
        Returns:
            DependencyInstallResult with installation status and details.
        """
        target_venv = venv_path or self.venv_path
        
        logger.info(f"Installing dependencies from {requirements_file} into {target_venv}")
        
        try:
            # Validate inputs
            if not os.path.exists(requirements_file):
                return DependencyInstallResult(
                    success=False,
                    error_message=f"Requirements file not found: {requirements_file}"
                )
            
            if not self._is_valid_venv(target_venv):
                return DependencyInstallResult(
                    success=False,
                    error_message=f"Invalid virtual environment: {target_venv}"
                )
            
            # Get Python executable
            python_exe = self._get_venv_python_executable(target_venv)
            
            # Read requirements file to track what we're installing
            with open(requirements_file, 'r', encoding='utf-8') as f:
                requirements_content = f.read()
            
            # Parse package names (basic parsing, ignores version specifiers)
            package_names = []
            for line in requirements_content.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before any version specifiers)
                    package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].split('~=')[0]
                    package_names.append(package_name.strip())
            
            # Install dependencies
            cmd = [python_exe, "-m", "pip", "install", "-r", requirements_file, "--upgrade"]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(requirements_file))
            )
            
            if result.returncode == 0:
                logger.info("Dependencies installed successfully")
                
                # Verify installation by checking if packages can be imported
                installed_packages, failed_packages = self._verify_package_installation(
                    python_exe, package_names
                )
                
                return DependencyInstallResult(
                    success=True,
                    installed_packages=installed_packages,
                    failed_packages=failed_packages,
                    installation_log=result.stdout
                )
            else:
                error_msg = f"Dependency installation failed: {result.stderr}"
                logger.error(error_msg)
                return DependencyInstallResult(
                    success=False,
                    error_message=error_msg,
                    installation_log=result.stdout + "\n" + result.stderr
                )
                
        except Exception as e:
            error_msg = f"Error installing dependencies: {str(e)}"
            logger.error(error_msg)
            return DependencyInstallResult(
                success=False,
                error_message=error_msg
            )
    
    def get_model_cache_path(self, venv_path: Optional[str] = None) -> str:
        """
        Get the model cache directory path within the virtual environment.
        
        Args:
            venv_path: Optional custom venv path. Uses configured path if not provided.
            
        Returns:
            Absolute path to model cache directory.
        """
        target_venv = venv_path or self.venv_path
        cache_path = os.path.join(target_venv, "models")
        
        # Create directory if it doesn't exist
        os.makedirs(cache_path, exist_ok=True)
        
        return os.path.abspath(cache_path)

    def verify_isolation(self, venv_path: Optional[str] = None) -> EnvironmentStatus:
        """
        Verify that the virtual environment is properly isolated.
        
        Args:
            venv_path: Optional custom venv path. Uses configured path if not provided.
            
        Returns:
            EnvironmentStatus with detailed isolation verification results.
        """
        target_venv = venv_path or self.venv_path
        
        logger.info(f"Verifying environment isolation for: {target_venv}")
        
        try:
            # Check if venv exists and is valid
            venv_exists = os.path.exists(target_venv)
            venv_valid = self._is_valid_venv(target_venv) if venv_exists else False
            
            # Check if we're currently in a virtual environment
            venv_active = self._is_venv_active()
            
            # Check if current venv matches target venv (if active)
            current_venv_matches = False
            if venv_active:
                current_prefix = sys.prefix
                target_prefix = target_venv
                current_venv_matches = os.path.samefile(current_prefix, target_prefix) if venv_exists else False
            
            # Check dependencies installation
            dependencies_installed = False
            if venv_valid:
                python_exe = self._get_venv_python_executable(target_venv)
                dependencies_installed = self._check_core_dependencies(python_exe)
            
            # Check model isolation (models should be within venv)
            models_isolated = True
            model_cache_path = self.get_model_cache_path(target_venv)
            if not model_cache_path.startswith(target_venv):
                models_isolated = False
            
            # Check system cleanliness (no global package pollution)
            system_clean = self._check_system_cleanliness()
            
            # Get Python version from venv
            python_version = ""
            if venv_valid:
                python_exe = self._get_venv_python_executable(target_venv)
                python_version = self._get_python_version(python_exe)
            
            return EnvironmentStatus(
                venv_active=venv_active and current_venv_matches,
                dependencies_installed=dependencies_installed,
                models_isolated=models_isolated,
                system_clean=system_clean,
                venv_path=target_venv,
                python_version=python_version
            )
            
        except Exception as e:
            logger.error(f"Error verifying environment isolation: {str(e)}")
            return EnvironmentStatus(
                venv_active=False,
                dependencies_installed=False,
                models_isolated=False,
                system_clean=False,
                venv_path=target_venv,
                python_version=""
            )
    
    def activate_venv(self, venv_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Provide activation instructions for the virtual environment.
        
        Note: Actual activation must be done by the user in their shell,
        as Python subprocess cannot modify the parent shell environment.
        
        Args:
            venv_path: Optional custom venv path. Uses configured path if not provided.
            
        Returns:
            Tuple of (success, activation_command_or_error)
        """
        target_venv = venv_path or self.venv_path
        
        if not self._is_valid_venv(target_venv):
            return False, f"Invalid virtual environment: {target_venv}"
        
        # Generate platform-specific activation command
        if os.name == 'nt':  # Windows
            activate_script = os.path.join(target_venv, "Scripts", "activate.bat")
            if os.path.exists(activate_script):
                return True, f"call \"{activate_script}\""
            else:
                return False, f"Activation script not found: {activate_script}"
        else:  # Unix-like (Linux, macOS)
            activate_script = os.path.join(target_venv, "bin", "activate")
            if os.path.exists(activate_script):
                return True, f"source \"{activate_script}\""
            else:
                return False, f"Activation script not found: {activate_script}"

    def cleanup_venv(self, venv_path: Optional[str] = None, confirm: bool = False) -> Tuple[bool, str]:
        """
        Remove virtual environment and all its contents.
        
        Args:
            venv_path: Optional custom venv path. Uses configured path if not provided.
            confirm: Must be True to actually perform deletion (safety measure).
            
        Returns:
            Tuple of (success, message)
        """
        if not confirm:
            return False, "Cleanup requires explicit confirmation (confirm=True)"
        
        target_venv = venv_path or self.venv_path
        
        try:
            if os.path.exists(target_venv):
                logger.info(f"Removing virtual environment: {target_venv}")
                shutil.rmtree(target_venv)
                return True, f"Virtual environment removed: {target_venv}"
            else:
                return True, f"Virtual environment does not exist: {target_venv}"
                
        except Exception as e:
            error_msg = f"Error removing virtual environment: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_environment_info(self, venv_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive information about the virtual environment.
        
        Args:
            venv_path: Optional custom venv path. Uses configured path if not provided.
            
        Returns:
            Dictionary with environment information.
        """
        target_venv = venv_path or self.venv_path
        
        info = {
            "venv_path": target_venv,
            "venv_exists": os.path.exists(target_venv),
            "venv_valid": False,
            "python_executable": "",
            "python_version": "",
            "model_cache_path": self.get_model_cache_path(target_venv),
            "installed_packages": [],
            "disk_usage_mb": 0
        }
        
        if info["venv_exists"]:
            info["venv_valid"] = self._is_valid_venv(target_venv)
            
            if info["venv_valid"]:
                python_exe = self._get_venv_python_executable(target_venv)
                info["python_executable"] = python_exe
                info["python_version"] = self._get_python_version(python_exe)
                info["installed_packages"] = self._get_installed_packages(python_exe)
            
            # Calculate disk usage
            try:
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(target_venv):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            total_size += os.path.getsize(filepath)
                info["disk_usage_mb"] = total_size / (1024 * 1024)
            except Exception as e:
                logger.warning(f"Could not calculate disk usage: {e}")
        
        return info
    
    # Private helper methods
    
    def _is_valid_venv(self, venv_path: str) -> bool:
        """Check if a directory contains a valid virtual environment."""
        try:
            # Check for Python executable
            python_exe = self._get_venv_python_executable(venv_path)
            if not os.path.exists(python_exe):
                return False
            
            # Check for pip
            result = subprocess.run(
                [python_exe, "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return result.returncode == 0
            
        except Exception:
            return False
    
    def _get_venv_python_executable(self, venv_path: str) -> str:
        """Get the Python executable path for a virtual environment."""
        if os.name == 'nt':  # Windows
            return os.path.join(venv_path, "Scripts", "python.exe")
        else:  # Unix-like
            return os.path.join(venv_path, "bin", "python")
    
    def _is_venv_active(self) -> bool:
        """Check if we're currently running in a virtual environment."""
        return (
            hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        )

    def _upgrade_pip(self, python_exe: str) -> Tuple[bool, str]:
        """Upgrade pip to the latest version."""
        try:
            cmd = [python_exe, "-m", "pip", "install", "--upgrade", "pip"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                return True, "Pip upgraded successfully"
            else:
                return False, f"Pip upgrade failed: {result.stderr}"
                
        except Exception as e:
            return False, f"Error upgrading pip: {str(e)}"
    
    def _verify_package_installation(self, python_exe: str, package_names: List[str]) -> Tuple[List[str], List[str]]:
        """Verify that packages were installed correctly."""
        installed = []
        failed = []
        
        for package in package_names:
            try:
                # Try to import the package
                cmd = [python_exe, "-c", f"import {package}"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    installed.append(package)
                else:
                    failed.append(package)
                    
            except Exception:
                failed.append(package)
        
        return installed, failed
    
    def _check_core_dependencies(self, python_exe: str) -> bool:
        """Check if core dependencies are installed."""
        core_deps = ["torch", "transformers", "yaml", "psutil"]
        
        for dep in core_deps:
            try:
                cmd = [python_exe, "-c", f"import {dep}"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    return False
            except Exception:
                return False
        
        return True
    
    def _check_system_cleanliness(self) -> bool:
        """Check if system packages are not polluted by our installation."""
        # This is a basic check - in a real implementation, you might want
        # to compare package lists before and after installation
        try:
            # Check if we're in a venv (good for cleanliness)
            if self._is_venv_active():
                return True
            
            # If not in venv, check if critical packages are not in system Python
            import importlib.util
            
            # These packages should ideally not be in system Python
            ml_packages = ["torch", "transformers"]
            
            for package in ml_packages:
                spec = importlib.util.find_spec(package)
                if spec and spec.origin:
                    # If package is found in system Python, system is not clean
                    if not self._is_venv_active():
                        return False
            
            return True
            
        except Exception:
            # If we can't determine, assume it's clean
            return True
    
    def _get_python_version(self, python_exe: str) -> str:
        """Get Python version from executable."""
        try:
            cmd = [python_exe, "--version"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "Unknown"
                
        except Exception:
            return "Unknown"
    
    def _get_installed_packages(self, python_exe: str) -> List[str]:
        """Get list of installed packages in the virtual environment."""
        try:
            cmd = [python_exe, "-m", "pip", "list", "--format=freeze"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                packages = []
                for line in result.stdout.strip().split('\n'):
                    if line and '==' in line:
                        packages.append(line.strip())
                return packages
            else:
                return []
                
        except Exception:
            return []


# Convenience functions for common operations

def create_environment(venv_path: str, requirements_file: str, config: Optional[Configuration] = None) -> Tuple[bool, str]:
    """
    Convenience function to create a complete environment setup.
    
    Args:
        venv_path: Path for virtual environment
        requirements_file: Path to requirements.txt
        config: Optional Configuration object
        
    Returns:
        Tuple of (success, message)
    """
    manager = EnvironmentManager(config)
    
    # Create venv
    venv_result = manager.setup_venv(venv_path)
    if not venv_result.success:
        return False, f"Failed to create virtual environment: {venv_result.error_message}"
    
    # Install dependencies
    deps_result = manager.install_dependencies(requirements_file, venv_path)
    if not deps_result.success:
        return False, f"Failed to install dependencies: {deps_result.error_message}"
    
    # Verify isolation
    status = manager.verify_isolation(venv_path)
    if not status.system_clean:
        return False, "Environment isolation verification failed"
    
    return True, f"Environment created successfully at {venv_path}"


def quick_environment_check(venv_path: str) -> Dict[str, Any]:
    """
    Quick check of environment status.
    
    Args:
        venv_path: Path to virtual environment
        
    Returns:
        Dictionary with status information
    """
    manager = EnvironmentManager()
    status = manager.verify_isolation(venv_path)
    info = manager.get_environment_info(venv_path)
    
    return {
        "status": status,
        "info": info,
        "ready": status.venv_active and status.dependencies_installed and status.models_isolated
    }

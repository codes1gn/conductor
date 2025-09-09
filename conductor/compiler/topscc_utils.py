"""
TOPSCC Compiler Utilities for GCU Backend.

This module provides utilities for using the topscc compiler toolchain
instead of standard g++/gcc for GCU-specific compilation and linking.
The topscc toolchain is required for proper GCU runtime integration.
"""

import os
import subprocess
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class TopsccEnvironment:
    """Manages topscc compiler environment and configuration."""

    def __init__(self, topscc_install: Optional[str] = None):
        """
        Initialize topscc environment.

        Args:
            topscc_install: Path to topscc installation (default: /opt/tops)
        """
        self.topscc_install = topscc_install or os.environ.get("TOPSCC_INSTALL", "/opt/tops")
        self.topscc_bin = os.path.join(self.topscc_install, "bin", "topscc")
        self.topscc_lib = os.path.join(self.topscc_install, "lib")
        self.include_dirs = [os.path.join(self.topscc_install, "include")]
        self._gcu_arch = None
        self._is_available = None

    def is_available(self) -> bool:
        """Check if topscc toolchain is available."""
        if self._is_available is not None:
            return self._is_available

        try:
            # Check if topscc binary exists and is executable
            if not os.path.isfile(self.topscc_bin) or not os.access(self.topscc_bin, os.X_OK):
                logger.debug(f"topscc binary not found or not executable: {self.topscc_bin}")
                self._is_available = False
                return False

            # Check if topscc lib directory exists
            if not os.path.isdir(self.topscc_lib):
                logger.debug(f"topscc lib directory not found: {self.topscc_lib}")
                self._is_available = False
                return False

            # Try to run topscc --help to verify it works
            result = subprocess.run([self.topscc_bin, "--help"], capture_output=True, timeout=10)

            self._is_available = result.returncode == 0
            if not self._is_available:
                logger.debug(f"topscc --help failed with return code {result.returncode}")

            return self._is_available

        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError) as e:
            logger.debug(f"topscc availability check failed: {e}")
            self._is_available = False
            return False

    def detect_gcu_architecture(self) -> str:
        """
        Detect GCU architecture for compilation flags.

        Returns:
            Architecture string (e.g., 'gcu300', 'gcu200')
        """
        if self._gcu_arch is not None:
            return self._gcu_arch

        # Try to detect from environment variable first
        arch = os.environ.get("GCU_ARCH")
        if arch:
            self._gcu_arch = arch
            return arch

        # Try to detect from system (this is a placeholder - actual detection
        # would depend on GCU hardware detection utilities)
        # For now, default to gcu300
        self._gcu_arch = "gcu300"
        logger.debug(f"Using default GCU architecture: {self._gcu_arch}")
        return self._gcu_arch

    def get_environment(self) -> Dict[str, str]:
        """
        Get environment variables needed for topscc compilation.

        Returns:
            Dictionary of environment variables
        """
        env = os.environ.copy()

        # Set topscc-specific environment variables
        env["TOPSCC_INSTALL"] = self.topscc_install
        env["TOPSCC"] = self.topscc_bin
        env["TOPSCC_LIB"] = self.topscc_lib

        # Update LD_LIBRARY_PATH to include topscc lib
        ld_library_path = env.get("LD_LIBRARY_PATH", "")
        if ld_library_path:
            env["LD_LIBRARY_PATH"] = f"{self.topscc_lib}:{ld_library_path}"
        else:
            env["LD_LIBRARY_PATH"] = self.topscc_lib

        return env

    def get_cflags(self, extra_flags: Optional[List[str]] = None) -> List[str]:
        """
        Get CFLAGS for topscc compilation.

        Args:
            extra_flags: Additional flags to include

        Returns:
            List of compiler flags
        """
        gcu_arch = self.detect_gcu_architecture()
        extra_target_cflags = os.environ.get("EXTRA_TARGET_CFLAGS", "")

        flags = [f"-arch", gcu_arch, "-std=c++17", "-ltops", "-lm", "-O3", "-fPIC", "-v"]

        if extra_target_cflags:
            flags.extend(extra_target_cflags.split())

        if extra_flags:
            flags.extend(extra_flags)

        return flags

    def get_link_command(
        self,
        object_files: List[str],
        output_path: str,
        shared: bool = True,
        extra_flags: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Get topscc link command.

        Args:
            object_files: List of object files to link
            output_path: Output file path
            shared: Whether to create shared library
            extra_flags: Additional flags

        Returns:
            Complete command list for subprocess
        """
        cmd = [self.topscc_bin, "-c"]

        # Add CFLAGS
        cmd.extend(self.get_cflags(extra_flags))

        if shared:
            cmd.append("-shared")

        # Add object files
        cmd.extend(object_files)

        # Add output
        cmd.extend(["-o", output_path])

        return cmd

    def get_compile_command(
        self,
        source_files: List[str],
        output_path: str,
        include_dirs: Optional[List[str]] = None,
        extra_flags: Optional[List[str]] = None,
        host_code: bool = True,
    ) -> List[str]:
        """
        Get topscc compile command.

        Args:
            source_files: List of source files to compile
            output_path: Output file path
            include_dirs: Include directories
            extra_flags: Additional flags
            host_code: Whether this is host code (True) or device code (False)

        Returns:
            Complete command list for subprocess
        """
        if host_code:
            # For host code compilation, use topscc without device flags to get proper GCU runtime symbols
            # This is essential for linking against GCU runtime libraries
            cmd = [self.topscc_bin, "-shared", "-fPIC", "-std=c++17"]

            # Add GCU runtime library path and linking flags
            cmd.extend([f"-L{self.topscc_lib}", "-ltopsrt", "-lrtcu", "-lefrt"])

            # Add runtime library path for execution
            cmd.extend([f"-Wl,-rpath,{self.topscc_lib}"])

            # Add include directories
            if include_dirs:
                for inc_dir in include_dirs:
                    cmd.extend(["-I", inc_dir])

            # Add topscc include directories for GCU runtime headers
            if self.include_dirs:
                for inc_dir in self.include_dirs:
                    cmd.extend(["-I", inc_dir])

            # Add extra flags
            if extra_flags:
                cmd.extend(extra_flags)

            # Add source files
            cmd.extend(source_files)

            # Add output
            cmd.extend(["-o", output_path])
        else:
            # For device code compilation, use topscc
            cmd = [self.topscc_bin, "-c"]

            # Add CFLAGS
            cmd.extend(self.get_cflags(extra_flags))

            # Add include directories
            if include_dirs:
                for inc_dir in include_dirs:
                    cmd.extend(["-I", inc_dir])

            # Add source files
            cmd.extend(source_files)

            # Add output
            cmd.extend(["-o", output_path])

        return cmd


# Global instance for reuse
_global_topscc_env = None


def get_topscc_environment() -> TopsccEnvironment:
    """Get global topscc environment instance."""
    global _global_topscc_env
    if _global_topscc_env is None:
        _global_topscc_env = TopsccEnvironment()
    return _global_topscc_env


def compile_with_topscc(
    source_files: List[str],
    output_path: str,
    include_dirs: Optional[List[str]] = None,
    extra_flags: Optional[List[str]] = None,
    timeout: int = 60,
    host_code: bool = True,
) -> Tuple[bool, str]:
    """
    Compile source files using topscc.

    Args:
        source_files: List of source files to compile
        output_path: Output file path
        include_dirs: Include directories
        extra_flags: Additional compiler flags
        timeout: Compilation timeout in seconds
        host_code: Whether this is host code (True) or device code (False)

    Returns:
        Tuple of (success, error_message)
    """
    topscc_env = get_topscc_environment()

    if not topscc_env.is_available():
        return False, "topscc toolchain not available"

    try:
        cmd = topscc_env.get_compile_command(
            source_files, output_path, include_dirs, extra_flags, host_code
        )
        env = topscc_env.get_environment()

        logger.debug(f"Running topscc compile: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)

        if result.returncode != 0:
            error_msg = f"topscc compilation failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr}"
            if result.stdout:
                error_msg += f"\nStdout: {result.stdout}"
            return False, error_msg

        return True, ""

    except subprocess.TimeoutExpired:
        return False, f"topscc compilation timed out after {timeout} seconds"
    except Exception as e:
        return False, f"topscc compilation failed: {e}"


def link_with_topscc(
    object_files: List[str],
    output_path: str,
    shared: bool = True,
    extra_flags: Optional[List[str]] = None,
    timeout: int = 60,
) -> Tuple[bool, str]:
    """
    Link object files using topscc.

    Args:
        object_files: List of object files to link
        output_path: Output file path
        shared: Whether to create shared library
        extra_flags: Additional linker flags
        timeout: Linking timeout in seconds

    Returns:
        Tuple of (success, error_message)
    """
    topscc_env = get_topscc_environment()

    if not topscc_env.is_available():
        return False, "topscc toolchain not available"

    try:
        cmd = topscc_env.get_link_command(object_files, output_path, shared, extra_flags)
        env = topscc_env.get_environment()

        logger.debug(f"Running topscc link: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)

        if result.returncode != 0:
            error_msg = f"topscc linking failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr}"
            if result.stdout:
                error_msg += f"\nStdout: {result.stdout}"
            return False, error_msg

        return True, ""

    except subprocess.TimeoutExpired:
        return False, f"topscc linking timed out after {timeout} seconds"
    except Exception as e:
        return False, f"topscc linking failed: {e}"

#!/usr/bin/env python3
"""
Debug Artifacts Management for GCU Compilation Pipeline.

This module provides utilities for saving and managing debugging artifacts
during the GCU compilation process.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DebugArtifactManager:
    """Manages debugging artifacts for GCU compilation pipeline."""
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize debug artifact manager.
        
        Args:
            project_root: Project root directory (auto-detected if None)
        """
        if project_root is None:
            # Auto-detect project root by finding conductor directory
            current_dir = Path(__file__).parent
            while current_dir.parent != current_dir:
                if (current_dir / 'conductor').exists():
                    project_root = str(current_dir)
                    break
                current_dir = current_dir.parent
            else:
                # Fallback to current working directory
                project_root = os.getcwd()
        
        self.project_root = Path(project_root)
        self.debug_dir = self.project_root / 'debug_dir'

        # Create debug directory if it doesn't exist
        self.debug_dir.mkdir(exist_ok=True)

        logger.info(f"Debug artifacts will be saved to: {self.debug_dir}")
    
    def get_host_wrapper_path(self, kernel_name: str) -> Path:
        """Get path for host wrapper C++ file."""
        return self.debug_dir / f"host_wrapper_{kernel_name}.cpp"
    
    def get_shared_library_path(self, kernel_name: str) -> Path:
        """Get path for compiled shared library."""
        return self.debug_dir / f"{kernel_name}.so"

    def get_intermediate_object_path(self, kernel_name: str) -> Path:
        """Get path for intermediate object file (compiled from Choreo DSL)."""
        return self.debug_dir / f"{kernel_name}.o"
    
    def get_dsl_source_path(self, kernel_name: str) -> Path:
        """Get path for Choreo DSL source file."""
        return self.debug_dir / f"{kernel_name}.co"
    
    def save_host_wrapper(self, kernel_name: str, wrapper_code: str) -> Path:
        """
        Save host wrapper C++ code to debug directory.
        
        Args:
            kernel_name: Name of the kernel
            wrapper_code: C++ wrapper code content
            
        Returns:
            Path to saved file
        """
        file_path = self.get_host_wrapper_path(kernel_name)
        
        with open(file_path, 'w') as f:
            f.write(wrapper_code)
        
        logger.info(f"Saved host wrapper C++ code: {file_path}")
        return file_path
    

    
    def save_shared_library(self, kernel_name: str, source_path: str) -> Path:
        """
        Save compiled shared library to debug directory.
        
        Args:
            kernel_name: Name of the kernel
            source_path: Path to source shared library
            
        Returns:
            Path to saved file
        """
        dest_path = self.get_shared_library_path(kernel_name)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            logger.info(f"Saved compiled shared library: {dest_path}")
        else:
            logger.warning(f"Source shared library not found: {source_path}")
        
        return dest_path
    
    def save_intermediate_object(self, kernel_name: str, source_path: str) -> Path:
        """
        Save intermediate object file to debug directory.
        
        Args:
            kernel_name: Name of the kernel
            source_path: Path to source object file
            
        Returns:
            Path to saved file
        """
        dest_path = self.get_intermediate_object_path(kernel_name)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            logger.info(f"Saved intermediate object file: {dest_path}")
        else:
            logger.warning(f"Source object file not found: {source_path}")
        
        return dest_path
    
    def save_dsl_source(self, kernel_name: str, dsl_content: str) -> Path:
        """
        Save Choreo DSL source code to debug directory.

        Args:
            kernel_name: Name of the kernel
            dsl_content: Choreo DSL source content

        Returns:
            Path to saved file
        """
        file_path = self.get_dsl_source_path(kernel_name)

        with open(file_path, 'w') as f:
            f.write(dsl_content)

        logger.info(f"Saved Choreo DSL source: {file_path}")
        return file_path

    def save_compilation_dsl_source(self, kernel_name: str, temp_dsl_path: str) -> Path:
        """
        Save the actual DSL file used for compilation to debug directory.

        This preserves the exact file that was passed to the choreo compiler,
        which may be useful for debugging compilation issues.

        Args:
            kernel_name: Name of the kernel
            temp_dsl_path: Path to the temporary DSL file used for compilation

        Returns:
            Path to saved file
        """
        dest_path = self.debug_dir / f"{kernel_name}_compilation.co"

        if os.path.exists(temp_dsl_path):
            shutil.copy2(temp_dsl_path, dest_path)
            logger.info(f"Saved compilation DSL source: {dest_path}")
        else:
            logger.warning(f"Compilation DSL source file not found: {temp_dsl_path}")

        return dest_path
    
    def list_artifacts(self, kernel_name: Optional[str] = None) -> list:
        """
        List all debug artifacts, optionally filtered by kernel name.
        
        Args:
            kernel_name: Optional kernel name to filter by
            
        Returns:
            List of artifact file paths
        """
        if kernel_name:
            pattern = f"*{kernel_name}*"
        else:
            pattern = "*"
        
        artifacts = list(self.debug_dir.glob(pattern))
        artifacts.sort()
        
        return artifacts
    
    def clean_artifacts(self, kernel_name: Optional[str] = None) -> int:
        """
        Clean debug artifacts, optionally filtered by kernel name.
        
        Args:
            kernel_name: Optional kernel name to filter by
            
        Returns:
            Number of files cleaned
        """
        artifacts = self.list_artifacts(kernel_name)
        
        cleaned_count = 0
        for artifact in artifacts:
            try:
                artifact.unlink()
                cleaned_count += 1
                logger.debug(f"Cleaned artifact: {artifact}")
            except Exception as e:
                logger.warning(f"Failed to clean artifact {artifact}: {e}")
        
        logger.info(f"Cleaned {cleaned_count} debug artifacts")
        return cleaned_count
    
    def get_artifact_summary(self, kernel_name: str) -> dict:
        """
        Get summary of artifacts for a specific kernel.
        
        Args:
            kernel_name: Name of the kernel
            
        Returns:
            Dictionary with artifact information
        """
        artifacts = {
            'host_wrapper': self.get_host_wrapper_path(kernel_name),
            'shared_library': self.get_shared_library_path(kernel_name),
            'intermediate_object': self.get_intermediate_object_path(kernel_name),
            'dsl_source': self.get_dsl_source_path(kernel_name),
        }
        
        summary = {}
        for artifact_type, path in artifacts.items():
            summary[artifact_type] = {
                'path': str(path),
                'exists': path.exists(),
                'size': path.stat().st_size if path.exists() else 0,
            }
        
        return summary


# Global debug artifact manager instance
_global_debug_manager = None


def get_debug_manager() -> DebugArtifactManager:
    """Get global debug artifact manager instance."""
    global _global_debug_manager
    if _global_debug_manager is None:
        _global_debug_manager = DebugArtifactManager()
    return _global_debug_manager

"""
GCU Compiled Function - Executable callable for GCU hardware.

This module provides the GCUCompiledFunction class that represents a compiled
function ready for execution on GCU hardware.
"""

import torch
import torch.fx as fx
from typing import Any, Dict


from .loader import CompiledArtifact
from .executor import GCUKernelExecutor
from ..utils.exceptions import ExecutionError
from ..config.logging import get_logger

logger = get_logger(__name__)


class Callable:
    """
    Represents a compiled function that can execute on GCU hardware.

    This class encapsulates a compiled choreo kernel and provides a PyTorch-compatible
    callable interface for execution. It follows Python naming conventions for
    callable objects.
    """
    
    def __init__(self, artifact: CompiledArtifact, original_graph: fx.GraphModule):
        """
        Initialize compiled function.
        
        Args:
            artifact: Compiled choreo artifact
            original_graph: Original FX graph for fallback
        """
        self.artifact = artifact
        self.original_graph = original_graph
        self.executor = GCUKernelExecutor(artifact)
        self._is_loaded = False
        
    def __call__(self, *args, **kwargs) -> Any:
        """
        Execute the compiled function on GCU hardware.
        
        Args:
            *args: Input tensors and arguments
            **kwargs: Keyword arguments
            
        Returns:
            Output tensors from GCU execution
        """
        try:
            # Convert args to list of tensors
            input_tensors = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    input_tensors.append(arg)
                else:
                    # For non-tensor arguments, we might need special handling
                    logger.warning(f"Non-tensor argument ignored: {type(arg)}")
            
            if not input_tensors:
                raise ExecutionError("No tensor inputs provided for GCU execution")
            
            # Execute on GCU
            if not self._is_loaded:
                self.executor.load_kernel()
                self._is_loaded = True
            
            outputs = self.executor.execute(input_tensors)
            
            # PyTorch's FX graph always expects tuple returns for consistency
            # We need to return a tuple to match the FX graph's expectations
            if len(outputs) == 1:
                # Return tuple with single element to match FX graph format
                return (outputs[0],)
            else:
                return tuple(outputs)
                
        except Exception as e:
            logger.error(f"GCU execution failed: {e}")

            # Provide additional debugging information for runtime check failures
            error_msg = str(e)
            if "choreo runtime check failed" in error_msg or "zero is detected" in error_msg:
                debug_info = self._get_debug_info_for_runtime_error(error_msg)
                logger.error(f"Runtime check failure debug info:\n{debug_info}")
                raise ExecutionError(f"GCU execution failed with runtime check error: {e}\n\nDebug Info:\n{debug_info}")

            # For now, re-raise the error. In production, we might want to fallback
            raise ExecutionError(f"GCU execution failed: {e}")
    
    def _get_debug_info_for_runtime_error(self, error_msg: str) -> str:
        """
        Generate debug information for runtime check failures.

        Args:
            error_msg: The runtime error message

        Returns:
            Formatted debug information string
        """
        debug_lines = []
        debug_lines.append("=== GCU Runtime Check Failure Debug Information ===")
        debug_lines.append(f"Kernel: {self.artifact.entry_point}")
        debug_lines.append(f"Artifact Path: {self.artifact.path}")

        # Extract file and line information from error message
        if ".co:" in error_msg:
            import re
            match = re.search(r'([^:]+\.co):(\d+)\.(\d+)', error_msg)
            if match:
                dsl_file, line_num, col_num = match.groups()
                debug_lines.append(f"Error Location: {dsl_file}:{line_num}.{col_num}")
                debug_lines.append(f"DSL File: {dsl_file}")

        # Show available debug files
        if 'debug_dsl_path' in self.artifact.metadata:
            debug_lines.append(f"Debug DSL Source: {self.artifact.metadata['debug_dsl_path']}")

        if 'compilation_dsl_path' in self.artifact.metadata:
            debug_lines.append(f"Compilation DSL Source: {self.artifact.metadata['compilation_dsl_path']}")

        if 'object_file_path' in self.artifact.metadata:
            debug_lines.append(f"Object File: {self.artifact.metadata['object_file_path']}")

        # Show DSL content snippet if available
        if 'dsl_content' in self.artifact.metadata:
            dsl_content = self.artifact.metadata['dsl_content']
            debug_lines.append("\n=== DSL Source Code ===")
            lines = dsl_content.split('\n')
            for i, line in enumerate(lines[:20], 1):  # Show first 20 lines
                debug_lines.append(f"{i:2d}: {line}")
            if len(lines) > 20:
                debug_lines.append(f"... ({len(lines) - 20} more lines)")

        return '\n'.join(debug_lines)

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the compiled function."""
        return {
            'artifact_path': self.artifact.path,
            'entry_point': self.artifact.entry_point,
            'is_loaded': self._is_loaded,
            'metadata': self.artifact.metadata
        }

"""
Artifact loading and execution.

This module provides classes for loading and executing compiled artifacts,
including both shared libraries and object files.
"""

import ctypes
import os
from typing import Any, List, Optional
from dataclasses import dataclass


@dataclass
class CompiledArtifact:
    """
    Represents a compiled artifact with metadata.
    
    This class encapsulates information about a compiled artifact,
    including its path, type, and execution interface.
    """
    path: str                           # Path to the compiled artifact
    artifact_type: str                  # Type: 'shared_library' or 'object_file'
    entry_point: str                    # Main execution function name
    metadata: dict                      # Additional artifact metadata
    
    def is_valid(self) -> bool:
        """Check if artifact file exists and is accessible."""
        return os.path.exists(self.path) and os.access(self.path, os.R_OK)


class ExecutableKernel:
    """
    Wrapper for loaded executable kernels.
    
    This class provides a unified interface for executing compiled
    kernels regardless of their underlying format (shared library, etc.).
    """
    
    def __init__(self, library_handle: Any, entry_function: Any):
        """
        Initialize executable kernel.
        
        Args:
            library_handle: Handle to loaded library
            entry_function: Function pointer to kernel entry point
        """
        self._library = library_handle
        self._entry_function = entry_function
        self._is_loaded = True
        
    @classmethod
    def load_from_file(cls, artifact_path: str) -> 'ExecutableKernel':
        """
        Load executable kernel from compiled artifact.
        
        Args:
            artifact_path: Path to compiled artifact (.so or .o file)
            
        Returns:
            ExecutableKernel instance ready for execution
            
        Raises:
            RuntimeError: If loading fails
        """
        if not os.path.exists(artifact_path):
            raise RuntimeError(f"Artifact not found: {artifact_path}")
            
        if not os.access(artifact_path, os.R_OK):
            raise RuntimeError(f"Artifact not readable: {artifact_path}")
            
        try:
            if artifact_path.endswith('.so'):
                # Load shared library using ctypes
                lib = ctypes.CDLL(artifact_path)
                
                # Try to get the standard entry function
                entry_func = None
                entry_function_names = [
                    'conductor_kernel_main',
                    'kernel_main', 
                    'main',
                    'execute'
                ]
                
                for func_name in entry_function_names:
                    try:
                        entry_func = getattr(lib, func_name)
                        break
                    except AttributeError:
                        continue
                
                if entry_func is None:
                    # Try to find any exported function
                    # This is a fallback - in practice, we'd have a known interface
                    raise RuntimeError(
                        f"No suitable entry function found in {artifact_path}. "
                        f"Expected one of: {', '.join(entry_function_names)}"
                    )
                
                # Set up function signature if possible
                # This would typically be defined by the Conductor compiler interface
                try:
                    # Example signature: int function(float* inputs, float* outputs, int* shapes)
                    entry_func.argtypes = [
                        ctypes.POINTER(ctypes.c_float),  # inputs
                        ctypes.POINTER(ctypes.c_float),  # outputs  
                        ctypes.POINTER(ctypes.c_int)     # shapes/metadata
                    ]
                    entry_func.restype = ctypes.c_int  # return code
                except:
                    # If signature setup fails, continue without it
                    # The function will still be callable but with less type safety
                    pass
                    
                return cls(lib, entry_func)
                
            elif artifact_path.endswith('.o'):
                # Object files need to be linked into a shared library first
                # This is more complex and would require a linker step
                raise NotImplementedError(
                    "Object file loading not yet implemented. "
                    "Object files must be linked into shared libraries first."
                )
                
            else:
                raise RuntimeError(
                    f"Unsupported artifact type: {artifact_path}. "
                    f"Supported types: .so (shared library), .o (object file - not yet implemented)"
                )
                
        except OSError as e:
            # This catches ctypes.CDLL loading errors
            raise RuntimeError(f"Failed to load shared library {artifact_path}: {e}")
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Unexpected error loading kernel from {artifact_path}: {e}")
            
    def execute(self, inputs: List[Any]) -> List[Any]:
        """
        Execute the kernel with given inputs.
        
        Args:
            inputs: List of input tensors/data
            
        Returns:
            List of output tensors/data
            
        Raises:
            RuntimeError: If execution fails
        """
        if not self._is_loaded:
            raise RuntimeError("Kernel is not loaded")
            
        if not inputs:
            raise RuntimeError("No inputs provided for kernel execution")
        
        try:
            # Convert inputs to ctypes-compatible format
            # This is a simplified implementation - real implementation would
            # need to handle various tensor types, shapes, and memory layouts
            
            # For now, assume inputs are PyTorch tensors
            import torch
            
            input_arrays = []
            input_shapes = []
            
            for i, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor):
                    # Convert tensor to contiguous float array
                    if inp.dtype != torch.float32:
                        inp = inp.float()
                    
                    if not inp.is_contiguous():
                        inp = inp.contiguous()
                    
                    # Get data pointer
                    data_ptr = inp.data_ptr()
                    array = (ctypes.c_float * inp.numel()).from_address(data_ptr)
                    input_arrays.append(array)
                    input_shapes.extend(inp.shape)
                else:
                    raise RuntimeError(f"Unsupported input type: {type(inp)}")
            
            # Prepare output buffers (this would need to be determined from the kernel metadata)
            # For now, assume single output with same shape as first input
            if inputs and isinstance(inputs[0], torch.Tensor):
                output_tensor = torch.zeros_like(inputs[0])
                output_array = (ctypes.c_float * output_tensor.numel()).from_address(
                    output_tensor.data_ptr()
                )
            else:
                raise RuntimeError("Cannot determine output shape without tensor inputs")
            
            # Prepare shapes array
            shapes_array = (ctypes.c_int * len(input_shapes))(*input_shapes)
            
            # Call the kernel function
            # This assumes the function signature: int func(float* inputs, float* outputs, int* shapes)
            if len(input_arrays) == 1:
                result_code = self._entry_function(
                    input_arrays[0],
                    output_array,
                    shapes_array
                )
            else:
                # For multiple inputs, we'd need a different calling convention
                # This is a limitation of the current simplified implementation
                raise RuntimeError("Multiple input execution not yet fully implemented")
            
            # Check result code
            if result_code != 0:
                raise RuntimeError(f"Kernel execution failed with code: {result_code}")
            
            return [output_tensor]
            
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Kernel execution failed: {e}")
        
    def unload(self) -> None:
        """
        Unload the kernel and free resources.
        """
        if self._is_loaded:
            # TODO: Implement proper cleanup
            self._library = None
            self._entry_function = None
            self._is_loaded = False
            
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.unload()
        
    def get_metadata(self) -> dict:
        """
        Get kernel metadata and information.
        
        Returns:
            Dictionary containing kernel metadata
        """
        return {
            'loaded': self._is_loaded,
            'library': str(self._library) if self._library else None,
            'entry_function': str(self._entry_function) if self._entry_function else None
        }
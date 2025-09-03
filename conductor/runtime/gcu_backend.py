"""
GCU Backend Integration.

This module provides the high-level integration between PyTorch and GCU hardware
execution, coordinating the compilation pipeline and device execution.
"""

import torch
import torch.fx as fx
from typing import List, Any, Dict, Optional, Callable
import tempfile
import os

from .choreo_jit import ChoreoJITCompiler
from .gcu_executor import GCUKernelExecutor
from .loader import CompiledArtifact
from ..utils.exceptions import DeviceError, ExecutionError, CompilationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class GCUCompiledFunction:
    """
    Represents a compiled function that can execute on GCU hardware.
    
    This class encapsulates a compiled choreo kernel and provides a PyTorch-compatible
    interface for execution.
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

        debug_lines.append("\n=== Troubleshooting Tips ===")
        debug_lines.append("1. Check the DSL source files in the debug directory")
        debug_lines.append("2. Verify tensor dimensions match the DSL expectations")
        debug_lines.append("3. Look for zero dimensions in mdspan declarations")
        debug_lines.append("4. Check if the input tensor shapes are correct")

        return '\n'.join(debug_lines)

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the compiled function."""
        return {
            'artifact_path': self.artifact.path,
            'entry_point': self.artifact.entry_point,
            'is_loaded': self._is_loaded,
            'metadata': self.artifact.metadata
        }


class GCUBackend:
    """
    Main GCU backend that integrates with PyTorch's compilation system.
    
    This class provides the primary interface for compiling and executing
    PyTorch models on GCU hardware using the choreo compiler.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize GCU backend.
        
        Args:
            cache_dir: Optional directory for compilation cache
        """
        self.jit_compiler = ChoreoJITCompiler(cache_dir)
        self._compiled_functions = {}  # Cache of compiled functions
        
    def compile_graph(self, graph_module: fx.GraphModule, example_inputs: List[torch.Tensor]) -> GCUCompiledFunction:
        """
        Compile a PyTorch FX graph for GCU execution.
        
        Args:
            graph_module: FX graph to compile
            example_inputs: Example inputs for shape inference
            
        Returns:
            GCUCompiledFunction ready for execution
            
        Raises:
            CompilationError: If compilation fails
        """
        logger.info(f"Compiling graph for GCU execution: {len(list(graph_module.graph.nodes))} nodes")
        
        try:
            # Generate a hash for caching
            graph_hash = self._generate_graph_hash(graph_module, example_inputs)
            
            # Check if already compiled
            if graph_hash in self._compiled_functions:
                logger.debug("Using cached compiled function")
                return self._compiled_functions[graph_hash]
            
            # Compile using choreo JIT compiler with example inputs for shape inference
            artifact = self.jit_compiler.compile_graph(graph_module, example_inputs)
            
            # Create compiled function
            compiled_func = GCUCompiledFunction(artifact, graph_module)
            
            # Cache the result
            self._compiled_functions[graph_hash] = compiled_func
            
            logger.info(f"Successfully compiled graph for GCU: {artifact.entry_point}")
            return compiled_func
            
        except Exception as e:
            logger.error(f"GCU compilation failed: {e}")
            raise CompilationError(f"Failed to compile graph for GCU: {e}")
    
    def _generate_graph_hash(self, graph_module: fx.GraphModule, example_inputs: List[torch.Tensor]) -> str:
        """Generate a hash for the graph and inputs for caching."""
        import hashlib
        
        # Create hash from graph structure and input shapes/dtypes
        hasher = hashlib.md5()
        
        # Add graph structure
        graph_str = str(graph_module.graph)
        hasher.update(graph_str.encode())
        
        # Add input shapes and dtypes
        for inp in example_inputs:
            hasher.update(str(inp.shape).encode())
            hasher.update(str(inp.dtype).encode())
        
        return hasher.hexdigest()
    
    def clear_cache(self) -> None:
        """Clear the compilation cache."""
        for func in self._compiled_functions.values():
            try:
                func.executor.unload()
            except:
                pass
        
        self._compiled_functions.clear()
        logger.info("GCU backend cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        return {
            'compiled_functions': len(self._compiled_functions),
            'cache_hits': 0,  # TODO: Implement cache hit tracking
            'compilation_failures': 0,  # TODO: Implement failure tracking
        }


def gcu_compile_function(graph_module: fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    """
    Compile a PyTorch FX graph for GCU execution.
    
    This is the main entry point used by PyTorch's compilation system.
    
    Args:
        graph_module: FX graph to compile
        example_inputs: Example inputs for compilation
        
    Returns:
        Callable that executes on GCU hardware
    """
    # Create backend instance (could be cached globally)
    backend = GCUBackend()
    
    # Compile the graph
    compiled_func = backend.compile_graph(graph_module, example_inputs)
    
    return compiled_func


# Register the GCU backend with PyTorch
def register_gcu_backend():
    """Register the GCU backend with PyTorch's compilation system."""
    try:
        # Register as a custom backend with a different name to avoid conflicts
        torch._dynamo.register_backend("gcu_direct", gcu_compile_function)
        logger.info("GCU backend registered successfully as 'gcu_direct'")
        return True
    except Exception as e:
        logger.error(f"Failed to register GCU backend: {e}")
        return False


# Global backend instance for reuse
_global_backend = None

def get_gcu_backend() -> GCUBackend:
    """Get the global GCU backend instance."""
    global _global_backend
    if _global_backend is None:
        _global_backend = GCUBackend()
    return _global_backend


def test_gcu_execution(model: torch.nn.Module, example_inputs: List[torch.Tensor]) -> bool:
    """
    Test GCU execution with a simple model.
    
    Args:
        model: PyTorch model to test
        example_inputs: Example inputs for testing
        
    Returns:
        True if test passes, False otherwise
    """
    try:
        # Compile model for GCU
        compiled_model = torch.compile(model, backend="gcu")
        
        # Run inference
        with torch.no_grad():
            output = compiled_model(*example_inputs)
        
        logger.info("GCU execution test passed")
        return True
        
    except Exception as e:
        logger.error(f"GCU execution test failed: {e}")
        return False


# Note: Backend registration is now handled by conductor/backend.py
# This legacy registration is disabled to avoid conflicts

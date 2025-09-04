"""
Consolidated GCU Backend for PyTorch torch.compile integration.

This module provides the main GCU backend that integrates with PyTorch's compilation system
to provide GCU hardware acceleration through the Conductor framework.
"""

import torch
import torch.fx as fx
from typing import Callable, Any, Optional, List, Dict
import logging
import warnings
import sys

from .choreo_jit import ChoreoJITCompiler
from .loader import CompiledArtifact
from .callable import Callable
from .exceptions import (
    ConductorError,
    CompilationError,
    DeviceError,
    ExecutionError
)
from .logging import get_logger

logger = get_logger(__name__)


class GCUInductorBackendImpl:
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
        self.enable_fallback = True
        
    def compile_graph(self, graph_module: fx.GraphModule, example_inputs: Optional[List[torch.Tensor]] = None) -> Callable:
        """
        Compile a PyTorch FX Graph for execution on GCU hardware.
        
        Args:
            graph_module: The FX Graph representation of the model
            example_inputs: Sample inputs for shape inference and validation
            
        Returns:
            Compiled callable that executes on GCU hardware
            
        Raises:
            CompilationError: If compilation fails
        """
        try:
            logger.debug(f"Compiling FX Graph with {len(list(graph_module.graph.nodes))} nodes for GCU")
            
            # Generate a hash for caching
            graph_hash = self._compute_graph_hash(graph_module, example_inputs)
            
            # Check cache first
            if graph_hash in self._compiled_functions:
                logger.debug("Using cached compiled function")
                return self._compiled_functions[graph_hash]
            
            # Compile using JIT compiler
            artifact = self.jit_compiler.compile_graph(graph_module, example_inputs)
            
            # Create compiled function
            compiled_fn = Callable(artifact, graph_module)
            
            # Cache the result
            self._compiled_functions[graph_hash] = compiled_fn
            
            logger.info("GCU compilation successful")
            return compiled_fn
            
        except Exception as e:
            logger.warning(f"GCU compilation failed: {e}")
            
            if self.enable_fallback:
                logger.info("Falling back to PyTorch eager execution")
                return graph_module.forward
            else:
                raise CompilationError(f"GCU compilation failed: {e}") from e
    
    def _compute_graph_hash(self, graph_module: fx.GraphModule, example_inputs: Optional[List[torch.Tensor]] = None) -> str:
        """Compute hash for graph caching."""
        import hashlib
        
        # Create hash from graph structure
        graph_str = str(graph_module.graph)
        
        # Include input shapes if available
        if example_inputs:
            shapes_str = str([tuple(inp.shape) for inp in example_inputs])
            graph_str += shapes_str
        
        return hashlib.md5(graph_str.encode()).hexdigest()
    
    def enable_fallback_mechanism(self, enable: bool = True) -> None:
        """
        Enable or disable fallback to PyTorch eager execution.
        
        Args:
            enable: Whether to enable fallback
        """
        self.enable_fallback = enable
        logger.info(f"Fallback mechanism {'enabled' if enable else 'disabled'}")


class GCUInductorBackend:
    """
    PyTorch torch.compile integration wrapper for GCU hardware acceleration.

    This class serves as the PyTorch-specific integration layer that wraps the core
    GCUInductorBackendImpl implementation. It handles torch.compile registration and provides
    the interface expected by PyTorch's compilation system.

    The relationship is:
    - GCUInductorBackendImpl: Core GCU compilation and execution implementation
    - GCUInductorBackend: PyTorch torch.compile integration wrapper
    """

    def __init__(self):
        self.name = "gcu"
        self._registered = False
        self.gcu_backend = get_global_gcu_backend()  # Use singleton
        self.enable_fallback = True
    
    def __call__(self, graph_module: torch.fx.GraphModule, example_inputs) -> Callable:
        """
        Compile a PyTorch FX Graph for execution on GCU hardware.

        Args:
            graph_module: The FX Graph representation of the model
            example_inputs: Sample inputs for shape inference and validation

        Returns:
            Compiled callable that executes on GCU hardware

        Raises:
            ConductorError: If compilation fails and fallback is disabled
        """
        try:
            logger.debug(f"Compiling FX Graph with {len(list(graph_module.graph.nodes))} nodes for GCU")

            # Compile directly using GCU backend
            compiled_fn = self.gcu_backend.compile_graph(graph_module, example_inputs)
            logger.info("GCU compilation successful")
            return compiled_fn

        except Exception as e:
            logger.warning(f"GCU compilation failed: {e}")

            if self.enable_fallback:
                logger.info("Falling back to PyTorch eager execution")
                return graph_module.forward
            else:
                raise ConductorError(f"GCU compilation failed: {e}") from e
    
    def enable_fallback_mechanism(self, enable: bool = True) -> None:
        """
        Enable or disable fallback to PyTorch eager execution.

        Args:
            enable: Whether to enable fallback
        """
        self.enable_fallback = enable
        logger.info(f"Fallback mechanism {'enabled' if enable else 'disabled'}")


# Singleton instance management
_global_gcu_backend: Optional[GCUInductorBackendImpl] = None


def get_global_gcu_backend() -> GCUInductorBackendImpl:
    """
    Get the global GCU backend instance using the singleton pattern.

    This function implements the singleton pattern to ensure only one GCU backend
    instance exists throughout the application lifecycle. The singleton is created
    lazily on first access and reused for all subsequent calls.

    Returns:
        GCUInductorBackendImpl: The global singleton GCU backend instance
    """
    global _global_gcu_backend

    if _global_gcu_backend is None:
        _global_gcu_backend = GCUInductorBackendImpl()

    return _global_gcu_backend


def register_backend() -> None:
    """
    Register the GCU backend with PyTorch's torch.compile system.

    This function creates a GCUInductorBackend (PyTorch integration wrapper) that
    internally uses the core GCUInductorBackendImpl implementation. The wrapper handles
    PyTorch-specific concerns while delegating actual compilation to GCUInductorBackendImpl.

    Raises:
        RuntimeError: If backend registration fails
    """
    try:
        # Check PyTorch version compatibility
        torch_version = torch.__version__
        major, minor = map(int, torch_version.split('.')[:2])

        if major < 2:
            raise RuntimeError(f"PyTorch 2.0+ required for torch.compile, found {torch_version}")

        # Create PyTorch integration backend instance
        backend = GCUInductorBackend()

        # Register with PyTorch's backend system
        if hasattr(torch._dynamo, 'register_backend'):
            torch._dynamo.register_backend(name="gcu", compiler_fn=backend)
            backend._registered = True
            logger.info(f"Successfully registered GCU backend for torch.compile (PyTorch {torch_version})")
        else:
            raise RuntimeError(f"PyTorch {torch_version} does not support backend registration")

    except Exception as e:
        error_msg = f"Failed to register GCU backend: {e}"
        logger.error(error_msg)

        # Issue warning instead of raising exception to allow package import
        warnings.warn(error_msg, UserWarning)


def is_backend_registered() -> bool:
    """
    Check if the Conductor backend is successfully registered.
    
    Returns:
        True if backend is registered, False otherwise
    """
    try:
        # Check if backend is in the list of available backends
        if hasattr(torch._dynamo, 'list_backends'):
            backends = torch._dynamo.list_backends()
            return 'gcu' in backends
        return False
    except Exception:
        return False


def get_backend_info() -> dict:
    """
    Get information about the registered GCU backend.

    Returns:
        Dictionary with backend information
    """
    info = {
        'name': 'gcu',
        'registered': is_backend_registered(),
        'pytorch_version': torch.__version__,
        'python_version': sys.version,
        'integration': 'torch.compile with inductor backend'
    }

    if info['registered']:
        try:
            # Get additional backend details
            backends = torch._dynamo.list_backends()
            if 'gcu' in backends:
                info['backend_function'] = str(backends['gcu'])
        except Exception:
            pass

    return info


def list_supported_operations() -> List[str]:
    """
    List operations supported by the GCU backend.

    Returns:
        List of supported operation names
    """
    return [
        # Elementwise operations (primary focus)
        'add', 'sub', 'mul', 'div',
        'relu', 'sigmoid', 'tanh',
        'abs', 'neg', 'exp', 'log', 'sqrt',

        # Basic tensor operations
        'reshape', 'transpose', 'permute',
    ]

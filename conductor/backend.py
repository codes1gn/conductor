"""
PyTorch torch.compile backend integration for Conductor GCU acceleration.

This module provides a streamlined interface for PyTorch's torch.compile system
to use GCU hardware acceleration through the Conductor framework.
"""

import torch
from typing import Callable, Any, Optional, List
import logging
import warnings
import sys

from .runtime.gcu_backend import GCUBackend
from .utils.exceptions import (
    ConductorError,
    CompilationError,
    DeviceError
)

logger = logging.getLogger(__name__)


class GCUInductorBackend:
    """
    Streamlined GCU backend for PyTorch's torch.compile system.

    This backend integrates with PyTorch's inductor backend to provide
    GCU hardware acceleration through the Conductor framework.
    """

    def __init__(self):
        self.name = "gcu"
        self._registered = False

        # Initialize GCU backend for hardware execution
        self.gcu_backend = GCUBackend()

        # Simple configuration
        self.enable_fallback = True
    
    def __call__(self, graph_module: torch.fx.GraphModule, example_inputs) -> Callable:
        """
        Compile a PyTorch FX Graph for execution on GCU hardware.

        This method integrates with PyTorch's torch.compile system to provide
        GCU acceleration through the Conductor framework.

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


def register_backend() -> None:
    """
    Register the GCU backend with PyTorch's torch.compile system.

    This function makes the 'gcu' backend available for use with torch.compile.
    It integrates with PyTorch's inductor backend for streamlined compilation.

    Raises:
        RuntimeError: If backend registration fails
    """
    try:
        # Check PyTorch version compatibility
        torch_version = torch.__version__
        major, minor = map(int, torch_version.split('.')[:2])

        if major < 2:
            raise RuntimeError(f"PyTorch 2.0+ required for torch.compile, found {torch_version}")

        # Create backend instance
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


# Global backend instance for configuration
_global_backend: Optional[GCUInductorBackend] = None


def get_backend() -> Optional[GCUInductorBackend]:
    """
    Get the global GCU backend instance.

    Returns:
        GCUInductorBackend instance if registered, None otherwise
    """
    global _global_backend

    if _global_backend is None and is_backend_registered():
        # Try to get the backend from PyTorch's registry
        try:
            backends = torch._dynamo.list_backends()
            if 'gcu' in backends:
                backend_fn = backends['gcu']
                if isinstance(backend_fn, GCUInductorBackend):
                    _global_backend = backend_fn
        except Exception:
            pass

    return _global_backend


def configure_backend(**kwargs) -> None:
    """
    Configure the global GCU backend instance.

    Args:
        **kwargs: Configuration options (enable_fallback)
    """
    backend = get_backend()
    if backend is None:
        raise RuntimeError("GCU backend not registered. Call register_backend() first.")

    if 'enable_fallback' in kwargs:
        backend.enable_fallback_mechanism(kwargs['enable_fallback'])
"""
PyTorch backend registration and integration for Conductor.

This module provides the main interface between PyTorch's compilation system
and the Conductor compiler backend.
"""

import torch
from typing import Callable, Any, Optional, List
import logging
import warnings
import sys

from .codegen.graph import GraphAnalyzer
from .codegen.fusion import FusionEngine
from .codegen.dsl import DSLGenerator
from .runtime.jit import JITCompiler
from .runtime.aot import AOTManager
from .utils.exceptions import (
    ConductorError, 
    UnsupportedOperationError, 
    CompilationError,
    DeviceError,
    get_fallback_handler
)

logger = logging.getLogger(__name__)


class ConductorBackend:
    """
    Main backend class that handles PyTorch FX Graph compilation.
    
    This class serves as the entry point for PyTorch's torch.compile system
    and orchestrates the conversion from FX Graph to executable GCU code.
    """
    
    def __init__(self):
        self.name = "gcu"
        self._registered = False
        
        # Initialize compilation pipeline components
        self.graph_analyzer = GraphAnalyzer()
        self.fusion_engine = FusionEngine()
        self.dsl_generator = DSLGenerator()
        self.jit_compiler = JITCompiler()
        self.aot_manager = AOTManager()
        
        # Configuration options
        self.enable_fusion = True
        self.enable_fallback = True
        self.compilation_mode = "jit"  # "jit" or "aot"
        
        # Fallback handler
        self.fallback_handler = get_fallback_handler()
    
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
            logger.debug(f"Compiling FX Graph with {len(list(graph_module.graph.nodes))} nodes")
            
            # Step 1: Parse FX Graph to internal DAG representation
            dag = self.graph_analyzer.parse_fx_graph(graph_module)
            
            # Step 2: Validate graph correctness
            if not self.graph_analyzer.validate_graph_correctness(dag):
                raise CompilationError("Graph validation failed", "", "Invalid graph structure")
            
            # Step 3: Apply fusion optimizations if enabled
            if self.enable_fusion:
                fusion_clusters = self.fusion_engine.identify_fusion_opportunities(dag)
                logger.debug(f"Identified {len(fusion_clusters)} fusion opportunities")
            
            # Step 4: Generate Conductor DSL
            dsl_code = self.dsl_generator.generate_dsl_file(dag)
            logger.debug(f"Generated DSL code ({len(dsl_code)} characters)")
            
            # Step 5: Compile based on mode
            if self.compilation_mode == "aot":
                # Try AOT mode first
                try:
                    compiled_fn = self._compile_aot_mode(graph_module, dsl_code, example_inputs)
                    if compiled_fn:
                        return compiled_fn
                except Exception as e:
                    logger.warning(f"AOT compilation failed, falling back to JIT: {e}")
            
            # JIT mode (default or fallback from AOT)
            return self._compile_jit_mode(graph_module, dsl_code, example_inputs)
            
        except (UnsupportedOperationError, CompilationError, DeviceError) as e:
            logger.warning(f"Conductor compilation failed: {e}")
            if self.enable_fallback and self.fallback_handler.should_fallback(e):
                return self._execute_fallback(graph_module, example_inputs, str(e))
            else:
                raise
        
        except Exception as e:
            logger.error(f"Unexpected compilation error: {e}")
            if self.enable_fallback:
                return self._execute_fallback(graph_module, example_inputs, f"Unexpected error: {type(e).__name__}")
            else:
                raise ConductorError(f"Backend compilation failed: {e}")
    
    def _compile_jit_mode(self, graph_module: torch.fx.GraphModule, dsl_code: str, example_inputs) -> Callable:
        """
        Compile using JIT mode.
        
        Args:
            graph_module: FX Graph module
            dsl_code: Generated DSL code
            example_inputs: Example inputs for validation
            
        Returns:
            Compiled callable
        """
        try:
            # Use JIT compiler to generate executable
            compiled_artifact = self.jit_compiler.compile_graph(graph_module)
            
            # Create wrapper function that executes the compiled kernel
            def compiled_fn(*args):
                # TODO: Implement actual kernel execution in task 5.1
                # For now, return a placeholder that maintains tensor shapes
                if args and hasattr(args[0], 'shape'):
                    # Return tensor with same shape as first input (placeholder)
                    return torch.zeros_like(args[0])
                return args[0] if args else torch.tensor(0.0)
            
            logger.info("JIT compilation successful")
            return compiled_fn
            
        except Exception as e:
            raise CompilationError(f"JIT compilation failed: {e}", dsl_code, str(e))
    
    def _compile_aot_mode(self, graph_module: torch.fx.GraphModule, dsl_code: str, example_inputs) -> Optional[Callable]:
        """
        Compile using AOT mode.
        
        Args:
            graph_module: FX Graph module
            dsl_code: Generated DSL code
            example_inputs: Example inputs for validation
            
        Returns:
            Compiled callable if successful, None if artifacts not found
        """
        try:
            # Generate graph signature for artifact lookup
            graph_signature = self._generate_graph_signature(graph_module)
            
            # Try to locate precompiled artifact
            artifact_path = self.aot_manager.locate_precompiled_artifact(graph_signature)
            
            if artifact_path:
                # Validate compatibility
                if self.aot_manager.validate_artifact_compatibility(artifact_path, graph_module):
                    # Load and return executable
                    executable = self.aot_manager.load_static_artifact(artifact_path)
                    
                    def compiled_fn(*args):
                        # TODO: Implement actual kernel execution in task 6.1
                        # For now, return a placeholder
                        if args and hasattr(args[0], 'shape'):
                            return torch.zeros_like(args[0])
                        return args[0] if args else torch.tensor(0.0)
                    
                    logger.info("AOT compilation successful")
                    return compiled_fn
                else:
                    logger.warning("Precompiled artifact incompatible with current graph")
            else:
                logger.debug("No precompiled artifact found for graph signature")
            
            return None
            
        except Exception as e:
            logger.warning(f"AOT mode failed: {e}")
            return None
    
    def _execute_fallback(self, graph_module: torch.fx.GraphModule, example_inputs, reason: str) -> Callable:
        """
        Execute fallback using the fallback handler.
        
        Args:
            graph_module: FX Graph module
            example_inputs: Example inputs
            reason: Reason for fallback
            
        Returns:
            Compiled function using fallback backend
        """
        try:
            return self.fallback_handler.execute_fallback(graph_module, reason)
        except Exception as e:
            logger.error(f"Fallback execution failed: {e}")
            # Last resort - return original function
            logger.warning("Using eager execution as final fallback")
            return graph_module.forward
    
    def _fallback_to_inductor(self, graph_module: torch.fx.GraphModule, example_inputs) -> Callable:
        """
        Fallback to PyTorch's Inductor backend.
        
        Args:
            graph_module: FX Graph module
            example_inputs: Example inputs
            
        Returns:
            Compiled function using Inductor backend
        """
        return self._execute_fallback(graph_module, example_inputs, "Manual fallback request")
    
    def _generate_graph_signature(self, graph_module: torch.fx.GraphModule) -> str:
        """
        Generate a unique signature for the graph for AOT artifact lookup.
        
        Args:
            graph_module: FX Graph module
            
        Returns:
            Unique graph signature string
        """
        import hashlib
        
        # Create signature from graph structure and operations
        graph_str = ""
        for node in graph_module.graph.nodes:
            graph_str += f"{node.op}:{node.target}:{len(node.args)}:{len(node.kwargs)};"
        
        # Add code hash for additional uniqueness
        if hasattr(graph_module, 'code'):
            graph_str += str(graph_module.code)
        
        return hashlib.sha256(graph_str.encode()).hexdigest()
    
    def set_compilation_mode(self, mode: str) -> None:
        """
        Set the compilation mode.
        
        Args:
            mode: Either "jit" or "aot"
        """
        if mode not in ("jit", "aot"):
            raise ValueError("Mode must be 'jit' or 'aot'")
        self.compilation_mode = mode
        logger.info(f"Compilation mode set to {mode}")
    
    def enable_fusion_optimization(self, enable: bool = True) -> None:
        """
        Enable or disable fusion optimization.
        
        Args:
            enable: Whether to enable fusion
        """
        self.enable_fusion = enable
        logger.info(f"Fusion optimization {'enabled' if enable else 'disabled'}")
    
    def enable_fallback_mechanism(self, enable: bool = True) -> None:
        """
        Enable or disable fallback to Inductor.
        
        Args:
            enable: Whether to enable fallback
        """
        self.enable_fallback = enable
        logger.info(f"Fallback mechanism {'enabled' if enable else 'disabled'}")
    
    def get_fallback_stats(self) -> dict:
        """
        Get fallback usage statistics.
        
        Returns:
            Dictionary with fallback statistics
        """
        return self.fallback_handler.get_fallback_stats()
    
    def reset_fallback_stats(self) -> None:
        """Reset fallback statistics."""
        self.fallback_handler.reset_stats()
        logger.info("Fallback statistics reset")


def register_backend() -> None:
    """
    Register the Conductor backend with PyTorch's compilation system.
    
    This function makes the 'gcu' backend available for use with torch.compile.
    It should be called automatically when the conductor package is imported.
    
    Raises:
        RuntimeError: If backend registration fails
    """
    try:
        # Check PyTorch version compatibility
        torch_version = torch.__version__
        major, minor = map(int, torch_version.split('.')[:2])
        
        if major < 2:
            raise RuntimeError(f"PyTorch 2.0+ required, found {torch_version}")
        
        # Create backend instance
        backend = ConductorBackend()
        
        # Register with PyTorch's backend system
        if hasattr(torch._dynamo, 'register_backend'):
            torch._dynamo.register_backend(name="gcu", compiler_fn=backend)
            backend._registered = True
            logger.info(f"Successfully registered Conductor backend as 'gcu' (PyTorch {torch_version})")
        else:
            raise RuntimeError(f"PyTorch {torch_version} does not support backend registration")
            
    except Exception as e:
        error_msg = f"Failed to register Conductor backend: {e}"
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
    Get information about the registered backend.
    
    Returns:
        Dictionary with backend information
    """
    info = {
        'name': 'gcu',
        'registered': is_backend_registered(),
        'pytorch_version': torch.__version__,
        'python_version': sys.version,
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
    List operations supported by the Conductor backend.
    
    Returns:
        List of supported operation names
    """
    # Based on the operations implemented in graph.py
    return [
        # Elementwise operations
        'add', 'sub', 'mul', 'div',
        'relu', 'sigmoid', 'tanh',
        'abs', 'neg', 'exp', 'log', 'sqrt',
        'sin', 'cos',
        
        # Reduction operations
        'sum', 'mean', 'max', 'min',
        'argmax', 'argmin',
        
        # Matrix operations
        'matmul', 'linear',
        
        # Shape operations
        'reshape', 'transpose', 'permute',
    ]


# Global backend instance for configuration
_global_backend: Optional[ConductorBackend] = None


def get_backend() -> Optional[ConductorBackend]:
    """
    Get the global backend instance.
    
    Returns:
        ConductorBackend instance if registered, None otherwise
    """
    global _global_backend
    
    if _global_backend is None and is_backend_registered():
        # Try to get the backend from PyTorch's registry
        try:
            backends = torch._dynamo.list_backends()
            if 'gcu' in backends:
                backend_fn = backends['gcu']
                if isinstance(backend_fn, ConductorBackend):
                    _global_backend = backend_fn
        except Exception:
            pass
    
    return _global_backend


def configure_backend(**kwargs) -> None:
    """
    Configure the global backend instance.
    
    Args:
        **kwargs: Configuration options (mode, enable_fusion, enable_fallback)
    """
    backend = get_backend()
    if backend is None:
        raise RuntimeError("Backend not registered. Call register_backend() first.")
    
    if 'mode' in kwargs:
        backend.set_compilation_mode(kwargs['mode'])
    
    if 'enable_fusion' in kwargs:
        backend.enable_fusion_optimization(kwargs['enable_fusion'])
    
    if 'enable_fallback' in kwargs:
        backend.enable_fallback_mechanism(kwargs['enable_fallback'])
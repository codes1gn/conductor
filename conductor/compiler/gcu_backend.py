"""
Simple GCU Backend for PyTorch torch.compile integration.

Provides GCU hardware acceleration through the Conductor framework.
"""

import torch
import torch.fx as fx
from typing import Callable as TypingCallable, Optional, List
import hashlib

from .jit_compiler import ChoreoJITCompiler
from .callable import Callable as GCUCallable
from .backend_registry import register_backend
from ..utils.tracer import get_tracer
from ..utils.trace import trace_fx_graph
from ..utils.logging import get_logger
from ..context import conductor_context

logger = get_logger(__name__)


class GCUBackend:
    """GCU backend for PyTorch torch.compile integration."""

    def __init__(self):
        self.jit_compiler = ChoreoJITCompiler()
        self._compiled_functions = {}

    def __call__(
        self, graph_module: fx.GraphModule, example_inputs: List[torch.Tensor]
    ) -> TypingCallable:
        """
        Compile FX graph for GCU execution.

        Args:
            graph_module: FX graph to compile
            example_inputs: Example input tensors

        Returns:
            Compiled callable
        """
        # Use context manager to ensure all global state is properly initialized
        with conductor_context() as context:
            logger.info(f"GCU backend called with {len(list(graph_module.graph.nodes))} nodes for GCU")

            # Debug tracing (config-driven)
            tracer = get_tracer()
            if tracer.should_trace_dag():
                trace_fx_graph(graph_module, example_inputs or [])

            # Check cache (config-driven)
            graph_hash = self._compute_graph_hash(graph_module, example_inputs)
            logger.info(f"Graph hash: {graph_hash}")
            cache_config = tracer._config
            if cache_config and graph_hash in self._compiled_functions:
                logger.debug("Using cached compiled function")
                return self._compiled_functions[graph_hash]

            try:
                # Compile using JIT compiler
                artifact = self.jit_compiler.compile_graph(graph_module, example_inputs)
                compiled_fn = GCUCallable(artifact, graph_module)

                # Cache result (config-driven)
                if cache_config:
                    self._compiled_functions[graph_hash] = compiled_fn
                logger.info("GCU compilation successful")
                return compiled_fn

            except Exception as e:
                # Import here to avoid circular imports
                from ..utils.exceptions import CompilationError

                # Provide detailed error information for debugging
                if isinstance(e, CompilationError):
                    logger.error(f"GCU compilation failed: {e}")
                    if hasattr(e, 'dsl_code') and e.dsl_code:
                        logger.error(f"DSL code length: {len(e.dsl_code)} characters")
                    if hasattr(e, 'compiler_output') and e.compiler_output:
                        logger.error(f"Choreo compiler output:\n{e.compiler_output}")
                    else:
                        logger.error("No compiler output available")
                else:
                    logger.error(f"GCU compilation failed with unexpected error: {e}")

                logger.warning(f"GCU compilation failed: {e}")
                logger.info("Falling back to PyTorch eager execution")
                return graph_module.forward

    def _compute_graph_hash(
        self, graph_module: fx.GraphModule, example_inputs: Optional[List[torch.Tensor]] = None
    ) -> str:
        """Compute hash for graph caching."""
        graph_str = str(graph_module.graph)
        if example_inputs:
            shapes_str = str([tuple(inp.shape) for inp in example_inputs])
            graph_str += shapes_str
        return hashlib.md5(graph_str.encode()).hexdigest()


# Global backend instance
_gcu_backend = None


def get_gcu_backend() -> GCUBackend:
    """Get the global GCU backend instance."""
    global _gcu_backend
    if _gcu_backend is None:
        _gcu_backend = GCUBackend()
    return _gcu_backend


def register_gcu_backend() -> bool:
    """Register the GCU backend with PyTorch."""
    backend = get_gcu_backend()
    return register_backend("gcu", backend)

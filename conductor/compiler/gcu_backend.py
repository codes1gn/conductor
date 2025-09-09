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
from ..utils.trace import get_debug_tracer, trace_fx_graph
from ..utils.logging import get_logger

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
        logger.debug(f"Compiling FX Graph with {len(list(graph_module.graph.nodes))} nodes for GCU")

        # Debug tracing
        debug_tracer = get_debug_tracer()
        if debug_tracer.is_enabled():
            trace_fx_graph(graph_module, example_inputs or [])

        # Check cache (skip if debug mode)
        graph_hash = self._compute_graph_hash(graph_module, example_inputs)
        if not debug_tracer.is_enabled() and graph_hash in self._compiled_functions:
            logger.debug("Using cached compiled function")
            return self._compiled_functions[graph_hash]

        try:
            # Compile using JIT compiler
            artifact = self.jit_compiler.compile_graph(graph_module, example_inputs)
            compiled_fn = GCUCallable(artifact, graph_module)

            # Cache result (skip if debug mode)
            if not debug_tracer.is_enabled():
                self._compiled_functions[graph_hash] = compiled_fn
            logger.info("GCU compilation successful")
            return compiled_fn

        except Exception as e:
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

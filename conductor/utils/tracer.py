"""
Modern Configuration-Driven Tracer for Conductor Framework.

This module provides a clean, config-driven tracing system that replaces
the legacy debug_tracer with a more systematic approach.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import json

from .config import get_config
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class TraceConfig:
    """Configuration for tracing behavior."""
    
    enabled: bool = False
    trace_dag: bool = True
    trace_dsl: bool = True
    trace_compilation: bool = True
    trace_execution: bool = False
    verbose_level: int = 1  # 0=minimal, 1=normal, 2=verbose, 3=debug


class ConductorTracer:
    """
    Modern configuration-driven tracer for Conductor framework.
    
    This class provides systematic tracing capabilities controlled by
    configuration rather than runtime state, making it more predictable
    and easier to control.
    """
    
    def __init__(self, config: Optional[TraceConfig] = None):
        """Initialize the tracer with configuration."""
        self._config = config or self._load_trace_config()
        
    def _load_trace_config(self) -> TraceConfig:
        """Load trace configuration from the main config."""
        conductor_config = get_config()
        debug_config = conductor_config.debug
        
        return TraceConfig(
            enabled=debug_config.enabled,
            trace_dag=debug_config.trace_dag,
            trace_dsl=debug_config.trace_dsl,
            trace_compilation=debug_config.trace_compilation,
            trace_execution=debug_config.trace_execution,
            verbose_level=debug_config.verbose_level,
        )
    
    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._config.enabled
    
    def should_trace_dag(self) -> bool:
        """Check if DAG tracing is enabled."""
        return self._config.enabled and self._config.trace_dag
    
    def should_trace_dsl(self) -> bool:
        """Check if DSL tracing is enabled."""
        return self._config.enabled and self._config.trace_dsl
    
    def should_trace_compilation(self) -> bool:
        """Check if compilation tracing is enabled."""
        return self._config.enabled and self._config.trace_compilation
    
    def should_trace_execution(self) -> bool:
        """Check if execution tracing is enabled."""
        return self._config.enabled and self._config.trace_execution
    
    def get_verbose_level(self) -> int:
        """Get the current verbose level."""
        return self._config.verbose_level
    
    def trace_dag(self, dag: Any, title: str = "Internal DAG"):
        """Trace DAG representation if enabled."""
        if not self.should_trace_dag():
            return
            
        logger.info(f"=== {title} ===")
        
        if hasattr(dag, 'nodes'):
            logger.info(f"DAG has {len(dag.nodes)} nodes:")
            for i, node in enumerate(dag.nodes):
                node_info = f"  {i}: {node.op_name}"
                if hasattr(node, 'input_buffers') and node.input_buffers:
                    input_names = [buf.name for buf in node.input_buffers]
                    node_info += f" inputs={input_names}"
                if hasattr(node, 'output_buffers') and node.output_buffers:
                    output_names = [buf.name for buf in node.output_buffers]
                    node_info += f" outputs={output_names}"
                logger.info(node_info)
        
        if hasattr(dag, 'buffers'):
            logger.info(f"DAG has {len(dag.buffers)} buffers:")
            for buf in dag.buffers:
                buf_info = f"  {buf.name}: {getattr(buf, 'shape', 'unknown_shape')}"
                if hasattr(buf, 'dtype'):
                    buf_info += f" dtype={buf.dtype}"
                logger.info(buf_info)
    
    def trace_dsl(self, dsl_content: str, kernel_code: Optional[str] = None, title: str = "Generated DSL"):
        """Trace DSL code if enabled."""
        if not self.should_trace_dsl():
            return

        logger.info(f"=== {title} ===")

        # Show summary
        lines = dsl_content.split('\n')
        logger.info(f"DSL has {len(lines)} lines")

        # Always show complete DSL content without logging prefixes for readability
        logger.info("Complete DSL content:")

        # Print DSL content directly to stdout without logging prefixes
        print("=" * 60)
        for i, line in enumerate(lines, 1):
            print(f"{i:3d}: {line}")
        print("=" * 60)
    
    def trace_compilation(self, command: List[str], result: Any, title: str = "Compilation"):
        """Trace compilation process if enabled."""
        if not self.should_trace_compilation():
            return
            
        logger.info(f"=== {title} ===")
        logger.info(f"Command: {' '.join(command)}")
        
        if hasattr(result, 'returncode'):
            logger.info(f"Return code: {result.returncode}")
            if result.stdout:
                logger.info(f"Stdout: {result.stdout}")
            if result.stderr:
                logger.info(f"Stderr: {result.stderr}")
    
    def trace_execution(self, function_name: str, inputs: List[Any], outputs: List[Any], duration_ms: float):
        """Trace execution if enabled."""
        if not self.should_trace_execution():
            return
            
        logger.info(f"=== Execution: {function_name} ===")
        logger.info(f"Inputs: {len(inputs)} tensors")
        logger.info(f"Outputs: {len(outputs)} tensors")
        logger.info(f"Duration: {duration_ms:.2f}ms")
        
        if self._config.verbose_level >= 2:
            for i, inp in enumerate(inputs):
                if hasattr(inp, 'shape'):
                    logger.info(f"  Input {i}: shape={inp.shape} dtype={getattr(inp, 'dtype', 'unknown')}")
            for i, out in enumerate(outputs):
                if hasattr(out, 'shape'):
                    logger.info(f"  Output {i}: shape={out.shape} dtype={getattr(out, 'dtype', 'unknown')}")
    
    def trace_graph_hash(self, graph_hash: str, graph_info: Dict[str, Any]):
        """Trace graph hash generation if enabled."""
        if not self.should_trace_dag():
            return
            
        logger.info(f"=== Graph Hash: {graph_hash} ===")
        for key, value in graph_info.items():
            logger.info(f"  {key}: {value}")
    
    def trace_cache_operation(self, operation: str, graph_hash: str, success: bool):
        """Trace cache operations if enabled."""
        if not self.should_trace_compilation():
            return
            
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"=== Cache {operation.upper()}: {status} ===")
        logger.info(f"Graph hash: {graph_hash}")


# Global tracer instance
_global_tracer: Optional[ConductorTracer] = None


def get_tracer() -> ConductorTracer:
    """Get the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = ConductorTracer()
    return _global_tracer


def reset_tracer():
    """Reset the global tracer (useful for testing)."""
    global _global_tracer
    _global_tracer = None


# Convenience functions for backward compatibility
def trace_internal_dag(dag: Any):
    """Trace internal DAG representation."""
    get_tracer().trace_dag(dag, "Internal DAG")


def trace_choreo_dsl(dsl_content: str, kernel_code: Optional[str] = None):
    """Trace Choreo DSL code."""
    get_tracer().trace_dsl(dsl_content, kernel_code, "Generated Choreo DSL")


def trace_compilation_result(command: List[str], result: Any):
    """Trace compilation result."""
    get_tracer().trace_compilation(command, result, "Choreo Compilation")


def trace_execution_result(function_name: str, inputs: List[Any], outputs: List[Any], duration_ms: float):
    """Trace execution result."""
    get_tracer().trace_execution(function_name, inputs, outputs, duration_ms)

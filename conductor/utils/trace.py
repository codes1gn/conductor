"""
Comprehensive Debug Tracing System for Conductor.

This module provides detailed debug tracing for the end-to-end compilation pipeline,
showing complete data flow and code generation at each processing stage.
"""

import os
import sys
import json
import torch
import torch.fx as fx
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class DebugTraceConfig:
    """Simplified configuration for debug tracing."""

    enabled: bool = False
    max_tensor_elements: int = 100  # Max elements to show in tensor dumps
    indent_size: int = 2


class DebugTracer:
    """Centralized debug tracing system for the compilation pipeline."""

    def __init__(self, config: Optional[DebugTraceConfig] = None):
        self.config = config or self._get_config_from_env()
        self.trace_data = {}
        self.section_counter = 0

    def _get_config_from_env(self) -> DebugTraceConfig:
        """Get debug configuration from the unified config system."""
        try:
            from .config import get_config

            config = get_config()
            return DebugTraceConfig(
                enabled=config.debug.enabled,
                max_tensor_elements=config.debug.max_tensor_elements,
                indent_size=config.debug.indent_size,
            )
        except ImportError:
            # Fallback to environment variables if config system not available
            return DebugTraceConfig(
                enabled=os.getenv("CONDUCTOR_DEBUG", "0").lower() in ("1", "true", "yes"),
                max_tensor_elements=int(os.getenv("CONDUCTOR_DEBUG_MAX_ELEMENTS", "100")),
                indent_size=int(os.getenv("CONDUCTOR_DEBUG_INDENT", "2")),
            )

    def is_enabled(self) -> bool:
        """Check if debug tracing is enabled."""
        return self.config.enabled

    def print_section_header(self, title: str, level: int = 1) -> None:
        """Print a formatted section header."""
        if not self.is_enabled():
            return

        self.section_counter += 1

        if level == 1:
            separator = "=" * 80
            print(f"\n{separator}")
            print(f"=== {title.upper()} ===")
            print(f"{separator}")
        elif level == 2:
            separator = "-" * 60
            print(f"\n{separator}")
            print(f"--- {title} ---")
            print(f"{separator}")
        else:
            print(f"\n{'  ' * (level - 1)}>>> {title}")

    def print_fx_graph_module(self, fx_graph: fx.GraphModule, inputs: List[torch.Tensor]) -> None:
        """Print complete FX GraphModule representation."""
        if not self.is_enabled():
            return

        self.print_section_header("FX Graph Module", 1)

        # Print graph structure
        print("Graph Structure:")
        print(fx_graph.graph)

        # Print code representation
        self.print_section_header("Python Code", 2)
        print(fx_graph.code)

        # Print node details
        self.print_section_header("Node Details", 2)
        for i, node in enumerate(fx_graph.graph.nodes):
            print(f"Node {i}: {node.name}")
            print(f"  Op: {node.op}")
            print(f"  Target: {node.target}")
            print(f"  Args: {node.args}")
            print(f"  Kwargs: {node.kwargs}")
            if hasattr(node, "meta") and node.meta:
                print(f"  Meta: {node.meta}")
            print()

        # Print input tensor information
        if True:  # Always show metadata when debug is enabled
            self.print_section_header("Input Tensor Information", 2)
            for i, tensor in enumerate(inputs):
                print(f"Input {i}:")
                print(f"  Shape: {tensor.shape}")
                print(f"  Dtype: {tensor.dtype}")
                print(f"  Device: {tensor.device}")
                print(f"  Requires grad: {tensor.requires_grad}")
                if tensor.numel() <= self.config.max_tensor_elements:
                    print(f"  Data: {tensor.flatten()}")
                else:
                    print(
                        f"  Data: {tensor.flatten()[:self.config.max_tensor_elements]}... (truncated)"
                    )
                print()

    def print_internal_dag(self, dag) -> None:
        """Print complete internal DAG representation."""
        if not self.is_enabled():
            return

        self.print_section_header("Internal DAG Representation", 1)

        # Print DAG overview
        print(f"Total nodes: {len(dag.nodes)}")
        print(f"Input buffers: {len(dag.inputs)}")
        print(f"Output buffers: {len(dag.outputs)}")
        print()

        # Print input buffers
        self.print_section_header("Input Buffers", 2)
        for i, buf in enumerate(dag.inputs):
            self._print_buffer_details(buf, f"Input {i}")

        # Print output buffers
        self.print_section_header("Output Buffers", 2)
        for i, buf in enumerate(dag.outputs):
            self._print_buffer_details(buf, f"Output {i}")

        # Print computation nodes
        self.print_section_header("Computation Nodes", 2)
        for i, node in enumerate(dag.nodes):
            self._print_node_details(node, f"Node {i}")

        # Print data flow connections
        self.print_section_header("Data Flow Connections", 2)
        self._print_data_flow_connections(dag)

    def _print_buffer_details(self, buffer, label: str) -> None:
        """Print detailed buffer information."""
        print(f"{label}: {buffer.name}")
        print(f"  Shape: {buffer.shape}")
        print(f"  Dtype: {buffer.dtype}")
        print(f"  Scope: {buffer.scope}")
        if hasattr(buffer, "size") and buffer.size:
            print(f"  Size: {buffer.size} bytes")
        if hasattr(buffer, "metadata") and buffer.metadata:
            print(f"  Metadata: {buffer.metadata}")
        print()

    def _print_node_details(self, node, label: str) -> None:
        """Print detailed node information."""
        print(f"{label}: {node.op_name}")
        print(f"  Inputs: {[str(inp) for inp in node.inputs]}")
        print(f"  Outputs: {[str(out) for out in node.outputs]}")

        if node.metadata:
            print(f"  Metadata:")
            for key, value in node.metadata.items():
                try:
                    if isinstance(value, (dict, list)):
                        print(
                            f"    {key}: {json.dumps(value, indent=self.config.indent_size, default=str)}"
                        )
                    else:
                        print(f"    {key}: {value}")
                except (TypeError, ValueError):
                    print(f"    {key}: {str(value)}")

        # Show operation-specific information
        if hasattr(node, "is_elementwise") and node.is_elementwise():
            print(f"  Type: Elementwise operation")
        elif hasattr(node, "is_reduction") and node.is_reduction():
            print(f"  Type: Reduction operation")
        else:
            print(f"  Type: General operation")

        print()

    def _print_data_flow_connections(self, dag) -> None:
        """Print data flow connections between nodes."""
        print("Data Flow Graph:")

        # Create a simple adjacency representation
        connections = {}
        for i, node in enumerate(dag.nodes):
            connections[f"Node_{i}_{node.op_name}"] = []

            # Find connections to other nodes
            for j, other_node in enumerate(dag.nodes):
                if i != j:
                    # Check if any output of current node is input to other node
                    node_outputs = {str(out) for out in node.outputs}
                    other_inputs = {str(inp) for inp in other_node.inputs}

                    if node_outputs.intersection(other_inputs):
                        connections[f"Node_{i}_{node.op_name}"].append(
                            f"Node_{j}_{other_node.op_name}"
                        )

        for node, targets in connections.items():
            if targets:
                print(f"  {node} -> {', '.join(targets)}")
            else:
                print(f"  {node} -> (no connections)")
        print()

    def print_choreo_dsl_code(self, dsl_code: str, kernel_code: Optional[str] = None) -> None:
        """Print complete generated Choreo DSL code."""
        if not self.is_enabled():
            return

        self.print_section_header("Generated Choreo DSL Code", 1)

        # Print main DSL code
        self.print_section_header("Complete DSL Implementation", 2)
        print(dsl_code)

        # Print kernel code if available
        if kernel_code:
            self.print_section_header("Device Kernel Code", 2)
            print(kernel_code)

    def print_host_wrapper_details(
        self, wrapper_code: str, function_signature: str, buffer_info: Dict[str, Any]
    ) -> None:
        """Print host wrapper integration details."""
        if not self.is_enabled():
            return

        self.print_section_header("Host Wrapper Integration", 1)

        # Print function signature
        self.print_section_header("Function Signature", 2)
        print(f"Generated signature: {function_signature}")
        print()

        # Print buffer allocation details
        self.print_section_header("Buffer Allocation & Memory Management", 2)
        for buffer_name, info in buffer_info.items():
            print(f"Buffer: {buffer_name}")
            print(f"  Size: {info.get('size', 'unknown')} bytes")
            print(f"  Alignment: {info.get('alignment', 'default')}")
            print(f"  Memory type: {info.get('memory_type', 'device')}")
            print()

        # Print critical wrapper code sections
        self.print_section_header("Critical Host Wrapper Code", 2)

        # Extract and highlight key sections
        lines = wrapper_code.split("\n")
        in_critical_section = False
        critical_keywords = [
            "tensor_to_gcu_buffer",
            "gcu_buffer_to_tensor",
            "allocate_buffer",
            "free_buffer",
            "launch_kernel",
            "synchronize",
        ]

        print("Data Marshalling & Execution Code:")
        for i, line in enumerate(lines):
            if any(keyword in line for keyword in critical_keywords):
                in_critical_section = True
                print(f"  {i+1:3d}: {line}")
            elif in_critical_section and (line.strip() == "" or line.startswith("}")):
                print(f"  {i+1:3d}: {line}")
                in_critical_section = False
            elif in_critical_section:
                print(f"  {i+1:3d}: {line}")
        print()

    def print_compilation_summary(self, compilation_stats: Dict[str, Any]) -> None:
        """Print compilation summary and statistics."""
        if not self.is_enabled():
            return

        self.print_section_header("Compilation Summary", 1)

        print("Compilation Statistics:")
        for key, value in compilation_stats.items():
            print(f"  {key}: {value}")
        print()

        if "errors" in compilation_stats and compilation_stats["errors"]:
            self.print_section_header("Compilation Errors", 2)
            for error in compilation_stats["errors"]:
                print(f"  ERROR: {error}")
            print()

        if "warnings" in compilation_stats and compilation_stats["warnings"]:
            self.print_section_header("Compilation Warnings", 2)
            for warning in compilation_stats["warnings"]:
                print(f"  WARNING: {warning}")
            print()

    def store_trace_data(self, key: str, data: Any) -> None:
        """Store trace data for later analysis."""
        if self.is_enabled():
            self.trace_data[key] = data

    def get_trace_data(self, key: str) -> Any:
        """Retrieve stored trace data."""
        return self.trace_data.get(key)

    def save_trace_to_file(self, filepath: str) -> None:
        """Save complete trace data to file."""
        if not self.is_enabled():
            return

        try:
            with open(filepath, "w") as f:
                json.dump(self.trace_data, f, indent=self.config.indent_size, default=str)
            print(f"Debug trace saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save debug trace: {e}")


# Global debug tracer instance
debug_tracer = DebugTracer()


def get_debug_tracer() -> DebugTracer:
    """Get the global debug tracer instance."""
    return debug_tracer


def enable_debug_tracing(config: Optional[DebugTraceConfig] = None) -> None:
    """Enable debug tracing with optional configuration."""
    global debug_tracer
    if config:
        debug_tracer.config = config
    else:
        debug_tracer.config.enabled = True


def disable_debug_tracing() -> None:
    """Disable debug tracing."""
    global debug_tracer
    debug_tracer.config.enabled = False


def trace_fx_graph(fx_graph: fx.GraphModule, inputs: List[torch.Tensor]) -> None:
    """Convenience function to trace FX graph."""
    debug_tracer.print_fx_graph_module(fx_graph, inputs)


def trace_internal_dag(dag) -> None:
    """Convenience function to trace internal DAG."""
    debug_tracer.print_internal_dag(dag)


def trace_choreo_dsl(dsl_code: str, kernel_code: Optional[str] = None) -> None:
    """Convenience function to trace Choreo DSL code."""
    debug_tracer.print_choreo_dsl_code(dsl_code, kernel_code)


def trace_host_wrapper(
    wrapper_code: str, function_signature: str, buffer_info: Dict[str, Any]
) -> None:
    """Convenience function to trace host wrapper."""
    debug_tracer.print_host_wrapper_details(wrapper_code, function_signature, buffer_info)

"""
FX Graph analysis and optimization.

This module provides classes and functions for parsing PyTorch FX Graphs
and converting them to Conductor's internal DAG representation with optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import re
import logging

import torch
from .buffers import Buffer, BufferManager, BufferScope
from .graph_nodes import ConductorNode
from ..utils.tracer import get_tracer

logger = logging.getLogger(__name__)


@dataclass
class ComputationDAG:
    """
    Represents the complete computation graph as a directed acyclic graph.

    This class maintains the full graph structure with nodes, edges,
    and metadata needed for optimization and code generation.
    """

    nodes: list[ConductorNode] = field(default_factory=list)
    buffers: list[Buffer] = field(default_factory=list)
    inputs: list[Buffer] = field(default_factory=list)
    outputs: list[Buffer] = field(default_factory=list)

    def add_node(self, node: ConductorNode) -> None:
        """Add a node to the computation graph."""
        self.nodes.append(node)

    def add_buffer(self, buffer: Buffer) -> None:
        """Add a buffer to the computation graph."""
        self.buffers.append(buffer)

    def validate_graph_correctness(self) -> bool:
        """
        Verify graph integrity and detect potential issues.

        Returns:
            True if graph is valid, False otherwise
        """
        # Basic structural validation
        try:
            # Check that all nodes have valid inputs/outputs
            for node in self.nodes:
                # All input buffers should exist in the DAG
                for input_buffer in node.inputs:
                    if input_buffer not in self.buffers:
                        return False

                # All output buffers should exist in the DAG
                for output_buffer in node.outputs:
                    if output_buffer not in self.buffers:
                        return False

            # Check that input/output lists contain valid buffers
            for input_buffer in self.inputs:
                if input_buffer not in self.buffers:
                    return False

            for output_buffer in self.outputs:
                if output_buffer not in self.buffers:
                    return False

            return True

        except Exception:
            return False


class GraphAnalyzer:
    """
    Analyzes FX Graph and builds internal representation.

    This class handles the conversion from PyTorch's FX Graph format
    to Conductor's internal DAG representation, including data dependency
    analysis and graph validation.
    """

    def __init__(self):
        """Initialize the GraphAnalyzer with buffer management."""
        self.buffer_manager = BufferManager()
        self._node_to_buffer = {}  # Maps FX nodes to output buffers
        self._buffer_to_node = {}  # Maps buffers to producing nodes

    def parse_fx_graph(
        self,
        graph_module: torch.fx.GraphModule,
        example_inputs: Optional[list[torch.Tensor]] = None,
        enable_shape_inference: bool = True,
    ) -> ComputationDAG:
        """
        Convert FX Graph to internal DAG representation.

        Args:
            graph_module: PyTorch FX Graph to convert
            example_inputs: Example inputs for shape inference
            enable_shape_inference: Whether to enable shape inference

        Returns:
            ComputationDAG representing the input graph
        """
        if graph_module is None:
            raise ValueError("graph_module cannot be None")

        # Initialize DAG with comprehensive tracing
        tracer = get_tracer()
        dag = ComputationDAG()

        if tracer.should_trace_dag():
            logger.info("=== DAG Construction Tracing ===")
            logger.info(f"Starting DAG construction from FX graph with {len(list(graph_module.graph.nodes))} nodes")

        # Reset internal state
        self._node_to_buffer.clear()
        self._buffer_to_node.clear()

        # First pass: create buffers for all FX nodes with tracing
        if tracer.should_trace_dag():
            logger.info("Phase 1: Creating buffers for FX nodes")
            self._dump_buffer_state_table(dag, "Buffer State Before Phase 1")

        buffer_count = 0
        for node in graph_module.graph.nodes:
            self._create_buffer_for_node(node, dag)
            buffer_count += 1

            if tracer.should_trace_dag():
                logger.info(f"  Buffer {buffer_count}: {node.name} ({node.op}) -> {len(dag.buffers)} total buffers")

        if tracer.should_trace_dag():
            self._dump_buffer_state_table(dag, "Buffer State After Phase 1")

        # Second pass: create ConductorNodes and establish connections with tracing
        if tracer.should_trace_dag():
            logger.info("Phase 2: Creating ConductorNodes and establishing connections")

        node_count = 0
        for node in graph_module.graph.nodes:
            conductor_node = self._convert_fx_node(node, dag)
            if conductor_node:
                dag.add_node(conductor_node)
                node_count += 1

                if tracer.should_trace_dag():
                    # Get a meaningful name for the node
                    node_name = getattr(conductor_node, 'name', f"node_{node_count}")
                    logger.info(f"  Node {node_count}: {node_name} ({conductor_node.op_name})")
                    logger.info(f"    Inputs: {[getattr(buf, 'name', str(buf)) for buf in conductor_node.inputs]}")
                    logger.info(f"    Outputs: {[getattr(buf, 'name', str(buf)) for buf in conductor_node.outputs]}")
                    logger.info(f"    Total DAG nodes: {len(dag.nodes)}")

        # Identify inputs and outputs with shape information from example inputs
        if tracer.should_trace_dag():
            logger.info("Phase 3: Identifying graph inputs and outputs")

        self._identify_graph_inputs_outputs(graph_module.graph, dag, example_inputs)

        if tracer.should_trace_dag():
            logger.info(f"  Identified {len(dag.inputs)} input buffers: {[buf.name for buf in dag.inputs]}")
            logger.info(f"  Identified {len(dag.outputs)} output buffers: {[buf.name for buf in dag.outputs]}")
            self._dump_buffer_state_table(dag, "Buffer State After Phase 3")

        # CRITICAL FIX: Third pass - propagate shapes after input shapes are set
        # FIXME: This is a temporary solution. Ideally, we should be able to get shape info from torch.dynamo/fx graph
        # I am not familiar with torch.fx, will it be possible to get shape info from fx graph in JIT mode?
        # Propagate shapes for elementwise operations (config-driven)
        if enable_shape_inference:
            if tracer.should_trace_dag():
                logger.info("Phase 4: Propagating shapes for elementwise operations")

            self._propagate_shapes_for_elementwise_ops(dag)

            if tracer.should_trace_dag():
                logger.info("  Shape propagation completed")
                self._dump_buffer_state_table(dag, "Buffer State After Phase 4")

        # Analyze data dependencies
        if tracer.should_trace_dag():
            logger.info("Phase 5: Analyzing data dependencies")

        self.identify_data_dependencies(dag)

        # Perform buffer lifetime analysis
        if tracer.should_trace_dag():
            logger.info("Phase 6: Buffer lifetime analysis")

        lifetime_map = self.buffer_manager.analyze_buffer_lifetimes(dag)

        if tracer.should_trace_dag():
            logger.info("  Lifetime analysis completed")

        if tracer.should_trace_dag():
            logger.info("=== DAG Construction Complete ===")
            logger.info(f"Final DAG summary:")
            logger.info(f"  Total nodes: {len(dag.nodes)}")
            logger.info(f"  Total buffers: {len(dag.buffers)}")
            logger.info(f"  Input buffers: {len(dag.inputs)}")
            logger.info(f"  Output buffers: {len(dag.outputs)}")

            # Calculate total memory usage
            total_memory = sum(buf.memory_size for buf in dag.buffers)
            logger.info(f"  Total memory usage: {total_memory} bytes ({total_memory / 1024:.1f} KB)")

        return dag

    def _propagate_shapes_for_elementwise_ops(self, dag: ComputationDAG) -> None:
        """
        Propagate shapes from input buffers to output buffers for operations.
        This runs after input shapes are set from example_inputs.
        """
        tracer = get_tracer()
        shape_updates = 0

        for node in dag.nodes:
            if node.op_name in [
                "add",
                "mul",
                "sub",
                "div",
                "relu",
                "sigmoid",
                "custom_add",
                "custom_mul",
            ]:
                # For element-wise operations, output shape should match input shape
                if node.inputs and node.outputs:
                    input_buffer = node.inputs[0]
                    output_buffer = node.outputs[0]

                    if input_buffer.shape and not output_buffer.shape:
                        old_shape = output_buffer.shape
                        old_dtype = output_buffer.dtype

                        output_buffer.shape = input_buffer.shape
                        output_buffer.dtype = input_buffer.dtype
                        shape_updates += 1

                        if tracer.should_trace_dag():
                            logger.info(f"    Shape update {shape_updates}: {output_buffer.name}")
                            logger.info(f"      Shape: {old_shape} -> {output_buffer.shape}")
                            logger.info(f"      Dtype: {old_dtype} -> {output_buffer.dtype}")

                        logger.debug(
                            f"Propagated shape for {output_buffer.name}: {output_buffer.shape}"
                        )

            elif node.op_name == "matmul":
                # For matmul operations, infer output shape from input shapes
                if len(node.inputs) >= 2 and node.outputs:
                    input1 = node.inputs[0]
                    input2 = node.inputs[1]
                    output_buffer = node.outputs[0]

                    if input1.shape and input2.shape and not output_buffer.shape:
                        # For 2D matmul: (M, K) @ (K, N) -> (M, N)
                        if len(input1.shape) == 2 and len(input2.shape) == 2:
                            old_shape = output_buffer.shape
                            old_dtype = output_buffer.dtype

                            output_shape = (input1.shape[0], input2.shape[1])
                            output_buffer.shape = output_shape
                            output_buffer.dtype = input1.dtype
                            shape_updates += 1

                            if tracer.should_trace_dag():
                                logger.info(f"    Shape update {shape_updates}: {output_buffer.name} (matmul)")
                                logger.info(f"      Input1: {input1.shape}, Input2: {input2.shape}")
                                logger.info(f"      Shape: {old_shape} -> {output_buffer.shape}")
                                logger.info(f"      Dtype: {old_dtype} -> {output_buffer.dtype}")

                            logger.debug(
                                f"Propagated matmul shape for {output_buffer.name}: {output_buffer.shape}"
                            )

            elif node.op_name == "reduce_mean":
                # For reduce_mean operations, infer output shape based on reduction dimensions
                if node.inputs and node.outputs:
                    input_buffer = node.inputs[0]
                    output_buffer = node.outputs[0]

                    if input_buffer.shape and not output_buffer.shape:
                        # Get reduction parameters from node metadata
                        dim = node.metadata.get('dim', None)
                        keepdim = node.metadata.get('keepdim', False)

                        old_shape = output_buffer.shape
                        old_dtype = output_buffer.dtype

                        if dim is None:
                            # Reduce all dimensions
                            output_shape = (1,) if keepdim else ()
                        else:
                            # Reduce specific dimension(s)
                            input_shape = list(input_buffer.shape)
                            if isinstance(dim, int):
                                dims_to_reduce = [dim]
                            else:
                                dims_to_reduce = list(dim)

                            # Handle negative dimensions
                            dims_to_reduce = [d if d >= 0 else len(input_shape) + d for d in dims_to_reduce]

                            for d in sorted(dims_to_reduce, reverse=True):
                                if keepdim:
                                    input_shape[d] = 1
                                else:
                                    input_shape.pop(d)

                            output_shape = tuple(input_shape) if input_shape else (1,)

                        output_buffer.shape = output_shape
                        output_buffer.dtype = input_buffer.dtype
                        shape_updates += 1

                        if tracer.should_trace_dag():
                            logger.info(f"    Shape update {shape_updates}: {output_buffer.name} (reduce_mean)")
                            logger.info(f"      Input: {input_buffer.shape}, dim={dim}, keepdim={keepdim}")
                            logger.info(f"      Shape: {old_shape} -> {output_buffer.shape}")
                            logger.info(f"      Dtype: {old_dtype} -> {output_buffer.dtype}")

                        logger.debug(
                            f"Propagated shape for reduce_mean {output_buffer.name}: {output_buffer.shape}"
                        )

        # Summary of shape propagation and update memory sizes
        if tracer.should_trace_dag():
            logger.info(f"  Shape propagation summary: {shape_updates} buffers updated")

        # Update memory sizes for all buffers after shape propagation
        for buffer in dag.buffers:
            if buffer.shape:
                buffer.memory_size = buffer._calculate_memory_size()

    def _create_buffer_for_node(self, fx_node: torch.fx.Node, dag: ComputationDAG) -> None:
        """
        Create a buffer for an FX node's output.

        Args:
            fx_node: FX Graph node
            dag: Computation DAG being built
        """
        # Get tracer for potential buffer dumps
        tracer = get_tracer()

        if fx_node.op in ("placeholder", "get_attr", "call_function", "call_method", "call_module"):
            # Determine buffer properties
            buffer_name = self._generate_buffer_name(fx_node)
            dtype, shape = self._infer_buffer_properties(fx_node)

            # Create buffer
            buffer = self.buffer_manager.allocate_buffer(buffer_name, dtype, shape)

            # Add to DAG and tracking
            dag.add_buffer(buffer)
            self._node_to_buffer[fx_node] = buffer
            self._buffer_to_node[buffer] = fx_node

    def _convert_fx_node(
        self, fx_node: torch.fx.Node, dag: ComputationDAG
    ) -> Optional[ConductorNode]:
        """
        Convert an FX node to a ConductorNode.

        Args:
            fx_node: FX Graph node to convert
            dag: Computation DAG being built

        Returns:
            ConductorNode if conversion successful, None otherwise
        """
        if fx_node.op in ("call_function", "call_method", "call_module"):
            return self._convert_operation_node(fx_node, dag)

        return None


    def _convert_operation_node(
        self, fx_node: torch.fx.Node, dag: ComputationDAG
    ) -> Optional[ConductorNode]:
        """
        Convert an FX operation node to ConductorNode.

        Args:
            fx_node: FX operation node
            dag: Computation DAG being built

        Returns:
            ConductorNode representing the operation
        """
        # Get operation name
        op_name = self._extract_operation_name(fx_node)

        # Get input buffers
        input_buffers = []
        for arg in fx_node.args:
            if isinstance(arg, torch.fx.Node) and arg in self._node_to_buffer:
                input_buffers.append(self._node_to_buffer[arg])

        # Handle keyword arguments
        for key, value in fx_node.kwargs.items():
            if isinstance(value, torch.fx.Node) and value in self._node_to_buffer:
                input_buffers.append(self._node_to_buffer[value])

        # Get output buffer
        output_buffer = self._node_to_buffer.get(fx_node)
        output_buffers = [output_buffer] if output_buffer else []

        # Note: Shape inference is now handled in _propagate_shapes_for_elementwise_ops

        # Extract metadata
        metadata = self._extract_node_metadata(fx_node)

        # Create ConductorNode
        conductor_node = ConductorNode(
            op_name=op_name, inputs=input_buffers, outputs=output_buffers, metadata=metadata
        )

        # Update buffer producer/consumer relationships
        for buf in input_buffers:
            buf.consumers.append(conductor_node)

        for buf in output_buffers:
            buf.producer = conductor_node

        return conductor_node

    def _extract_operation_name(self, fx_node: torch.fx.Node) -> str:
        """
        Extract operation name from FX node using centralized operation factory.

        Args:
            fx_node: FX Graph node

        Returns:
            Operation name string
        """
        # FIXME: Import here to avoid circular imports
        from ..codegen.operation_factory import get_operation_factory

        # Use centralized operation factory for consistent operation name extraction
        operation_factory = get_operation_factory()
        return operation_factory.extract_operation_name(fx_node)

    def _extract_node_metadata(self, fx_node: torch.fx.Node) -> dict[str, Any]:
        """
        Extract metadata from FX node.

        Args:
            fx_node: FX Graph node

        Returns:
            Dictionary of metadata
        """
        metadata = {}

        # Add keyword arguments as metadata
        if fx_node.kwargs:
            metadata.update(fx_node.kwargs)

        # Add positional arguments that are not nodes (constants, etc.)
        non_node_args = []
        for arg in fx_node.args:
            if not isinstance(arg, torch.fx.Node):
                non_node_args.append(arg)

        if non_node_args:
            metadata["args"] = non_node_args

        # Add node-specific metadata
        if hasattr(fx_node, "meta"):
            metadata.update(fx_node.meta)

        return metadata

    def _generate_buffer_name(self, fx_node: torch.fx.Node) -> str:
        """
        Generate a unique buffer name for an FX node.

        Args:
            fx_node: FX Graph node

        Returns:
            Unique buffer name
        """
        # Use the FX node name as base, ensuring uniqueness
        base_name = fx_node.name if fx_node.name else f"node_{id(fx_node)}"
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", base_name)

        return clean_name

    def _infer_buffer_properties(self, fx_node: torch.fx.Node) -> tuple:
        """
        Infer buffer dtype and shape from FX node.

        Args:
            fx_node: FX Graph node

        Returns:
            Tuple of (dtype, shape)
        """
        dtype = torch.float32  # Default dtype
        shape = None  # Unknown shape by default

        # Try to get type information from node metadata
        if hasattr(fx_node, "meta") and "tensor_meta" in fx_node.meta:
            tensor_meta = fx_node.meta["tensor_meta"]
            if hasattr(tensor_meta, "dtype"):
                dtype = tensor_meta.dtype
            if hasattr(tensor_meta, "shape"):
                shape = tuple(tensor_meta.shape)

        # Try to get type from 'val' metadata (common in some FX graphs)
        elif hasattr(fx_node, "meta") and "val" in fx_node.meta:
            val = fx_node.meta["val"]
            if hasattr(val, "dtype"):
                dtype = val.dtype
            if hasattr(val, "shape"):
                shape = tuple(val.shape)

        return dtype, shape

    def _identify_graph_inputs_outputs(
        self,
        fx_graph: torch.fx.Graph,
        dag: ComputationDAG,
        example_inputs: Optional[list[torch.Tensor]] = None,
    ) -> None:
        """
        Identify input and output buffers of the graph.

        Args:
            fx_graph: FX Graph
            dag: Computation DAG being built
            example_inputs: Example input tensors for shape inference
        """
        # Count placeholder nodes first
        placeholder_count = sum(1 for node in fx_graph.nodes if node.op == "placeholder")

        # Validate example inputs if provided
        if example_inputs is not None and len(example_inputs) != placeholder_count:
            raise ValueError(
                f"Number of example inputs ({len(example_inputs)}) must match number of placeholders ({placeholder_count})"
            )

        # Find input nodes (placeholders) and update their shapes from example inputs
        input_index = 0
        for node in fx_graph.nodes:
            if node.op == "placeholder" and node in self._node_to_buffer:
                buffer = self._node_to_buffer[node]

                # Update buffer shape from example inputs if available
                if example_inputs and input_index < len(example_inputs):
                    example_tensor = example_inputs[input_index]
                    buffer.shape = tuple(example_tensor.shape)
                    buffer.dtype = example_tensor.dtype
                    logger.debug(
                        f"Updated input buffer {buffer.name} shape to {buffer.shape} from example input"
                    )
                    input_index += 1

                # Mark as input buffer for lifetime analysis
                buffer.is_input = True
                buffer.memory_size = buffer._calculate_memory_size()
                dag.inputs.append(buffer)

        # Find output nodes
        for node in fx_graph.nodes:
            if node.op == "output":
                # Output node args contain the actual output values
                for arg in node.args:
                    if isinstance(arg, (list, tuple)):
                        # Multiple outputs
                        for output_node in arg:
                            if (
                                isinstance(output_node, torch.fx.Node)
                                and output_node in self._node_to_buffer
                            ):
                                buffer = self._node_to_buffer[output_node]
                                buffer.is_output = True  # Mark as output for lifetime analysis
                                buffer.memory_size = buffer._calculate_memory_size()
                                dag.outputs.append(buffer)
                    elif isinstance(arg, torch.fx.Node) and arg in self._node_to_buffer:
                        # Single output
                        buffer = self._node_to_buffer[arg]
                        buffer.is_output = True  # Mark as output for lifetime analysis
                        buffer.memory_size = buffer._calculate_memory_size()
                        dag.outputs.append(buffer)

    def identify_data_dependencies(self, dag: ComputationDAG) -> None:
        """
        Analyze data flow and establish buffer dependencies.

        Args:
            dag: Computation DAG to analyze
        """
        # Data dependencies are already established during node conversion
        # through the producer/consumer relationships in buffers

        # Additional validation: ensure all buffers have proper connections
        for buffer in dag.buffers:
            # Input buffers should have no producer
            if buffer in dag.inputs:
                if buffer.producer is not None:
                    raise ValueError(f"Input buffer {buffer.name} should not have a producer")

            # Non-input buffers should have exactly one producer
            elif buffer.producer is None:
                # This might be a constant or parameter buffer, which is okay
                pass

            # Output buffers should have at least one consumer or be graph outputs
            if not buffer.consumers and buffer not in dag.outputs:
                # Mark as temporary buffer
                buffer.is_temporary = True

        # Promote buffer scopes based on usage patterns
        self._optimize_buffer_scopes(dag)

    def _optimize_buffer_scopes(self, dag: ComputationDAG) -> None:
        """
        Optimize buffer scopes based on usage patterns.

        Args:
            dag: Computation DAG to optimize
        """
        for buffer in dag.buffers:
            # Input and output buffers should be GLOBAL scope
            if buffer in dag.inputs or buffer in dag.outputs:
                buffer.promote_scope(BufferScope.GLOBAL)

            # Buffers with multiple consumers should be SHARED scope
            elif len(buffer.consumers) > 1:
                buffer.promote_scope(BufferScope.SHARED)

            # Single-consumer buffers can remain LOCAL
            # (already set by default in BufferManager)

    def validate_graph_correctness(self, dag: ComputationDAG) -> bool:
        """
        Verify graph integrity and detect potential issues.

        Args:
            dag: Computation DAG to validate

        Returns:
            True if graph is valid, False otherwise
        """
        # Check for cycles (DAG property)
        if self._has_cycles(dag):
            return False

        # Check buffer consistency
        if not self._validate_buffer_consistency(dag):
            return False

        # Check node connectivity
        if not self._validate_node_connectivity(dag):
            return False

        return True

    def _has_cycles(self, dag: ComputationDAG) -> bool:
        """
        Check if the graph has cycles using DFS.

        Args:
            dag: Computation DAG to check

        Returns:
            True if cycles exist, False otherwise
        """
        # Use DFS with coloring: white (0), gray (1), black (2)
        color = {node: 0 for node in dag.nodes}

        def dfs(node):
            if color[node] == 1:  # Gray - back edge found (cycle)
                return True
            if color[node] == 2:  # Black - already processed
                return False

            color[node] = 1  # Mark as gray (visiting)

            # Visit all successors
            for output_buffer in node.outputs:
                for consumer in output_buffer.consumers:
                    if dfs(consumer):
                        return True

            color[node] = 2  # Mark as black (processed)
            return False

        # Check all nodes
        for node in dag.nodes:
            if color[node] == 0 and dfs(node):
                return True

        return False

    def _validate_buffer_consistency(self, dag: ComputationDAG) -> bool:
        """
        Validate buffer producer/consumer consistency.

        Args:
            dag: Computation DAG to validate

        Returns:
            True if consistent, False otherwise
        """
        for buffer in dag.buffers:
            # Check producer consistency
            if buffer.producer:
                if buffer not in buffer.producer.outputs:
                    return False

            # Check consumer consistency
            for consumer in buffer.consumers:
                if buffer not in consumer.inputs:
                    return False

        return True

    def _validate_node_connectivity(self, dag: ComputationDAG) -> bool:
        """
        Validate node input/output connectivity.

        Args:
            dag: Computation DAG to validate

        Returns:
            True if properly connected, False otherwise
        """
        for node in dag.nodes:
            # Check input buffer consistency
            for input_buffer in node.inputs:
                if node not in input_buffer.consumers:
                    return False

            # Check output buffer consistency
            for output_buffer in node.outputs:
                if output_buffer.producer != node:
                    return False

        return True

    def _dump_buffer_state_table(self, dag: ComputationDAG, title: str, context: str = "") -> None:
        """
        Dump current buffer state in tabular format for debugging.

        Args:
            dag: Current computation DAG
            title: Title for the dump
            context: Additional context information
        """
        tracer = get_tracer()
        if not tracer.should_trace_dag():
            return

        logger.info(f"=== {title} ===")
        if context:
            logger.info(f"Context: {context}")

        if not dag.buffers:
            logger.info("No buffers in DAG")
            return

        # Calculate column widths
        name_width = max(len("Buffer Name"), max(len(buf.name) for buf in dag.buffers))
        shape_width = max(len("Shape"), max(len(str(buf.shape)) if buf.shape else len("None") for buf in dag.buffers))
        dtype_width = max(len("Dtype"), max(len(str(buf.dtype)) if buf.dtype else len("None") for buf in dag.buffers))
        scope_width = max(len("Scope"), max(len(str(buf.scope)) if hasattr(buf, 'scope') else len("Unknown") for buf in dag.buffers))

        # Ensure minimum widths
        name_width = max(name_width, 12)
        shape_width = max(shape_width, 15)
        dtype_width = max(dtype_width, 15)
        scope_width = max(scope_width, 10)

        # Create table header
        header = f"| {'Buffer Name':<{name_width}} | {'Shape':<{shape_width}} | {'Dtype':<{dtype_width}} | {'Scope':<{scope_width}} |"
        separator = f"|{'-' * (name_width + 2)}|{'-' * (shape_width + 2)}|{'-' * (dtype_width + 2)}|{'-' * (scope_width + 2)}|"

        logger.info(separator)
        logger.info(header)
        logger.info(separator)

        # Create table rows
        for i, buffer in enumerate(dag.buffers):
            name = buffer.name
            shape = str(buffer.shape) if buffer.shape else "None"
            dtype = str(buffer.dtype) if buffer.dtype else "None"
            scope = str(buffer.scope) if hasattr(buffer, 'scope') else "Unknown"

            row = f"| {name:<{name_width}} | {shape:<{shape_width}} | {dtype:<{dtype_width}} | {scope:<{scope_width}} |"
            logger.info(row)

        logger.info(separator)
        logger.info(f"Total buffers: {len(dag.buffers)}")
        logger.info("")

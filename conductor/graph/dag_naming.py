"""
DAG Naming and Topological Sorting for Conductor.

This module provides systematic naming and topological analysis of computation DAGs,
ensuring consistent identifier generation and proper execution ordering.
"""

from __future__ import annotations

from typing import Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Avoid circular imports by using TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_nodes import ConductorNode
    from .graph_analyzer import ComputationDAG
from .buffer_naming import buffer_naming_manager, MemoryLevel as BufferMemoryLevel
from ..codegen.dsl_constants import (
    DSLIdentifierGenerator,
    MemoryLevel,
    DSLGenerationContext,
    ContextKeys,
)


class TopologicalSortError(Exception):
    """Exception raised when topological sorting fails."""

    pass


@dataclass
class NodeNaming:
    """Naming information for a DAG node."""

    node_id: str  # Unique node identifier
    operation_name: str  # Operation name (add, mul, etc.)
    input_vars: list[str] = field(default_factory=list)  # Input variable names
    output_vars: list[str] = field(default_factory=list)  # Output variable names
    load_vars: list[str] = field(default_factory=list)  # DMA load variable names
    memory_level: MemoryLevel = MemoryLevel.L1  # Memory level for this operation
    parallel_var: Optional[str] = None  # Parallel loop variable
    index_vars: list[str] = field(default_factory=list)  # Loop index variables


@dataclass
class DAGNamingAnnotation:
    """Complete naming annotation for a DAG."""

    node_naming: dict[str, NodeNaming] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)  # Topologically sorted node IDs
    global_context: DSLGenerationContext = field(default_factory=DSLGenerationContext)
    identifier_generator: DSLIdentifierGenerator = field(default_factory=DSLIdentifierGenerator)


class DAGTopologicalSorter:
    """Implements proper topological sorting with cycle detection."""

    def __init__(self):
        """Initialize the topological sorter."""
        pass

    def sort(self, dag: "ComputationDAG") -> list["ConductorNode"]:
        """
        Perform topological sorting on the DAG.

        Args:
            dag: The computation DAG to sort

        Returns:
            List of nodes in topological order

        Raises:
            TopologicalSortError: If the DAG contains cycles or is invalid
        """
        if not dag.nodes:
            return []

        # Build adjacency list and in-degree count
        adj_list = defaultdict(list)
        in_degree = defaultdict(int)
        node_map = {id(node): node for node in dag.nodes}

        # Initialize in-degree for all nodes
        for node in dag.nodes:
            in_degree[id(node)] = 0

        # Build graph from buffer dependencies
        for node in dag.nodes:
            node_id = id(node)

            # For each input buffer, find its producer
            if hasattr(node, "inputs") and node.inputs:
                for input_buffer in node.inputs:
                    if hasattr(input_buffer, "producer") and input_buffer.producer:
                        producer_id = id(input_buffer.producer)
                        if producer_id in node_map:
                            adj_list[producer_id].append(node_id)
                            in_degree[node_id] += 1

        # Kahn's algorithm for topological sorting
        queue = deque()

        # Find all nodes with no incoming edges
        for node_id in node_map:
            if in_degree[node_id] == 0:
                queue.append(node_id)

        result = []

        while queue:
            current_id = queue.popleft()
            result.append(node_map[current_id])

            # Remove this node from the graph
            for neighbor_id in adj_list[current_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        # Check for cycles
        if len(result) != len(dag.nodes):
            remaining_nodes = [node for node in dag.nodes if node not in result]
            raise TopologicalSortError(
                f"DAG contains cycles. Remaining nodes: {[node.op_name for node in remaining_nodes]}"
            )

        return result

    def validate_dag(self, dag: "ComputationDAG") -> bool:
        """
        Validate that the DAG is well-formed.

        Args:
            dag: The computation DAG to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            self.sort(dag)
            return True
        except TopologicalSortError:
            return False


class DAGNamingAnnotator:
    """
    Annotates a DAG with consistent naming before DSL generation.

    This class implements the pre-DSL naming stage that ensures all identifiers
    are consistent and derived from the DAG structure and operator templates.
    """

    def __init__(self):
        """Initialize the naming annotator."""
        self.sorter = DAGTopologicalSorter()

    def annotate(self, dag: "ComputationDAG") -> DAGNamingAnnotation:
        """
        Annotate the DAG with consistent naming.

        Args:
            dag: The computation DAG to annotate

        Returns:
            Complete naming annotation for the DAG
        """
        # Reset naming state
        buffer_naming_manager.reset()

        # Create annotation
        annotation = DAGNamingAnnotation()

        # Get topological order
        try:
            sorted_nodes = self.sorter.sort(dag)
            annotation.execution_order = [self._get_node_id(node) for node in sorted_nodes]
        except TopologicalSortError as e:
            raise ValueError(f"Cannot annotate invalid DAG: {e}")

        # Annotate each node with naming information
        for i, node in enumerate(sorted_nodes):
            node_id = self._get_node_id(node)
            naming = self._annotate_node(node, i, annotation)
            annotation.node_naming[node_id] = naming

        # Ensure naming consistency across the DAG
        self._ensure_naming_consistency(annotation, sorted_nodes)

        return annotation

    def _get_node_id(self, node: "ConductorNode") -> str:
        """Generate unique node ID."""
        return f"{node.op_name}_{id(node)}"

    def _annotate_node(
        self, node: "ConductorNode", index: int, annotation: DAGNamingAnnotation
    ) -> NodeNaming:
        """Annotate a single node with naming information."""
        from ..codegen.operator_registry import operator_registry

        node_id = self._get_node_id(node)

        # Get operator information
        op_info = operator_registry.get_operation(node.op_name)

        # Generate naming
        naming = NodeNaming(
            node_id=node_id,
            operation_name=node.op_name,
            memory_level=MemoryLevel.L1,  # Default, can be customized
        )

        # Generate input variable names based on operator template
        if op_info and op_info.input_buffers:
            for buf_spec in op_info.input_buffers:
                var_name = annotation.identifier_generator.generate_load_var(
                    buf_spec.name, naming.memory_level
                )
                naming.input_vars.append(var_name)
                naming.load_vars.append(var_name)
        else:
            # Fallback
            var_name = annotation.identifier_generator.generate_load_var(
                "input", naming.memory_level
            )
            naming.input_vars.append(var_name)
            naming.load_vars.append(var_name)

        # Generate output variable names based on operator template
        if op_info and op_info.output_buffers:
            for buf_spec in op_info.output_buffers:
                var_name = annotation.identifier_generator.generate_buffer_var(
                    node.op_name, naming.memory_level, str(index)
                )
                naming.output_vars.append(var_name)
        else:
            # Fallback
            var_name = annotation.identifier_generator.generate_buffer_var(
                node.op_name, naming.memory_level, str(index)
            )
            naming.output_vars.append(var_name)

        # Generate parallel and index variables
        naming.parallel_var = annotation.identifier_generator.generate_parallel_var()
        naming.index_vars = [
            annotation.identifier_generator.generate_index_var("i"),
            annotation.identifier_generator.generate_index_var("j"),
            annotation.identifier_generator.generate_index_var("k"),
        ]

        return naming

    def _ensure_naming_consistency(
        self, annotation: DAGNamingAnnotation, sorted_nodes: list["ConductorNode"]
    ):
        """Ensure naming consistency across the entire DAG."""
        # Build producer-consumer relationships
        producer_map = {}  # buffer_name -> producer_node_id

        for node in sorted_nodes:
            node_id = self._get_node_id(node)
            naming = annotation.node_naming[node_id]

            # Register this node as producer of its output buffers
            for output_var in naming.output_vars:
                producer_map[output_var] = node_id

        # Update consumer input names to match producer output names
        for node in sorted_nodes:
            node_id = self._get_node_id(node)
            naming = annotation.node_naming[node_id]

            # Check if this node consumes outputs from previous nodes
            if hasattr(node, "inputs") and node.inputs:
                for i, input_buffer in enumerate(node.inputs):
                    if hasattr(input_buffer, "producer") and input_buffer.producer:
                        producer_node_id = self._get_node_id(input_buffer.producer)
                        if producer_node_id in annotation.node_naming:
                            producer_naming = annotation.node_naming[producer_node_id]
                            if producer_naming.output_vars and i < len(naming.input_vars):
                                # Use the producer's output name as this node's input name
                                naming.input_vars[i] = producer_naming.output_vars[0]


# Global instance for use across the system
dag_naming_annotator = DAGNamingAnnotator()

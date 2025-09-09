"""
Operation fusion optimization and heuristics.

This module implements the fusion engine that identifies opportunities
to combine compatible operations for performance optimization.
Core optimization component for the Conductor compilation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .buffers import Buffer, BufferScope
from .graph_nodes import ConductorNode
from .graph_analyzer import ComputationDAG

# Import unified operator system for fusion decisions
try:
    from .operator_registry import get_fusion_metadata, can_operators_fuse

    UNIFIED_OPERATORS_AVAILABLE = True
except ImportError:
    UNIFIED_OPERATORS_AVAILABLE = False


class FusionType(Enum):
    """Categorizes different types of operation fusion."""

    ELEMENTWISE = "elementwise"  # Pure elementwise operation chains
    REDUCTION = "reduction"  # Elementwise followed by reduction
    MIXED = "mixed"  # Complex fusion patterns
    MEMORY_BOUND = "memory_bound"  # Memory bandwidth limited operations
    COMPUTE_BOUND = "compute_bound"  # Computation intensive operations


@dataclass
class FusionCluster:
    """
    Groups compatible operations for optimization.

    This class represents a collection of operations that can be
    fused together to reduce kernel launch overhead and improve
    memory locality.
    """

    nodes: list[ConductorNode] = field(default_factory=list)  # Operations included in this cluster
    cluster_type: FusionType = FusionType.ELEMENTWISE  # Type of fusion
    external_inputs: list[Buffer] = field(default_factory=list)  # Inputs from outside the cluster
    external_outputs: list[Buffer] = field(
        default_factory=list
    )  # Outputs consumed outside the cluster
    internal_buffers: list[Buffer] = field(
        default_factory=list
    )  # Temporary buffers within the cluster
    dsl_function_name: str = ""  # Generated DSL function identifier

    def validate_fusion_safety(self) -> bool:
        """
        Verify that fusion preserves mathematical correctness.

        Returns:
            True if fusion is mathematically safe, False otherwise
        """
        if not self.nodes:
            return True

        # Check that all nodes in cluster are compatible
        for i, node in enumerate(self.nodes):
            for _, other_node in enumerate(self.nodes[i + 1 :], i + 1):
                if not node.can_fuse_with(other_node):
                    return False

        # Validate data dependencies within cluster
        if not self._validate_data_dependencies():
            return False

        # Check for cycles in the fusion cluster
        if self._has_cycles():
            return False

        # Validate buffer usage patterns
        if not self._validate_buffer_usage():
            return False

        return True

    def _validate_data_dependencies(self) -> bool:
        """
        Validate that data dependencies are preserved in fusion.

        Returns:
            True if dependencies are valid, False otherwise
        """
        # Build dependency graph within cluster
        node_outputs = {}
        node_inputs = {}

        for node in self.nodes:
            node_outputs[node] = {buf.name for buf in node.outputs}
            node_inputs[node] = {buf.name for buf in node.inputs}

        # Check that internal dependencies are satisfied
        for node in self.nodes:
            for input_name in node_inputs[node]:
                # Check if input is produced by another node in cluster
                producer_found = False
                for other_node in self.nodes:
                    if other_node != node and input_name in node_outputs[other_node]:
                        producer_found = True
                        break

                # If not produced internally, must be external input
                if not producer_found:
                    external_input_names = {buf.name for buf in self.external_inputs}
                    if input_name not in external_input_names:
                        return False

        return True

    def _has_cycles(self) -> bool:
        """
        Check for cycles in the fusion cluster dependency graph.

        Returns:
            True if cycles exist, False otherwise
        """
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            # Find nodes that depend on this node's outputs
            node_output_names = {buf.name for buf in node.outputs}
            for other_node in self.nodes:
                if other_node != node:
                    other_input_names = {buf.name for buf in other_node.inputs}
                    if node_output_names.intersection(other_input_names):
                        if other_node in rec_stack:
                            return True
                        if other_node not in visited and dfs(other_node):
                            return True

            rec_stack.remove(node)
            return False

        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    return True

        return False

    def _validate_buffer_usage(self) -> bool:
        """
        Validate buffer usage patterns within the cluster.

        Returns:
            True if buffer usage is valid, False otherwise
        """
        # Check that internal buffers are only used within cluster
        internal_buffer_names = {buf.name for buf in self.internal_buffers}

        for node in self.nodes:
            # Check inputs
            for buf in node.inputs:
                if buf.name in internal_buffer_names:
                    # Internal buffer must be produced by another node in cluster
                    producer_found = False
                    for other_node in self.nodes:
                        if other_node != node:
                            for output_buf in other_node.outputs:
                                if output_buf.name == buf.name:
                                    producer_found = True
                                    break
                    if not producer_found:
                        return False

        return True

    def generate_fused_dsl(self) -> str:
        """
        Generate optimized DSL code for the entire cluster.

        Returns:
            DSL code string for the fused operations
        """
        if not self.nodes:
            return ""

        # Generate function signature
        input_names = [buf.name for buf in self.external_inputs]
        output_names = [buf.name for buf in self.external_outputs]

        function_signature = f"function {self.dsl_function_name}({', '.join(input_names)}) -> \
        ({', '.join(output_names)})"

        # Generate function body
        body_lines = []

        # Declare internal buffers
        for buf in self.internal_buffers:
            if buf.shape:
                shape_str = f"[{', '.join(map(str, buf.shape))}]"
                body_lines.append(f"  {buf.scope.value} {buf.dtype} {buf.name}{shape_str};")
            else:
                body_lines.append(f"  {buf.scope.value} {buf.dtype} {buf.name};")

        # Generate operations in topological order
        ordered_nodes = self._topological_sort()
        for node in ordered_nodes:
            node_dsl = node.generate_dsl()
            body_lines.append(f"  {node_dsl};")

        # Combine into complete function
        if body_lines:
            body = "\n".join(body_lines)
            return f"{function_signature} {{\n{body}\n}}"
        else:
            return f"{function_signature} {{}}"

    def _topological_sort(self) -> list[ConductorNode]:
        """
        Sort nodes in topological order for correct execution sequence.

        Returns:
            List of nodes in topological order
        """
        # Build dependency graph
        in_degree = {node: 0 for node in self.nodes}
        dependencies = {node: [] for node in self.nodes}

        for node in self.nodes:
            node_input_names = {buf.name for buf in node.inputs}
            for other_node in self.nodes:
                if other_node != node:
                    other_output_names = {buf.name for buf in other_node.outputs}
                    if node_input_names.intersection(other_output_names):
                        dependencies[other_node].append(node)
                        in_degree[node] += 1

        # Kahn's algorithm for topological sorting
        queue = [node for node in self.nodes if in_degree[node] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for dependent in dependencies[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # If not all nodes are processed, there's a cycle
        if len(result) != len(self.nodes):
            # Fallback to original order
            return self.nodes

        return result

    def estimate_performance_gain(self) -> float:
        """
        Estimate performance improvement from fusion.

        Returns:
            Estimated performance gain ratio (>1.0 means improvement)
        """
        if len(self.nodes) <= 1:
            return 1.0  # No gain from single operation

        # Base gain from reduced kernel launches
        kernel_launch_savings = (len(self.nodes) - 1) * 0.15

        # Additional gain from memory locality
        memory_locality_gain = 0.0
        if self.internal_buffers:
            # More internal buffers = better memory locality
            memory_locality_gain = len(self.internal_buffers) * 0.05

        # Penalty for complex fusion (compilation overhead)
        complexity_penalty = 0.0
        if len(self.nodes) > 5:
            complexity_penalty = (len(self.nodes) - 5) * 0.02

        # Type-specific gains
        type_gain = 0.0
        if self.cluster_type == FusionType.ELEMENTWISE:
            type_gain = 0.1  # Elementwise fusion is very effective
        elif self.cluster_type == FusionType.REDUCTION:
            type_gain = 0.2  # Reduction fusion saves significant bandwidth
        elif self.cluster_type == FusionType.MEMORY_BOUND:
            type_gain = 0.15  # Good for memory-bound operations

        total_gain = (
            1.0 + kernel_launch_savings + memory_locality_gain + type_gain - complexity_penalty
        )
        return max(1.0, total_gain)  # Ensure gain is at least 1.0 (no loss)


class FusionEngine:
    """
    Implements operation fusion heuristics and optimization.

    This class analyzes the computation DAG to identify fusion
    opportunities and creates optimized fusion clusters.
    """

    def identify_fusion_opportunities(self, dag: ComputationDAG) -> list[FusionCluster]:
        """
        Find groups of operations that can be safely fused.

        Args:
            dag: Computation DAG to analyze for fusion opportunities

        Returns:
            List of fusion clusters representing optimization opportunities
        """
        clusters = []
        visited_nodes = set()

        # Find elementwise chains first
        for node in dag.nodes:
            if node not in visited_nodes and self._is_elementwise_op(node.op_name):
                chain = self._find_elementwise_chain(node, dag, visited_nodes)
                if len(chain) > 1:  # Only create cluster if multiple operations
                    cluster = self.apply_elementwise_fusion(chain)
                    self._populate_cluster_buffers(cluster, dag)
                    if cluster.validate_fusion_safety():
                        clusters.append(cluster)
                        visited_nodes.update(chain)

        # Find elementwise + reduction patterns
        for node in dag.nodes:
            if node not in visited_nodes and self._is_reduction_op(node.op_name):
                # Look for elementwise operations that feed into this reduction
                elementwise_producers = self._find_elementwise_producers(node, dag, visited_nodes)
                if elementwise_producers:
                    fusion_nodes = elementwise_producers + [node]
                    cluster = self.apply_reduction_fusion(fusion_nodes)
                    self._populate_cluster_buffers(cluster, dag)
                    if cluster.validate_fusion_safety():
                        clusters.append(cluster)
                        visited_nodes.update(fusion_nodes)

        return clusters

    def _is_elementwise_op(self, op_name: str) -> bool:
        """Check if operation is elementwise."""
        elementwise_ops = {
            "add",
            "sub",
            "mul",
            "div",
            "relu",
            "sigmoid",
            "tanh",
            "abs",
            "neg",
            "exp",
            "log",
            "sqrt",
            "sin",
            "cos",
        }
        return op_name in elementwise_ops

    def _is_reduction_op(self, op_name: str) -> bool:
        """Check if operation is a reduction."""
        reduction_ops = {"sum", "mean", "max", "min", "argmax", "argmin"}
        return op_name in reduction_ops

    def _find_elementwise_chain(
        self, start_node: ConductorNode, dag: ComputationDAG, visited: set
    ) -> list[ConductorNode]:
        """Find a chain of consecutive elementwise operations starting from a node."""
        chain = [start_node]
        current = start_node

        while True:
            # Find the next elementwise operation in the chain
            next_node = None

            # Look for a single consumer that is also elementwise
            if len(current.outputs) == 1:
                output_buffer = current.outputs[0]
                elementwise_consumers = [
                    consumer
                    for consumer in output_buffer.consumers
                    if (
                        consumer not in visited
                        and self._is_elementwise_op(consumer.op_name)
                        and len(consumer.inputs) == 1
                    )  # Single input for simple chaining
                ]

                if len(elementwise_consumers) == 1:
                    candidate = elementwise_consumers[0]
                    # Check if this consumer only has one producer (our current node)
                    if len(candidate.inputs) == 1 and candidate.inputs[0] == output_buffer:
                        next_node = candidate

            if next_node is None:
                break

            chain.append(next_node)
            current = next_node

        return chain

    def _find_elementwise_producers(
        self, reduction_node: ConductorNode, dag: ComputationDAG, visited: set
    ) -> list[ConductorNode]:
        """Find elementwise operations that feed into a reduction."""
        producers = []

        for input_buffer in reduction_node.inputs:
            if input_buffer.producer and input_buffer.producer not in visited:
                producer = input_buffer.producer
                if self._is_elementwise_op(producer.op_name):
                    # Check if this producer only feeds into the reduction
                    if len(producer.outputs) == 1 and len(producer.outputs[0].consumers) == 1:
                        producers.append(producer)

        return producers

    def _populate_cluster_buffers(self, cluster: FusionCluster, dag: "ComputationDAG") -> None:
        """Populate external inputs, outputs, and internal buffers for a cluster."""
        if not cluster.nodes:
            return

        cluster_nodes_set = set(cluster.nodes)
        all_inputs = set()
        all_outputs = set()

        # Collect all inputs and outputs from cluster nodes
        for node in cluster.nodes:
            all_inputs.update(node.inputs)
            all_outputs.update(node.outputs)

        # Determine external inputs (not produced by cluster nodes)
        for buffer in all_inputs:
            if buffer.producer is None or buffer.producer not in cluster_nodes_set:
                cluster.external_inputs.append(buffer)

        # Determine external outputs (consumed outside cluster or are graph outputs)
        for buffer in all_outputs:
            is_external = False

            # Check if consumed outside cluster
            for consumer in buffer.consumers:
                if consumer not in cluster_nodes_set:
                    is_external = True
                    break

            # Check if it's a graph output
            if buffer in dag.outputs:
                is_external = True

            if is_external:
                cluster.external_outputs.append(buffer)

        # Determine internal buffers (produced and consumed within cluster)
        for buffer in all_outputs:
            if buffer not in cluster.external_outputs:
                cluster.internal_buffers.append(buffer)

    def _analyze_buffer_lifetimes(self, cluster: FusionCluster) -> dict:
        """Analyze buffer lifetimes within a fusion cluster."""
        lifetimes = {}

        # Simple lifetime analysis: when buffer is created and last used
        for i, node in enumerate(cluster.nodes):
            # Buffers created by this node
            for output_buffer in node.outputs:
                if output_buffer not in lifetimes:
                    lifetimes[output_buffer] = {"start": i, "end": i}

            # Buffers used by this node
            for input_buffer in node.inputs:
                if input_buffer in lifetimes:
                    lifetimes[input_buffer]["end"] = i

        return lifetimes

    def _find_buffer_reuse_opportunities(self, lifetimes: dict) -> dict:
        """Find opportunities to reuse buffers with non-overlapping lifetimes."""
        reuse_map = {}

        # Simple heuristic: find buffers with same shape/dtype and non-overlapping lifetimes
        buffers = list(lifetimes.keys())

        for i, buffer1 in enumerate(buffers):
            for _, buffer2 in enumerate(buffers[i + 1 :], i + 1):
                if (
                    buffer1.dtype == buffer2.dtype
                    and buffer1.shape == buffer2.shape
                    and self._lifetimes_non_overlapping(lifetimes[buffer1], lifetimes[buffer2])
                ):

                    # Reuse the earlier buffer for the later one
                    if lifetimes[buffer1]["start"] < lifetimes[buffer2]["start"]:
                        reuse_map[buffer2] = buffer1
                    else:
                        reuse_map[buffer1] = buffer2

        return reuse_map

    def _lifetimes_non_overlapping(self, lifetime1: dict, lifetime2: dict) -> bool:
        """Check if two buffer lifetimes don't overlap."""
        return lifetime1["end"] < lifetime2["start"] or lifetime2["end"] < lifetime1["start"]

    # TODO: implement
    def _apply_buffer_reuse(self, cluster: FusionCluster, original: Buffer, reuse: Buffer) -> None:
        """Apply buffer reuse optimization within cluster."""
        # This would update the cluster's internal representation
        # For now, just mark the optimization opportunity
        pass

    def _optimize_internal_buffer_scopes(self, cluster: FusionCluster) -> None:
        """Optimize buffer scopes for internal buffers."""
        # Internal buffers can use LOCAL scope for better performance
        for buffer in cluster.internal_buffers:
            if buffer.scope != BufferScope.LOCAL:
                buffer.promote_scope(BufferScope.LOCAL)

    def apply_elementwise_fusion(self, nodes: list[ConductorNode]) -> FusionCluster:
        """
        Fuse consecutive elementwise operations.

        Args:
            nodes: List of elementwise nodes to fuse

        Returns:
            FusionCluster containing the fused operations
        """
        if not nodes:
            return FusionCluster()

        # Generate unique function name
        op_names = [node.op_name for node in nodes]
        function_name = f"fused_{'_'.join(op_names)}_{id(nodes[0])}"

        cluster = FusionCluster(
            nodes=nodes, cluster_type=FusionType.ELEMENTWISE, dsl_function_name=function_name
        )

        return cluster

    def apply_reduction_fusion(self, nodes: list[ConductorNode]) -> FusionCluster:
        """
        Fuse elementwise operations with following reductions.

        Args:
            nodes: List of nodes including elementwise and reduction operations

        Returns:
            FusionCluster containing the fused operations
        """
        if not nodes:
            return FusionCluster()

        # Generate unique function name
        op_names = [node.op_name for node in nodes]
        function_name = f"fused_{'_'.join(op_names)}_{id(nodes[0])}"

        cluster = FusionCluster(
            nodes=nodes, cluster_type=FusionType.REDUCTION, dsl_function_name=function_name
        )

        return cluster

    def optimize_buffer_usage(self, cluster: FusionCluster) -> None:
        """
        Optimize memory usage within fusion clusters.

        Args:
            cluster: FusionCluster to optimize
        """
        if not cluster.nodes:
            return

        # Identify buffers that can be reused within the cluster
        buffer_lifetimes = self._analyze_buffer_lifetimes(cluster)
        reuse_opportunities = self._find_buffer_reuse_opportunities(buffer_lifetimes)

        # Apply buffer reuse optimizations
        for original_buffer, reuse_buffer in reuse_opportunities.items():
            self._apply_buffer_reuse(cluster, original_buffer, reuse_buffer)

        # Promote internal buffer scopes appropriately
        self._optimize_internal_buffer_scopes(cluster)


class FusionHeuristics:
    """
    Implements fusion decision logic and compatibility checking.

    This class contains the rules and heuristics used to determine
    when operations can be safely and beneficially fused.
    """

    def can_fuse_elementwise(self, op1: str, op2: str) -> bool:
        """
        Determine if two elementwise operations can be fused using unified operator system.

        Args:
            op1: First operation name
            op2: Second operation name

        Returns:
            True if operations can be fused, False otherwise
        """
        # Try unified operator system first
        if UNIFIED_OPERATORS_AVAILABLE:
            return can_operators_fuse(op1, op2)

        # Legacy fusion logic for operations without templates
        elementwise_ops = {
            "add",
            "sub",
            "mul",
            "div",
            "sigmoid",
            "tanh",
            "abs",
            "neg",
            "exp",
            "log",
            "sqrt",
            "sin",
            "cos",
        }

        # ReLU is excluded from fusion due to Choreo syntax limitations
        non_fusible_ops = {"relu"}

        # Both operations must be elementwise and fusible
        if not (op1 in elementwise_ops and op2 in elementwise_ops):
            return False

        # Prevent fusion with ReLU
        if op1 in non_fusible_ops or op2 in non_fusible_ops:
            return False

        # Check for incompatible operation combinations
        incompatible_pairs = {
            ("div", "log"),
            ("log", "div"),  # Numerical stability issues
        }

        return (op1, op2) not in incompatible_pairs and (op2, op1) not in incompatible_pairs

    def estimate_fusion_benefit(self, nodes: list[ConductorNode]) -> float:
        """
        Estimate performance benefit of fusing given nodes using unified operator metadata.

        Args:
            nodes: List of nodes to potentially fuse

        Returns:
            Estimated benefit score (higher is better)
        """
        if len(nodes) <= 1:
            return 0.0

        # Base benefit from kernel launch reduction
        kernel_launch_benefit = (len(nodes) - 1) * 0.3

        # Memory bandwidth benefit
        memory_benefit = min(len(nodes) * 0.1, 0.5)

        # Use unified operator metadata if available
        if UNIFIED_OPERATORS_AVAILABLE:
            return self._estimate_benefit_with_unified_metadata(
                nodes, kernel_launch_benefit, memory_benefit
            )

        # Legacy benefit estimation
        op_types = [node.op_name for node in nodes]

        # Elementwise chains are highly beneficial
        elementwise_ops = {"add", "sub", "mul", "div", "relu", "sigmoid", "tanh", "abs", "neg"}
        elementwise_count = sum(1 for op in op_types if op in elementwise_ops)
        elementwise_benefit = elementwise_count * 0.15

        # Reduction fusion is very beneficial
        reduction_ops = {"sum", "mean", "max", "min"}
        has_reduction = any(op in reduction_ops for op in op_types)
        reduction_benefit = 0.4 if has_reduction else 0.0

        # Penalty for complex operations
        complex_ops = {"matmul", "conv2d", "linear"}
        complex_count = sum(1 for op in op_types if op in complex_ops)
        complexity_penalty = complex_count * 0.2

        total_benefit = (
            kernel_launch_benefit
            + memory_benefit
            + elementwise_benefit
            + reduction_benefit
            - complexity_penalty
        )
        return max(0.0, total_benefit)

    def _estimate_benefit_with_unified_metadata(
        self, nodes: list[ConductorNode], base_kernel_benefit: float, base_memory_benefit: float
    ) -> float:
        """Estimate fusion benefit using unified operator metadata."""
        total_benefit = base_kernel_benefit + base_memory_benefit

        # Analyze each operation using unified metadata
        for node in nodes:
            metadata = get_fusion_metadata(node.op_name)
            if metadata:
                # Element-wise operations are highly fusable
                if metadata.element_wise:
                    total_benefit += 0.15

                # Memory-bound operations benefit more from fusion
                if metadata.memory_bound:
                    total_benefit += 0.1

                # Higher compute intensity operations benefit less
                compute_penalty = max(0, (metadata.compute_intensity - 1.0) * 0.05)
                total_benefit -= compute_penalty

                # Use fusion priority
                priority_bonus = (metadata.fusion_priority - 1) * 0.05
                total_benefit += priority_bonus

        return max(0.0, total_benefit)

    def check_memory_constraints(self, cluster: FusionCluster) -> bool:
        """
        Verify fusion doesn't exceed memory limitations.

        Args:
            cluster: FusionCluster to check

        Returns:
            True if memory constraints are satisfied, False otherwise
        """
        # Calculate total memory usage for the cluster
        total_memory = 0

        # Memory for external inputs and outputs
        for buffer in cluster.external_inputs + cluster.external_outputs:
            memory = buffer.get_memory_footprint()
            if memory > 0:
                total_memory += memory

        # Memory for internal buffers
        for buffer in cluster.internal_buffers:
            memory = buffer.get_memory_footprint()
            if memory > 0:
                total_memory += memory

        # Conservative memory limit: 1GB for fusion clusters
        # This prevents excessive memory usage that could hurt performance
        memory_limit = 1024 * 1024 * 1024  # 1GB in bytes

        # If we can't determine memory usage, assume it's okay
        if total_memory <= 0:
            return True

        return total_memory <= memory_limit

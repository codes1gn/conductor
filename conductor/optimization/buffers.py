"""
Buffer management and optimization.

This module provides classes and utilities for managing data buffers
in the computation graph, including scope management, memory optimization,
and lifetime analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Set, List, Tuple
import torch
from .graph_nodes import ConductorNode
from ..utils.constants import BufferScope
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Buffer:
    """
    Represents data flow and memory management in the computation graph.

    This class encapsulates all information needed to track data buffers
    throughout the compilation pipeline, including scope management and
    dependency tracking.
    """

    name: str  # Unique identifier for the buffer
    scope: BufferScope  # Memory scope (LOCAL, SHARED, GLOBAL)
    dtype: torch.dtype  # Data type information
    shape: Optional[tuple[int, ...]] = None  # Shape when statically known
    producer: Optional[ConductorNode] = None  # Node that produces this buffer
    consumers: list[ConductorNode] = None  # Nodes that consume this buffer
    is_temporary: bool = False  # Whether this is a temporary intermediate buffer

    # Lifetime analysis fields
    birth_time: int = -1  # When buffer is first produced (topological order)
    death_time: int = -1  # When buffer is last consumed (topological order)
    live_range: Tuple[int, int] = field(default_factory=lambda: (-1, -1))  # (birth, death)
    memory_size: int = 0  # Estimated memory size in bytes
    is_input: bool = False  # Whether this is a graph input
    is_output: bool = False  # Whether this is a graph output

    def __post_init__(self):
        if self.consumers is None:
            self.consumers = []

        # Calculate memory size if shape is known
        if self.shape:
            self.memory_size = self._calculate_memory_size()

    def _calculate_memory_size(self) -> int:
        """Calculate estimated memory size in bytes."""
        if not self.shape:
            return 0

        # Calculate total elements
        total_elements = 1
        for dim in self.shape:
            total_elements *= dim

        # Get dtype size in bytes
        dtype_size = self._get_dtype_size_bytes()

        return total_elements * dtype_size

    def _get_dtype_size_bytes(self) -> int:
        """Get size of dtype in bytes."""
        dtype_sizes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float64: 8,
            torch.int32: 4,
            torch.int64: 8,
            torch.int8: 1,
            torch.uint8: 1,
            torch.bool: 1,
            torch.complex64: 8,
            torch.complex128: 16,
        }
        return dtype_sizes.get(self.dtype, 4)  # Default to 4 bytes

    def set_lifetime(self, birth_time: int, death_time: int) -> None:
        """
        Set the lifetime of this buffer.

        Args:
            birth_time: When buffer is first produced (topological order)
            death_time: When buffer is last consumed (topological order)
        """
        self.birth_time = birth_time
        self.death_time = death_time
        self.live_range = (birth_time, death_time)

    def is_alive_at(self, time: int) -> bool:
        """
        Check if buffer is alive at given time.

        Args:
            time: Time point to check

        Returns:
            True if buffer is alive at given time
        """
        if self.birth_time == -1 or self.death_time == -1:
            return False
        return self.birth_time <= time <= self.death_time

    def overlaps_with(self, other: 'Buffer') -> bool:
        """
        Check if this buffer's lifetime overlaps with another buffer.

        Args:
            other: Other buffer to check overlap with

        Returns:
            True if lifetimes overlap
        """
        if (self.birth_time == -1 or self.death_time == -1 or
            other.birth_time == -1 or other.death_time == -1):
            return False

        return not (self.death_time < other.birth_time or other.death_time < self.birth_time)

    def can_reuse_memory_with(self, other: 'Buffer') -> bool:
        """
        Check if this buffer can reuse memory with another buffer.

        Args:
            other: Other buffer to check reuse compatibility

        Returns:
            True if memory can be reused
        """
        # Cannot reuse if lifetimes overlap
        if self.overlaps_with(other):
            return False

        # Cannot reuse if different scopes
        if self.scope != other.scope:
            return False

        # Cannot reuse if different dtypes
        if self.dtype != other.dtype:
            return False

        # Cannot reuse if one is input/output and other is not
        if (self.is_input or self.is_output) != (other.is_input or other.is_output):
            return False

        # Can reuse if memory sizes are compatible
        return self.memory_size <= other.memory_size or other.memory_size <= self.memory_size

    def __hash__(self):
        """Make Buffer hashable for use in sets and dictionaries."""
        # Use id() to ensure unique hash for each instance
        return hash(id(self))

    def __eq__(self, other):
        """Define equality based on object identity."""
        return self is other

    def promote_scope(self, new_scope: BufferScope) -> None:
        """
        Promote buffer to higher scope when needed for sharing.

        Args:
            new_scope: Target scope to promote to
        """
        # Define scope hierarchy: LOCAL < SHARED < GLOBAL
        scope_hierarchy = {BufferScope.LOCAL: 0, BufferScope.SHARED: 1, BufferScope.GLOBAL: 2}

        if scope_hierarchy[new_scope] > scope_hierarchy[self.scope]:
            self.scope = new_scope

    def get_memory_footprint(self) -> int:
        """
        Calculate memory requirements for this buffer.

        Returns:
            Memory footprint in bytes, or -1 if shape is unknown
        """
        if self.shape is None:
            return -1  # Unknown size

        # Calculate total elements
        total_elements = 1
        for dim in self.shape:
            total_elements *= dim

        # Get dtype size in bytes
        dtype_sizes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.int32: 4,
            torch.int64: 8,
            torch.bool: 1,
        }

        dtype_size = dtype_sizes.get(self.dtype, 4)  # Default to 4 bytes
        return total_elements * dtype_size


class BufferManager:
    """
    Manages buffer allocation and scope promotion.

    This class handles the lifecycle of buffers throughout the compilation
    process, including automatic scope promotion and memory optimization.
    """

    def __init__(self):
        self._buffers: dict[str, Buffer] = {}
        self._buffer_counter = 0

    def allocate_buffer(
        self, name: str, dtype: torch.dtype, shape: Optional[tuple[int, ...]] = None
    ) -> Buffer:
        """
        Allocate new buffer with appropriate initial scope.

        Args:
            name: Buffer identifier
            dtype: Data type for the buffer
            shape: Optional shape information

        Returns:
            Newly allocated Buffer instance
        """
        # Generate unique name if needed
        if name in self._buffers:
            self._buffer_counter += 1
            name = f"{name}_{self._buffer_counter}"

        # Start with LOCAL scope by default
        buffer = Buffer(name=name, scope=BufferScope.LOCAL, dtype=dtype, shape=shape)

        self._buffers[name] = buffer
        return buffer

    def promote_buffer_scope(self, buffer: Buffer, required_scope: BufferScope) -> None:
        """
        Promote buffer to higher scope when sharing is required.

        Args:
            buffer: Buffer to promote
            required_scope: Minimum required scope
        """
        buffer.promote_scope(required_scope)

    def optimize_buffer_reuse(self, buffers: list[Buffer]) -> dict[str, str]:
        """
        Identify opportunities for buffer reuse to reduce memory footprint.

        Args:
            buffers: List of buffers to analyze for reuse opportunities

        Returns:
            Dictionary mapping original buffer names to reused buffer names
        """
        reuse_map = {}

        # Simple heuristic: reuse buffers with same shape and dtype
        # that don't have overlapping lifetimes
        for i, buffer1 in enumerate(buffers):
            if buffer1.name in reuse_map:
                continue

            for j, buffer2 in enumerate(buffers[i + 1 :], i + 1):
                if (
                    buffer2.name not in reuse_map
                    and buffer1.dtype == buffer2.dtype
                    and buffer1.shape == buffer2.shape
                    and self._can_reuse_buffers(buffer1, buffer2)
                ):
                    reuse_map[buffer2.name] = buffer1.name
                    break

        return reuse_map

    def _can_reuse_buffers(self, buffer1: Buffer, buffer2: Buffer) -> bool:
        """
        Check if two buffers can safely reuse the same memory using proper lifetime analysis.

        Args:
            buffer1: First buffer
            buffer2: Second buffer

        Returns:
            True if buffers can be reused, False otherwise
        """
        return buffer1.can_reuse_memory_with(buffer2)

    def get_buffer(self, name: str) -> Optional[Buffer]:
        """
        Retrieve buffer by name.

        Args:
            name: Buffer identifier

        Returns:
            Buffer instance if found, None otherwise
        """
        return self._buffers.get(name)

    def list_buffers(self) -> list[Buffer]:
        """
        Get list of all managed buffers.

        Returns:
            List of all Buffer instances
        """
        return list(self._buffers.values())

    def analyze_buffer_lifetimes(self, dag) -> Dict[str, Tuple[int, int]]:
        """
        Perform comprehensive lifetime analysis for all buffers in the DAG.

        Args:
            dag: Computation DAG to analyze

        Returns:
            Dictionary mapping buffer names to (birth_time, death_time) tuples
        """
        from ..utils.tracer import get_tracer
        tracer = get_tracer()

        if tracer.should_trace_dag():
            logger.info("=== Buffer Lifetime Analysis ===")

        # Step 1: Create topological ordering of nodes
        topo_order = self._create_topological_order(dag)

        if tracer.should_trace_dag():
            logger.info(f"Topological order: {[node.op_name for node in topo_order]}")

        # Step 2: Assign time indices to nodes
        node_times = {node: i for i, node in enumerate(topo_order)}

        # Step 3: Calculate birth and death times for each buffer
        lifetime_map = {}

        for buffer in dag.buffers:
            birth_time = self._calculate_birth_time(buffer, node_times)
            death_time = self._calculate_death_time(buffer, node_times)

            # Set lifetime in buffer object
            buffer.set_lifetime(birth_time, death_time)
            lifetime_map[buffer.name] = (birth_time, death_time)

            if tracer.should_trace_dag():
                logger.info(f"  {buffer.name}: birth={birth_time}, death={death_time}, size={buffer.memory_size}B")

        # Step 4: Identify reuse opportunities
        if tracer.should_trace_dag():
            self._trace_reuse_opportunities(dag.buffers)

        return lifetime_map

    def _create_topological_order(self, dag) -> List[ConductorNode]:
        """Create topological ordering of nodes in the DAG."""
        # Simple topological sort using Kahn's algorithm
        in_degree = {node: 0 for node in dag.nodes}

        # Calculate in-degrees
        for node in dag.nodes:
            for output_buffer in node.outputs:
                for consumer in output_buffer.consumers:
                    if consumer in in_degree:
                        in_degree[consumer] += 1

        # Start with nodes that have no dependencies
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # Update in-degrees of dependent nodes
            for output_buffer in current.outputs:
                for consumer in output_buffer.consumers:
                    if consumer in in_degree:
                        in_degree[consumer] -= 1
                        if in_degree[consumer] == 0:
                            queue.append(consumer)

        return result

    def _calculate_birth_time(self, buffer: Buffer, node_times: Dict[ConductorNode, int]) -> int:
        """Calculate when a buffer is born (first produced)."""
        if buffer.producer and buffer.producer in node_times:
            return node_times[buffer.producer]
        elif buffer.is_input:
            return 0  # Input buffers are born at time 0
        else:
            return -1  # Unknown birth time

    def _calculate_death_time(self, buffer: Buffer, node_times: Dict[ConductorNode, int]) -> int:
        """Calculate when a buffer dies (last consumed)."""
        if buffer.is_output:
            # Output buffers live until the end
            return max(node_times.values()) if node_times else 0
        elif buffer.consumers:
            # Buffer dies after its last consumer
            consumer_times = [node_times[consumer] for consumer in buffer.consumers if consumer in node_times]
            return max(consumer_times) if consumer_times else -1
        else:
            # No consumers, dies immediately after birth
            birth_time = self._calculate_birth_time(buffer, node_times)
            return birth_time if birth_time != -1 else -1

    def _trace_reuse_opportunities(self, buffers: List[Buffer]) -> None:
        """Trace potential buffer reuse opportunities."""
        logger.info("  Reuse opportunities:")

        reuse_count = 0
        for i, buffer1 in enumerate(buffers):
            for buffer2 in buffers[i+1:]:
                if buffer1.can_reuse_memory_with(buffer2):
                    reuse_count += 1
                    logger.info(f"    {buffer1.name} ↔ {buffer2.name} (save {min(buffer1.memory_size, buffer2.memory_size)}B)")

        if reuse_count == 0:
            logger.info("    No reuse opportunities found")
        else:
            logger.info(f"  Total reuse opportunities: {reuse_count}")

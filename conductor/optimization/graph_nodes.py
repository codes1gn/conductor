"""
Graph Node Definitions for Conductor.

This module contains the core node classes used in the computation graph
for optimization and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from ..codegen.operator_registry import is_elementwise, is_reduction


@dataclass
class ConductorNode:
    """
    Represents a single operation node in the computation graph.

    This is the internal representation of operations that will be
    compiled to Choreo DSL.
    """

    op_name: str
    inputs: list[Any] = field(default_factory=list)  # Can be Buffer objects or strings
    outputs: list[Any] = field(default_factory=list)  # Can be Buffer objects or strings
    metadata: dict[str, Any] = field(default_factory=dict)
    fusion_group: Optional[Any] = None  # Fusion cluster membership

    def __post_init__(self):
        """Post-initialization processing."""
        if not self.inputs:
            self.inputs = []
        if not self.outputs:
            self.outputs = []
        if not self.metadata:
            self.metadata = {}

    def add_input(self, input_name: str) -> None:
        """Add an input to this node."""
        if input_name not in self.inputs:
            self.inputs.append(input_name)

    def add_output(self, output_name: str) -> None:
        """Add an output to this node."""
        if output_name not in self.outputs:
            self.outputs.append(output_name)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for this node."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata for this node."""
        return self.metadata.get(key, default)

    def is_elementwise(self) -> bool:
        """Check if this operation is elementwise."""
        return is_elementwise(self.op_name)

    def is_reduction(self) -> bool:
        """Check if this operation is a reduction."""
        return is_reduction(self.op_name)

    def can_fuse_with(self, other: 'ConductorNode') -> bool:
        """
        Check if this node can be fused with another node.

        Args:
            other: Another ConductorNode to check fusion compatibility with

        Returns:
            True if nodes can be fused, False otherwise
        """
        # Import fusion heuristics to check compatibility
        from .fusion import FusionHeuristics
        heuristics = FusionHeuristics()

        # Check different fusion patterns
        # 1. Both elementwise operations
        if self.is_elementwise() and other.is_elementwise():
            return heuristics.can_fuse_elementwise(self.op_name, other.op_name)

        # 2. Reduction + elementwise operations
        if self.is_reduction() and other.is_elementwise():
            return heuristics.can_fuse_elementwise(self.op_name, other.op_name)

        # 3. Elementwise + reduction operations
        if self.is_elementwise() and other.is_reduction():
            return heuristics.can_fuse_elementwise(self.op_name, other.op_name)

        # Other combinations are not supported
        return False

    def generate_dsl(self) -> str:
        """
        Generate DSL code for this node.

        Returns:
            DSL code string for this operation
        """
        # Simple DSL generation - this would be expanded based on operation type
        if self.op_name == "add":
            return f"{self.outputs[0].name} = {self.inputs[0].name} + {self.inputs[1].name}"
        elif self.op_name == "mul":
            return f"{self.outputs[0].name} = {self.inputs[0].name} * {self.inputs[1].name}"
        else:
            # Generic operation
            input_names = [inp.name if hasattr(inp, 'name') else str(inp) for inp in self.inputs]
            output_names = [out.name if hasattr(out, 'name') else str(out) for out in self.outputs]
            return f"{', '.join(output_names)} = {self.op_name}({', '.join(input_names)})"

    def __str__(self) -> str:
        return f"ConductorNode(op={self.op_name}, inputs={self.inputs}, outputs={self.outputs})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self):
        """Make ConductorNode hashable for use in sets and dictionaries."""
        return hash(id(self))

    def __eq__(self, other):
        """Define equality based on object identity."""
        return self is other

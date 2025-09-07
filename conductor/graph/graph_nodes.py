"""
Graph Node Definitions for Conductor.

This module contains the core node classes used in the computation graph
to avoid circular import issues.
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

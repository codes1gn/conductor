#!/usr/bin/env python3
"""
Registry-Based Template System

This module implements a clean DSL generation system that properly integrates
with the existing operator registry, eliminating duplication and providing
a single source of truth for DSL generation.

Key principles:
1. Use existing OperatorTemplate definitions from operator_registry
2. Extract computation logic from templates when needed for fusion
3. Generate consistent DSL output format
4. Eliminate all code duplication
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import re

from .operator_registry import (
    operator_registry,
    get_operator_template,
    OperatorInfo as OperatorTemplate,
)
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..graph.graph_nodes import ConductorNode
    from ..graph.buffers import Buffer

logger = get_logger(__name__)


@dataclass
class TemplateContext:
    """Context for template rendering with all necessary parameters."""

    function_name: str
    input_buffers: list[Buffer] = field(default_factory=list)
    output_buffers: list[Buffer] = field(default_factory=list)
    dimensions: dict[str, int] = field(
        default_factory=lambda: {"M": 16, "N": 16, "P": 4, "buffer_m": 16, "buffer_n": 8}
    )
    custom_params: dict[str, Any] = field(default_factory=dict)

    def get_template_params(self) -> dict[str, Any]:
        """Get all parameters for template substitution."""
        params = {"function_name": self.function_name, **self.dimensions, **self.custom_params}

        # Add buffer-derived parameters
        if self.input_buffers and self.input_buffers[0].shape:
            shape = self.input_buffers[0].shape
            if len(shape) >= 2:
                params["M"] = shape[0] if isinstance(shape[0], int) else 16
                params["N"] = shape[1] if isinstance(shape[1], int) else 16

        return params


class RegistryBasedTemplateEngine:
    """Template engine that uses the operator registry as the single source of truth."""

    def __init__(self):
        logger.info("Initialized registry-based template engine")

    def render_operation(self, node: "ConductorNode", context: TemplateContext) -> str:
        """Render a single operation using operator registry templates."""
        # Import at runtime to avoid circular imports
        from ..graph.graph_nodes import ConductorNode

        # Get template from registry
        template = get_operator_template(node.op_name)
        if not template:
            raise ValueError(f"No template found for operation: {node.op_name}")

        # Use the new generate_code method with proper variable names
        # For now, use simple variable names - this should be improved to use systematic naming
        input_vars = ["input0", "input1"]  # TODO: Use systematic naming from DAGNamingAnnotator
        output_var = "output"  # TODO: Use systematic naming from DAGNamingAnnotator
        index_var = "i"  # TODO: Use systematic naming from DAGNamingAnnotator

        # Generate code using the new operator template system
        result = template.generate_code(input_vars, output_var, index_var)

        logger.info(f"Rendered operation {node.op_name} using registry template")
        return result

    def render_fused_operations(self, nodes: list[ConductorNode], context: TemplateContext) -> str:
        """Render fused operations by combining computations."""
        if len(nodes) == 1:
            return self.render_operation(nodes[0], context)

        # Simple fusion: just chain the operations
        fused_computation = ""
        for i, node in enumerate(nodes):
            if i == 0:
                fused_computation += f"temp{i} = {node.op_name}(input0, input1);"
            elif i == len(nodes) - 1:
                fused_computation += f" output = {node.op_name}(temp{i-1});"
            else:
                fused_computation += f" temp{i} = {node.op_name}(temp{i-1});"

        # Create simple fused template
        params = context.get_template_params()
        params["fused_computation"] = fused_computation

        # Use first template as base and substitute fused computation
        base_template = get_operator_template(nodes[0].op_name)
        if base_template and base_template.code_template:
            result = base_template.code_template.replace(
                "output[i] = input0[i] + input1[i];", fused_computation
            )
            logger.info(f"Rendered fused operations: {[n.op_name for n in nodes]}")
            return result
        else:
            # Fallback to simple concatenation
            return (
                f"// Fused operations: {', '.join(n.op_name for n in nodes)}\n{fused_computation}"
            )

    def list_available_operations(self) -> list[str]:
        """List all operations available in the registry."""
        return operator_registry.list_operators()

    def get_operation_metadata(self, op_name: str) -> Optional[Any]:
        """Get metadata for an operation."""
        template = get_operator_template(op_name)
        return template.metadata if template else None


# Global instance
registry_template_engine = RegistryBasedTemplateEngine()


def render_node_with_registry_templates(
    node: "ConductorNode", function_name: str = "generated_function"
) -> str:
    """Convenience function to render a single node."""
    context = TemplateContext(function_name=function_name)
    return registry_template_engine.render_operation(node, context)


def render_fused_nodes_with_registry_templates(
    nodes: list[ConductorNode], function_name: str = "generated_function"
) -> str:
    """Convenience function to render fused nodes."""
    context = TemplateContext(function_name=function_name)
    return registry_template_engine.render_fused_operations(nodes, context)


# Initialize the registry to ensure templates are loaded
def initialize_registry_templates():
    """Initialize the operator registry templates."""
    # The operator_registry is automatically initialized when imported
    # This function exists for explicit initialization if needed
    available_ops = registry_template_engine.list_available_operations()
    logger.info(
        f"Registry-based template engine initialized with {len(available_ops)} operations: {available_ops}"
    )


# Auto-initialize when module is imported
initialize_registry_templates()

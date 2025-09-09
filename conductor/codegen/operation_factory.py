"""
Operation Factory for Conductor.

This module provides a direct approach to operation handling
following production-quality design principles.
"""

from __future__ import annotations

from typing import Optional, Any
import torch.fx as fx
from .operator_registry import get_op_desc, get_operator_registry, get_operator_template
from ..utils.logging import get_logger

# Delayed import to avoid circular dependency; used only for typing and construction
# from ..graph.graph_nodes import ConductorNode

logger = get_logger(__name__)


class OperationFactory:
    """
    Operation factory that directly uses operator templates
    without complex handler abstractions.
    """

    def __init__(self):
        self.operation_registry = get_operator_registry()
        logger.info("Initialized operation factory")

    def extract_operation_name(self, fx_node: fx.Node) -> str:
        """Extract operation name from FX node."""
        if fx_node.op == "call_function":
            if hasattr(fx_node.target, "__name__"):
                name = fx_node.target.__name__
            else:
                name = str(fx_node.target).split(".")[-1]
        elif fx_node.op == "call_method":
            name = fx_node.target
        else:
            return "unknown"

        # Handle common aliases
        if name in ["mm", "bmm"]:
            return "matmul"

        return name

    def create_conductor_node(
        self, fx_node: fx.Node, metadata: Optional[dict[str, Any]] = None
    ) -> ConductorNode:
        """Create a ConductorNode from an FX node."""

        op_name = self.extract_operation_name(fx_node)

        return ConductorNode(
            op_name=op_name,
            inputs=[],  # Will be populated by graph analyzer
            outputs=[],  # Will be populated by graph analyzer
            metadata=metadata or {},
        )

    def generate_dsl_code(self, node: ConductorNode, **kwargs) -> list[str]:
        """Generate DSL code for a ConductorNode using operator templates."""
        # Try to get template from operator registry first
        template = get_operator_template(node.op_name)
        if template:
            # Use template-based generation
            params = {
                "function_name": kwargs.get("function_name", "generated_function"),
                "input_vars": kwargs.get("input_vars", ["l1_input0"]),
                "output_var": kwargs.get("output_var", "l1_output"),
                "index_vars": kwargs.get("index_vars", "i, j"),
                **kwargs,
            }
            try:
                dsl_code = template.substitute_parameters(**params)
                return dsl_code.split("\n")
            except Exception as e:
                logger.warning(f"Template substitution failed for {node.op_name}: {e}")

        # Fallback to simple generation based on operation type
        return self._generate_simple_dsl(node, **kwargs)

    def _generate_simple_dsl(self, node: ConductorNode, **kwargs) -> list[str]:
        """Generate simple DSL code based on operation properties."""
        op_desc = get_op_desc(node.op_name)
        input_vars = kwargs.get("input_vars", ["l1_input0"])
        output_var = kwargs.get("output_var", "l1_output")
        index_vars = kwargs.get("index_vars", "i, j")

        if not op_desc:
            return [
                f"{output_var}[{index_vars}] = {input_vars[0]}[{index_vars}]; // Unknown: {node.op_name}"
            ]

        # Simple operation mapping
        op_symbols = {"add": "+", "mul": "*", "sub": "-", "div": "/", "relu": "max(0.0f, {})"}

        if op_desc.operation_type.value == "elementwise":
            if node.op_name in op_symbols:
                if node.op_name == "relu":
                    return [
                        f"{output_var}[{index_vars}] = {op_symbols[node.op_name].format(input_vars[0] + '[' + index_vars + ']')};"
                    ]
                elif len(input_vars) >= 2:
                    return [
                        f"{output_var}[{index_vars}] = {input_vars[0]}[{index_vars}] {op_symbols[node.op_name]} {input_vars[1]}[{index_vars}];"
                    ]
                else:
                    return [
                        f"{output_var}[{index_vars}] = {input_vars[0]}[{index_vars}]; // {node.op_name} (single input)"
                    ]
            else:
                return [
                    f"{output_var}[{index_vars}] = {node.op_name}({input_vars[0]}[{index_vars}]);"
                ]

        elif op_desc.operation_type.value == "matrix":
            if node.op_name in ["matmul", "mm", "bmm"]:
                return [
                    "// Matrix multiplication",
                    f"for (k: [0:K]) {{ {output_var}[{index_vars}] += {input_vars[0]}[i, k] * {input_vars[1]}[k, j]; }}",
                ]
            else:
                return [
                    f"{output_var}[{index_vars}] = {node.op_name}({input_vars[0]}[{index_vars}]); // matrix"
                ]

        elif op_desc.operation_type.value == "reduction":
            return [
                f"{output_var}[0] = {node.op_name}({input_vars[0]}[{index_vars}]); // reduction"
            ]

        else:
            return [
                f"{output_var}[{index_vars}] = {node.op_name}({input_vars[0]}[{index_vars}]); // {op_desc.operation_type.value}"
            ]

    def is_supported_operation(self, op_name: str) -> bool:
        """Check if an operation is supported."""
        return get_operator_template(op_name) is not None or get_op_desc(op_name) is not None

    def list_supported_operations(self) -> list[str]:
        """List all supported operations."""
        # Get all operations from unified registry
        return list(self.operation_registry.list_all_operations())

    def get_operation_info(self, op_name: str) -> Optional[dict[str, Any]]:
        """Get information about an operation."""
        op_desc = get_op_desc(op_name)
        template = get_operator_template(op_name)

        info = {}
        if op_desc:
            info.update(
                {
                    "name": op_desc.canonical_name,
                    "type": op_desc.operation_type.value,
                    "fusable": op_desc.fusable,
                    "memory_bound": op_desc.memory_bound,
                    "compute_intensity": op_desc.compute_intensity,
                    "compute_bound": op_desc.compute_bound,
                }
            )

        if template:
            info.update(
                {
                    "has_template": True,
                    "template_metadata": template.metadata.__dict__ if template.metadata else None,
                }
            )

        return info if info else None


# Global factory instance
def get_operation_factory() -> OperationFactory:
    """Get the operation factory from the global context."""
    from ..context import ensure_context_initialized
    context = ensure_context_initialized()
    return context.get_operation_factory()


# Convenience functions
def create_conductor_node(
    fx_node: fx.Node, metadata: Optional[dict[str, Any]] = None
) -> ConductorNode:
    """Create a ConductorNode from an FX node."""
    return operation_factory.create_conductor_node(fx_node, metadata)


def generate_dsl_code(node: ConductorNode, **kwargs) -> list[str]:
    """Generate DSL code for a ConductorNode."""
    return operation_factory.generate_dsl_code(node, **kwargs)


def extract_operation_name(fx_node: fx.Node) -> str:
    """Extract operation name from FX node."""
    return operation_factory.extract_operation_name(fx_node)


def is_supported_operation(op_name: str) -> bool:
    """Check if an operation is supported."""
    return operation_factory.is_supported_operation(op_name)


def list_supported_operations() -> list[str]:
    """List all supported operations."""
    return operation_factory.list_supported_operations()


def get_operation_info(op_name: str) -> Optional[dict[str, Any]]:
    """Get information about an operation."""
    return operation_factory.get_operation_info(op_name)

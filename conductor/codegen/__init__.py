"""
Codegen module for DSL generation and code templating.

This module provides the DSL generation pipeline for converting internal
DAG representations to executable Choreo DSL code.
"""

from .dslgen import ChoreoDslGen
from .registry_based_templates import (
    get_registry_template_engine,
    TemplateContext,
    render_node_with_registry_templates,
    render_fused_nodes_with_registry_templates,
)

# Operator system (moved from utils and simplified)
from .operator_registry import (
    get_operator_registry,
    OperatorInfo as OperatorTemplate,
    BufferSpec,
    get_operator_template,
    OpType,
    get_op_desc,
    is_elementwise,
    is_reduction,
)
from .operation_factory import get_operation_factory, OperationFactory

__all__ = [
    # Unified DSL generation
    "ChoreoDslGen",
    # Registry-based template system (current)
    "get_registry_template_engine",
    "TemplateContext",
    "render_node_with_registry_templates",
    "render_fused_nodes_with_registry_templates",
    # Operator system (moved from utils and simplified)
    "get_operator_registry",
    "OperatorTemplate",
    "OperatorMetadata",
    "get_operator_template",
    "OpType",
    "get_op_desc",
    "is_elementwise",
    "is_reduction",
    "get_operation_factory",
    "OperationFactory",
]

"""
Codegen module for DSL generation and code templating.

This module provides the DSL generation pipeline for converting internal
DAG representations to executable Choreo DSL code.
"""

from .dslgen import ChoreoDslGen, DSLGenerator
from .dsl_visitor import DSLCodeGenerator, dsl_codegen
from .registry_based_templates import (
    registry_template_engine, TemplateContext, ComputationExtractor,
    render_node_with_registry_templates, render_fused_nodes_with_registry_templates
)

# Operator system (moved from utils and simplified)
from .operator_registry import (
    operator_registry, OperatorInfo as OperatorTemplate, BufferSpec,
    get_operator_template, OpType, get_op_desc, is_elementwise, is_reduction
)
from .operation_factory import operation_factory, OperationFactory

# Legacy imports - deprecated, will be removed
from .dslgen_base import DslGenerator, OperationHandler, OperationHandlerRegistry
# DSLTemplate and DSLTemplateEngine removed - use registry_template_engine instead

__all__ = [
    # Main DSL generation
    'ChoreoDslGen',
    'DSLGenerator',  # Alias for backward compatibility
    'DSLCodeGenerator',
    'dsl_codegen',

    # Registry-based template system (current)
    'registry_template_engine',
    'TemplateContext',
    'ComputationExtractor',
    'render_node_with_registry_templates',
    'render_fused_nodes_with_registry_templates',

    # Operator system (moved from utils and simplified)
    'operator_registry',
    'OperatorTemplate',
    'OperatorMetadata',
    'get_operator_template',
    'OpType',
    'get_op_desc',
    'is_elementwise',
    'is_reduction',
    'operation_factory',
    'OperationFactory',

    # Legacy components (deprecated)
    'DslGenerator',
    'OperationHandler',
    'OperationHandlerRegistry',
    # 'DSLTemplate', 'DSLTemplateEngine' removed - use registry_template_engine
]
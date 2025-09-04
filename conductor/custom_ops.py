"""
Custom Operation Registration System for Conductor GCU Backend.

This module provides the infrastructure to register custom operations
that can be compiled to GCU using Choreo DSL templates, enabling
seamless integration with PyTorch's torch.compile() system.
"""

import torch
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from .operator_registry import (
    OperatorTemplate, OperatorMetadata, ParallelStructure,
    BufferSpec, unified_registry
)

logger = logging.getLogger(__name__)


@dataclass
class CustomOperatorSpec:
    """Specification for a custom operator."""
    name: str
    torch_op: Callable  # The actual PyTorch operation function
    template: str       # Choreo DSL template
    metadata: OperatorMetadata
    parameter_substitutions: Dict[str, str] = None


class CustomOperatorRegistry:
    """Registry for custom operations that integrates with Conductor's compilation pipeline."""
    
    def __init__(self):
        self.custom_ops: Dict[str, CustomOperatorSpec] = {}
        self.torch_ops: Dict[Callable, str] = {}  # Map torch ops to custom op names
        
    def register_custom_op(self, spec: CustomOperatorSpec) -> None:
        """
        Register a custom operation with the Conductor system.
        
        Args:
            spec: Custom operator specification
        """
        # Register with our custom registry
        self.custom_ops[spec.name] = spec
        self.torch_ops[spec.torch_op] = spec.name
        
        # Register with the unified operator registry for DSL generation
        operator_template = OperatorTemplate(
            name=spec.name,
            template=spec.template,
            metadata=spec.metadata,
            parameter_substitutions=spec.parameter_substitutions or {}
        )
        unified_registry.register_operator(operator_template)
        
        logger.info(f"Registered custom operator: {spec.name}")
    
    def get_custom_op(self, name: str) -> Optional[CustomOperatorSpec]:
        """Get custom operator specification by name."""
        return self.custom_ops.get(name)
    
    def get_op_name_from_torch_function(self, torch_func: Callable) -> Optional[str]:
        """Get custom operation name from PyTorch function."""
        return self.torch_ops.get(torch_func)
    
    def is_custom_op(self, op_name: str) -> bool:
        """Check if an operation is a registered custom operation."""
        return op_name in self.custom_ops
    
    def list_custom_ops(self) -> List[str]:
        """List all registered custom operations."""
        return list(self.custom_ops.keys())


# Global custom operator registry
custom_op_registry = CustomOperatorRegistry()


def register_custom_operator(
    name: str,
    torch_op: Callable,
    template: str,
    inputs: int,
    outputs: int,
    element_wise: bool = True,
    fusable: bool = True,
    parallel_structure: ParallelStructure = ParallelStructure.CHUNKED_PARALLEL,
    buffer_specs: List[BufferSpec] = None,
    parameter_substitutions: Dict[str, str] = None,
    **metadata_kwargs
) -> None:
    """
    Register a custom operator with the Conductor system.
    
    Args:
        name: Name of the custom operator
        torch_op: PyTorch operation function
        template: Choreo DSL template
        inputs: Number of input tensors
        outputs: Number of output tensors
        element_wise: Whether the operation is element-wise
        fusable: Whether the operation can be fused
        parallel_structure: Parallelization strategy
        buffer_specs: Buffer specifications for the operation
        parameter_substitutions: Template parameter substitutions
        **metadata_kwargs: Additional metadata
    """
    metadata = OperatorMetadata(
        inputs=inputs,
        outputs=outputs,
        element_wise=element_wise,
        fusable=fusable,
        parallel_structure=parallel_structure,
        buffer_specs=buffer_specs or [],
        **metadata_kwargs
    )
    
    spec = CustomOperatorSpec(
        name=name,
        torch_op=torch_op,
        template=template,
        metadata=metadata,
        parameter_substitutions=parameter_substitutions
    )
    
    custom_op_registry.register_custom_op(spec)


def create_torch_custom_op(name: str, impl_fn: Callable) -> Callable:
    """
    Create a PyTorch custom operator that can be traced by FX.
    
    Args:
        name: Name of the custom operator
        impl_fn: Implementation function
        
    Returns:
        PyTorch custom operator function
    """
    # Create a namespace for custom operations
    if not hasattr(torch.ops, 'conductor_custom'):
        torch.library.define("conductor_custom", None)
    
    # Define the custom operator
    op_name = f"conductor_custom::{name}"
    
    # Register the operator with PyTorch
    try:
        # Define the operator signature
        torch.library.define(op_name, "(Tensor x, Tensor y) -> Tensor")
        
        # Register the implementation
        torch.library.impl(op_name, "default", impl_fn)
        
        # Get the operator function
        custom_op = getattr(torch.ops.conductor_custom, name)
        
        logger.info(f"Created PyTorch custom operator: {op_name}")
        return custom_op
        
    except Exception as e:
        logger.warning(f"Failed to create PyTorch custom operator {op_name}: {e}")
        # Fallback to regular function
        return impl_fn


def get_custom_op_name_from_fx_node(fx_node: torch.fx.Node) -> Optional[str]:
    """
    Extract custom operation name from FX node if it's a custom operation.
    
    Args:
        fx_node: FX graph node
        
    Returns:
        Custom operation name if found, None otherwise
    """
    if fx_node.op == 'call_function':
        target = fx_node.target
        
        # Check if it's a custom operator
        if hasattr(target, '_name') and 'conductor_custom' in str(target._name):
            # Extract the operation name from conductor_custom::op_name
            op_name = str(target._name).split('::')[-1]
            if custom_op_registry.is_custom_op(op_name):
                return op_name
        
        # Check if it's a registered function
        op_name = custom_op_registry.get_op_name_from_torch_function(target)
        if op_name:
            return op_name
    
    return None


def is_custom_operation_node(fx_node: torch.fx.Node) -> bool:
    """Check if an FX node represents a custom operation."""
    return get_custom_op_name_from_fx_node(fx_node) is not None


# Integration with existing operation name extraction
def enhance_operation_name_extraction():
    """
    Enhance the existing operation name extraction to support custom operations.
    This function patches the graph analyzer to recognize custom operations.
    """
    from .graph_analyzer import GraphAnalyzer
    
    # Store the original method
    original_extract_operation_name = GraphAnalyzer._extract_operation_name
    
    def enhanced_extract_operation_name(self, fx_node: torch.fx.Node) -> str:
        """Enhanced operation name extraction with custom operation support."""
        # First check if it's a custom operation
        custom_op_name = get_custom_op_name_from_fx_node(fx_node)
        if custom_op_name:
            return custom_op_name
        
        # Fall back to original extraction
        return original_extract_operation_name(self, fx_node)
    
    # Patch the method
    GraphAnalyzer._extract_operation_name = enhanced_extract_operation_name
    logger.info("Enhanced operation name extraction with custom operation support")


# Auto-enhance when module is imported
enhance_operation_name_extraction()

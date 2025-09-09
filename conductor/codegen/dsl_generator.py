"""
Modern DSL Generator for Choreo Code Generation.

This module provides a clean, modern DSL generator that fixes the variable
naming issues and generates correct Choreo DSL code.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from .dsl_builder import ModernDSLGenerator, DSLBuilder, VariableScope
from ..utils.constants import MemoryLevel
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KernelSpec:
    """Specification for a kernel to be generated."""

    function_name: str
    input_shapes: List[List[int]]
    input_names: List[str]
    output_shape: List[int]
    operation_type: str
    nodes: List[Any]  # Use Any to avoid circular import


class ChoreoGenerator:
    """
    Modern Choreo DSL Generator.
    
    This class generates clean, correct Choreo DSL code using modern
    patterns and proper variable management.
    """
    
    def __init__(self):
        """Initialize the generator."""
        self.dsl_generator = ModernDSLGenerator()
        
    def generate_dsl(self, dag: Any, function_name: str) -> str:
        """
        Generate Choreo DSL code from a computation DAG.

        Args:
            dag: The computation DAG
            function_name: Name for the generated function

        Returns:
            Generated Choreo DSL code
        """
        logger.info(f"Generating DSL for function '{function_name}' with {len(dag.nodes)} nodes")

        # Extract kernel specifications directly from DAG
        kernel_specs = self._extract_kernel_specs_simple(dag, function_name)

        # Generate DSL for the primary kernel
        if kernel_specs:
            primary_spec = kernel_specs[0]  # Use the first (and typically only) kernel
            return self._generate_kernel_dsl(primary_spec)
        else:
            # Fallback for simple cases
            return self._generate_simple_kernel(dag, function_name)
    
    def _extract_kernel_specs_simple(self, dag: Any, function_name: str) -> List[KernelSpec]:
        """Extract kernel specifications from DAG without complex analysis."""
        specs = []

        # Extract input information
        input_shapes = []
        input_names = []

        for buffer in dag.inputs:
            if hasattr(buffer, 'shape') and buffer.shape:
                input_shapes.append(list(buffer.shape))
                input_names.append(buffer.name)

        # Extract output information
        output_shape = []
        if dag.outputs and hasattr(dag.outputs[0], 'shape') and dag.outputs[0].shape:
            output_shape = list(dag.outputs[0].shape)

        # Determine operation type from nodes
        operation_type = self._determine_operation_type_simple(dag.nodes)

        spec = KernelSpec(
            function_name=function_name,
            input_shapes=input_shapes,
            input_names=input_names,
            output_shape=output_shape,
            operation_type=operation_type,
            nodes=dag.nodes
        )

        specs.append(spec)
        return specs

    def _extract_kernel_specs(self, dag: Any, analysis: Any) -> List[KernelSpec]:
        """Extract kernel specifications from the DAG analysis."""
        specs = []
        
        # For now, create a single kernel spec for the entire DAG
        # In the future, this could be extended to handle multiple kernels
        
        # Extract input information
        input_shapes = []
        input_names = []
        
        for buffer in dag.inputs:
            if hasattr(buffer, 'shape') and buffer.shape:
                input_shapes.append(list(buffer.shape))
                input_names.append(buffer.name)
        
        # Extract output information
        output_shape = []
        if dag.outputs and hasattr(dag.outputs[0], 'shape') and dag.outputs[0].shape:
            output_shape = list(dag.outputs[0].shape)
        
        # Determine operation type
        operation_type = self._determine_operation_type(analysis.execution_order)
        
        spec = KernelSpec(
            function_name="kernel_function",
            input_shapes=input_shapes,
            input_names=input_names,
            output_shape=output_shape,
            operation_type=operation_type,
            nodes=analysis.execution_order
        )
        
        specs.append(spec)
        return specs
    
    def _determine_operation_type_simple(self, nodes: List[Any]) -> str:
        """Determine the primary operation type from the nodes (simplified)."""
        if not nodes:
            return "identity"

        # Look for the primary operation
        for node in nodes:
            if hasattr(node, 'op_name') and node.op_name in ['add', 'mul', 'sub', 'div']:
                return node.op_name

        # Default to the first node's operation
        if nodes and hasattr(nodes[0], 'op_name'):
            return nodes[0].op_name
        return "identity"

    def _determine_operation_type(self, nodes: List[Any]) -> str:
        """Determine the primary operation type from the nodes."""
        if not nodes:
            return "identity"
        
        # Look for the primary operation
        for node in nodes:
            if node.op_name in ['add', 'mul', 'sub', 'div']:
                return node.op_name
        
        # Default to the first node's operation
        return nodes[0].op_name if nodes else "identity"
    
    def _generate_kernel_dsl(self, spec: KernelSpec) -> str:
        """Generate DSL code for a kernel specification."""
        logger.debug(f"Generating kernel DSL for operation '{spec.operation_type}'")
        
        # Map operation types to DSL operations
        operation_map = {
            'add': '+',
            'mul': '*',
            'sub': '-',
            'div': '/',
        }
        
        dsl_operation = operation_map.get(spec.operation_type, '+')
        
        # Generate the kernel using the modern DSL generator
        return self.dsl_generator.generate_elementwise_kernel(
            function_name=spec.function_name,
            input_shapes=spec.input_shapes,
            input_names=spec.input_names,
            operation=dsl_operation,
            output_shape=spec.output_shape
        )
    
    def _generate_simple_kernel(self, dag: Any, function_name: str) -> str:
        """Generate a simple kernel for basic cases."""
        logger.debug("Generating simple fallback kernel")
        
        # Create a basic elementwise kernel
        input_shapes = [[16, 16]] * 2  # Default shapes
        input_names = ["lhs", "rhs"]
        output_shape = [16, 16]
        
        return self.dsl_generator.generate_elementwise_kernel(
            function_name=function_name,
            input_shapes=input_shapes,
            input_names=input_names,
            operation='+',
            output_shape=output_shape
        )


class ChoreoCompiler:
    """
    Choreo Compiler that integrates with the existing system.
    
    This class provides a drop-in replacement for the existing DSL generator
    while using the modern architecture internally.
    """
    
    def __init__(self):
        """Initialize the compiler."""
        self.generator = ChoreoGenerator()
    
    def generate_dsl_from_dag(self, dag: Any, function_name: str) -> str:
        """
        Generate DSL from DAG - compatible with existing interface.
        
        Args:
            dag: Computation DAG
            function_name: Function name
            
        Returns:
            Generated DSL code
        """
        try:
            return self.generator.generate_dsl(dag, function_name)
        except Exception as e:
            logger.error(f"Modern DSL generation failed: {e}")
            # Fallback to simple generation
            return self._generate_fallback_dsl(function_name)
    
    def _generate_fallback_dsl(self, function_name: str) -> str:
        """Generate a minimal working DSL as fallback."""
        builder = DSLBuilder()
        
        builder.add_header()
        builder.begin_function(function_name, ["f32 [16, 16] lhs", "f32 [16, 16] rhs"])
        
        # Declare output
        output_var = builder.declare_variable("output", "f32", [16, 16], MemoryLevel.GLOBAL)
        
        # Simple parallel computation
        parallel_var = builder.var_manager.create_variable("p", "int", VariableScope.PARALLEL)
        builder.begin_parallel(parallel_var.name)

        # Outer loop
        outer_var = builder.var_manager.create_variable("chunk_idx", "int", VariableScope.LOOP)
        builder.begin_foreach(outer_var.name, "[2]")

        # DMA loads
        load_lhs = builder.var_manager.create_variable("load_lhs", "auto", VariableScope.LOOP)
        load_rhs = builder.var_manager.create_variable("load_rhs", "auto", VariableScope.LOOP)
        
        builder.add_dma_copy_async(f"lhs.chunkat({parallel_var.name}, {outer_var.name})", load_lhs.name)
        builder.add_dma_copy_async(f"rhs.chunkat({parallel_var.name}, {outer_var.name})", load_rhs.name)
        builder.add_wait(load_lhs.name, load_rhs.name)
        
        # Local computation
        local_var = builder.declare_variable("local_result", "f32", [8, 8], MemoryLevel.L1)
        
        # Nested loops
        i_var = builder.var_manager.create_variable("i", "int", VariableScope.LOOP)
        j_var = builder.var_manager.create_variable("j", "int", VariableScope.LOOP)
        
        builder.begin_foreach(i_var.name, "[8]")
        builder.begin_foreach(j_var.name, "[8]")
        
        # Computation
        index_expr = f"{i_var.name}, {j_var.name}"
        expr = f"{load_lhs.name}.data.at({index_expr}) + {load_rhs.name}.data.at({index_expr})"
        builder.add_assignment(f"{local_var.name}.at({index_expr})", expr)
        
        builder.end_foreach()  # j
        builder.end_foreach()  # i
        
        # Store result
        builder.add_dma_copy(local_var.name, f"{output_var.name}.chunkat({parallel_var.name}, {outer_var.name})")
        
        builder.end_foreach()  # chunk
        builder.end_parallel()
        
        builder.add_return(output_var.name)
        builder.end_function()
        
        return builder.build()


# Global instance for compatibility
modern_choreo_compiler = ChoreoCompiler()

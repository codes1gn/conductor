#!/usr/bin/env python3
"""
Unified Operator System for Conductor

This module provides a unified template-based system where both built-in
and custom operators share the same architecture, enabling seamless fusion
and optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ParallelStructure(Enum):
    """Types of parallel execution structures."""
    CHUNKED_PARALLEL = "chunked_parallel"    # Parallel with chunked processing
    SIMPLE_PARALLEL = "simple_parallel"      # Basic parallel execution
    SEQUENTIAL = "sequential"                # No parallelization
    REDUCTION = "reduction"                  # Reduction-style parallelization


@dataclass
class BufferSpec:
    """Specification for buffer declarations in templates."""
    name: str
    scope: str  # "local", "shared", "global"
    dtype: str = "f32"
    shape: List[int] = field(default_factory=list)
    is_temporary: bool = True


@dataclass
class OperatorMetadata:
    """Metadata for operator fusion and optimization."""
    inputs: int
    outputs: int
    element_wise: bool = False
    fusable: bool = True
    parallel_structure: ParallelStructure = ParallelStructure.SIMPLE_PARALLEL
    buffer_specs: List[BufferSpec] = field(default_factory=list)
    memory_bound: bool = True
    compute_intensity: float = 1.0  # Relative compute vs memory ratio
    fusion_priority: int = 1  # Higher priority operations fuse first


@dataclass
class OperatorTemplate:
    """Template for generating Choreo DSL code."""
    name: str
    template: str
    metadata: OperatorMetadata
    parameter_substitutions: Dict[str, str] = field(default_factory=dict)
    
    def substitute_parameters(self, **kwargs) -> str:
        """Substitute template parameters with actual values."""
        result = self.template
        
        # Apply default substitutions
        for param, value in self.parameter_substitutions.items():
            result = result.replace(f"{{{param}}}", str(value))
        
        # Apply runtime substitutions
        for param, value in kwargs.items():
            result = result.replace(f"{{{param}}}", str(value))
        
        return result
    
    def can_fuse_with(self, other: 'OperatorTemplate') -> bool:
        """Check if this operator can fuse with another."""
        if not (self.metadata.fusable and other.metadata.fusable):
            return False
        
        # Element-wise operations can generally fuse together
        if self.metadata.element_wise and other.metadata.element_wise:
            return True
        
        # Check parallel structure compatibility
        if self.metadata.parallel_structure != other.metadata.parallel_structure:
            return False
        
        # Check buffer compatibility
        return self._check_buffer_compatibility(other)
    
    def _check_buffer_compatibility(self, other: 'OperatorTemplate') -> bool:
        """Check if buffer specifications are compatible for fusion."""
        # For now, simple check - same buffer sizes
        self_buffers = {(spec.scope, tuple(spec.shape)) for spec in self.metadata.buffer_specs}
        other_buffers = {(spec.scope, tuple(spec.shape)) for spec in other.metadata.buffer_specs}
        
        # Compatible if they use similar buffer patterns
        return len(self_buffers.intersection(other_buffers)) > 0 or not (self_buffers and other_buffers)


class UnifiedOperatorRegistry:
    """Registry for both built-in and custom operators using unified template system."""
    
    def __init__(self):
        self.operators: Dict[str, OperatorTemplate] = {}
        self._initialize_builtin_operators()
    
    def register_operator(self, template: OperatorTemplate) -> None:
        """Register an operator template."""
        self.operators[template.name] = template
        logger.info(f"Registered operator: {template.name}")
    
    def get_operator(self, name: str) -> Optional[OperatorTemplate]:
        """Get operator template by name."""
        return self.operators.get(name)
    
    def list_operators(self) -> List[str]:
        """List all registered operators."""
        return list(self.operators.keys())
    
    def get_fusable_operators(self) -> List[str]:
        """Get list of operators that support fusion."""
        return [name for name, template in self.operators.items() 
                if template.metadata.fusable]
    
    def can_fuse(self, op1: str, op2: str) -> bool:
        """Check if two operators can be fused."""
        template1 = self.get_operator(op1)
        template2 = self.get_operator(op2)
        
        if not (template1 and template2):
            return False
        
        return template1.can_fuse_with(template2)
    
    def _initialize_builtin_operators(self) -> None:
        """Initialize built-in operators with template-based approach."""
        
        # Built-in Addition Template
        add_template = OperatorTemplate(
            name="add",
            template="""
// Element-wise addition
func {function_name}(input0: f32 mdspan<2> [{M}, {N}], input1: f32 mdspan<2> [{M}, {N}]) -> (output: f32 mdspan<2> [{M}, {N}]) {{
    parallel (p: [0:{P}]) {{
        local l1_input0: f32 mdspan<2> [{buffer_m}, {buffer_n}];
        local l1_input1: f32 mdspan<2> [{buffer_m}, {buffer_n}];
        local l1_output: f32 mdspan<2> [{buffer_m}, {buffer_n}];
        
        for (index: [0:{M}*{N}/({P}*{buffer_m}*{buffer_n})]) {{
            dma.copy input0.chunkat(p, index) => l1_input0;
            dma.copy input1.chunkat(p, index) => l1_input1;
            
            for (i: [0:{buffer_m}], j: [0:{buffer_n}]) {{
                l1_output[i, j] = l1_input0[i, j] + l1_input1[i, j];
            }}
            
            dma.copy l1_output => output.chunkat(p, index);
        }}
    }}
}}""",
            metadata=OperatorMetadata(
                inputs=2,
                outputs=1,
                element_wise=True,
                fusable=True,
                parallel_structure=ParallelStructure.CHUNKED_PARALLEL,
                buffer_specs=[
                    BufferSpec("l1_input0", "local", "f32", [16, 8]),
                    BufferSpec("l1_input1", "local", "f32", [16, 8]),
                    BufferSpec("l1_output", "local", "f32", [16, 8])
                ],
                memory_bound=True,
                compute_intensity=1.0,
                fusion_priority=1
            ),
            parameter_substitutions={
                "buffer_m": "16",
                "buffer_n": "8",
                "P": "4"
            }
        )
        
        # Built-in Multiplication Template
        mul_template = OperatorTemplate(
            name="mul",
            template="""
// Element-wise multiplication
func {function_name}(input0: f32 mdspan<2> [{M}, {N}], input1: f32 mdspan<2> [{M}, {N}]) -> (output: f32 mdspan<2> [{M}, {N}]) {{
    parallel (p: [0:{P}]) {{
        local l1_input0: f32 mdspan<2> [{buffer_m}, {buffer_n}];
        local l1_input1: f32 mdspan<2> [{buffer_m}, {buffer_n}];
        local l1_output: f32 mdspan<2> [{buffer_m}, {buffer_n}];
        
        for (index: [0:{M}*{N}/({P}*{buffer_m}*{buffer_n})]) {{
            dma.copy input0.chunkat(p, index) => l1_input0;
            dma.copy input1.chunkat(p, index) => l1_input1;
            
            for (i: [0:{buffer_m}], j: [0:{buffer_n}]) {{
                l1_output[i, j] = l1_input0[i, j] * l1_input1[i, j];
            }}
            
            dma.copy l1_output => output.chunkat(p, index);
        }}
    }}
}}""",
            metadata=OperatorMetadata(
                inputs=2,
                outputs=1,
                element_wise=True,
                fusable=True,
                parallel_structure=ParallelStructure.CHUNKED_PARALLEL,
                buffer_specs=[
                    BufferSpec("l1_input0", "local", "f32", [16, 8]),
                    BufferSpec("l1_input1", "local", "f32", [16, 8]),
                    BufferSpec("l1_output", "local", "f32", [16, 8])
                ],
                memory_bound=True,
                compute_intensity=1.2,  # Slightly more compute than add
                fusion_priority=1
            ),
            parameter_substitutions={
                "buffer_m": "16",
                "buffer_n": "8",
                "P": "4"
            }
        )
        
        # Register built-in operators
        self.register_operator(add_template)
        self.register_operator(mul_template)


# Global registry instance
unified_registry = UnifiedOperatorRegistry()


def register_custom_operator(name: str, template: str, metadata: OperatorMetadata) -> None:
    """Register a custom operator with the unified system."""
    operator_template = OperatorTemplate(
        name=name,
        template=template,
        metadata=metadata
    )
    unified_registry.register_operator(operator_template)


def get_operator_template(name: str) -> Optional[OperatorTemplate]:
    """Get operator template by name."""
    return unified_registry.get_operator(name)


def can_operators_fuse(op1: str, op2: str) -> bool:
    """Check if two operators can be fused."""
    return unified_registry.can_fuse(op1, op2)


def list_all_operators() -> List[str]:
    """List all registered operators."""
    return unified_registry.list_operators()


def get_fusion_metadata(op_name: str) -> Optional[OperatorMetadata]:
    """Get fusion metadata for an operator."""
    template = unified_registry.get_operator(op_name)
    return template.metadata if template else None


class FusionAwareDSLGenerator:
    """DSL generator that uses unified operator templates and supports fusion."""

    def __init__(self):
        self.registry = unified_registry

    def generate_operation_dsl(self, op_name: str, function_name: str, **params) -> str:
        """Generate DSL for a single operation using templates."""
        template = self.registry.get_operator(op_name)
        if not template:
            raise ValueError(f"Unknown operator: {op_name}")

        # Add function name to parameters
        params['function_name'] = function_name

        return template.substitute_parameters(**params)

    def generate_fused_operation_dsl(self, op_names: List[str], function_name: str, **params) -> str:
        """Generate DSL for fused operations."""
        if len(op_names) == 1:
            return self.generate_operation_dsl(op_names[0], function_name, **params)

        # Check if all operations can be fused
        for i in range(len(op_names) - 1):
            if not self.registry.can_fuse(op_names[i], op_names[i + 1]):
                raise ValueError(f"Cannot fuse {op_names[i]} with {op_names[i + 1]}")

        # Generate fused template
        return self._generate_fused_template(op_names, function_name, **params)

    def _generate_fused_template(self, op_names: List[str], function_name: str, **params) -> str:
        """Generate fused template for multiple operations."""
        # For element-wise operations, we can fuse the computation
        templates = [self.registry.get_operator(name) for name in op_names]

        # Use the first template as base structure
        base_template = templates[0]

        # Generate fused computation
        fused_computation = self._generate_fused_computation(op_names)

        # Create fused template
        fused_template = f"""
// Fused operations: {', '.join(op_names)}
func {function_name}(input0: f32 mdspan<2> [{{M}}, {{N}}], input1: f32 mdspan<2> [{{M}}, {{N}}]) -> (output: f32 mdspan<2> [{{M}}, {{N}}]) {{
    parallel (p: [0:{{P}}]) {{
        local l1_input0: f32 mdspan<2> [{{buffer_m}}, {{buffer_n}}];
        local l1_input1: f32 mdspan<2> [{{buffer_m}}, {{buffer_n}}];
        local l1_output: f32 mdspan<2> [{{buffer_m}}, {{buffer_n}}];

        for (index: [0:{{M}}*{{N}}/({{P}}*{{buffer_m}}*{{buffer_n}})]) {{
            dma.copy input0.chunkat(p, index) => l1_input0;
            dma.copy input1.chunkat(p, index) => l1_input1;

            for (i: [0:{{buffer_m}}], j: [0:{{buffer_n}}]) {{
                {fused_computation}
            }}

            dma.copy l1_output => output.chunkat(p, index);
        }}
    }}
}}"""

        # Substitute parameters
        params['function_name'] = function_name
        params.update(base_template.parameter_substitutions)

        result = fused_template
        for param, value in params.items():
            result = result.replace(f"{{{param}}}", str(value))

        return result

    def _generate_fused_computation(self, op_names: List[str]) -> str:
        """Generate fused computation for element-wise operations."""
        if len(op_names) == 1:
            op = op_names[0]
            if op == "add":
                return "l1_output[i, j] = l1_input0[i, j] + l1_input1[i, j];"
            elif op == "mul":
                return "l1_output[i, j] = l1_input0[i, j] * l1_input1[i, j];"

        # For multiple operations, chain them
        # Example: add then mul -> (input0 + input1) * input0
        if op_names == ["add", "mul"]:
            return "l1_output[i, j] = (l1_input0[i, j] + l1_input1[i, j]) * l1_input0[i, j];"

        # Default: just use the last operation
        return self._generate_fused_computation([op_names[-1]])

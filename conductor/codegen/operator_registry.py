"""
Operator Registry for Conductor Operations.

This module provides a consolidated registry that combines operation
properties, DSL templates, and metadata in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from enum import Enum
import logging
import torch

logger = logging.getLogger(__name__)


@dataclass
class BufferSpec:
    """Specification for input/output buffers of an operation."""

    name: str  # Buffer identifier (e.g., "lhs", "rhs", "output")
    dtype: Optional[torch.dtype] = None  # Data type (None means infer from context)
    shape_fn: Optional[Callable] = None  # Function to compute shape from input shapes
    is_optional: bool = False  # Whether this buffer is optional

    def infer_shape(
        self, input_shapes: list[tuple[int, ...]], context: dict[str, Any] = None
    ) -> tuple[int, ...]:
        """Infer buffer shape from input shapes and context."""
        if self.shape_fn:
            return self.shape_fn(input_shapes, context or {})
        elif input_shapes:
            # Default: use first input shape
            return input_shapes[0]
        else:
            # Fallback
            return (4, 4)


class OpType(Enum):
    """Unified operation type classification."""

    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    MATRIX = "matrix"
    ACTIVATION = "activation"
    SHAPE = "shape"
    MATH = "math"
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BOUND = "memory_bound"
    UNKNOWN = "unknown"


class ParallelStructure(Enum):
    """Types of parallel execution structures."""

    CHUNKED_PARALLEL = "chunked_parallel"
    SIMPLE_PARALLEL = "simple_parallel"
    SEQUENTIAL = "sequential"
    REDUCTION = "reduction"


# Old BufferSpec removed - using the new one above


@dataclass
class OperatorInfo:
    """
    Complete operator information that dslgen uses to generate code.
    This is the single source of truth for all operator-related information.
    """

    name: str  # Operation name (e.g., "add", "matmul")
    canonical_name: str  # Canonical name for templates

    # Buffer specifications - the key information dslgen needs
    input_buffers: list[BufferSpec] = field(default_factory=list)  # Input buffer requirements
    output_buffers: list[BufferSpec] = field(default_factory=list)  # Output buffer requirements

    # Code generation information
    code_template: Optional[str] = None  # Choreo DSL template
    code_gen_fn: Optional[Callable] = None  # Custom code generation function
    parameter_substitutions: dict[str, str] = field(default_factory=dict)  # Parameter substitutions

    # Operation properties
    operation_type: OpType = OpType.ELEMENTWISE
    requires_device_kernel: bool = False
    fusable: bool = True

    def get_input_buffer_specs(self) -> list[BufferSpec]:
        """Get input buffer specifications."""
        return self.input_buffers

    def get_output_buffer_specs(self) -> list[BufferSpec]:
        """Get output buffer specifications."""
        return self.output_buffers

    def generate_code(
        self,
        input_vars: list[str],
        output_var: str,
        index_vars: list[str] = None,
        tensor_rank: int = 2,
    ) -> str:
        """Generate code for this operation with proper multi-dimensional indexing."""
        if index_vars is None:
            index_vars = ["i", "j"]  # Default for 2D tensors

        # Generate proper multi-dimensional index string
        index_str = ", ".join(index_vars[:tensor_rank])

        if self.code_gen_fn:
            return self.code_gen_fn(input_vars, output_var, index_vars, tensor_rank)
        elif self.code_template:
            # Smart variable access - DMA loads have .data, local buffers don't
            def format_input_access(var_name: str) -> str:
                if var_name.startswith("l1_load__") or var_name.startswith("l2_load__"):
                    return f"{var_name}.data.at({index_str})"
                else:
                    return f"{var_name}.at({index_str})"

            # Format inputs with smart access pattern
            input0_access = format_input_access(input_vars[0]) if len(input_vars) > 0 else "input"
            input1_access = format_input_access(input_vars[1]) if len(input_vars) > 1 else "input"

            # Use template with smart variable access
            return self.code_template.format(
                output=output_var,
                input0_access=input0_access,
                input1_access=input1_access,
                index=index_str,
            )
        else:
            # Fallback with smart access pattern
            if len(input_vars) > 0:
                if input_vars[0].startswith("l1_load__") or input_vars[0].startswith("l2_load__"):
                    input_access = f"{input_vars[0]}.data.at({index_str})"
                else:
                    input_access = f"{input_vars[0]}.at({index_str})"
                return f"{output_var}.at({index_str}) = {input_access};"
            else:
                return f"{output_var}.at({index_str}) = 0.0; // No inputs"

    def substitute_parameters(self, **params) -> str:
        """Substitute parameters in the template if available."""
        if not self.code_template:
            raise ValueError(f"No template available for operator: {self.name}")

        result = self.code_template

        # Apply default substitutions first
        for key, value in self.parameter_substitutions.items():
            result = result.replace(f"{{{key}}}", value)

        # Apply provided parameters
        for key, value in params.items():
            result = result.replace(f"{{{key}}}", str(value))

        return result


class OperatorRegistry:
    """
    Registry for all operator information.

    Provides a single source of truth for operator properties and templates.
    """

    def __init__(self):
        self._operators: Dict[str, OperatorInfo] = {}
        self._function_mappings: Dict[str, OperatorInfo] = {}
        self._method_mappings: Dict[str, OperatorInfo] = {}
        self._initialize_all_operators()

    def _initialize_all_operators(self) -> None:
        """Initialize all operators with buffer specs and code generation."""

        # Helper functions for shape inference
        def same_as_input(
            input_shapes: list[tuple[int, ...]], context: dict = None
        ) -> tuple[int, ...]:
            """Output shape same as first input."""
            return input_shapes[0] if input_shapes else (4, 4)

        def matmul_shape(
            input_shapes: list[tuple[int, ...]], context: dict = None
        ) -> tuple[int, ...]:
            """Matrix multiplication output shape."""
            if len(input_shapes) >= 2:
                return (input_shapes[0][0], input_shapes[1][1])
            return (4, 4)

        # Elementwise operations
        add_op = OperatorInfo(
            name="add",
            canonical_name="add",
            operation_type=OpType.ELEMENTWISE,
            input_buffers=[
                BufferSpec("lhs", dtype=torch.float32, shape_fn=same_as_input),
                BufferSpec("rhs", dtype=torch.float32, shape_fn=same_as_input),
            ],
            output_buffers=[BufferSpec("output", dtype=torch.float32, shape_fn=same_as_input)],
            code_template="{output}.at({index}) = {input0_access} + {input1_access};",
            fusable=True,
        )

        mul_op = OperatorInfo(
            name="mul",
            canonical_name="mul",
            operation_type=OpType.ELEMENTWISE,
            input_buffers=[
                BufferSpec("lhs", dtype=torch.float32, shape_fn=same_as_input),
                BufferSpec("rhs", dtype=torch.float32, shape_fn=same_as_input),
            ],
            output_buffers=[BufferSpec("output", dtype=torch.float32, shape_fn=same_as_input)],
            code_template="{output}.at({index}) = {input0_access} * {input1_access};",
            fusable=True,
        )

        sub_op = OperatorInfo(
            name="sub",
            canonical_name="sub",
            operation_type=OpType.ELEMENTWISE,
            input_buffers=[
                BufferSpec("lhs", dtype=torch.float32, shape_fn=same_as_input),
                BufferSpec("rhs", dtype=torch.float32, shape_fn=same_as_input),
            ],
            output_buffers=[BufferSpec("output", dtype=torch.float32, shape_fn=same_as_input)],
            code_template="{output}.at({index}) = {input0_access} - {input1_access};",
            fusable=True,
        )

        div_op = OperatorInfo(
            name="div",
            canonical_name="div",
            operation_type=OpType.ELEMENTWISE,
            input_buffers=[
                BufferSpec("lhs", dtype=torch.float32, shape_fn=same_as_input),
                BufferSpec("rhs", dtype=torch.float32, shape_fn=same_as_input),
            ],
            output_buffers=[BufferSpec("output", dtype=torch.float32, shape_fn=same_as_input)],
            code_template="{output}.at({index}) = {input0_access} / {input1_access};",
            fusable=True,
        )

        relu_op = OperatorInfo(
            name="relu",
            canonical_name="relu",
            operation_type=OpType.ACTIVATION,
            input_buffers=[BufferSpec("input", dtype=torch.float32, shape_fn=same_as_input)],
            output_buffers=[BufferSpec("output", dtype=torch.float32, shape_fn=same_as_input)],
            code_template="{output}.at({index}) = {input0_access}; // ReLU placeholder",
            fusable=True,
        )

        # Matrix operations
        matmul_op = OperatorInfo(
            name="matmul",
            canonical_name="matmul",
            operation_type=OpType.MATRIX,
            input_buffers=[
                BufferSpec(
                    "lhs",
                    dtype=torch.float32,
                    shape_fn=lambda shapes, ctx: shapes[0] if shapes else (4, 4),
                ),
                BufferSpec(
                    "rhs",
                    dtype=torch.float32,
                    shape_fn=lambda shapes, ctx: shapes[1] if len(shapes) > 1 else (4, 4),
                ),
            ],
            output_buffers=[BufferSpec("output", dtype=torch.float32, shape_fn=matmul_shape)],
            code_template="// Matrix multiplication - requires device kernel",
            requires_device_kernel=True,
            fusable=False,
        )

        # Reduction operations
        sum_op = OperatorInfo(
            name="sum",
            canonical_name="sum",
            operation_type=OpType.REDUCTION,
            input_buffers=[BufferSpec("input", dtype=torch.float32, shape_fn=same_as_input)],
            output_buffers=[
                BufferSpec(
                    "output",
                    dtype=torch.float32,
                    shape_fn=lambda shapes, ctx: (1,) if shapes else (1,),
                )
            ],
            code_template="// Sum reduction - requires special handling",
            fusable=False,
        )

        # Define reduce_mean shape function
        def reduce_mean_shape(shapes: list[tuple[int, ...]], ctx: dict[str, Any]) -> tuple[int, ...]:
            """Compute output shape for reduce_mean operation."""
            if not shapes:
                return (1,)

            input_shape = shapes[0]
            dim = ctx.get('dim', None)
            keepdim = ctx.get('keepdim', False)

            if dim is None:
                # Reduce all dimensions
                return (1,) if keepdim else ()

            # Reduce specific dimension
            if isinstance(dim, int):
                dims_to_reduce = [dim]
            else:
                dims_to_reduce = list(dim)

            output_shape = list(input_shape)
            for d in sorted(dims_to_reduce, reverse=True):
                if keepdim:
                    output_shape[d] = 1
                else:
                    output_shape.pop(d)

            return tuple(output_shape) if output_shape else (1,)

        reduce_mean_op = OperatorInfo(
            name="reduce_mean",
            canonical_name="reduce_mean",
            operation_type=OpType.REDUCTION,
            input_buffers=[BufferSpec("input", dtype=torch.float32, shape_fn=same_as_input)],
            output_buffers=[
                BufferSpec(
                    "output",
                    dtype=torch.float32,
                    shape_fn=reduce_mean_shape,
                )
            ],
            code_template="// Reduce mean - requires special handling",
            fusable=True,  # Allow fusion with elementwise operations
        )

        # Complex operations requiring device kernels
        conv2d_op = OperatorInfo(
            name="conv2d",
            canonical_name="conv2d",
            operation_type=OpType.COMPUTE_BOUND,
            input_buffers=[
                BufferSpec("input", dtype=torch.float32),
                BufferSpec("weight", dtype=torch.float32),
            ],
            output_buffers=[BufferSpec("output", dtype=torch.float32)],
            requires_device_kernel=True,
            fusable=False,
        )

        attention_op = OperatorInfo(
            name="attention",
            canonical_name="attention",
            operation_type=OpType.COMPUTE_BOUND,
            input_buffers=[
                BufferSpec("query", dtype=torch.float32),
                BufferSpec("key", dtype=torch.float32),
                BufferSpec("value", dtype=torch.float32),
            ],
            output_buffers=[BufferSpec("output", dtype=torch.float32)],
            requires_device_kernel=True,
            fusable=False,
        )

        # Register all operations
        all_ops = [
            add_op,
            mul_op,
            sub_op,
            div_op,
            relu_op,
            matmul_op,
            sum_op,
            reduce_mean_op,
            conv2d_op,
            attention_op,
        ]

        for op in all_ops:
            self._operators[op.name] = op
            self._function_mappings[op.name] = op
            self._method_mappings[op.name] = op
            # Also register in-place versions for methods
            self._method_mappings[f"{op.name}_"] = op

            # Register aliases
            if op.name == "matmul":
                for alias in ["mm", "bmm"]:
                    self._function_mappings[alias] = op
                    self._method_mappings[alias] = op

    # Operation property methods
    def get_operation(self, name: str) -> Optional[OperatorInfo]:
        """Get complete operation information by name."""
        return self._operators.get(name)

    def get_function_mapping(self, function_name: str) -> Optional[OperatorInfo]:
        """Get operation info for a function call."""
        return self._function_mappings.get(function_name)

    def get_method_mapping(self, method_name: str) -> Optional[OperatorInfo]:
        """Get operation info for a method call."""
        return self._method_mappings.get(method_name)

    def register_operation(
        self, op_info: OperatorInfo, for_functions: bool = True, for_methods: bool = True
    ) -> None:
        """Register a custom operation."""
        self._operators[op_info.name] = op_info
        if for_functions:
            self._function_mappings[op_info.name] = op_info
        if for_methods:
            self._method_mappings[op_info.name] = op_info
        logger.info(f"Registered unified operator: {op_info.name}")

    # Template methods (backward compatibility)
    def get_operator(self, name: str) -> Optional[OperatorInfo]:
        """Get operator template by name (backward compatibility)."""
        return self.get_operation(name)

    def list_operators(self) -> List[str]:
        """List all registered operator names."""
        return list(self._operators.keys())

    def can_fuse(self, op1: str, op2: str) -> bool:
        """Check if two operators can be fused."""
        info1 = self.get_operation(op1)
        info2 = self.get_operation(op2)

        if not info1 or not info2:
            return False

        return (
            info1.fusable
            and info2.fusable
            and info1.operation_type in [OpType.ELEMENTWISE, OpType.ACTIVATION]
            and info2.operation_type in [OpType.ELEMENTWISE, OpType.ACTIVATION]
        )

    # Utility methods
    def get_operations_by_type(self, op_type: OpType) -> Dict[str, OperatorInfo]:
        """Get all operations of a specific type."""
        return {
            name: info for name, info in self._operators.items() if info.operation_type == op_type
        }

    def get_fusable_operations(self) -> Set[str]:
        """Get set of operations that can be fused."""
        return {name for name, info in self._operators.items() if info.fusable}

    def get_elementwise_operations(self) -> Set[str]:
        """Get all elementwise operation names."""
        return {
            name
            for name, info in self._operators.items()
            if info.operation_type == OpType.ELEMENTWISE
        }

    def get_reduction_operations(self) -> Set[str]:
        """Get all reduction operation names."""
        return {
            name
            for name, info in self._operators.items()
            if info.operation_type == OpType.REDUCTION
        }

    def get_device_kernel_operations(self) -> Set[str]:
        """Get all operations that require device kernels."""
        return {name for name, info in self._operators.items() if info.requires_device_kernel}

    def requires_device_kernel(self, op_name: str) -> bool:
        """Check if an operation requires a device kernel."""
        info = self.get_operator_info(op_name)
        return info is not None and info.requires_device_kernel

    def list_all_operations(self) -> Set[str]:
        """Get all registered operation names."""
        return set(self._operators.keys())


# Convenience functions for accessing the registry through context
def get_operator_registry() -> OperatorRegistry:
    """Get the operator registry from the global context."""
    from ..context import ensure_context_initialized
    context = ensure_context_initialized()
    return context.get_operator_registry()


def get_op_desc(op_name: str) -> Optional[OperatorInfo]:
    """Get operation descriptor by name."""
    return get_operator_registry().get_operation(op_name)


def get_operator_template(name: str) -> Optional[OperatorInfo]:
    """Get operator template by name."""
    return get_operator_registry().get_operation(name)


def is_elementwise(op_name: str) -> bool:
    """Check if operation is elementwise."""
    info = get_op_desc(op_name)
    return info is not None and info.operation_type == OpType.ELEMENTWISE


def is_reduction(op_name: str) -> bool:
    """Check if operation is a reduction."""
    info = get_op_desc(op_name)
    return info is not None and info.operation_type == OpType.REDUCTION


def requires_device_kernel(op_name: str) -> bool:
    """Check if operation requires a device kernel."""
    info = get_op_desc(op_name)
    return info is not None and info.requires_device_kernel


def get_device_kernel_operations() -> Set[str]:
    """Get all operations that require device kernels."""
    return get_operator_registry().get_device_kernel_operations()


# Backward compatibility aliases
def get_unified_registry():
    """Get unified registry (backward compatibility)."""
    return get_operator_registry()


def get_simple_unified_registry():
    """Get simple unified registry (backward compatibility)."""
    return get_operator_registry()
# Backward compatibility alias
def get_unified_operation_registry():
    """Get unified operation registry (backward compatibility)."""
    return get_operator_registry()

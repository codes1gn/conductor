"""
Core Data Structures and Protocols for Modern DSL Generation.

This module defines the fundamental types and interfaces used throughout
the modern DSL generation architecture. All data structures are immutable
and fully typed for safety and clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any, Dict, Tuple, Optional, List
from enum import Enum
import torch

from ...utils.constants import MemoryLevel


class BufferScope(Enum):
    """Scope of a buffer in the computation."""
    LOCAL = "local"
    GLOBAL = "global"
    SHARED = "shared"


class ValidationLevel(Enum):
    """Level of validation to perform."""
    NONE = "none"
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    FULL = "full"


@dataclass(frozen=True)
class DSLGenerationConfig:
    """Configuration for DSL generation."""
    indent_size: int = 2
    parallel_factor: int = 1
    memory_level: MemoryLevel = MemoryLevel.L1
    enable_fusion: bool = True
    target_dialect: str = "choreo"
    validation_level: ValidationLevel = ValidationLevel.SYNTAX
    enable_optimization: bool = True
    debug_mode: bool = False


@dataclass(frozen=True)
class BufferInfo:
    """Information about a buffer in the computation."""
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    scope: BufferScope
    memory_level: MemoryLevel
    is_input: bool = False
    is_output: bool = False


@dataclass(frozen=True)
class OperationInfo:
    """Information about an operation to generate."""
    op_name: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    attributes: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class ParallelConfig:
    """Configuration for parallel execution."""
    factor: int
    variable: str
    chunk_size: int
    strategy: str = "simple"


@dataclass(frozen=True)
class CodeGenerationContext:
    """Context for code generation operations."""
    function_name: str
    input_buffers: Tuple[BufferInfo, ...]
    output_buffers: Tuple[BufferInfo, ...]
    parallel_config: ParallelConfig
    config: DSLGenerationConfig
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class CodeFragment:
    """A fragment of generated code with metadata."""
    content: str
    dependencies: Tuple[str, ...]
    declarations: Tuple[str, ...]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate the code fragment."""
        if not isinstance(self.content, str):
            raise TypeError("Content must be a string")
        if not self.content.strip():
            raise ValueError("Content cannot be empty")


@dataclass(frozen=True)
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class DSLResult:
    """Complete result of DSL generation."""
    content: str
    validation: ValidationResult
    metadata: Dict[str, Any]
    fragments: Tuple[CodeFragment, ...]


# Protocols for interfaces

class CodeGenerator(Protocol):
    """Protocol for code generation components."""
    
    def generate(self, context: CodeGenerationContext) -> CodeFragment:
        """Generate a code fragment for the given context."""
        ...


class TemplateRenderer(Protocol):
    """Protocol for template rendering."""
    
    def render(self, template: str, context: Dict[str, Any]) -> str:
        """Render a template with the given context."""
        ...
    
    def render_file(self, template_path: str, context: Dict[str, Any]) -> str:
        """Render a template file with the given context."""
        ...


class NamingStrategy(Protocol):
    """Protocol for naming strategies."""
    
    def get_buffer_name(self, buffer_info: BufferInfo) -> str:
        """Get a name for a buffer."""
        ...
    
    def get_variable_name(self, operation: OperationInfo) -> str:
        """Get a variable name for an operation."""
        ...
    
    def get_function_name(self, base_name: str) -> str:
        """Get a function name."""
        ...


class SyntaxValidator(Protocol):
    """Protocol for syntax validation."""
    
    def validate(self, code: str) -> ValidationResult:
        """Validate the syntax of generated code."""
        ...
    
    def validate_fragment(self, fragment: CodeFragment) -> ValidationResult:
        """Validate a code fragment."""
        ...


# Utility functions for type validation

def validate_buffer_info(buffer: BufferInfo) -> None:
    """Validate a BufferInfo object."""
    if not buffer.name:
        raise ValueError("Buffer name cannot be empty")
    if not buffer.shape:
        raise ValueError("Buffer shape cannot be empty")
    if any(dim <= 0 for dim in buffer.shape):
        raise ValueError("All shape dimensions must be positive")


def validate_operation_info(operation: OperationInfo) -> None:
    """Validate an OperationInfo object."""
    if not operation.op_name:
        raise ValueError("Operation name cannot be empty")
    if not operation.inputs:
        raise ValueError("Operation must have at least one input")
    if not operation.outputs:
        raise ValueError("Operation must have at least one output")


def validate_generation_context(context: CodeGenerationContext) -> None:
    """Validate a CodeGenerationContext object."""
    if not context.function_name:
        raise ValueError("Function name cannot be empty")
    if not context.input_buffers:
        raise ValueError("Context must have at least one input buffer")
    if not context.output_buffers:
        raise ValueError("Context must have at least one output buffer")
    
    # Validate all buffers
    for buffer in context.input_buffers:
        validate_buffer_info(buffer)
    for buffer in context.output_buffers:
        validate_buffer_info(buffer)

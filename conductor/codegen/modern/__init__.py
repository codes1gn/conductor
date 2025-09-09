"""
Modern DSL Code Generation Architecture.

This module provides a clean, type-safe, and composable architecture for
generating DSL code from computation graphs. It replaces the monolithic
ChoreoDslGen with a modular, testable design.

Key Features:
- Type-safe data structures with comprehensive type hints
- Protocol-based interfaces for flexibility and testability
- Immutable configuration and context objects
- Composable generators for different DSL components
- Template-based code generation with validation
- Clear separation of concerns

Architecture Overview:
- types.py: Core data structures and protocols
- config.py: Configuration management
- context.py: Generation context and state
- generators/: Specialized code generators
- templates/: Template rendering system
- naming/: Naming strategies and validation
- main.py: Main DSL generator orchestrator
"""

from .types import (
    DSLGenerationConfig,
    CodeGenerationContext,
    BufferInfo,
    OperationInfo,
    CodeFragment,
    DSLResult,
    ValidationResult,
    BufferScope,
    ValidationLevel,
    ParallelConfig,
    # Protocols
    CodeGenerator,
    TemplateRenderer,
    NamingStrategy,
    SyntaxValidator,
)

from .config import (
    create_default_config,
    create_config_from_dict,
    validate_config,
)

from .context import (
    create_generation_context,
    ContextBuilder,
    create_buffer_info,
    create_parallel_config,
    context_from_dag,
)

from .templates import (
    create_template_renderer,
    validate_template_syntax,
)

# Main generator will be added in later phases
# from .main import ModernDSLGenerator

__all__ = [
    # Core types
    "DSLGenerationConfig",
    "CodeGenerationContext",
    "BufferInfo",
    "OperationInfo",
    "CodeFragment",
    "DSLResult",
    "ValidationResult",
    "BufferScope",
    "ValidationLevel",
    "ParallelConfig",
    # Protocols
    "CodeGenerator",
    "TemplateRenderer",
    "NamingStrategy",
    "SyntaxValidator",
    # Configuration
    "create_default_config",
    "create_config_from_dict",
    "validate_config",
    # Context
    "create_generation_context",
    "ContextBuilder",
    "create_buffer_info",
    "create_parallel_config",
    "context_from_dag",
    # Templates
    "create_template_renderer",
    "validate_template_syntax",
    # Main generator (to be added)
    # "ModernDSLGenerator",
]

"""
Utils package for Conductor.

This module provides utility functions and classes for operation handling,
device management, configuration, and other common functionality.
"""

# Core utilities
from .exceptions import ConductorError, CompilationError, ExecutionError
from .constants import *
from .naming import *
from .string_utils import *

# Configuration and system utilities
from .config import (
    ConductorConfig,
    DebugConfig,
    CacheConfig,
    CompilationConfig,
    RuntimeConfig,
    LoggingConfig,
    get_config,
    set_config,
    load_config,
)

from .trace import (
    DebugTracer,
    get_debug_tracer,
    enable_debug_tracing,
    disable_debug_tracing,
    trace_fx_graph,
)

from .tracer import (
    ConductorTracer,
    get_tracer,
    trace_internal_dag,
    trace_choreo_dsl,
    trace_compilation_result,
    trace_execution_result,
)

from .type_mapping import (
    TypeMapper,
    ChoreoDataType,
    TypeInfo,
    get_type_mapper,
    torch_to_choreo_string,
    choreo_to_torch_dtype,
    is_supported_dtype,
    get_dtype_size_bytes,
)

from .symbolic_shapes import (
    SymbolicDimension,
    SymbolicDimType,
    SymbolicShapeResolver,
    get_symbolic_shape_resolver,
    resolve_symbolic_shape,
    infer_shape_context,
)

from .cache import CompilationCache
from .artifacts import get_debug_manager
from .logging import get_logger

__all__ = [
    # Core exceptions
    "ConductorError",
    "CompilationError",
    "ExecutionError",

    # Constants (exported via *)
    # Naming utilities (exported via *)
    # String utilities (exported via *)

    # Configuration
    "ConductorConfig",
    "DebugConfig",
    "CacheConfig",
    "CompilationConfig",
    "RuntimeConfig",
    "LoggingConfig",
    "get_config",
    "set_config",
    "load_config",

    # Debug and tracing (legacy)
    "DebugTracer",
    "get_debug_tracer",
    "enable_debug_tracing",
    "disable_debug_tracing",
    "trace_fx_graph",

    # Modern tracing
    "ConductorTracer",
    "get_tracer",
    "trace_internal_dag",
    "trace_choreo_dsl",
    "trace_compilation_result",
    "trace_execution_result",

    # Type mapping
    "TypeMapper",
    "ChoreoDataType",
    "TypeInfo",
    "get_type_mapper",
    "torch_to_choreo_string",
    "choreo_to_torch_dtype",
    "is_supported_dtype",
    "get_dtype_size_bytes",

    # Symbolic shapes
    "SymbolicDimension",
    "SymbolicDimType",
    "SymbolicShapeResolver",
    "get_symbolic_shape_resolver",
    "resolve_symbolic_shape",
    "infer_shape_context",

    # Caching and artifacts
    "CompilationCache",
    "get_debug_manager",

    # Logging
    "get_logger",
]

"""
Constants and Enumerations for Conductor Framework.

This module consolidates all constant definitions from across the project,
providing a single source of truth for configuration values, enums, and
other constant data.
"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Any


# =============================================================================
# Memory Hierarchy Constants
# =============================================================================

class MemoryLevel(Enum):
    """Memory hierarchy levels in GCU architecture."""
    
    GLOBAL = "global"  # Global memory (DDR)
    L2 = "l2"  # L2 cache
    L1 = "l1"  # L1 cache/local memory
    REGISTER = "reg"  # Register file


class BufferScope(Enum):
    """Defines memory scope hierarchy for buffer management."""
    
    LOCAL = "local"  # Temporary variables within single kernel
    SHARED = "shared"  # Inter-kernel communication within model execution
    GLOBAL = "global"  # Persistent data across multiple model invocations


class BufferType(Enum):
    """Types of buffers in computation."""
    
    INPUT = "input"  # Input parameter buffer
    OUTPUT = "output"  # Output result buffer
    INTERMEDIATE = "temp"  # Intermediate computation buffer
    LOAD = "load"  # DMA load buffer
    STORE = "store"  # DMA store buffer


# =============================================================================
# DSL Generation Constants
# =============================================================================

class ContextKeys(Enum):
    """Enumeration of all valid context parameter keys."""
    
    # Matrix dimensions
    MATRIX_M = "M"
    MATRIX_N = "N"
    MATRIX_K = "K"
    
    # Buffer dimensions
    BUFFER_M = "buffer_m"
    BUFFER_N = "buffer_n"
    BUFFER_K = "buffer_k"
    
    # Tensor shapes
    BATCH_SIZE = "batch_size"
    HEIGHT = "height"
    WIDTH = "width"
    CHANNELS = "channels"
    
    # Parallel execution
    PARALLEL_FACTOR = "parallel_factor"
    CHUNK_SIZE = "chunk_size"
    
    # Memory hierarchy
    MEMORY_LEVEL = "memory_level"
    BUFFER_SIZE = "buffer_size"


class DSLKeywords(Enum):
    """Choreo DSL keywords and constructs."""
    
    # Function declarations
    CO_FUNCTION = "__co__"
    COK_SECTION = "__cok__"
    CO_DEVICE = "__co_device__"
    
    # Control flow
    PARALLEL = "parallel"
    FOREACH = "foreach"
    IF = "if"
    ELSE = "else"
    
    # Memory operations
    DMA_COPY = "dma.copy"
    DMA_COPY_ASYNC = "dma.copy.async"
    WAIT = "wait"
    LOCAL = "local"
    
    # Data access
    CHUNKAT = "chunkat"
    AT = "at"
    SPAN = "span"
    DATA = "data"


# =============================================================================
# Operation and Fusion Constants
# =============================================================================

class FusionType(Enum):
    """Categorizes different types of operation fusion."""
    
    ELEMENTWISE = "elementwise"  # Element-wise operations (add, mul, relu)
    REDUCTION = "reduction"  # Reduction operations (sum, max, mean)
    MATMUL = "matmul"  # Matrix multiplication operations
    CONV = "conv"  # Convolution operations
    ATTENTION = "attention"  # Attention mechanism operations


class OpType(Enum):
    """Operation type classification."""
    
    ELEMENTWISE = auto()
    REDUCTION = auto()
    MATMUL = auto()
    CONV = auto()
    ATTENTION = auto()
    CUSTOM = auto()


# =============================================================================
# Configuration Constants
# =============================================================================

# Default configuration values
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_OPTIMIZATION_LEVEL = "O2"
DEFAULT_MEMORY_POOL_SIZE_MB = 512
DEFAULT_CACHE_SIZE_MB = 1024
DEFAULT_MAX_TENSOR_ELEMENTS = 100
DEFAULT_INDENT_SIZE = 2

# File extensions and paths
DSL_FILE_EXTENSION = ".co"
OBJECT_FILE_EXTENSION = ".o"
SHARED_LIBRARY_EXTENSION = ".so"
CPP_FILE_EXTENSION = ".cpp"

# Debug and logging constants
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = "conductor.log"
DEBUG_DIR_NAME = "debug_dir"

# Compilation constants
DEFAULT_PARALLEL_FACTOR = 1
DEFAULT_CHUNK_SIZE = 4
DEFAULT_BUFFER_SIZE = 16

# Template and naming constants
DEFAULT_FUNCTION_PREFIX = "kernel_"
DEFAULT_BUFFER_PREFIX = "buf_"
DEFAULT_INDEX_PREFIX = "idx_"
DEFAULT_PARALLEL_PREFIX = "p"
DEFAULT_LOAD_PREFIX = "load_"

# Separators and formatting
LEVEL_SEPARATOR = "_"
INDEX_SEPARATOR = "_"
TEMPLATE_INDENT = "  "

# Numerical constants
DEFAULT_TOLERANCE = 1e-5
DEFAULT_WARMUP_ITERATIONS = 10
DEFAULT_BENCHMARK_ITERATIONS = 100


# =============================================================================
# Error and Status Constants
# =============================================================================

class CompilationStatus(Enum):
    """Status of compilation process."""
    
    PENDING = "pending"
    COMPILING = "compiling"
    SUCCESS = "success"
    FAILED = "failed"
    CACHED = "cached"


class ExecutionStatus(Enum):
    """Status of kernel execution."""
    
    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


# =============================================================================
# Hardware and Device Constants
# =============================================================================

DEFAULT_DEVICE = "gcu"
SUPPORTED_DEVICES = ["gcu", "cpu", "cuda"]

# GCU specific constants
GCU_MAX_THREADS = 1024
GCU_MAX_BLOCKS = 65535
GCU_WARP_SIZE = 32

# Memory alignment constants
DEFAULT_MEMORY_ALIGNMENT = 32
CACHE_LINE_SIZE = 64


# =============================================================================
# Utility Constants
# =============================================================================

# String manipulation
CAMEL_CASE_PATTERN = r'([a-z0-9])([A-Z])'
SNAKE_CASE_REPLACEMENT = r'\1_\2'

# File and path constants
CONFIG_FILE_NAMES = ["conductor_config.yaml", "conductor_config.json", "config.yaml", "config.json"]
TEMP_FILE_PREFIX = "conductor_tmp_"

# Version and compatibility
MIN_PYTHON_VERSION = (3, 8)
MIN_TORCH_VERSION = "1.12.0"

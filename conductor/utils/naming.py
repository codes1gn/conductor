"""
Naming Utilities for Conductor Framework.

This module provides systematic naming and identifier generation utilities
for buffers, variables, functions, and other code constructs. It consolidates
naming logic from across the project into a single, reusable module.
"""

from __future__ import annotations

from typing import Optional, Set, Dict, Any
from dataclasses import dataclass
import re

from .constants import (
    MemoryLevel,
    BufferType,
    DEFAULT_FUNCTION_PREFIX,
    DEFAULT_BUFFER_PREFIX,
    DEFAULT_INDEX_PREFIX,
    DEFAULT_PARALLEL_PREFIX,
    DEFAULT_LOAD_PREFIX,
    LEVEL_SEPARATOR,
    INDEX_SEPARATOR,
    CAMEL_CASE_PATTERN,
    SNAKE_CASE_REPLACEMENT,
)


# =============================================================================
# Naming Context and Configuration
# =============================================================================

@dataclass
class NamingConfig:
    """Configuration for naming and identifier generation."""
    
    # Variable prefixes
    parallel_var_prefix: str = DEFAULT_PARALLEL_PREFIX
    index_var_prefix: str = DEFAULT_INDEX_PREFIX
    buffer_var_prefix: str = DEFAULT_BUFFER_PREFIX
    load_var_prefix: str = DEFAULT_LOAD_PREFIX
    
    # Separators
    level_separator: str = LEVEL_SEPARATOR
    index_separator: str = INDEX_SEPARATOR
    
    # Naming patterns
    use_operation_names: bool = True
    use_memory_levels: bool = True
    include_node_ids: bool = True


@dataclass
class BufferNamingContext:
    """Context for buffer naming decisions."""
    
    memory_level: MemoryLevel = MemoryLevel.L1
    buffer_type: BufferType = BufferType.INTERMEDIATE
    operation_name: Optional[str] = None
    sequence_id: Optional[int] = None
    is_fused: bool = False


# =============================================================================
# Core Naming Utilities
# =============================================================================

def camel_to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    return re.sub(CAMEL_CASE_PATTERN, SNAKE_CASE_REPLACEMENT, name).lower()


def snake_to_camel_case(name: str) -> str:
    """Convert snake_case to CamelCase."""
    components = name.split('_')
    return ''.join(word.capitalize() for word in components)


def sanitize_identifier(name: str) -> str:
    """Sanitize a string to be a valid identifier."""
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed"
    
    return sanitized


def generate_unique_name(base_name: str, used_names: Set[str], separator: str = "_") -> str:
    """Generate a unique name by appending a counter if needed."""
    if base_name not in used_names:
        return base_name
    
    counter = 1
    while True:
        candidate = f"{base_name}{separator}{counter}"
        if candidate not in used_names:
            return candidate
        counter += 1


# =============================================================================
# Specialized Naming Managers
# =============================================================================

class IdentifierGenerator:
    """
    Generates unique identifiers for DSL constructs.
    
    This class provides systematic identifier generation with configurable
    patterns and automatic uniqueness guarantees.
    """
    
    def __init__(self, config: Optional[NamingConfig] = None):
        """Initialize the identifier generator."""
        self.config = config or NamingConfig()
        self._used_names: Set[str] = set()
        self._name_counters: Dict[str, int] = {}
    
    def generate_parallel_var(self, level: int = 0) -> str:
        """Generate parallel variable name (e.g., 'p', 'p_1')."""
        base_name = f"{self.config.parallel_var_prefix}"
        if level > 0:
            base_name = f"{base_name}{self.config.level_separator}{level}"
        return self._ensure_unique(base_name)
    
    def generate_index_var(self, dimension: str = "") -> str:
        """Generate index variable name (e.g., 'idx_i', 'idx_j')."""
        if dimension:
            base_name = f"{self.config.index_var_prefix}_{dimension}"
        else:
            base_name = self.config.index_var_prefix
        return self._ensure_unique(base_name)
    
    def generate_buffer_var(self, memory_level: MemoryLevel, operation: str = "") -> str:
        """Generate buffer variable name."""
        parts = [memory_level.value]
        if operation:
            parts.append(operation)
        base_name = self.config.level_separator.join(parts)
        return self._ensure_unique(base_name)
    
    def generate_load_var(self, source_name: str) -> str:
        """Generate load variable name for DMA operations."""
        base_name = f"{self.config.load_var_prefix}{source_name}"
        return self._ensure_unique(base_name)
    
    def _ensure_unique(self, base_name: str) -> str:
        """Ensure name is unique by adding counter if needed."""
        if base_name not in self._used_names:
            self._used_names.add(base_name)
            return base_name
        
        counter = self._name_counters.get(base_name, 0)
        while True:
            counter += 1
            candidate = f"{base_name}{self.config.index_separator}{counter}"
            if candidate not in self._used_names:
                self._name_counters[base_name] = counter
                self._used_names.add(candidate)
                return candidate
    
    def reset(self):
        """Reset the generator for a new generation session."""
        self._used_names.clear()
        self._name_counters.clear()


class BufferNamingManager:
    """
    Manages unique buffer naming across DSL generation.
    
    This class provides systematic buffer naming with context awareness
    and automatic uniqueness guarantees.
    """
    
    def __init__(self):
        """Initialize the buffer naming manager."""
        self._used_names: Set[str] = set()
        self._name_counters: Dict[str, int] = {}
        self._buffer_registry: Dict[str, BufferNamingContext] = {}
    
    def generate_buffer_name(self, context: BufferNamingContext) -> str:
        """
        Generate a unique buffer name based on context.
        
        Args:
            context: Naming context with memory level, type, etc.
        
        Returns:
            Unique buffer name following Choreo conventions
        """
        # Build base name from context
        base_name = self._build_base_name(context)
        
        # Ensure uniqueness
        unique_name = self._ensure_unique_name(base_name)
        
        # Register the name
        self._used_names.add(unique_name)
        self._buffer_registry[unique_name] = context
        
        return unique_name
    
    def get_output_buffer_name(self, memory_level: MemoryLevel = MemoryLevel.L1) -> str:
        """Get standard output buffer name for the given memory level."""
        context = BufferNamingContext(memory_level=memory_level, buffer_type=BufferType.OUTPUT)
        return self.generate_buffer_name(context)
    
    def get_load_buffer_name(
        self, source_name: str, memory_level: MemoryLevel = MemoryLevel.L1
    ) -> str:
        """Get load buffer name for DMA operations."""
        context = BufferNamingContext(
            memory_level=memory_level, buffer_type=BufferType.LOAD, operation_name=source_name
        )
        return self.generate_buffer_name(context)
    
    def get_intermediate_buffer_name(
        self, operation: str, memory_level: MemoryLevel = MemoryLevel.L1
    ) -> str:
        """Get intermediate buffer name for computation results."""
        context = BufferNamingContext(
            memory_level=memory_level, buffer_type=BufferType.INTERMEDIATE, operation_name=operation
        )
        return self.generate_buffer_name(context)
    
    def _build_base_name(self, context: BufferNamingContext) -> str:
        """Build base name from context."""
        parts = []
        
        # Memory level prefix
        parts.append(context.memory_level.value)
        
        # Buffer type or operation
        if context.operation_name:
            # Use operation name for specificity
            parts.append(context.operation_name)
        else:
            # Use buffer type
            parts.append(context.buffer_type.value)
        
        # Join with underscore
        return "_".join(parts)
    
    def _ensure_unique_name(self, base_name: str) -> str:
        """Ensure name is unique by adding counter if needed."""
        if base_name not in self._used_names:
            return base_name
        
        # Get or initialize counter for this base name
        counter = self._name_counters.get(base_name, 0)
        
        while True:
            counter += 1
            candidate = f"{base_name}_{counter}"
            if candidate not in self._used_names:
                self._name_counters[base_name] = counter
                return candidate
    
    def reset(self):
        """Reset the naming manager for a new generation session."""
        self._used_names.clear()
        self._name_counters.clear()
        self._buffer_registry.clear()


# =============================================================================
# Function and Kernel Naming
# =============================================================================

def generate_kernel_name(graph_hash: str, prefix: str = DEFAULT_FUNCTION_PREFIX) -> str:
    """Generate a unique kernel function name."""
    return f"{prefix}{graph_hash}"


def generate_function_signature_name(operation: str, inputs: int, outputs: int) -> str:
    """Generate a descriptive function signature name."""
    return f"{operation}_{inputs}in_{outputs}out"


# =============================================================================
# Global Instances and Convenience Functions
# =============================================================================

# Global instances for use across the framework
default_identifier_generator = IdentifierGenerator()
default_buffer_naming_manager = BufferNamingManager()


def get_output_buffer_name(memory_level: MemoryLevel = MemoryLevel.L1) -> str:
    """Convenience function to get output buffer name."""
    return default_buffer_naming_manager.get_output_buffer_name(memory_level)


def get_load_buffer_name(source_name: str, memory_level: MemoryLevel = MemoryLevel.L1) -> str:
    """Convenience function to get load buffer name."""
    return default_buffer_naming_manager.get_load_buffer_name(source_name, memory_level)


def get_intermediate_buffer_name(operation: str, memory_level: MemoryLevel = MemoryLevel.L1) -> str:
    """Convenience function to get intermediate buffer name."""
    return default_buffer_naming_manager.get_intermediate_buffer_name(operation, memory_level)


def reset_naming_state():
    """Reset all naming state for new generation session."""
    default_identifier_generator.reset()
    default_buffer_naming_manager.reset()

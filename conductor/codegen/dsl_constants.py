"""
DSL Constants and Context Keys for Conductor.

This module defines all constants and context keys used in DSL generation,
eliminating hardcoded strings and providing a structured approach.

Note: This module is being deprecated in favor of conductor.utils.constants
"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Any

# Import from new utils location
from ..utils.constants import (
    MemoryLevel,
    ContextKeys,
    DSLKeywords,
    DEFAULT_PARALLEL_FACTOR,
    DEFAULT_CHUNK_SIZE,
    LEVEL_SEPARATOR,
    INDEX_SEPARATOR,
)
from ..utils.naming import NamingConfig, IdentifierGenerator


# These are now imported from utils.constants - keeping for backward compatibility


# Use NamingConfig from utils.naming for backward compatibility
DSLIdentifierConfig = NamingConfig


class DSLIdentifierGenerator:
    """Generates consistent DSL identifiers following naming conventions."""

    def __init__(self, config: Optional[DSLIdentifierConfig] = None):
        """Initialize with configuration."""
        self.config = config or DSLIdentifierConfig()
        self._used_names: set[str] = set()
        self._name_counters: dict[str, int] = {}

    def generate_parallel_var(self, level: int = 0) -> str:
        """Generate parallel variable name."""
        base_name = (
            f"{self.config.parallel_var_prefix}{level}"
            if level > 0
            else self.config.parallel_var_prefix
        )
        return self._ensure_unique(base_name)

    def generate_index_var(self, dimension: str = "", level: int = 0) -> str:
        """Generate index variable name."""
        if dimension:
            base_name = f"{self.config.index_var_prefix}_{dimension}"
        else:
            base_name = (
                f"{self.config.index_var_prefix}{level}"
                if level > 0
                else self.config.index_var_prefix
            )
        return self._ensure_unique(base_name)

    def generate_buffer_var(
        self, operation: str, memory_level: MemoryLevel, node_id: Optional[str] = None
    ) -> str:
        """Generate buffer variable name."""
        parts = []

        if self.config.use_memory_levels:
            parts.append(memory_level.value)

        if self.config.use_operation_names:
            parts.append(operation)

        if self.config.include_node_ids and node_id:
            parts.append(node_id)

        base_name = self.config.level_separator.join(parts)
        return self._ensure_unique(base_name)

    def generate_load_var(self, source_name: str, memory_level: MemoryLevel) -> str:
        """Generate load variable name."""
        base_name = f"{memory_level.value}_{self.config.load_var_prefix}_{source_name}"
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


@dataclass
class DSLGenerationContext:
    """Context for DSL generation with structured parameters."""

    # Matrix dimensions
    matrix_m: int = 4
    matrix_n: int = 4
    matrix_k: int = 4

    # Buffer dimensions
    buffer_m: int = 16
    buffer_n: int = 8
    buffer_k: int = 8

    # Parallel execution
    parallel_factor: int = 1
    chunk_size: int = 4

    # Memory configuration
    default_memory_level: MemoryLevel = MemoryLevel.L1

    def get_value(self, key: ContextKeys) -> Any:
        """Get context value by key."""
        mapping = {
            ContextKeys.MATRIX_M: self.matrix_m,
            ContextKeys.MATRIX_N: self.matrix_n,
            ContextKeys.MATRIX_K: self.matrix_k,
            ContextKeys.BUFFER_M: self.buffer_m,
            ContextKeys.BUFFER_N: self.buffer_n,
            ContextKeys.BUFFER_K: self.buffer_k,
            ContextKeys.PARALLEL_FACTOR: self.parallel_factor,
            ContextKeys.CHUNK_SIZE: self.chunk_size,
            ContextKeys.MEMORY_LEVEL: self.default_memory_level,
        }
        return mapping.get(key)

    def set_value(self, key: ContextKeys, value: Any):
        """Set context value by key."""
        if key == ContextKeys.MATRIX_M:
            self.matrix_m = value
        elif key == ContextKeys.MATRIX_N:
            self.matrix_n = value
        elif key == ContextKeys.MATRIX_K:
            self.matrix_k = value
        elif key == ContextKeys.BUFFER_M:
            self.buffer_m = value
        elif key == ContextKeys.BUFFER_N:
            self.buffer_n = value
        elif key == ContextKeys.BUFFER_K:
            self.buffer_k = value
        elif key == ContextKeys.PARALLEL_FACTOR:
            self.parallel_factor = value
        elif key == ContextKeys.CHUNK_SIZE:
            self.chunk_size = value
        elif key == ContextKeys.MEMORY_LEVEL:
            self.default_memory_level = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for template substitution."""
        return {
            ContextKeys.MATRIX_M.value: self.matrix_m,
            ContextKeys.MATRIX_N.value: self.matrix_n,
            ContextKeys.MATRIX_K.value: self.matrix_k,
            ContextKeys.BUFFER_M.value: self.buffer_m,
            ContextKeys.BUFFER_N.value: self.buffer_n,
            ContextKeys.BUFFER_K.value: self.buffer_k,
            ContextKeys.PARALLEL_FACTOR.value: self.parallel_factor,
            ContextKeys.CHUNK_SIZE.value: self.chunk_size,
        }


# Global instances for use across DSL generation
default_identifier_generator = DSLIdentifierGenerator()
default_generation_context = DSLGenerationContext()


def reset_dsl_generation():
    """Reset all global DSL generation state."""
    default_identifier_generator.reset()

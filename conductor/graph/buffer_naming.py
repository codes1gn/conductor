"""
Buffer Naming Utilities for Conductor.

This module provides utilities for generating unique and consistent buffer names
across DSL generation, addressing the TODO in dslgen.py about hardcoded identifiers.
"""

from __future__ import annotations

from typing import Optional, Set
from dataclasses import dataclass
from enum import Enum


class MemoryLevel(Enum):
    """Memory hierarchy levels in GCU architecture."""
    GLOBAL = "global"      # Global memory (DDR)
    L2 = "l2"             # L2 cache
    L1 = "l1"             # L1 cache/local memory
    REGISTER = "reg"       # Register file


class BufferType(Enum):
    """Types of buffers in computation."""
    INPUT = "input"        # Input parameter buffer
    OUTPUT = "output"      # Output result buffer
    INTERMEDIATE = "temp"  # Intermediate computation buffer
    LOAD = "load"         # DMA load buffer
    STORE = "store"       # DMA store buffer


@dataclass
class BufferNamingContext:
    """Context for buffer naming decisions."""
    memory_level: MemoryLevel = MemoryLevel.L1
    buffer_type: BufferType = BufferType.INTERMEDIATE
    operation_name: Optional[str] = None
    sequence_id: Optional[int] = None
    is_fused: bool = False


class BufferNamingManager:
    """
    Manages unique buffer naming across DSL generation.
    
    This class addresses the TODO about hardcoded buffer identifiers like 'l1_out'
    by providing a systematic approach to buffer naming.
    """
    
    def __init__(self):
        """Initialize the buffer naming manager."""
        self._used_names: Set[str] = set()
        self._name_counters: dict[str, int] = {}
        self._buffer_registry: dict[str, BufferNamingContext] = {}
    
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
        context = BufferNamingContext(
            memory_level=memory_level,
            buffer_type=BufferType.OUTPUT
        )
        return self.generate_buffer_name(context)
    
    def get_load_buffer_name(self, source_name: str, memory_level: MemoryLevel = MemoryLevel.L1) -> str:
        """Get load buffer name for DMA operations."""
        context = BufferNamingContext(
            memory_level=memory_level,
            buffer_type=BufferType.LOAD,
            operation_name=source_name
        )
        return self.generate_buffer_name(context)
    
    def get_intermediate_buffer_name(self, operation: str, memory_level: MemoryLevel = MemoryLevel.L1) -> str:
        """Get intermediate buffer name for computation results."""
        context = BufferNamingContext(
            memory_level=memory_level,
            buffer_type=BufferType.INTERMEDIATE,
            operation_name=operation
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
    
    def get_buffer_context(self, buffer_name: str) -> Optional[BufferNamingContext]:
        """Get the context for a previously generated buffer name."""
        return self._buffer_registry.get(buffer_name)
    
    def is_output_buffer(self, buffer_name: str) -> bool:
        """Check if a buffer is an output buffer."""
        context = self.get_buffer_context(buffer_name)
        return context is not None and context.buffer_type == BufferType.OUTPUT
    
    def get_memory_level(self, buffer_name: str) -> Optional[MemoryLevel]:
        """Get the memory level for a buffer."""
        context = self.get_buffer_context(buffer_name)
        return context.memory_level if context else None


# Global instance for use across DSL generation
buffer_naming_manager = BufferNamingManager()


def get_output_buffer_name(memory_level: MemoryLevel = MemoryLevel.L1) -> str:
    """Convenience function to get output buffer name."""
    return buffer_naming_manager.get_output_buffer_name(memory_level)


def get_load_buffer_name(source_name: str, memory_level: MemoryLevel = MemoryLevel.L1) -> str:
    """Convenience function to get load buffer name."""
    return buffer_naming_manager.get_load_buffer_name(source_name, memory_level)


def get_intermediate_buffer_name(operation: str, memory_level: MemoryLevel = MemoryLevel.L1) -> str:
    """Convenience function to get intermediate buffer name."""
    return buffer_naming_manager.get_intermediate_buffer_name(operation, memory_level)


def reset_buffer_naming():
    """Reset buffer naming for new generation session."""
    buffer_naming_manager.reset()

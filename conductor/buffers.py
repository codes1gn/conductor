"""
Buffer management and scoping.

This module provides classes and utilities for managing data buffers
in the computation graph, including scope management and memory optimization.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING
from enum import Enum
import torch

if TYPE_CHECKING:
    from .graph_analyzer import ConductorNode


class BufferScope(Enum):
    """Defines memory scope hierarchy for buffer management."""
    LOCAL = "local"      # Temporary variables within single kernel
    SHARED = "shared"    # Inter-kernel communication within model execution
    GLOBAL = "global"    # Persistent data across multiple model invocations


@dataclass
class Buffer:
    """
    Represents data flow and memory management in the computation graph.
    
    This class encapsulates all information needed to track data buffers
    throughout the compilation pipeline, including scope management and
    dependency tracking.
    """
    name: str                                       # Unique identifier for the buffer
    scope: BufferScope                              # Memory scope (LOCAL, SHARED, GLOBAL)
    dtype: torch.dtype                              # Data type information
    shape: Optional[Tuple[int, ...]] = None         # Shape when statically known
    producer: Optional['ConductorNode'] = None      # Node that produces this buffer
    consumers: List['ConductorNode'] = None         # Nodes that consume this buffer
    is_temporary: bool = False                      # Whether this is a temporary intermediate buffer
    
    def __post_init__(self):
        if self.consumers is None:
            self.consumers = []
    
    def __hash__(self):
        """Make Buffer hashable for use in sets and dictionaries."""
        # Use id() to ensure unique hash for each instance
        return hash(id(self))
    
    def __eq__(self, other):
        """Define equality based on object identity."""
        return self is other
    
    def promote_scope(self, new_scope: BufferScope) -> None:
        """
        Promote buffer to higher scope when needed for sharing.
        
        Args:
            new_scope: Target scope to promote to
        """
        # Define scope hierarchy: LOCAL < SHARED < GLOBAL
        scope_hierarchy = {
            BufferScope.LOCAL: 0,
            BufferScope.SHARED: 1,
            BufferScope.GLOBAL: 2
        }
        
        if scope_hierarchy[new_scope] > scope_hierarchy[self.scope]:
            self.scope = new_scope
            
    def get_memory_footprint(self) -> int:
        """
        Calculate memory requirements for this buffer.
        
        Returns:
            Memory footprint in bytes, or -1 if shape is unknown
        """
        if self.shape is None:
            return -1  # Unknown size
            
        # Calculate total elements
        total_elements = 1
        for dim in self.shape:
            total_elements *= dim
            
        # Get dtype size in bytes
        dtype_sizes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.int32: 4,
            torch.int64: 8,
            torch.bool: 1,
        }
        
        dtype_size = dtype_sizes.get(self.dtype, 4)  # Default to 4 bytes
        return total_elements * dtype_size


class BufferManager:
    """
    Manages buffer allocation and scope promotion.
    
    This class handles the lifecycle of buffers throughout the compilation
    process, including automatic scope promotion and memory optimization.
    """
    
    def __init__(self):
        self._buffers: Dict[str, Buffer] = {}
        self._buffer_counter = 0
    
    def allocate_buffer(self, name: str, dtype: torch.dtype, shape: Optional[Tuple[int, ...]] = None) -> Buffer:
        """
        Allocate new buffer with appropriate initial scope.
        
        Args:
            name: Buffer identifier
            dtype: Data type for the buffer
            shape: Optional shape information
            
        Returns:
            Newly allocated Buffer instance
        """
        # Generate unique name if needed
        if name in self._buffers:
            self._buffer_counter += 1
            name = f"{name}_{self._buffer_counter}"
        
        # Start with LOCAL scope by default
        buffer = Buffer(
            name=name,
            scope=BufferScope.LOCAL,
            dtype=dtype,
            shape=shape
        )
        
        self._buffers[name] = buffer
        return buffer
        
    def promote_buffer_scope(self, buffer: Buffer, required_scope: BufferScope) -> None:
        """
        Promote buffer to higher scope when sharing is required.
        
        Args:
            buffer: Buffer to promote
            required_scope: Minimum required scope
        """
        buffer.promote_scope(required_scope)
        
    def optimize_buffer_reuse(self, buffers: List[Buffer]) -> Dict[str, str]:
        """
        Identify opportunities for buffer reuse to reduce memory footprint.
        
        Args:
            buffers: List of buffers to analyze for reuse opportunities
            
        Returns:
            Dictionary mapping original buffer names to reused buffer names
        """
        # TODO: Implement sophisticated buffer reuse analysis
        reuse_map = {}
        
        # Simple heuristic: reuse buffers with same shape and dtype
        # that don't have overlapping lifetimes
        for i, buffer1 in enumerate(buffers):
            if buffer1.name in reuse_map:
                continue
                
            for j, buffer2 in enumerate(buffers[i+1:], i+1):
                if (buffer2.name not in reuse_map and
                    buffer1.dtype == buffer2.dtype and
                    buffer1.shape == buffer2.shape and
                    self._can_reuse_buffers(buffer1, buffer2)):
                    reuse_map[buffer2.name] = buffer1.name
                    break
        
        return reuse_map
        
    def _can_reuse_buffers(self, buffer1: Buffer, buffer2: Buffer) -> bool:
        """
        Check if two buffers can safely reuse the same memory.
        
        Args:
            buffer1: First buffer
            buffer2: Second buffer
            
        Returns:
            True if buffers can be reused, False otherwise
        """
        # TODO: Implement proper lifetime analysis
        # For now, only allow reuse of temporary buffers
        return buffer1.is_temporary and buffer2.is_temporary
        
    def get_buffer(self, name: str) -> Optional[Buffer]:
        """
        Retrieve buffer by name.
        
        Args:
            name: Buffer identifier
            
        Returns:
            Buffer instance if found, None otherwise
        """
        return self._buffers.get(name)
        
    def list_buffers(self) -> List[Buffer]:
        """
        Get list of all managed buffers.
        
        Returns:
            List of all Buffer instances
        """
        return list(self._buffers.values())
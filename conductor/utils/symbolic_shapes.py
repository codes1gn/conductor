"""
Symbolic Shape Handling for Conductor Framework.

This module provides robust handling of symbolic shapes in DSL generation and execution,
supporting dynamic shape resolution, shape constraints, and symbolic dimension mapping.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from .logging import get_logger

logger = get_logger(__name__)


class SymbolicDimType(Enum):
    """Types of symbolic dimensions."""
    
    BATCH_SIZE = "batch_size"  # Batch dimension (N)
    SEQUENCE_LENGTH = "seq_len"  # Sequence length (L, T)
    FEATURE_SIZE = "feature_size"  # Feature dimension (D, H, W)
    DYNAMIC = "dynamic"  # General dynamic dimension
    UNKNOWN = "unknown"  # Unknown symbolic dimension


@dataclass
class SymbolicDimension:
    """Represents a symbolic dimension with metadata."""
    
    name: str
    dim_type: SymbolicDimType = SymbolicDimType.UNKNOWN
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    default_value: int = 16
    constraints: List[str] = field(default_factory=list)
    
    def resolve(self, context: Optional[Dict[str, int]] = None) -> int:
        """
        Resolve symbolic dimension to concrete value.
        
        Args:
            context: Context with known dimension values
            
        Returns:
            Concrete dimension value
        """
        if context and self.name in context:
            value = context[self.name]
            if self.min_value is not None and value < self.min_value:
                logger.warning(f"Dimension {self.name}={value} below minimum {self.min_value}")
                return self.min_value
            if self.max_value is not None and value > self.max_value:
                logger.warning(f"Dimension {self.name}={value} above maximum {self.max_value}")
                return self.max_value
            return value
        
        return self.default_value


class SymbolicShapeResolver:
    """
    Resolves symbolic shapes to concrete shapes with context awareness.
    """
    
    def __init__(self):
        """Initialize the symbolic shape resolver."""
        self._dimension_registry: Dict[str, SymbolicDimension] = {}
        self._shape_context: Dict[str, int] = {}
        self._common_patterns = self._build_common_patterns()
    
    def _build_common_patterns(self) -> Dict[str, SymbolicDimension]:
        """Build registry of common symbolic dimension patterns."""
        return {
            # Batch dimensions
            "N": SymbolicDimension("N", SymbolicDimType.BATCH_SIZE, min_value=1, default_value=16),
            "B": SymbolicDimension("B", SymbolicDimType.BATCH_SIZE, min_value=1, default_value=16),
            "batch": SymbolicDimension("batch", SymbolicDimType.BATCH_SIZE, min_value=1, default_value=16),
            
            # Sequence dimensions
            "L": SymbolicDimension("L", SymbolicDimType.SEQUENCE_LENGTH, min_value=1, default_value=32),
            "T": SymbolicDimension("T", SymbolicDimType.SEQUENCE_LENGTH, min_value=1, default_value=32),
            "seq_len": SymbolicDimension("seq_len", SymbolicDimType.SEQUENCE_LENGTH, min_value=1, default_value=32),
            
            # Feature dimensions
            "D": SymbolicDimension("D", SymbolicDimType.FEATURE_SIZE, min_value=1, default_value=64),
            "H": SymbolicDimension("H", SymbolicDimType.FEATURE_SIZE, min_value=1, default_value=64),
            "W": SymbolicDimension("W", SymbolicDimType.FEATURE_SIZE, min_value=1, default_value=64),
            "C": SymbolicDimension("C", SymbolicDimType.FEATURE_SIZE, min_value=1, default_value=32),
            
            # Matrix dimensions
            "M": SymbolicDimension("M", SymbolicDimType.FEATURE_SIZE, min_value=1, default_value=32),
            "K": SymbolicDimension("K", SymbolicDimType.FEATURE_SIZE, min_value=1, default_value=32),
        }
    
    def register_dimension(self, dim: SymbolicDimension) -> None:
        """Register a symbolic dimension."""
        self._dimension_registry[dim.name] = dim
        logger.debug(f"Registered symbolic dimension: {dim.name}")
    
    def set_context(self, context: Dict[str, int]) -> None:
        """Set shape resolution context."""
        self._shape_context.update(context)
        logger.debug(f"Updated shape context: {context}")
    
    def clear_context(self) -> None:
        """Clear shape resolution context."""
        self._shape_context.clear()
    
    def parse_shape_string(self, shape_str: str) -> List[Union[int, SymbolicDimension]]:
        """
        Parse shape string into list of concrete and symbolic dimensions.
        
        Args:
            shape_str: Shape string like "16, 32" or "N, M, 64"
            
        Returns:
            List of dimensions (int for concrete, SymbolicDimension for symbolic)
        """
        dimensions = []
        parts = [part.strip() for part in shape_str.split(",")]
        
        for part in parts:
            if part.isdigit():
                # Concrete dimension
                dimensions.append(int(part))
            else:
                # Symbolic dimension
                symbolic_dim = self._resolve_symbolic_name(part)
                dimensions.append(symbolic_dim)
        
        return dimensions
    
    def _resolve_symbolic_name(self, name: str) -> SymbolicDimension:
        """
        Resolve symbolic dimension name to SymbolicDimension object.
        
        Args:
            name: Symbolic dimension name
            
        Returns:
            SymbolicDimension object
        """
        # Check registered dimensions first
        if name in self._dimension_registry:
            return self._dimension_registry[name]
        
        # Check common patterns
        if name in self._common_patterns:
            return self._common_patterns[name]
        
        # Create new symbolic dimension for unknown names
        logger.debug(f"Creating new symbolic dimension for unknown name: {name}")
        dim = SymbolicDimension(name, SymbolicDimType.UNKNOWN, default_value=16)
        self._dimension_registry[name] = dim
        return dim
    
    def resolve_shape(self, shape: List[Union[int, SymbolicDimension]]) -> List[int]:
        """
        Resolve shape to concrete integers.
        
        Args:
            shape: List of dimensions (int or SymbolicDimension)
            
        Returns:
            List of concrete integer dimensions
        """
        resolved = []
        for dim in shape:
            if isinstance(dim, int):
                resolved.append(dim)
            elif isinstance(dim, SymbolicDimension):
                resolved.append(dim.resolve(self._shape_context))
            else:
                logger.warning(f"Unknown dimension type: {type(dim)}, using default 16")
                resolved.append(16)
        
        return resolved
    
    def resolve_shape_string(self, shape_str: str, context: Optional[Dict[str, int]] = None) -> List[int]:
        """
        Parse and resolve shape string to concrete dimensions.
        
        Args:
            shape_str: Shape string like "N, M, 64"
            context: Optional context for dimension resolution
            
        Returns:
            List of concrete integer dimensions
        """
        if context:
            old_context = self._shape_context.copy()
            self.set_context(context)
        
        try:
            parsed_shape = self.parse_shape_string(shape_str)
            resolved_shape = self.resolve_shape(parsed_shape)
            return resolved_shape
        finally:
            if context:
                self._shape_context = old_context
    
    def infer_context_from_tensors(self, tensors: List[Any]) -> Dict[str, int]:
        """
        Infer symbolic dimension context from actual tensor shapes.
        
        Args:
            tensors: List of tensors with shape information
            
        Returns:
            Inferred context mapping
        """
        context = {}
        
        for i, tensor in enumerate(tensors):
            if hasattr(tensor, 'shape'):
                shape = tensor.shape
                # Simple heuristic: map common dimension positions
                if len(shape) >= 1:
                    context.setdefault("N", shape[0])  # Batch dimension
                if len(shape) >= 2:
                    context.setdefault("M", shape[1])  # First feature dimension
                if len(shape) >= 3:
                    context.setdefault("K", shape[2])  # Second feature dimension
                if len(shape) >= 4:
                    context.setdefault("L", shape[3])  # Sequence/channel dimension
        
        logger.debug(f"Inferred context from tensors: {context}")
        return context
    
    def validate_shape_compatibility(self, expected_shape: str, actual_shape: List[int]) -> bool:
        """
        Validate that actual shape is compatible with expected symbolic shape.
        
        Args:
            expected_shape: Expected shape string with symbolic dimensions
            actual_shape: Actual concrete shape
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            parsed_expected = self.parse_shape_string(expected_shape)
            
            if len(parsed_expected) != len(actual_shape):
                return False
            
            # Build context from actual shape
            context = {}
            for i, (expected_dim, actual_dim) in enumerate(zip(parsed_expected, actual_shape)):
                if isinstance(expected_dim, SymbolicDimension):
                    if expected_dim.name in context:
                        # Check consistency
                        if context[expected_dim.name] != actual_dim:
                            return False
                    else:
                        context[expected_dim.name] = actual_dim
                elif expected_dim != actual_dim:
                    return False
            
            return True
        except Exception as e:
            logger.warning(f"Shape compatibility check failed: {e}")
            return False


# Global symbolic shape resolver instance
_global_resolver: Optional[SymbolicShapeResolver] = None


def get_symbolic_shape_resolver() -> SymbolicShapeResolver:
    """Get the global symbolic shape resolver instance."""
    global _global_resolver
    if _global_resolver is None:
        _global_resolver = SymbolicShapeResolver()
    return _global_resolver


def resolve_symbolic_shape(shape_str: str, context: Optional[Dict[str, int]] = None) -> List[int]:
    """
    Convenience function to resolve symbolic shape string.
    
    Args:
        shape_str: Shape string with symbolic dimensions
        context: Optional context for resolution
        
    Returns:
        List of concrete integer dimensions
    """
    resolver = get_symbolic_shape_resolver()
    return resolver.resolve_shape_string(shape_str, context)


def infer_shape_context(tensors: List[Any]) -> Dict[str, int]:
    """
    Convenience function to infer shape context from tensors.
    
    Args:
        tensors: List of tensors
        
    Returns:
        Inferred context mapping
    """
    resolver = get_symbolic_shape_resolver()
    return resolver.infer_context_from_tensors(tensors)

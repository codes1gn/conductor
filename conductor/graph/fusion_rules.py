"""
Fusion Rules Configuration for GCU Operations.

This module contains the fusion rules and compatibility logic that determines
which operations can be fused together for optimization. The rules are
extracted from the graph analyzer to make them easily modifiable.
"""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass
from ..codegen.operator_registry import operator_registry, is_elementwise, is_reduction


@dataclass
class FusionRule:
    """Represents a fusion rule between operation types."""
    source_ops: set[str]
    target_ops: set[str]
    compatibility_check: str  # Method name for compatibility checking
    description: str


class FusionRules:
    """
    Centralized fusion rules configuration for GCU operations.
    """
    
    def __init__(self):
        """Initialize fusion rules configuration."""
        # Use operator registry for operation definitions
        self.elementwise_ops = operator_registry.get_elementwise_operations()
        self.reduction_ops = operator_registry.get_reduction_operations()
        
        # Define fusion rules
        self.fusion_rules = [
            FusionRule(
                source_ops=self.elementwise_ops,
                target_ops=self.elementwise_ops,
                compatibility_check='check_elementwise_elementwise',
                description='Elementwise operations can fuse with other elementwise operations'
            ),
            FusionRule(
                source_ops=self.elementwise_ops,
                target_ops=self.reduction_ops,
                compatibility_check='check_elementwise_reduction',
                description='Elementwise operations can fuse with reductions'
            )
        ]
    
    def can_operations_fuse(self, op1: str, op2: str) -> bool:
        """
        Check if two operations can be fused based on their names.
        
        Args:
            op1: First operation name
            op2: Second operation name
            
        Returns:
            True if operations can be fused, False otherwise
        """
        for rule in self.fusion_rules:
            if op1 in rule.source_ops and op2 in rule.target_ops:
                return True
            if op2 in rule.source_ops and op1 in rule.target_ops:
                return True
        return False
    
    def get_fusion_compatibility_check(self, op1: str, op2: str) -> Optional[str]:
        """
        Get the compatibility check method for two operations.
        
        Args:
            op1: First operation name
            op2: Second operation name
            
        Returns:
            Method name for compatibility checking, or None if not fusable
        """
        for rule in self.fusion_rules:
            if op1 in rule.source_ops and op2 in rule.target_ops:
                return rule.compatibility_check
            if op2 in rule.source_ops and op1 in rule.target_ops:
                return rule.compatibility_check
        return None
    
    def is_elementwise_operation(self, op_name: str) -> bool:
        """Check if an operation is elementwise."""
        return is_elementwise(op_name)

    def is_reduction_operation(self, op_name: str) -> bool:
        """Check if an operation is a reduction."""
        return is_reduction(op_name)
    
    def add_custom_elementwise_op(self, op_name: str) -> None:
        """Add a custom elementwise operation to the fusion rules."""
        self.elementwise_ops.add(op_name)
    
    def add_custom_reduction_op(self, op_name: str) -> None:
        """Add a custom reduction operation to the fusion rules."""
        self.reduction_ops.add(op_name)
    
    def get_fusable_operations(self) -> set[str]:
        """Get all operations that can participate in fusion."""
        return self.elementwise_ops | self.reduction_ops
    
    def get_operation_category(self, op_name: str) -> Optional[str]:
        """
        Get the category of an operation.
        
        Args:
            op_name: Operation name
            
        Returns:
            'elementwise', 'reduction', or None if unknown
        """
        if op_name in self.elementwise_ops:
            return 'elementwise'
        elif op_name in self.reduction_ops:
            return 'reduction'
        return None


# Global fusion rules instance
_fusion_rules = None


def get_fusion_rules() -> FusionRules:
    """
    Get the global fusion rules instance (singleton pattern).
    
    Returns:
        FusionRules: The global fusion rules configuration
    """
    global _fusion_rules
    if _fusion_rules is None:
        _fusion_rules = FusionRules()
    return _fusion_rules


def can_fuse_operations(op1: str, op2: str) -> bool:
    """
    Convenience function to check if two operations can be fused.
    
    Args:
        op1: First operation name
        op2: Second operation name
        
    Returns:
        True if operations can be fused, False otherwise
    """
    return get_fusion_rules().can_operations_fuse(op1, op2)


def is_elementwise_op(op_name: str) -> bool:
    """
    Convenience function to check if an operation is elementwise.

    Args:
        op_name: Operation name

    Returns:
        True if operation is elementwise, False otherwise
    """
    return is_elementwise(op_name)

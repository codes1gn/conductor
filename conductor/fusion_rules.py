"""
Fusion Rules Configuration for GCU Operations.

This module contains the fusion rules and compatibility logic that determines
which operations can be fused together for optimization. The rules are
extracted from the graph analyzer to make them easily modifiable.
"""

from typing import Set, Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class FusionRule:
    """Represents a fusion rule between operation types."""
    source_ops: Set[str]
    target_ops: Set[str]
    compatibility_check: str  # Method name for compatibility checking
    description: str


class FusionRules:
    """
    Centralized fusion rules configuration for GCU operations.
    
    This class contains all the fusion rules and compatibility logic,
    making it easy for developers to modify fusion behavior without
    deep code understanding.
    """
    
    def __init__(self):
        """Initialize fusion rules configuration."""
        # Define operation categories
        self.elementwise_ops = {
            'add', 'sub', 'mul', 'div', 'sigmoid', 'tanh',
            'abs', 'neg', 'exp', 'log', 'sqrt',
            'custom_add', 'custom_mul'  # Include custom operations
        }
        
        self.reduction_ops = {
            'sum', 'mean', 'max', 'min'
        }
        
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
        return op_name in self.elementwise_ops
    
    def is_reduction_operation(self, op_name: str) -> bool:
        """Check if an operation is a reduction."""
        return op_name in self.reduction_ops
    
    def add_custom_elementwise_op(self, op_name: str) -> None:
        """Add a custom elementwise operation to the fusion rules."""
        self.elementwise_ops.add(op_name)
    
    def add_custom_reduction_op(self, op_name: str) -> None:
        """Add a custom reduction operation to the fusion rules."""
        self.reduction_ops.add(op_name)
    
    def get_fusable_operations(self) -> Set[str]:
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
    return get_fusion_rules().is_elementwise_operation(op_name)

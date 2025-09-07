"""
Abstract Base Classes for DSL Generation using Template Method Pattern.

This module provides the abstract base classes and interfaces for DSL generation,
implementing the Template Method Pattern to allow for flexible and extensible
code generation while maintaining a consistent structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
from ..config.logging import get_logger

logger = get_logger(__name__)


class DslGenerator(ABC):
    """
    Abstract base class for DSL generators using Template Method Pattern.
    
    This class defines the template method for DSL generation and delegates
    specific implementation details to subclasses.
    """
    
    def __init__(self):
        self.indent_level = 0
        self.indent_size = 4
    
    def generate_dsl_file(self, dag: ComputationDAG, function_name: str = "conductor_kernel") -> str:
        """
        Template method for generating complete DSL file.
        
        This method defines the overall structure and delegates specific
        generation steps to abstract methods that subclasses must implement.
        
        Args:
            dag: Computation DAG to generate DSL for
            function_name: Name of the generated function
            
        Returns:
            Complete DSL file content as string
        """
        logger.info(f"Generating DSL file for function: {function_name}")
        
        dsl_lines = []
        
        # Template method steps - each can be overridden by subclasses
        dsl_lines.extend(self.generate_header())
        
        device_section = self.generate_device_section(dag)
        if device_section:
            dsl_lines.append("")
            dsl_lines.extend(device_section)
        
        dsl_lines.append("")
        dsl_lines.extend(self.generate_co_func(dag, function_name))
        
        footer = self.generate_footer()
        if footer:
            dsl_lines.append("")
            dsl_lines.extend(footer)
        
        return '\n'.join(dsl_lines)
    
    @abstractmethod
    def generate_header(self) -> list[str]:
        """Generate DSL file header (includes, imports, etc.)."""
        pass
    
    @abstractmethod
    def generate_device_section(self, dag: ComputationDAG) -> Optional[list[str]]:
        """Generate device-specific code section if needed."""
        pass
    
    @abstractmethod
    def generate_co_func(self, dag: ComputationDAG, function_name: str) -> list[str]:
        """Generate the main function containing the computation."""
        pass
    
    def generate_footer(self) -> Optional[list[str]]:
        """Generate DSL file footer (optional, default is None)."""
        return None
    
    @abstractmethod
    def generate_operation(self, node: ConductorNode, context: dict[str, Any]) -> list[str]:
        """Generate code for a single operation."""
        pass
    
    @abstractmethod
    def generate_buffer_declaration(self, buffer: Buffer) -> str:
        """Generate buffer declaration code."""
        pass
    
    @abstractmethod
    def generate_parallel_structure(self, dag: ComputationDAG) -> list[str]:
        """Generate parallel execution structure."""
        pass
    
    def _indent(self, line: str) -> str:
        """Add indentation to a line."""
        return " " * (self.indent_level * self.indent_size) + line
    
    def _increase_indent(self) -> None:
        """Increase indentation level."""
        self.indent_level += 1
    
    def _decrease_indent(self) -> None:
        """Decrease indentation level."""
        self.indent_level = max(0, self.indent_level - 1)


class OperationHandler(ABC):
    """
    Abstract base class for operation-specific code generation handlers.
    
    This allows for clean separation of operation-specific logic and
    makes it easy to add new operations without modifying the main generator.
    """
    
    @abstractmethod
    def get_operation_name(self) -> str:
        """Get the name of the operation this handler supports."""
        pass
    
    @abstractmethod
    def can_handle(self, node: ConductorNode) -> bool:
        """Check if this handler can generate code for the given node."""
        pass
    
    @abstractmethod
    def generate_code(self, node: ConductorNode, context: dict[str, Any]) -> list[str]:
        """Generate code for the operation."""
        pass
    
    @abstractmethod
    def get_buffer_requirements(self, node: ConductorNode) -> list[str]:
        """Get list of buffer names required by this operation."""
        pass
    
    def validate_inputs(self, node: ConductorNode) -> bool:
        """Validate that the node has the correct inputs for this operation."""
        return True
    
    def get_fusion_compatibility(self) -> dict[str, Any]:
        """Get information about fusion compatibility."""
        return {
            'fusable': True,
            'fusion_group': 'elementwise',
            'memory_bound': True
        }


class OperationHandlerRegistry:
    """Registry for operation handlers using the Strategy Pattern."""
    
    def __init__(self):
        self.handlers: dict[str, OperationHandler] = {}
    
    def register_handler(self, handler: OperationHandler) -> None:
        """Register an operation handler."""
        op_name = handler.get_operation_name()
        self.handlers[op_name] = handler
        logger.info(f"Registered operation handler: {op_name}")
    
    def get_handler(self, operation_name: str) -> Optional[OperationHandler]:
        """Get handler for a specific operation."""
        return self.handlers.get(operation_name)
    
    def get_handler_for_node(self, node: ConductorNode) -> Optional[OperationHandler]:
        """Get the appropriate handler for a node."""
        # First try exact match
        handler = self.get_handler(node.op_name)
        if handler and handler.can_handle(node):
            return handler
        
        # Then try all handlers to see if any can handle this node
        for handler in self.handlers.values():
            if handler.can_handle(node):
                return handler
        
        return None
    
    def list_supported_operations(self) -> list[str]:
        """Get list of all supported operations."""
        return list(self.handlers.keys())


# Global handler registry
operation_handler_registry = OperationHandlerRegistry()


def register_operation_handler(handler: OperationHandler) -> None:
    """Register an operation handler globally."""
    operation_handler_registry.register_handler(handler)


def get_operation_handler(operation_name: str) -> Optional[OperationHandler]:
    """Get operation handler by name."""
    return operation_handler_registry.get_handler(operation_name)

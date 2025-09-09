"""
Modern DSL Builder for Choreo Code Generation.

This module implements a clean, modern architecture for generating Choreo DSL code
using the Builder pattern with proper variable scoping and naming consistency.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from ..utils.constants import MemoryLevel, DSLKeywords
from ..utils.naming import generate_unique_name
from ..utils.string_utils import indent_text, format_code_block
from ..utils.logging import get_logger

logger = get_logger(__name__)


class VariableScope(Enum):
    """Variable scope levels for proper scoping."""
    GLOBAL = "global"
    FUNCTION = "function"
    PARALLEL = "parallel"
    LOOP = "loop"


@dataclass
class Variable:
    """Represents a variable in the DSL with proper scoping."""
    name: str
    var_type: str
    scope: VariableScope
    dimensions: Optional[List[int]] = None
    memory_level: MemoryLevel = MemoryLevel.L1
    
    def get_declaration(self) -> str:
        """Get the variable declaration string."""
        if self.dimensions:
            dims_str = ", ".join(str(d) for d in self.dimensions)
            if self.memory_level == MemoryLevel.L1:
                return f"local {self.var_type} [{dims_str}] {self.name};"
            else:
                return f"{self.var_type} [{dims_str}] {self.name};"
        else:
            return f"{self.var_type} {self.name};"


@dataclass
class VariableManager:
    """Manages variable naming and scoping."""
    
    _variables: Dict[str, Variable] = field(default_factory=dict)
    _scope_counters: Dict[VariableScope, Dict[str, int]] = field(default_factory=dict)
    _used_names: set = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize scope counters."""
        for scope in VariableScope:
            self._scope_counters[scope] = {}
    
    def create_variable(
        self, 
        base_name: str, 
        var_type: str, 
        scope: VariableScope,
        dimensions: Optional[List[int]] = None,
        memory_level: MemoryLevel = MemoryLevel.L1
    ) -> Variable:
        """Create a new variable with unique naming."""
        # Generate unique name
        unique_name = generate_unique_name(base_name, self._used_names)
        self._used_names.add(unique_name)
        
        # Create variable
        var = Variable(
            name=unique_name,
            var_type=var_type,
            scope=scope,
            dimensions=dimensions,
            memory_level=memory_level
        )
        
        self._variables[unique_name] = var
        return var
    
    def get_variable(self, name: str) -> Optional[Variable]:
        """Get a variable by name."""
        return self._variables.get(name)
    
    def get_variables_in_scope(self, scope: VariableScope) -> List[Variable]:
        """Get all variables in a specific scope."""
        return [var for var in self._variables.values() if var.scope == scope]


class DSLBuilder:
    """
    Modern DSL Builder using Builder pattern for clean code generation.
    
    This class provides a fluent interface for building Choreo DSL code
    with proper variable management and scoping.
    """
    
    def __init__(self):
        """Initialize the DSL builder."""
        self.var_manager = VariableManager()
        self._lines: List[str] = []
        self._indent_level = 0
        self._current_scope = VariableScope.GLOBAL
        
    def reset(self):
        """Reset the builder for a new generation."""
        self.var_manager = VariableManager()
        self._lines = []
        self._indent_level = 0
        self._current_scope = VariableScope.GLOBAL
        
    def add_header(self) -> 'DSLBuilder':
        """Add standard DSL header."""
        self._lines.extend([
            "// Generated Choreo DSL",
            "// Auto-generated from PyTorch FX Graph via Conductor",
            "",
            '#include "choreo.h"'
        ])
        return self
    
    def begin_function(self, name: str, params: List[str], return_type: str = "auto") -> 'DSLBuilder':
        """Begin a function definition."""
        param_str = ", ".join(params)
        self._lines.append(f"__co__ {return_type} {name}({param_str}) {{")
        self._indent_level += 1
        self._current_scope = VariableScope.FUNCTION
        return self
    
    def end_function(self) -> 'DSLBuilder':
        """End the current function."""
        self._indent_level -= 1
        self._lines.append("}")
        self._current_scope = VariableScope.GLOBAL
        return self
    
    def declare_variable(
        self, 
        base_name: str, 
        var_type: str, 
        dimensions: Optional[List[int]] = None,
        memory_level: MemoryLevel = MemoryLevel.GLOBAL
    ) -> Variable:
        """Declare a new variable."""
        var = self.var_manager.create_variable(
            base_name, var_type, self._current_scope, dimensions, memory_level
        )
        
        # Add declaration to code
        declaration = var.get_declaration()
        self._add_line(declaration)
        
        return var
    
    def begin_parallel(self, parallel_var: str, factor: int = 1) -> 'DSLBuilder':
        """Begin a parallel block."""
        self._add_line(f"parallel {parallel_var} by {factor}")
        self._indent_level += 1
        self._current_scope = VariableScope.PARALLEL
        return self
    
    def end_parallel(self) -> 'DSLBuilder':
        """End the current parallel block."""
        self._indent_level -= 1
        self._current_scope = VariableScope.FUNCTION
        return self
    
    def begin_foreach(self, loop_var: str, range_expr: str) -> 'DSLBuilder':
        """Begin a foreach loop."""
        self._add_line(f"foreach {loop_var} in {range_expr} {{")
        self._indent_level += 1
        self._current_scope = VariableScope.LOOP
        return self
    
    def end_foreach(self) -> 'DSLBuilder':
        """End the current foreach loop."""
        self._indent_level -= 1
        self._add_line("}")
        self._current_scope = VariableScope.PARALLEL
        return self
    
    def add_dma_copy_async(self, source: str, target_var: str, location: str = "local") -> 'DSLBuilder':
        """Add an async DMA copy operation."""
        self._add_line(f"{target_var} = dma.copy.async {source} => {location};")
        return self
    
    def add_wait(self, *variables: str) -> 'DSLBuilder':
        """Add a wait statement for variables."""
        var_list = ", ".join(variables)
        self._add_line(f"wait {var_list};")
        return self
    
    def add_dma_copy(self, source: str, target: str) -> 'DSLBuilder':
        """Add a synchronous DMA copy operation."""
        self._add_line(f"dma.copy {source} => {target};")
        return self
    
    def add_assignment(self, target: str, expression: str) -> 'DSLBuilder':
        """Add an assignment statement."""
        self._add_line(f"{target} = {expression};")
        return self
    
    def add_return(self, expression: str) -> 'DSLBuilder':
        """Add a return statement."""
        self._add_line(f"return {expression};")
        return self
    
    def add_comment(self, comment: str) -> 'DSLBuilder':
        """Add a comment."""
        self._add_line(f"// {comment}")
        return self
    
    def add_empty_line(self) -> 'DSLBuilder':
        """Add an empty line."""
        self._lines.append("")
        return self
    
    def _add_line(self, line: str):
        """Add a line with proper indentation."""
        if line.strip():
            indented_line = "  " * self._indent_level + line
            self._lines.append(indented_line)
        else:
            self._lines.append("")
    
    def build(self) -> str:
        """Build the final DSL code."""
        return "\n".join(self._lines)
    
    def get_variable_by_name(self, name: str) -> Optional[Variable]:
        """Get a variable by name."""
        return self.var_manager.get_variable(name)


class ModernDSLGenerator:
    """
    Modern DSL Generator using the Builder pattern.
    
    This class generates clean, correct Choreo DSL code with proper
    variable scoping and naming consistency.
    """
    
    def __init__(self):
        """Initialize the generator."""
        self.builder = DSLBuilder()
        
    def generate_elementwise_kernel(
        self, 
        function_name: str,
        input_shapes: List[List[int]],
        input_names: List[str],
        operation: str,
        output_shape: List[int]
    ) -> str:
        """
        Generate a clean elementwise kernel.
        
        Args:
            function_name: Name of the kernel function
            input_shapes: Shapes of input tensors
            input_names: Names of input parameters
            operation: Operation to perform (e.g., '+', '*')
            output_shape: Shape of output tensor
            
        Returns:
            Generated DSL code
        """
        self.builder.reset()
        
        # Build function signature
        params = []
        for i, (shape, name) in enumerate(zip(input_shapes, input_names)):
            shape_str = ", ".join(str(d) for d in shape)
            params.append(f"f32 [{shape_str}] {name}")
        
        # Start building
        self.builder.add_header()
        self.builder.begin_function(function_name, params)
        
        # Declare output variable
        output_shape_str = ", ".join(str(d) for d in output_shape)
        output_var = self.builder.declare_variable("output", "f32", output_shape, MemoryLevel.GLOBAL)
        
        # Create parallel execution
        parallel_var = self.builder.var_manager.create_variable("p", "int", VariableScope.PARALLEL)
        self.builder.begin_parallel(parallel_var.name)
        
        # Calculate chunking for the last dimension
        last_dim = output_shape[-1]
        chunk_size = 8  # Standard chunk size
        num_chunks = (last_dim + chunk_size - 1) // chunk_size
        
        # Create outer loop variable
        outer_loop_var = self.builder.var_manager.create_variable("chunk_idx", "int", VariableScope.LOOP)
        self.builder.begin_foreach(outer_loop_var.name, f"[{num_chunks}]")
        
        # Generate DMA loads
        load_vars = []
        for i, name in enumerate(input_names):
            load_var = self.builder.var_manager.create_variable(f"load_{name}", "auto", VariableScope.LOOP)
            load_vars.append(load_var)
            chunkat_expr = f"{name}.chunkat({parallel_var.name}, {outer_loop_var.name})"
            self.builder.add_dma_copy_async(chunkat_expr, load_var.name)
        
        # Wait for loads
        load_var_names = [var.name for var in load_vars]
        self.builder.add_wait(*load_var_names)
        
        self.builder.add_empty_line()
        
        # Declare local computation buffer
        local_shape = output_shape[:-1] + [chunk_size]  # Replace last dim with chunk_size
        local_var = self.builder.declare_variable("local_result", "f32", local_shape, MemoryLevel.L1)
        
        # Generate nested loops for computation
        loop_vars = []
        for i, dim in enumerate(local_shape):
            loop_var = self.builder.var_manager.create_variable(f"i{i}", "int", VariableScope.LOOP)
            loop_vars.append(loop_var)
            self.builder.begin_foreach(loop_var.name, f"[{dim}]")
        
        # Generate the actual computation
        index_expr = ", ".join(var.name for var in loop_vars)
        
        if operation == "+":
            expr = f"{load_vars[0].name}.data.at({index_expr}) + {load_vars[1].name}.data.at({index_expr})"
        elif operation == "*":
            expr = f"{load_vars[0].name}.data.at({index_expr}) * {load_vars[1].name}.data.at({index_expr})"
        else:
            expr = f"{load_vars[0].name}.data.at({index_expr})"
        
        self.builder.add_assignment(f"{local_var.name}.at({index_expr})", expr)
        
        # Close nested loops
        for _ in loop_vars:
            self.builder.end_foreach()
        
        self.builder.add_empty_line()
        
        # Store result back
        store_target = f"{output_var.name}.chunkat({parallel_var.name}, {outer_loop_var.name})"
        self.builder.add_dma_copy(local_var.name, store_target)
        
        # Close outer loop and parallel
        self.builder.end_foreach()
        self.builder.end_parallel()
        
        # Return result
        self.builder.add_return(output_var.name)
        self.builder.end_function()
        
        return self.builder.build()


# Global instance
modern_dsl_generator = ModernDSLGenerator()

"""
Conductor DSL generation.

This module handles the generation of Conductor DSL (.co files) from
the internal DAG representation, including buffer declarations and
operation sequences.
"""

from typing import List
from .graph import ComputationDAG, ConductorNode
from .buffers import Buffer


class DSLGenerator:
    """
    Generates Conductor DSL code from processed graph.
    
    This class converts the internal DAG representation to valid
    Conductor DSL syntax, handling buffer declarations, operation
    sequences, and optimization annotations.
    """
    
    def generate_dsl_file(self, dag: ComputationDAG) -> str:
        """
        Generate complete DSL file for the computation graph.
        
        Args:
            dag: ComputationDAG to convert to DSL
            
        Returns:
            Complete DSL file content as string
        """
        if not dag.nodes:
            return "// Empty computation graph\n"
        
        dsl_parts = []
        
        # Add file header with metadata
        dsl_parts.append("// Generated Conductor DSL")
        dsl_parts.append("// Auto-generated from PyTorch FX Graph")
        dsl_parts.append(f"// Nodes: {len(dag.nodes)}, Buffers: {len(dag.buffers)}")
        dsl_parts.append("")
        
        # Generate main function signature
        input_names = [buf.name for buf in dag.inputs]
        output_names = [buf.name for buf in dag.outputs]
        
        if input_names and output_names:
            function_signature = f"function main({', '.join(input_names)}) -> ({', '.join(output_names)})"
            dsl_parts.append(function_signature + " {")
        else:
            dsl_parts.append("function main() {")
        
        # Add buffer declarations (only for non-input/output buffers)
        internal_buffers = [buf for buf in dag.buffers if buf not in dag.inputs and buf not in dag.outputs]
        if internal_buffers:
            buffer_decls = self.emit_buffer_declarations(internal_buffers)
            if buffer_decls.strip():
                dsl_parts.append("  // Buffer declarations")
                for line in buffer_decls.split('\n'):
                    if line.strip():
                        dsl_parts.append(f"  {line}")
                dsl_parts.append("")
        
        # Add operation sequence in topological order
        ordered_nodes = self._topological_sort_nodes(dag)
        if ordered_nodes:
            dsl_parts.append("  // Operation sequence")
            op_sequence = self.emit_operation_sequence(ordered_nodes)
            for line in op_sequence.split('\n'):
                if line.strip():
                    if line.strip().startswith('//'):
                        # Comments should be indented
                        dsl_parts.append(f"  {line}")
                    else:
                        # Operations should be indented and end with semicolon
                        dsl_parts.append(f"  {line};")
        
        # Close function
        dsl_parts.append("}")
        
        # Join and optimize
        raw_dsl = "\n".join(dsl_parts)
        return self.optimize_temporary_variables(raw_dsl)
        
    def emit_buffer_declarations(self, buffers: List[Buffer]) -> str:
        """
        Generate buffer declarations with appropriate scoping.
        
        Args:
            buffers: List of buffers to declare
            
        Returns:
            DSL code for buffer declarations
        """
        if not buffers:
            return ""
        
        decl_lines = []
        
        # Group buffers by scope for better organization
        from .buffers import BufferScope
        scope_groups = {
            BufferScope.GLOBAL: [],
            BufferScope.SHARED: [],
            BufferScope.LOCAL: []
        }
        
        for buffer in buffers:
            scope_groups[buffer.scope].append(buffer)
        
        # Generate declarations for each scope
        for scope in [BufferScope.GLOBAL, BufferScope.SHARED, BufferScope.LOCAL]:
            scope_buffers = scope_groups[scope]
            if not scope_buffers:
                continue
            
            for buffer in scope_buffers:
                decl = self._generate_buffer_declaration(buffer)
                if decl:
                    decl_lines.append(decl)
        
        return "\n".join(decl_lines)
        
    def emit_operation_sequence(self, nodes: List[ConductorNode]) -> str:
        """
        Generate operation sequence maintaining topological order.
        
        Args:
            nodes: List of nodes in topological order
            
        Returns:
            DSL code for operation sequence
        """
        if not nodes:
            return ""
        
        op_lines = []
        
        for i, node in enumerate(nodes):
            # Add comment for complex operations
            if node.op_name in ['matmul', 'conv2d', 'linear']:
                op_lines.append(f"// {node.op_name} operation")
            
            # Generate the operation DSL
            dsl_code = node.generate_dsl()
            if dsl_code:
                op_lines.append(dsl_code)
            
            # Add spacing between operation groups for readability
            if i < len(nodes) - 1 and self._should_add_spacing(node, nodes[i + 1]):
                op_lines.append("")
        
        return "\n".join(op_lines)
        
    def optimize_temporary_variables(self, dsl_code: str) -> str:
        """
        Optimize temporary variable usage in generated DSL.
        
        Args:
            dsl_code: Raw DSL code to optimize
            
        Returns:
            Optimized DSL code with reduced temporary variables
        """
        # For now, perform basic optimizations
        lines = dsl_code.split('\n')
        optimized_lines = []
        
        # Remove empty lines at the beginning and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        # Remove excessive blank lines (more than 2 consecutive)
        blank_count = 0
        for line in lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 2:  # Allow up to 2 blank lines
                    optimized_lines.append(line)
            else:
                blank_count = 0
                optimized_lines.append(line)
        
        # Remove trailing whitespace from lines
        optimized_lines = [line.rstrip() for line in optimized_lines]
        
        return '\n'.join(optimized_lines)
    
    def _generate_buffer_declaration(self, buffer: Buffer) -> str:
        """
        Generate DSL declaration for a single buffer.
        
        Args:
            buffer: Buffer to generate declaration for
            
        Returns:
            DSL declaration string
        """
        # Map PyTorch dtypes to Conductor DSL types
        dtype_mapping = {
            'torch.float32': 'float32',
            'torch.float16': 'float16',
            'torch.int32': 'int32',
            'torch.int64': 'int64',
            'torch.bool': 'bool',
        }
        
        dtype_str = dtype_mapping.get(str(buffer.dtype), 'float32')
        scope_str = buffer.scope.value
        
        # Generate shape specification
        if buffer.shape is not None:
            shape_str = f"[{', '.join(map(str, buffer.shape))}]"
            return f"{scope_str} {dtype_str} {buffer.name}{shape_str}"
        else:
            return f"{scope_str} {dtype_str} {buffer.name}"
    
    def _topological_sort_nodes(self, dag: ComputationDAG) -> List[ConductorNode]:
        """
        Sort nodes in topological order for correct execution sequence.
        
        Args:
            dag: Computation DAG containing nodes to sort
            
        Returns:
            List of nodes in topological order
        """
        if not dag.nodes:
            return []
        
        # Build dependency graph
        in_degree = {node: 0 for node in dag.nodes}
        dependencies = {node: [] for node in dag.nodes}
        
        # Calculate dependencies based on buffer producer/consumer relationships
        for node in dag.nodes:
            for input_buffer in node.inputs:
                if input_buffer.producer and input_buffer.producer in dag.nodes:
                    producer = input_buffer.producer
                    dependencies[producer].append(node)
                    in_degree[node] += 1
        
        # Kahn's algorithm for topological sorting
        queue = [node for node in dag.nodes if in_degree[node] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for dependent in dependencies[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # If not all nodes are processed, there's a cycle - use original order
        if len(result) != len(dag.nodes):
            return dag.nodes
        
        return result
    
    def _should_add_spacing(self, current_node: ConductorNode, next_node: ConductorNode) -> bool:
        """
        Determine if spacing should be added between two operations.
        
        Args:
            current_node: Current operation node
            next_node: Next operation node
            
        Returns:
            True if spacing should be added, False otherwise
        """
        # Add spacing between different operation types
        elementwise_ops = {'add', 'sub', 'mul', 'div', 'relu', 'sigmoid', 'tanh', 'abs', 'neg'}
        reduction_ops = {'sum', 'mean', 'max', 'min', 'argmax', 'argmin'}
        matrix_ops = {'matmul', 'conv2d', 'linear'}
        
        def get_op_category(op_name):
            if op_name in elementwise_ops:
                return 'elementwise'
            elif op_name in reduction_ops:
                return 'reduction'
            elif op_name in matrix_ops:
                return 'matrix'
            else:
                return 'other'
        
        current_category = get_op_category(current_node.op_name)
        next_category = get_op_category(next_node.op_name)
        
        # Add spacing when switching between different categories
        return current_category != next_category
"""
Unit tests for DSL generation.

Tests the DSLGenerator class functionality including buffer declarations,
operation sequence generation, and complete DSL file generation.
"""

import pytest
import torch
from conductor.codegen.dsl import DSLGenerator
from conductor.codegen.graph import ConductorNode, ComputationDAG
from conductor.codegen.buffers import Buffer, BufferScope


class TestDSLGenerator:
    """Test DSLGenerator class functionality."""
    
    def test_generator_creation(self):
        """Test DSLGenerator initialization."""
        generator = DSLGenerator()
        assert generator is not None
    
    def test_generate_dsl_file_empty_dag(self):
        """Test DSL generation for empty DAG."""
        generator = DSLGenerator()
        dag = ComputationDAG()
        
        dsl = generator.generate_dsl_file(dag)
        assert "Empty computation graph" in dsl
    
    def test_generate_dsl_file_single_operation(self):
        """Test DSL generation for single operation."""
        generator = DSLGenerator()
        
        # Create simple DAG with one ReLU operation
        input_buf = Buffer("input", BufferScope.GLOBAL, torch.float32, (10, 10))
        output_buf = Buffer("output", BufferScope.GLOBAL, torch.float32, (10, 10))
        
        relu_node = ConductorNode("relu", inputs=[input_buf], outputs=[output_buf])
        
        dag = ComputationDAG()
        dag.add_node(relu_node)
        dag.add_buffer(input_buf)
        dag.add_buffer(output_buf)
        dag.inputs = [input_buf]
        dag.outputs = [output_buf]
        
        dsl = generator.generate_dsl_file(dag)
        
        # Check function signature
        assert "function main(input) -> (output)" in dsl
        
        # Check operation
        assert "output = relu(input)" in dsl
        
        # Check structure
        assert dsl.startswith("// Generated Conductor DSL")
        assert dsl.endswith("}")
    
    def test_generate_dsl_file_multiple_operations(self):
        """Test DSL generation for multiple operations."""
        generator = DSLGenerator()
        
        # Create DAG with add -> relu chain
        input_buf = Buffer("input", BufferScope.GLOBAL, torch.float32, (10, 10))
        intermediate_buf = Buffer("temp", BufferScope.LOCAL, torch.float32, (10, 10))
        output_buf = Buffer("output", BufferScope.GLOBAL, torch.float32, (10, 10))
        
        add_node = ConductorNode("add", inputs=[input_buf], outputs=[intermediate_buf])
        relu_node = ConductorNode("relu", inputs=[intermediate_buf], outputs=[output_buf])
        
        # Set up buffer relationships
        intermediate_buf.producer = add_node
        intermediate_buf.consumers = [relu_node]
        
        dag = ComputationDAG()
        dag.add_node(add_node)
        dag.add_node(relu_node)
        dag.add_buffer(input_buf)
        dag.add_buffer(intermediate_buf)
        dag.add_buffer(output_buf)
        dag.inputs = [input_buf]
        dag.outputs = [output_buf]
        
        dsl = generator.generate_dsl_file(dag)
        
        # Check function signature
        assert "function main(input) -> (output)" in dsl
        
        # Check buffer declaration for intermediate buffer
        assert "local float32 temp[10, 10]" in dsl
        
        # Check operations in correct order
        assert "temp = add(input)" in dsl
        assert "output = relu(temp)" in dsl
        
        # Verify topological order (add should come before relu)
        add_pos = dsl.find("temp = add(input)")
        relu_pos = dsl.find("output = relu(temp)")
        assert add_pos < relu_pos
    
    def test_emit_buffer_declarations_empty(self):
        """Test buffer declaration generation with empty list."""
        generator = DSLGenerator()
        
        decl = generator.emit_buffer_declarations([])
        assert decl == ""
    
    def test_emit_buffer_declarations_single_buffer(self):
        """Test buffer declaration generation for single buffer."""
        generator = DSLGenerator()
        
        buffer = Buffer("test_buf", BufferScope.LOCAL, torch.float32, (5, 5))
        decl = generator.emit_buffer_declarations([buffer])
        
        assert "local float32 test_buf[5, 5]" in decl
    
    def test_emit_buffer_declarations_multiple_buffers(self):
        """Test buffer declaration generation for multiple buffers."""
        generator = DSLGenerator()
        
        buffers = [
            Buffer("global_buf", BufferScope.GLOBAL, torch.float32, (10, 10)),
            Buffer("shared_buf", BufferScope.SHARED, torch.int32, (5,)),
            Buffer("local_buf", BufferScope.LOCAL, torch.float16, None)
        ]
        
        decl = generator.emit_buffer_declarations(buffers)
        
        # Check all buffers are declared
        assert "global float32 global_buf[10, 10]" in decl
        assert "shared int32 shared_buf[5]" in decl
        assert "local float16 local_buf" in decl
    
    def test_emit_buffer_declarations_different_dtypes(self):
        """Test buffer declaration generation for different data types."""
        generator = DSLGenerator()
        
        buffers = [
            Buffer("float32_buf", BufferScope.LOCAL, torch.float32),
            Buffer("float16_buf", BufferScope.LOCAL, torch.float16),
            Buffer("int32_buf", BufferScope.LOCAL, torch.int32),
            Buffer("int64_buf", BufferScope.LOCAL, torch.int64),
            Buffer("bool_buf", BufferScope.LOCAL, torch.bool)
        ]
        
        decl = generator.emit_buffer_declarations(buffers)
        
        assert "local float32 float32_buf" in decl
        assert "local float16 float16_buf" in decl
        assert "local int32 int32_buf" in decl
        assert "local int64 int64_buf" in decl
        assert "local bool bool_buf" in decl
    
    def test_emit_operation_sequence_empty(self):
        """Test operation sequence generation with empty list."""
        generator = DSLGenerator()
        
        sequence = generator.emit_operation_sequence([])
        assert sequence == ""
    
    def test_emit_operation_sequence_single_operation(self):
        """Test operation sequence generation for single operation."""
        generator = DSLGenerator()
        
        input_buf = Buffer("input", BufferScope.LOCAL, torch.float32)
        output_buf = Buffer("output", BufferScope.LOCAL, torch.float32)
        node = ConductorNode("relu", inputs=[input_buf], outputs=[output_buf])
        
        sequence = generator.emit_operation_sequence([node])
        assert "output = relu(input)" in sequence
    
    def test_emit_operation_sequence_multiple_operations(self):
        """Test operation sequence generation for multiple operations."""
        generator = DSLGenerator()
        
        # Create different types of operations
        buf1 = Buffer("buf1", BufferScope.LOCAL, torch.float32)
        buf2 = Buffer("buf2", BufferScope.LOCAL, torch.float32)
        buf3 = Buffer("buf3", BufferScope.LOCAL, torch.float32)
        buf4 = Buffer("buf4", BufferScope.LOCAL, torch.float32)
        
        add_node = ConductorNode("add", inputs=[buf1], outputs=[buf2])
        matmul_node = ConductorNode("matmul", inputs=[buf2], outputs=[buf3])
        sum_node = ConductorNode("sum", inputs=[buf3], outputs=[buf4])
        
        sequence = generator.emit_operation_sequence([add_node, matmul_node, sum_node])
        
        # Check all operations are present
        assert "buf2 = add(buf1)" in sequence
        assert "buf3 = matmul(buf2)" in sequence
        assert "buf4 = sum(buf3)" in sequence
        
        # Check comments for complex operations
        assert "// matmul operation" in sequence
    
    def test_optimize_temporary_variables_basic(self):
        """Test basic temporary variable optimization."""
        generator = DSLGenerator()
        
        # Test with excessive blank lines
        input_code = """
        
        
        function main() {
        
        
          output = relu(input);
        
        
        }
        
        
        """
        
        optimized = generator.optimize_temporary_variables(input_code)
        
        # Should remove excessive blank lines
        lines = optimized.split('\n')
        blank_count = 0
        max_consecutive_blanks = 0
        
        for line in lines:
            if not line.strip():
                blank_count += 1
                max_consecutive_blanks = max(max_consecutive_blanks, blank_count)
            else:
                blank_count = 0
        
        # Should not have more than 2 consecutive blank lines
        assert max_consecutive_blanks <= 2
    
    def test_optimize_temporary_variables_whitespace(self):
        """Test whitespace optimization."""
        generator = DSLGenerator()
        
        input_code = "function main() {   \n  output = relu(input);  \n}   "
        optimized = generator.optimize_temporary_variables(input_code)
        
        # Should remove trailing whitespace
        for line in optimized.split('\n'):
            assert line == line.rstrip()
    
    def test_generate_buffer_declaration_with_shape(self):
        """Test buffer declaration generation with shape."""
        generator = DSLGenerator()
        
        buffer = Buffer("test", BufferScope.SHARED, torch.float32, (3, 4, 5))
        decl = generator._generate_buffer_declaration(buffer)
        
        assert decl == "shared float32 test[3, 4, 5]"
    
    def test_generate_buffer_declaration_without_shape(self):
        """Test buffer declaration generation without shape."""
        generator = DSLGenerator()
        
        buffer = Buffer("test", BufferScope.LOCAL, torch.int32, None)
        decl = generator._generate_buffer_declaration(buffer)
        
        assert decl == "local int32 test"
    
    def test_topological_sort_nodes_simple_chain(self):
        """Test topological sorting for simple chain."""
        generator = DSLGenerator()
        
        # Create chain: node1 -> node2 -> node3
        buf1 = Buffer("buf1", BufferScope.LOCAL, torch.float32)
        buf2 = Buffer("buf2", BufferScope.LOCAL, torch.float32)
        buf3 = Buffer("buf3", BufferScope.LOCAL, torch.float32)
        buf4 = Buffer("buf4", BufferScope.LOCAL, torch.float32)
        
        node1 = ConductorNode("add", inputs=[buf1], outputs=[buf2])
        node2 = ConductorNode("mul", inputs=[buf2], outputs=[buf3])
        node3 = ConductorNode("relu", inputs=[buf3], outputs=[buf4])
        
        # Set up buffer relationships
        buf2.producer = node1
        buf2.consumers = [node2]
        buf3.producer = node2
        buf3.consumers = [node3]
        
        dag = ComputationDAG()
        dag.nodes = [node3, node1, node2]  # Intentionally out of order
        
        sorted_nodes = generator._topological_sort_nodes(dag)
        
        # Should be in correct topological order
        assert sorted_nodes.index(node1) < sorted_nodes.index(node2)
        assert sorted_nodes.index(node2) < sorted_nodes.index(node3)
    
    def test_topological_sort_nodes_complex_dependencies(self):
        """Test topological sorting for complex dependencies."""
        generator = DSLGenerator()
        
        # Create diamond dependency pattern
        buf1 = Buffer("buf1", BufferScope.LOCAL, torch.float32)
        buf2 = Buffer("buf2", BufferScope.LOCAL, torch.float32)
        buf3 = Buffer("buf3", BufferScope.LOCAL, torch.float32)
        buf4 = Buffer("buf4", BufferScope.LOCAL, torch.float32)
        buf5 = Buffer("buf5", BufferScope.LOCAL, torch.float32)
        
        node1 = ConductorNode("split", inputs=[buf1], outputs=[buf2, buf3])
        node2 = ConductorNode("add", inputs=[buf2], outputs=[buf4])
        node3 = ConductorNode("mul", inputs=[buf3], outputs=[buf5])
        node4 = ConductorNode("concat", inputs=[buf4, buf5], outputs=[])
        
        # Set up buffer relationships
        buf2.producer = node1
        buf2.consumers = [node2]
        buf3.producer = node1
        buf3.consumers = [node3]
        buf4.producer = node2
        buf4.consumers = [node4]
        buf5.producer = node3
        buf5.consumers = [node4]
        
        dag = ComputationDAG()
        dag.nodes = [node4, node3, node2, node1]  # Reverse order
        
        sorted_nodes = generator._topological_sort_nodes(dag)
        
        # Verify dependencies are respected
        assert sorted_nodes.index(node1) < sorted_nodes.index(node2)
        assert sorted_nodes.index(node1) < sorted_nodes.index(node3)
        assert sorted_nodes.index(node2) < sorted_nodes.index(node4)
        assert sorted_nodes.index(node3) < sorted_nodes.index(node4)
    
    def test_should_add_spacing_same_category(self):
        """Test spacing decision for same operation category."""
        generator = DSLGenerator()
        
        node1 = ConductorNode("add")
        node2 = ConductorNode("mul")  # Both elementwise
        
        assert generator._should_add_spacing(node1, node2) is False
    
    def test_should_add_spacing_different_category(self):
        """Test spacing decision for different operation categories."""
        generator = DSLGenerator()
        
        elementwise_node = ConductorNode("add")
        matrix_node = ConductorNode("matmul")
        reduction_node = ConductorNode("sum")
        
        # Different categories should have spacing
        assert generator._should_add_spacing(elementwise_node, matrix_node) is True
        assert generator._should_add_spacing(matrix_node, reduction_node) is True
        assert generator._should_add_spacing(elementwise_node, reduction_node) is True


class TestDSLIntegration:
    """Integration tests for DSL generation."""
    
    def test_complete_dsl_generation_workflow(self):
        """Test complete DSL generation workflow."""
        generator = DSLGenerator()
        
        # Create a realistic computation graph: input -> add -> relu -> sum -> output
        input_buf = Buffer("input", BufferScope.GLOBAL, torch.float32, (10, 10))
        temp1_buf = Buffer("temp1", BufferScope.LOCAL, torch.float32, (10, 10))
        temp2_buf = Buffer("temp2", BufferScope.LOCAL, torch.float32, (10, 10))
        output_buf = Buffer("output", BufferScope.GLOBAL, torch.float32, (10,))
        
        add_node = ConductorNode("add", inputs=[input_buf], outputs=[temp1_buf])
        relu_node = ConductorNode("relu", inputs=[temp1_buf], outputs=[temp2_buf])
        sum_node = ConductorNode("sum", inputs=[temp2_buf], outputs=[output_buf])
        
        # Set up buffer relationships
        temp1_buf.producer = add_node
        temp1_buf.consumers = [relu_node]
        temp2_buf.producer = relu_node
        temp2_buf.consumers = [sum_node]
        
        dag = ComputationDAG()
        dag.add_node(add_node)
        dag.add_node(relu_node)
        dag.add_node(sum_node)
        dag.add_buffer(input_buf)
        dag.add_buffer(temp1_buf)
        dag.add_buffer(temp2_buf)
        dag.add_buffer(output_buf)
        dag.inputs = [input_buf]
        dag.outputs = [output_buf]
        
        dsl = generator.generate_dsl_file(dag)
        
        # Verify complete DSL structure
        assert "// Generated Conductor DSL" in dsl
        assert "function main(input) -> (output)" in dsl
        
        # Verify buffer declarations
        assert "local float32 temp1[10, 10]" in dsl
        assert "local float32 temp2[10, 10]" in dsl
        
        # Verify operations in correct order
        lines = dsl.split('\n')
        add_line = next(i for i, line in enumerate(lines) if "temp1 = add(input)" in line)
        relu_line = next(i for i, line in enumerate(lines) if "temp2 = relu(temp1)" in line)
        sum_line = next(i for i, line in enumerate(lines) if "output = sum(temp2)" in line)
        
        assert add_line < relu_line < sum_line
        
        # Verify function closure
        assert dsl.rstrip().endswith("}")
    
    def test_dsl_generation_with_multiple_inputs_outputs(self):
        """Test DSL generation with multiple inputs and outputs."""
        generator = DSLGenerator()
        
        # Create graph with multiple inputs and outputs
        input1 = Buffer("x", BufferScope.GLOBAL, torch.float32, (10, 10))
        input2 = Buffer("y", BufferScope.GLOBAL, torch.float32, (10, 10))
        output1 = Buffer("sum_result", BufferScope.GLOBAL, torch.float32, (10, 10))
        output2 = Buffer("mul_result", BufferScope.GLOBAL, torch.float32, (10, 10))
        
        add_node = ConductorNode("add", inputs=[input1, input2], outputs=[output1])
        mul_node = ConductorNode("mul", inputs=[input1, input2], outputs=[output2])
        
        dag = ComputationDAG()
        dag.add_node(add_node)
        dag.add_node(mul_node)
        dag.add_buffer(input1)
        dag.add_buffer(input2)
        dag.add_buffer(output1)
        dag.add_buffer(output2)
        dag.inputs = [input1, input2]
        dag.outputs = [output1, output2]
        
        dsl = generator.generate_dsl_file(dag)
        
        # Check function signature with multiple inputs/outputs
        assert "function main(x, y) -> (sum_result, mul_result)" in dsl
        
        # Check operations
        assert "sum_result = add(x, y)" in dsl
        assert "mul_result = mul(x, y)" in dsl
    
    def test_dsl_generation_with_fusion_clusters(self):
        """Test DSL generation that would work with fusion clusters."""
        generator = DSLGenerator()
        
        # Create a pattern that could be fused: add -> mul -> relu
        input_buf = Buffer("input", BufferScope.GLOBAL, torch.float32, (10, 10))
        temp1_buf = Buffer("temp1", BufferScope.LOCAL, torch.float32, (10, 10))
        temp2_buf = Buffer("temp2", BufferScope.LOCAL, torch.float32, (10, 10))
        output_buf = Buffer("output", BufferScope.GLOBAL, torch.float32, (10, 10))
        
        add_node = ConductorNode("add", inputs=[input_buf], outputs=[temp1_buf])
        mul_node = ConductorNode("mul", inputs=[temp1_buf], outputs=[temp2_buf])
        relu_node = ConductorNode("relu", inputs=[temp2_buf], outputs=[output_buf])
        
        # Set up buffer relationships
        temp1_buf.producer = add_node
        temp1_buf.consumers = [mul_node]
        temp2_buf.producer = mul_node
        temp2_buf.consumers = [relu_node]
        
        dag = ComputationDAG()
        dag.add_node(add_node)
        dag.add_node(mul_node)
        dag.add_node(relu_node)
        dag.add_buffer(input_buf)
        dag.add_buffer(temp1_buf)
        dag.add_buffer(temp2_buf)
        dag.add_buffer(output_buf)
        dag.inputs = [input_buf]
        dag.outputs = [output_buf]
        
        dsl = generator.generate_dsl_file(dag)
        
        # Verify that all elementwise operations are present and properly ordered
        assert "temp1 = add(input)" in dsl
        assert "temp2 = mul(temp1)" in dsl
        assert "output = relu(temp2)" in dsl
        
        # Verify no spacing between elementwise operations (same category)
        lines = [line.strip() for line in dsl.split('\n') if line.strip()]
        
        # Find operation lines
        op_lines = [line for line in lines if '=' in line and not line.startswith('//')]
        
        # Should have all three operations
        assert len(op_lines) == 3
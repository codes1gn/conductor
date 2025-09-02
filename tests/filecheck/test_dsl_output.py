"""
FileCheck-style tests for DSL output validation.

These tests validate the structure and correctness of generated DSL code
using pattern matching similar to LLVM FileCheck.
"""

import pytest
import torch
import re
from conductor.codegen.dsl import DSLGenerator
from conductor.codegen.graph import ConductorNode, ComputationDAG
from conductor.codegen.buffers import Buffer, BufferScope


class TestDSLStructureValidation:
    """Test DSL output structure using FileCheck-style validation."""
    
    def test_dsl_function_structure(self):
        """Test that generated DSL has correct function structure."""
        generator = DSLGenerator()
        
        # Create simple computation
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
        
        # FileCheck: Verify function structure
        # CHECK: // Generated Conductor DSL
        assert re.search(r'// Generated Conductor DSL', dsl)
        
        # CHECK: function main({{.*}}) -> ({{.*}}) {
        assert re.search(r'function main\([^)]*\) -> \([^)]*\) \{', dsl)
        
        # CHECK: {{.*}} = relu({{.*}});
        assert re.search(r'\w+ = relu\([^)]+\);', dsl)
        
        # CHECK: }
        assert dsl.rstrip().endswith('}')
    
    def test_dsl_buffer_declaration_format(self):
        """Test buffer declaration format validation."""
        generator = DSLGenerator()
        
        # Create computation with internal buffer
        input_buf = Buffer("input", BufferScope.GLOBAL, torch.float32, (10, 10))
        temp_buf = Buffer("temp", BufferScope.LOCAL, torch.float32, (10, 10))
        output_buf = Buffer("output", BufferScope.GLOBAL, torch.float32, (10, 10))
        
        add_node = ConductorNode("add", inputs=[input_buf], outputs=[temp_buf])
        relu_node = ConductorNode("relu", inputs=[temp_buf], outputs=[output_buf])
        
        # Set up relationships
        temp_buf.producer = add_node
        temp_buf.consumers = [relu_node]
        
        dag = ComputationDAG()
        dag.add_node(add_node)
        dag.add_node(relu_node)
        dag.add_buffer(input_buf)
        dag.add_buffer(temp_buf)
        dag.add_buffer(output_buf)
        dag.inputs = [input_buf]
        dag.outputs = [output_buf]
        
        dsl = generator.generate_dsl_file(dag)
        
        # FileCheck: Verify buffer declaration format
        # CHECK: local float32 {{name}}[{{dimensions}}]
        assert re.search(r'local float32 \w+\[\d+, \d+\]', dsl)
        
        # CHECK-NOT: global float32 temp
        # (temp should be local, not global)
        assert not re.search(r'global float32 temp', dsl)
    
    def test_dsl_operation_sequence_order(self):
        """Test operation sequence ordering validation."""
        generator = DSLGenerator()
        
        # Create chain: add -> mul -> relu
        input_buf = Buffer("input", BufferScope.GLOBAL, torch.float32, (10, 10))
        temp1_buf = Buffer("temp1", BufferScope.LOCAL, torch.float32, (10, 10))
        temp2_buf = Buffer("temp2", BufferScope.LOCAL, torch.float32, (10, 10))
        output_buf = Buffer("output", BufferScope.GLOBAL, torch.float32, (10, 10))
        
        add_node = ConductorNode("add", inputs=[input_buf], outputs=[temp1_buf])
        mul_node = ConductorNode("mul", inputs=[temp1_buf], outputs=[temp2_buf])
        relu_node = ConductorNode("relu", inputs=[temp2_buf], outputs=[output_buf])
        
        # Set up relationships
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
        
        # FileCheck: Verify operation ordering
        # Find positions of operations
        add_match = re.search(r'temp1 = add\(input\);', dsl)
        mul_match = re.search(r'temp2 = mul\(temp1\);', dsl)
        relu_match = re.search(r'output = relu\(temp2\);', dsl)
        
        assert add_match is not None
        assert mul_match is not None
        assert relu_match is not None
        
        # CHECK: add operation comes before mul
        assert add_match.start() < mul_match.start()
        
        # CHECK: mul operation comes before relu
        assert mul_match.start() < relu_match.start()
    
    def test_dsl_fusion_pattern_validation(self):
        """Test validation of fusion-friendly patterns."""
        generator = DSLGenerator()
        
        # Create elementwise chain that could be fused
        input_buf = Buffer("input", BufferScope.GLOBAL, torch.float32, (10, 10))
        temp1_buf = Buffer("temp1", BufferScope.LOCAL, torch.float32, (10, 10))
        temp2_buf = Buffer("temp2", BufferScope.LOCAL, torch.float32, (10, 10))
        output_buf = Buffer("output", BufferScope.GLOBAL, torch.float32, (10, 10))
        
        add_node = ConductorNode("add", inputs=[input_buf], outputs=[temp1_buf])
        mul_node = ConductorNode("mul", inputs=[temp1_buf], outputs=[temp2_buf])
        relu_node = ConductorNode("relu", inputs=[temp2_buf], outputs=[output_buf])
        
        # Set up relationships
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
        
        # FileCheck: Verify fusion pattern structure
        # CHECK: Consecutive elementwise operations
        lines = [line.strip() for line in dsl.split('\n') if '=' in line and ';' in line]
        
        # Should have exactly 3 operations
        assert len(lines) == 3
        
        # CHECK: All operations use consistent naming pattern
        assert re.search(r'temp1 = add\(input\);', dsl)
        assert re.search(r'temp2 = mul\(temp1\);', dsl)
        assert re.search(r'output = relu\(temp2\);', dsl)
        
        # CHECK: No gaps in the chain (each temp is used exactly once as input/output)
        # Count only in operation lines, not declarations
        operation_lines = [line for line in dsl.split('\n') if '=' in line and ';' in line]
        operation_text = '\n'.join(operation_lines)
        
        temp1_uses = len(re.findall(r'\btemp1\b', operation_text))
        temp2_uses = len(re.findall(r'\btemp2\b', operation_text))
        
        # temp1 should appear twice: once as output, once as input
        assert temp1_uses == 2
        # temp2 should appear twice: once as output, once as input
        assert temp2_uses == 2
    
    def test_dsl_reduction_pattern_validation(self):
        """Test validation of elementwise + reduction patterns."""
        generator = DSLGenerator()
        
        # Create elementwise -> reduction pattern
        input_buf = Buffer("input", BufferScope.GLOBAL, torch.float32, (10, 10))
        temp_buf = Buffer("temp", BufferScope.LOCAL, torch.float32, (10, 10))
        output_buf = Buffer("output", BufferScope.GLOBAL, torch.float32, (10,))
        
        mul_node = ConductorNode("mul", inputs=[input_buf], outputs=[temp_buf])
        sum_node = ConductorNode("sum", inputs=[temp_buf], outputs=[output_buf])
        
        # Set up relationships
        temp_buf.producer = mul_node
        temp_buf.consumers = [sum_node]
        
        dag = ComputationDAG()
        dag.add_node(mul_node)
        dag.add_node(sum_node)
        dag.add_buffer(input_buf)
        dag.add_buffer(temp_buf)
        dag.add_buffer(output_buf)
        dag.inputs = [input_buf]
        dag.outputs = [output_buf]
        
        dsl = generator.generate_dsl_file(dag)
        
        # FileCheck: Verify reduction pattern
        # CHECK: elementwise operation followed by reduction
        mul_match = re.search(r'temp = mul\(input\);', dsl)
        sum_match = re.search(r'output = sum\(temp\);', dsl)
        
        assert mul_match is not None
        assert sum_match is not None
        
        # CHECK: elementwise comes before reduction
        assert mul_match.start() < sum_match.start()
        
        # CHECK: intermediate buffer is properly scoped
        assert re.search(r'local float32 temp\[10, 10\]', dsl)
    
    def test_dsl_multiple_inputs_outputs_validation(self):
        """Test validation of multiple inputs/outputs format."""
        generator = DSLGenerator()
        
        # Create computation with multiple inputs and outputs
        input1 = Buffer("x", BufferScope.GLOBAL, torch.float32, (10, 10))
        input2 = Buffer("y", BufferScope.GLOBAL, torch.float32, (10, 10))
        output1 = Buffer("sum_out", BufferScope.GLOBAL, torch.float32, (10, 10))
        output2 = Buffer("mul_out", BufferScope.GLOBAL, torch.float32, (10, 10))
        
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
        
        # FileCheck: Verify multiple inputs/outputs format
        # CHECK: function main({{input1}}, {{input2}}) -> ({{output1}}, {{output2}})
        assert re.search(r'function main\(x, y\) -> \(sum_out, mul_out\)', dsl)
        
        # CHECK: Both operations use both inputs
        assert re.search(r'sum_out = add\(x, y\);', dsl)
        assert re.search(r'mul_out = mul\(x, y\);', dsl)
    
    def test_dsl_comments_and_formatting_validation(self):
        """Test validation of comments and formatting."""
        generator = DSLGenerator()
        
        # Create computation with different operation types
        input_buf = Buffer("input", BufferScope.GLOBAL, torch.float32, (10, 10))
        temp_buf = Buffer("temp", BufferScope.LOCAL, torch.float32, (10, 10))
        output_buf = Buffer("output", BufferScope.GLOBAL, torch.float32, (10, 10))
        
        add_node = ConductorNode("add", inputs=[input_buf], outputs=[temp_buf])
        matmul_node = ConductorNode("matmul", inputs=[temp_buf], outputs=[output_buf])
        
        # Set up relationships
        temp_buf.producer = add_node
        temp_buf.consumers = [matmul_node]
        
        dag = ComputationDAG()
        dag.add_node(add_node)
        dag.add_node(matmul_node)
        dag.add_buffer(input_buf)
        dag.add_buffer(temp_buf)
        dag.add_buffer(output_buf)
        dag.inputs = [input_buf]
        dag.outputs = [output_buf]
        
        dsl = generator.generate_dsl_file(dag)
        
        # FileCheck: Verify comments and formatting
        # CHECK: Header comments present
        assert re.search(r'// Generated Conductor DSL', dsl)
        assert re.search(r'// Auto-generated from PyTorch FX Graph', dsl)
        
        # CHECK: Complex operation has comment
        assert re.search(r'// matmul operation', dsl)
        
        # CHECK: Proper indentation (operations should be indented)
        lines = dsl.split('\n')
        operation_lines = [line for line in lines if '=' in line and ';' in line]
        
        for line in operation_lines:
            # Operations should be indented with 2 spaces
            assert line.startswith('  ')
        
        # CHECK: Function braces are properly formatted
        assert re.search(r'function main.*\) \{$', dsl, re.MULTILINE)
        assert dsl.rstrip().endswith('}')


class TestDSLCorrectnessValidation:
    """Test DSL correctness using FileCheck-style validation."""
    
    def test_dsl_variable_consistency(self):
        """Test that variable names are used consistently."""
        generator = DSLGenerator()
        
        # Create computation with specific buffer names
        input_buf = Buffer("input_tensor", BufferScope.GLOBAL, torch.float32, (10, 10))
        intermediate_buf = Buffer("intermediate_result", BufferScope.LOCAL, torch.float32, (10, 10))
        output_buf = Buffer("final_output", BufferScope.GLOBAL, torch.float32, (10, 10))
        
        add_node = ConductorNode("add", inputs=[input_buf], outputs=[intermediate_buf])
        relu_node = ConductorNode("relu", inputs=[intermediate_buf], outputs=[output_buf])
        
        # Set up relationships
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
        
        # FileCheck: Verify variable name consistency
        # CHECK: input_tensor appears in function signature
        assert re.search(r'function main\(input_tensor\)', dsl)
        
        # CHECK: intermediate_result is declared and used consistently
        assert re.search(r'local float32 intermediate_result\[10, 10\]', dsl)
        assert re.search(r'intermediate_result = add\(input_tensor\)', dsl)
        assert re.search(r'final_output = relu\(intermediate_result\)', dsl)
        
        # CHECK: final_output appears in function signature
        assert re.search(r'-> \(final_output\)', dsl)
    
    def test_dsl_type_consistency(self):
        """Test that data types are handled consistently."""
        generator = DSLGenerator()
        
        # Create buffers with different types
        float32_buf = Buffer("f32_buf", BufferScope.LOCAL, torch.float32, (5, 5))
        float16_buf = Buffer("f16_buf", BufferScope.LOCAL, torch.float16, (5, 5))
        int32_buf = Buffer("i32_buf", BufferScope.LOCAL, torch.int32, (5, 5))
        
        buffers = [float32_buf, float16_buf, int32_buf]
        decl = generator.emit_buffer_declarations(buffers)
        
        # FileCheck: Verify type declarations
        # CHECK: float32 type
        assert re.search(r'local float32 f32_buf\[5, 5\]', decl)
        
        # CHECK: float16 type
        assert re.search(r'local float16 f16_buf\[5, 5\]', decl)
        
        # CHECK: int32 type
        assert re.search(r'local int32 i32_buf\[5, 5\]', decl)
    
    def test_dsl_scope_consistency(self):
        """Test that buffer scopes are handled consistently."""
        generator = DSLGenerator()
        
        # Create buffers with different scopes
        global_buf = Buffer("global_data", BufferScope.GLOBAL, torch.float32, (10, 10))
        shared_buf = Buffer("shared_data", BufferScope.SHARED, torch.float32, (10, 10))
        local_buf = Buffer("local_data", BufferScope.LOCAL, torch.float32, (10, 10))
        
        buffers = [global_buf, shared_buf, local_buf]
        decl = generator.emit_buffer_declarations(buffers)
        
        # FileCheck: Verify scope declarations
        # CHECK: global scope
        assert re.search(r'global float32 global_data\[10, 10\]', decl)
        
        # CHECK: shared scope
        assert re.search(r'shared float32 shared_data\[10, 10\]', decl)
        
        # CHECK: local scope
        assert re.search(r'local float32 local_data\[10, 10\]', decl)
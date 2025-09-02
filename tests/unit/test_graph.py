"""
Unit tests for graph representation and analysis.

Tests the ConductorNode class, ComputationDAG, and GraphAnalyzer functionality
including fusion compatibility, DSL generation, and cost estimation.
"""

import pytest
import torch
from conductor.codegen.graph import ConductorNode, ComputationDAG, GraphAnalyzer
from conductor.codegen.buffers import Buffer, BufferScope


class TestConductorNode:
    """Test ConductorNode class functionality."""
    
    def test_node_creation_basic(self):
        """Test basic node creation with minimal parameters."""
        node = ConductorNode(op_name="add")
        
        assert node.op_name == "add"
        assert node.inputs == []
        assert node.outputs == []
        assert node.metadata == {}
        assert node.fusion_group is None
    
    def test_node_creation_with_buffers(self):
        """Test node creation with input and output buffers."""
        input_buf = Buffer("input", BufferScope.LOCAL, torch.float32, (10, 10))
        output_buf = Buffer("output", BufferScope.LOCAL, torch.float32, (10, 10))
        
        node = ConductorNode(
            op_name="relu",
            inputs=[input_buf],
            outputs=[output_buf]
        )
        
        assert node.op_name == "relu"
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        assert node.inputs[0] is input_buf
        assert node.outputs[0] is output_buf
    
    def test_node_creation_with_metadata(self):
        """Test node creation with operation metadata."""
        node = ConductorNode(
            op_name="sum",
            metadata={"dim": 1, "keepdim": True}
        )
        
        assert node.metadata["dim"] == 1
        assert node.metadata["keepdim"] is True
    
    def test_can_fuse_elementwise_operations(self):
        """Test fusion compatibility between elementwise operations."""
        # Create compatible elementwise nodes
        buf1 = Buffer("buf1", BufferScope.LOCAL, torch.float32, (10, 10))
        buf2 = Buffer("buf2", BufferScope.LOCAL, torch.float32, (10, 10))
        buf3 = Buffer("buf3", BufferScope.LOCAL, torch.float32, (10, 10))
        
        node1 = ConductorNode("add", outputs=[buf1])
        node2 = ConductorNode("relu", outputs=[buf2])
        
        # Should be fusible (both elementwise with compatible shapes)
        assert node1.can_fuse_with(node2)
        assert node2.can_fuse_with(node1)
    
    def test_can_fuse_elementwise_with_reduction(self):
        """Test fusion compatibility between elementwise and reduction operations."""
        # Create elementwise node that feeds into reduction
        intermediate_buf = Buffer("intermediate", BufferScope.LOCAL, torch.float32, (10, 10))
        output_buf = Buffer("output", BufferScope.LOCAL, torch.float32, (10,))
        
        elementwise_node = ConductorNode("mul", outputs=[intermediate_buf])
        reduction_node = ConductorNode("sum", inputs=[intermediate_buf], outputs=[output_buf])
        
        # Elementwise should be fusible with reduction
        assert elementwise_node.can_fuse_with(reduction_node)
    
    def test_cannot_fuse_incompatible_operations(self):
        """Test that incompatible operations cannot be fused."""
        node1 = ConductorNode("matmul")  # Not elementwise
        node2 = ConductorNode("conv2d")  # Not elementwise
        
        assert not node1.can_fuse_with(node2)
        assert not node2.can_fuse_with(node1)
    
    def test_cannot_fuse_incompatible_shapes(self):
        """Test that nodes with incompatible shapes cannot be fused."""
        buf1 = Buffer("buf1", BufferScope.LOCAL, torch.float32, (10, 10))
        buf2 = Buffer("buf2", BufferScope.LOCAL, torch.float32, (5, 5))
        
        node1 = ConductorNode("add", outputs=[buf1])
        node2 = ConductorNode("mul", outputs=[buf2])
        
        # Different shapes should not be fusible
        assert not node1.can_fuse_with(node2)
    
    def test_can_fuse_broadcastable_shapes(self):
        """Test that nodes with broadcastable shapes can be fused."""
        buf1 = Buffer("buf1", BufferScope.LOCAL, torch.float32, (10, 1))
        buf2 = Buffer("buf2", BufferScope.LOCAL, torch.float32, (10, 5))
        
        node1 = ConductorNode("add", outputs=[buf1])
        node2 = ConductorNode("mul", outputs=[buf2])
        
        # Broadcastable shapes should be fusible
        assert node1.can_fuse_with(node2)
    
    def test_can_fuse_unknown_shapes(self):
        """Test that nodes with unknown shapes are considered fusible."""
        buf1 = Buffer("buf1", BufferScope.LOCAL, torch.float32, None)
        buf2 = Buffer("buf2", BufferScope.LOCAL, torch.float32, None)
        
        node1 = ConductorNode("add", outputs=[buf1])
        node2 = ConductorNode("mul", outputs=[buf2])
        
        # Unknown shapes should be considered compatible
        assert node1.can_fuse_with(node2)
    
    def test_shapes_broadcastable_identical(self):
        """Test broadcastability check with identical shapes."""
        node = ConductorNode("add")
        assert node._shapes_broadcastable((10, 10), (10, 10))
    
    def test_shapes_broadcastable_with_ones(self):
        """Test broadcastability check with dimension of 1."""
        node = ConductorNode("add")
        assert node._shapes_broadcastable((10, 1), (10, 5))
        assert node._shapes_broadcastable((1, 10), (5, 10))
        assert node._shapes_broadcastable((10, 1, 5), (10, 3, 1))
    
    def test_shapes_not_broadcastable(self):
        """Test broadcastability check with incompatible shapes."""
        node = ConductorNode("add")
        assert not node._shapes_broadcastable((10, 5), (10, 3))
        assert not node._shapes_broadcastable((5, 10), (3, 10))
    
    def test_generate_dsl_add_operation(self):
        """Test DSL generation for add operation."""
        input1 = Buffer("a", BufferScope.LOCAL, torch.float32)
        input2 = Buffer("b", BufferScope.LOCAL, torch.float32)
        output = Buffer("c", BufferScope.LOCAL, torch.float32)
        
        node = ConductorNode("add", inputs=[input1, input2], outputs=[output])
        dsl = node.generate_dsl()
        
        assert dsl == "c = add(a, b)"
    
    def test_generate_dsl_mul_operation(self):
        """Test DSL generation for mul operation."""
        input1 = Buffer("x", BufferScope.LOCAL, torch.float32)
        input2 = Buffer("y", BufferScope.LOCAL, torch.float32)
        output = Buffer("z", BufferScope.LOCAL, torch.float32)
        
        node = ConductorNode("mul", inputs=[input1, input2], outputs=[output])
        dsl = node.generate_dsl()
        
        assert dsl == "z = mul(x, y)"
    
    def test_generate_dsl_relu_operation(self):
        """Test DSL generation for relu operation."""
        input_buf = Buffer("input", BufferScope.LOCAL, torch.float32)
        output_buf = Buffer("output", BufferScope.LOCAL, torch.float32)
        
        node = ConductorNode("relu", inputs=[input_buf], outputs=[output_buf])
        dsl = node.generate_dsl()
        
        assert dsl == "output = relu(input)"
    
    def test_generate_dsl_sum_operation_with_dim(self):
        """Test DSL generation for sum operation with dimension."""
        input_buf = Buffer("tensor", BufferScope.LOCAL, torch.float32)
        output_buf = Buffer("result", BufferScope.LOCAL, torch.float32)
        
        node = ConductorNode(
            "sum", 
            inputs=[input_buf], 
            outputs=[output_buf],
            metadata={"dim": 1}
        )
        dsl = node.generate_dsl()
        
        assert dsl == "result = sum(tensor, dim=1)"
    
    def test_generate_dsl_sum_operation_no_dim(self):
        """Test DSL generation for sum operation without dimension."""
        input_buf = Buffer("tensor", BufferScope.LOCAL, torch.float32)
        output_buf = Buffer("result", BufferScope.LOCAL, torch.float32)
        
        node = ConductorNode("sum", inputs=[input_buf], outputs=[output_buf])
        dsl = node.generate_dsl()
        
        assert dsl == "result = sum(tensor)"
    
    def test_generate_dsl_matmul_operation(self):
        """Test DSL generation for matmul operation."""
        input1 = Buffer("matrix1", BufferScope.LOCAL, torch.float32)
        input2 = Buffer("matrix2", BufferScope.LOCAL, torch.float32)
        output = Buffer("product", BufferScope.LOCAL, torch.float32)
        
        node = ConductorNode("matmul", inputs=[input1, input2], outputs=[output])
        dsl = node.generate_dsl()
        
        assert dsl == "product = matmul(matrix1, matrix2)"
    
    def test_generate_dsl_generic_operation(self):
        """Test DSL generation for generic operation."""
        input_buf = Buffer("input", BufferScope.LOCAL, torch.float32)
        output_buf = Buffer("output", BufferScope.LOCAL, torch.float32)
        
        node = ConductorNode("custom_op", inputs=[input_buf], outputs=[output_buf])
        dsl = node.generate_dsl()
        
        assert dsl == "output = custom_op(input)"
    
    def test_generate_dsl_multiple_outputs(self):
        """Test DSL generation for operation with multiple outputs."""
        input_buf = Buffer("input", BufferScope.LOCAL, torch.float32)
        output1 = Buffer("out1", BufferScope.LOCAL, torch.float32)
        output2 = Buffer("out2", BufferScope.LOCAL, torch.float32)
        
        node = ConductorNode("split", inputs=[input_buf], outputs=[output1, output2])
        dsl = node.generate_dsl()
        
        assert dsl == "out1, out2 = split(input)"
    
    def test_estimate_cost_elementwise_operations(self):
        """Test cost estimation for elementwise operations."""
        add_node = ConductorNode("add")
        relu_node = ConductorNode("relu")
        exp_node = ConductorNode("exp")
        
        assert add_node.estimate_cost() == 1.0
        assert relu_node.estimate_cost() == 0.5
        assert exp_node.estimate_cost() == 4.0
    
    def test_estimate_cost_reduction_operations(self):
        """Test cost estimation for reduction operations."""
        sum_node = ConductorNode("sum")
        mean_node = ConductorNode("mean")
        argmax_node = ConductorNode("argmax")
        
        assert sum_node.estimate_cost() == 2.0
        assert mean_node.estimate_cost() == 2.5
        assert argmax_node.estimate_cost() == 3.0
    
    def test_estimate_cost_matrix_operations(self):
        """Test cost estimation for matrix operations."""
        matmul_node = ConductorNode("matmul")
        conv2d_node = ConductorNode("conv2d")
        
        assert matmul_node.estimate_cost() == 10.0
        assert conv2d_node.estimate_cost() == 15.0
    
    def test_estimate_cost_unknown_operation(self):
        """Test cost estimation for unknown operation."""
        unknown_node = ConductorNode("unknown_op")
        assert unknown_node.estimate_cost() == 5.0  # Default cost
    
    def test_estimate_cost_with_tensor_size(self):
        """Test cost estimation scaling with tensor size."""
        # Small tensor
        small_buf = Buffer("small", BufferScope.LOCAL, torch.float32, (10, 10))
        small_node = ConductorNode("add", outputs=[small_buf])
        small_cost = small_node.estimate_cost()
        
        # Large tensor
        large_buf = Buffer("large", BufferScope.LOCAL, torch.float32, (1000, 1000))
        large_node = ConductorNode("add", outputs=[large_buf])
        large_cost = large_node.estimate_cost()
        
        # Large tensor should have higher cost
        assert large_cost > small_cost
        assert small_cost >= 1.0  # Base cost
    
    def test_estimate_cost_no_output_shape(self):
        """Test cost estimation with unknown output shape."""
        buf = Buffer("unknown", BufferScope.LOCAL, torch.float32, None)
        node = ConductorNode("add", outputs=[buf])
        
        # Should return base cost without size scaling
        assert node.estimate_cost() == 1.0


class TestComputationDAG:
    """Test ComputationDAG class functionality."""
    
    def test_dag_creation(self):
        """Test basic DAG creation."""
        dag = ComputationDAG()
        
        assert dag.nodes == []
        assert dag.buffers == []
        assert dag.inputs == []
        assert dag.outputs == []
    
    def test_add_node(self):
        """Test adding nodes to DAG."""
        dag = ComputationDAG()
        node = ConductorNode("add")
        
        dag.add_node(node)
        assert len(dag.nodes) == 1
        assert dag.nodes[0] is node
    
    def test_add_buffer(self):
        """Test adding buffers to DAG."""
        dag = ComputationDAG()
        buffer = Buffer("test", BufferScope.LOCAL, torch.float32)
        
        dag.add_buffer(buffer)
        assert len(dag.buffers) == 1
        assert dag.buffers[0] is buffer
    
    def test_validate_graph_correctness(self):
        """Test graph validation."""
        dag = ComputationDAG()
        # Basic validation should pass for empty graph
        assert dag.validate_graph_correctness() is True


class TestGraphAnalyzer:
    """Test GraphAnalyzer class functionality."""
    
    def test_analyzer_creation(self):
        """Test GraphAnalyzer initialization."""
        analyzer = GraphAnalyzer()
        assert analyzer is not None
    
    def test_parse_fx_graph_simple_relu(self):
        """Test FX Graph parsing with simple ReLU model."""
        analyzer = GraphAnalyzer()
        
        # Create a simple FX Graph module for testing
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)
        
        model = SimpleModel()
        example_input = torch.randn(1, 10)
        
        # Trace the model to get FX Graph
        traced = torch.fx.symbolic_trace(model)
        
        # Parse the FX Graph
        dag = analyzer.parse_fx_graph(traced)
        assert isinstance(dag, ComputationDAG)
        
        # Should have one ReLU operation
        assert len(dag.nodes) == 1
        assert dag.nodes[0].op_name == "relu"
        
        # Should have input and output buffers
        assert len(dag.inputs) == 1
        assert len(dag.outputs) == 1
        
        # Should have buffers for input, output
        assert len(dag.buffers) >= 2
    
    def test_parse_fx_graph_add_operation(self):
        """Test FX Graph parsing with addition operation."""
        analyzer = GraphAnalyzer()
        
        # Create a model with addition
        class AddModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.add(x, y)
        
        model = AddModel()
        
        # Trace the model
        traced = torch.fx.symbolic_trace(model)
        
        # Parse the FX Graph
        dag = analyzer.parse_fx_graph(traced)
        
        # Should have one add operation
        assert len(dag.nodes) == 1
        assert dag.nodes[0].op_name == "add"
        
        # Should have two inputs, one output
        assert len(dag.inputs) == 2
        assert len(dag.outputs) == 1
        
        # Add node should have two input buffers, one output buffer
        add_node = dag.nodes[0]
        assert len(add_node.inputs) == 2
        assert len(add_node.outputs) == 1
    
    def test_parse_fx_graph_chain_operations(self):
        """Test FX Graph parsing with chained operations."""
        analyzer = GraphAnalyzer()
        
        # Create a model with chained operations
        class ChainModel(torch.nn.Module):
            def forward(self, x):
                y = torch.add(x, x)
                z = torch.relu(y)
                return z
        
        model = ChainModel()
        
        # Trace the model
        traced = torch.fx.symbolic_trace(model)
        
        # Parse the FX Graph
        dag = analyzer.parse_fx_graph(traced)
        
        # Should have two operations: add and relu
        assert len(dag.nodes) == 2
        
        # Find the operations
        add_node = None
        relu_node = None
        for node in dag.nodes:
            if node.op_name == "add":
                add_node = node
            elif node.op_name == "relu":
                relu_node = node
        
        assert add_node is not None
        assert relu_node is not None
        
        # Check connectivity: add output should be relu input
        assert len(add_node.outputs) == 1
        assert len(relu_node.inputs) == 1
        assert add_node.outputs[0] == relu_node.inputs[0]
    
    def test_identify_data_dependencies(self):
        """Test data dependency analysis."""
        analyzer = GraphAnalyzer()
        
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)
        
        model = SimpleModel()
        traced = torch.fx.symbolic_trace(model)
        
        # Parse and analyze dependencies
        dag = analyzer.parse_fx_graph(traced)
        
        # Dependencies should be established
        relu_node = dag.nodes[0]
        input_buffer = dag.inputs[0]
        output_buffer = dag.outputs[0]
        
        # Input buffer should be consumed by relu node
        assert relu_node in input_buffer.consumers
        
        # Output buffer should be produced by relu node
        assert output_buffer.producer == relu_node
    
    def test_validate_graph_correctness_valid_graph(self):
        """Test graph validation with valid graph."""
        analyzer = GraphAnalyzer()
        
        # Create a valid model
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)
        
        model = SimpleModel()
        traced = torch.fx.symbolic_trace(model)
        
        # Parse and validate
        dag = analyzer.parse_fx_graph(traced)
        result = analyzer.validate_graph_correctness(dag)
        assert result is True
    
    def test_validate_graph_correctness_empty_graph(self):
        """Test graph validation with empty graph."""
        analyzer = GraphAnalyzer()
        dag = ComputationDAG()
        
        # Empty graph should be valid
        result = analyzer.validate_graph_correctness(dag)
        assert result is True
    
    def test_buffer_scope_optimization(self):
        """Test buffer scope optimization during parsing."""
        analyzer = GraphAnalyzer()
        
        # Create a model where one buffer feeds multiple operations
        class MultiConsumerModel(torch.nn.Module):
            def forward(self, x):
                y = torch.relu(x)
                z1 = torch.add(y, y)  # y consumed twice
                z2 = torch.mul(y, y)  # y consumed twice
                return z1 + z2
        
        model = MultiConsumerModel()
        traced = torch.fx.symbolic_trace(model)
        
        # Parse the graph
        dag = analyzer.parse_fx_graph(traced)
        
        # Find the relu output buffer (should have multiple consumers)
        relu_node = None
        for node in dag.nodes:
            if node.op_name == "relu":
                relu_node = node
                break
        
        assert relu_node is not None
        relu_output = relu_node.outputs[0]
        
        # Should have multiple consumers and SHARED scope
        assert len(relu_output.consumers) > 1
        from conductor.codegen.buffers import BufferScope
        assert relu_output.scope == BufferScope.SHARED


class TestNodeIntegration:
    """Integration tests for node functionality."""
    
    def test_complete_node_workflow(self):
        """Test complete workflow with node creation, fusion check, and DSL generation."""
        # Create buffers
        input1 = Buffer("a", BufferScope.LOCAL, torch.float32, (10, 10))
        input2 = Buffer("b", BufferScope.LOCAL, torch.float32, (10, 10))
        intermediate = Buffer("temp", BufferScope.LOCAL, torch.float32, (10, 10))
        output = Buffer("result", BufferScope.LOCAL, torch.float32, (10, 10))
        
        # Create nodes
        add_node = ConductorNode("add", inputs=[input1, input2], outputs=[intermediate])
        relu_node = ConductorNode("relu", inputs=[intermediate], outputs=[output])
        
        # Test fusion compatibility
        assert add_node.can_fuse_with(relu_node)
        
        # Test DSL generation
        add_dsl = add_node.generate_dsl()
        relu_dsl = relu_node.generate_dsl()
        
        assert add_dsl == "temp = add(a, b)"
        assert relu_dsl == "result = relu(temp)"
        
        # Test cost estimation
        add_cost = add_node.estimate_cost()
        relu_cost = relu_node.estimate_cost()
        
        assert add_cost > 1.0  # Should be scaled by tensor size
        assert relu_cost > 0.5  # Should be scaled by tensor size
        assert add_cost > relu_cost  # Add should be more expensive than ReLU
    
    def test_fusion_chain_compatibility(self):
        """Test fusion compatibility in a chain of operations."""
        # Create a chain: input -> add -> mul -> relu -> sum
        buffers = [
            Buffer(f"buf_{i}", BufferScope.LOCAL, torch.float32, (10, 10))
            for i in range(5)
        ]
        
        nodes = [
            ConductorNode("add", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("mul", inputs=[buffers[1]], outputs=[buffers[2]]),
            ConductorNode("relu", inputs=[buffers[2]], outputs=[buffers[3]]),
            ConductorNode("sum", inputs=[buffers[3]], outputs=[buffers[4]])
        ]
        
        # Elementwise operations should be fusible with each other
        assert nodes[0].can_fuse_with(nodes[1])  # add -> mul
        assert nodes[1].can_fuse_with(nodes[2])  # mul -> relu
        
        # Elementwise should be fusible with reduction
        assert nodes[2].can_fuse_with(nodes[3])  # relu -> sum
        
        # But reduction should not be fusible with elementwise
        assert not nodes[3].can_fuse_with(nodes[2])  # sum -> relu (reverse)
"""
Comprehensive tests for FX Graph to DAG conversion in graph_analyzer.py.

These tests focus on the robustness of the FX graph to internal DAG conversion,
which is a critical component that doesn't require code generation.
"""

import pytest
import torch
import torch.fx as fx
from typing import List, Tuple

from conductor.graph.graph_analyzer import GraphAnalyzer, ComputationDAG
from conductor.graph.graph_nodes import ConductorNode
from conductor.graph.buffers import Buffer


class TestFXToDAGConversion:
    """Test suite for FX graph to DAG conversion."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = GraphAnalyzer()
    
    def create_simple_model(self, op_func):
        """Create a simple model with a single operation."""
        def model(x, y):
            return op_func(x, y)
        return model
    
    def create_multi_op_model(self):
        """Create a model with multiple operations."""
        def model(x, y, z):
            temp1 = x + y
            temp2 = temp1 * z
            return torch.relu(temp2)
        return model
    
    def create_branching_model(self):
        """Create a model with branching computation."""
        def model(x):
            branch1 = torch.relu(x)
            branch2 = torch.sigmoid(x)
            return branch1 + branch2
        return model
    
    def create_nested_model(self):
        """Create a model with nested operations."""
        def model(x, y):
            temp1 = x + y
            temp2 = torch.relu(temp1)
            temp3 = temp2 * x
            return torch.sum(temp3)
        return model
    
    def trace_model(self, model, *inputs):
        """Trace a model to get FX graph."""
        return fx.symbolic_trace(model)
    
    def test_simple_add_conversion(self):
        """Test conversion of simple addition operation."""
        model = self.create_simple_model(torch.add)
        fx_graph = self.trace_model(model)
        
        inputs = [torch.randn(4, 4), torch.randn(4, 4)]
        dag = self.analyzer.parse_fx_graph(fx_graph, inputs)
        
        # Verify DAG structure
        assert isinstance(dag, ComputationDAG)
        assert len(dag.inputs) == 2
        assert len(dag.outputs) == 1
        assert len(dag.nodes) == 1
        
        # Verify node properties
        node = dag.nodes[0]
        assert node.op_name == "add"
        assert len(node.inputs) == 2
        assert len(node.outputs) == 1
        
        # Verify buffer shapes
        for input_buf in dag.inputs:
            assert input_buf.shape == (4, 4)
            assert input_buf.dtype == torch.float32
        
        output_buf = dag.outputs[0]
        assert output_buf.shape == (4, 4)
        assert output_buf.dtype == torch.float32
    
    def test_simple_mul_conversion(self):
        """Test conversion of simple multiplication operation."""
        model = self.create_simple_model(torch.mul)
        fx_graph = self.trace_model(model)
        
        inputs = [torch.randn(2, 8), torch.randn(2, 8)]
        dag = self.analyzer.parse_fx_graph(fx_graph, inputs)
        
        assert len(dag.nodes) == 1
        assert dag.nodes[0].op_name == "mul"
        
        # Verify shapes are preserved
        for buf in dag.inputs + dag.outputs:
            assert buf.shape == (2, 8)
    
    def test_multi_operation_conversion(self):
        """Test conversion of multi-operation graph."""
        model = self.create_multi_op_model()
        fx_graph = self.trace_model(model)
        
        inputs = [torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 3)]
        dag = self.analyzer.parse_fx_graph(fx_graph, inputs)
        
        # Should have 3 operations: add, mul, relu
        assert len(dag.inputs) == 3
        assert len(dag.outputs) == 1
        assert len(dag.nodes) == 3
        
        # Verify operation sequence
        op_names = [node.op_name for node in dag.nodes]
        assert "add" in op_names
        assert "mul" in op_names
        assert "relu" in op_names
        
        # Verify all shapes are consistent
        for buf in dag.inputs + dag.outputs:
            assert buf.shape == (3, 3)
    
    def test_branching_graph_conversion(self):
        """Test conversion of branching computation graph."""
        model = self.create_branching_model()
        fx_graph = self.trace_model(model)
        
        inputs = [torch.randn(5, 5)]
        dag = self.analyzer.parse_fx_graph(fx_graph, inputs)
        
        # Should have relu, sigmoid, and add operations
        assert len(dag.inputs) == 1
        assert len(dag.outputs) == 1
        assert len(dag.nodes) == 3
        
        op_names = [node.op_name for node in dag.nodes]
        assert "relu" in op_names
        assert "sigmoid" in op_names
        assert "add" in op_names
    
    def test_nested_operations_conversion(self):
        """Test conversion of nested operations."""
        model = self.create_nested_model()
        fx_graph = self.trace_model(model)
        
        inputs = [torch.randn(4, 6), torch.randn(4, 6)]
        dag = self.analyzer.parse_fx_graph(fx_graph, inputs)
        
        # Should have add, relu, mul, sum operations
        assert len(dag.inputs) == 2
        assert len(dag.outputs) == 1
        assert len(dag.nodes) == 4
        
        op_names = [node.op_name for node in dag.nodes]
        expected_ops = {"add", "relu", "mul", "sum"}
        assert set(op_names) == expected_ops
    
    def test_various_tensor_shapes(self):
        """Test conversion with various tensor shapes."""
        test_shapes = [
            (1,),           # 1D
            (10,),          # 1D larger
            (3, 4),         # 2D
            (2, 3, 4),      # 3D
            (1, 2, 3, 4),   # 4D
        ]
        
        for shape in test_shapes:
            model = self.create_simple_model(torch.add)
            fx_graph = self.trace_model(model)
            
            inputs = [torch.randn(shape), torch.randn(shape)]
            dag = self.analyzer.parse_fx_graph(fx_graph, inputs)
            
            # Verify shapes are preserved
            for buf in dag.inputs + dag.outputs:
                assert buf.shape == shape
    
    def test_various_data_types(self):
        """Test conversion with various data types."""
        test_dtypes = [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        ]
        
        for dtype in test_dtypes:
            model = self.create_simple_model(torch.add)
            fx_graph = self.trace_model(model)
            
            inputs = [torch.randn(3, 3).to(dtype), torch.randn(3, 3).to(dtype)]
            dag = self.analyzer.parse_fx_graph(fx_graph, inputs)
            
            # Verify dtypes are preserved
            for buf in dag.inputs + dag.outputs:
                assert buf.dtype == dtype
    
    def test_dag_validation(self):
        """Test DAG validation functionality."""
        model = self.create_multi_op_model()
        fx_graph = self.trace_model(model)
        
        inputs = [torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2)]
        dag = self.analyzer.parse_fx_graph(fx_graph, inputs)
        
        # DAG should be valid
        assert dag.validate_graph_correctness()
    
    def test_buffer_connections(self):
        """Test that buffer connections are correctly established."""
        model = self.create_multi_op_model()
        fx_graph = self.trace_model(model)
        
        inputs = [torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2)]
        dag = self.analyzer.parse_fx_graph(fx_graph, inputs)
        
        # Verify that intermediate buffers exist
        all_buffers = set()
        for node in dag.nodes:
            all_buffers.update(node.inputs)
            all_buffers.update(node.outputs)
        
        # Should have more buffers than just inputs and outputs
        assert len(all_buffers) >= len(dag.inputs) + len(dag.outputs)
    
    def test_node_metadata_preservation(self):
        """Test that node metadata is preserved during conversion."""
        model = self.create_simple_model(torch.add)
        fx_graph = self.trace_model(model)
        
        inputs = [torch.randn(2, 2), torch.randn(2, 2)]
        dag = self.analyzer.parse_fx_graph(fx_graph, inputs)
        
        # Verify nodes have metadata
        for node in dag.nodes:
            assert hasattr(node, 'metadata')
            assert isinstance(node.metadata, dict)
    
    def test_complex_matmul_conversion(self):
        """Test conversion of matrix multiplication operations."""
        def matmul_model(x, y):
            return torch.matmul(x, y)
        
        fx_graph = self.trace_model(matmul_model)
        
        inputs = [torch.randn(4, 3), torch.randn(3, 5)]
        dag = self.analyzer.parse_fx_graph(fx_graph, inputs)
        
        assert len(dag.nodes) == 1
        assert dag.nodes[0].op_name == "matmul"
        
        # Verify input and output shapes for matmul
        assert dag.inputs[0].shape == (4, 3)
        assert dag.inputs[1].shape == (3, 5)
        assert dag.outputs[0].shape == (4, 5)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with None inputs
        with pytest.raises((ValueError, TypeError)):
            self.analyzer.parse_fx_graph(None, [])
        
        # Test with empty inputs
        model = self.create_simple_model(torch.add)
        fx_graph = self.trace_model(model)
        
        with pytest.raises((ValueError, IndexError)):
            self.analyzer.parse_fx_graph(fx_graph, [])
    
    def test_large_graph_conversion(self):
        """Test conversion of larger, more complex graphs."""
        def large_model(x):
            # Create a chain of operations
            result = x
            for i in range(10):
                result = torch.relu(result + 0.1)
                result = result * 0.9
            return result
        
        fx_graph = self.trace_model(large_model)
        
        inputs = [torch.randn(8, 8)]
        dag = self.analyzer.parse_fx_graph(fx_graph, inputs)
        
        # Should have many operations
        assert len(dag.nodes) >= 20  # 10 iterations * 2 ops per iteration
        assert dag.validate_graph_correctness()
        
        # All intermediate shapes should be preserved
        for buf in dag.inputs + dag.outputs:
            assert buf.shape == (8, 8)

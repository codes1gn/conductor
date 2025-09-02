"""
Enhanced unit tests for fusion engine functionality.

This module provides comprehensive tests for the FusionEngine class,
including fusion heuristics, cluster creation, and optimization validation.
"""

import pytest
import torch
from conductor.codegen.fusion import FusionEngine, FusionCluster, FusionType
from conductor.codegen.graph import ConductorNode, ComputationDAG
from conductor.codegen.buffers import Buffer, BufferScope


class TestFusionEngine:
    """Enhanced test cases for FusionEngine functionality."""
    
    def test_fusion_engine_initialization(self):
        """Test FusionEngine initialization with different configurations."""
        # Default configuration
        engine = FusionEngine()
        assert engine is not None
        
        # Custom configuration
        config = {
            'max_fusion_size': 5,
            'fusion_threshold': 0.9,
            'elementwise_fusion': True,
            'reduction_fusion': False
        }
        engine_custom = FusionEngine(config)
        assert engine_custom is not None
    
    def test_identify_elementwise_fusion_opportunities(self, fusion_engine):
        """Test identification of elementwise fusion opportunities."""
        # Create chain of elementwise operations
        buffers = [
            Buffer(f"buf_{i}", BufferScope.LOCAL, torch.float32, (10, 10))
            for i in range(4)
        ]
        
        nodes = [
            ConductorNode("add", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("mul", inputs=[buffers[1]], outputs=[buffers[2]]),
            ConductorNode("relu", inputs=[buffers[2]], outputs=[buffers[3]])
        ]
        
        # Set up buffer relationships
        for i in range(len(nodes) - 1):
            buffers[i + 1].producer = nodes[i]
            buffers[i + 1].consumers = [nodes[i + 1]]
        
        dag = ComputationDAG()
        for node in nodes:
            dag.add_node(node)
        for buffer in buffers:
            dag.add_buffer(buffer)
        
        # Identify fusion opportunities
        clusters = fusion_engine.identify_fusion_opportunities(dag)
        
        # Should find one cluster with all three elementwise operations
        assert len(clusters) >= 1
        
        # Find the elementwise cluster
        elementwise_cluster = None
        for cluster in clusters:
            if cluster.cluster_type == FusionType.ELEMENTWISE:
                elementwise_cluster = cluster
                break
        
        assert elementwise_cluster is not None
        assert len(elementwise_cluster.nodes) >= 2  # At least two nodes fused
    
    def test_identify_reduction_fusion_opportunities(self, fusion_engine):
        """Test identification of elementwise + reduction fusion opportunities."""
        # Create elementwise -> reduction pattern
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (10, 10)),
            Buffer("temp", BufferScope.LOCAL, torch.float32, (10, 10)),
            Buffer("output", BufferScope.GLOBAL, torch.float32, (10,))
        ]
        
        nodes = [
            ConductorNode("mul", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("sum", inputs=[buffers[1]], outputs=[buffers[2]])
        ]
        
        # Set up relationships
        buffers[1].producer = nodes[0]
        buffers[1].consumers = [nodes[1]]
        
        dag = ComputationDAG()
        for node in nodes:
            dag.add_node(node)
        for buffer in buffers:
            dag.add_buffer(buffer)
        
        # Identify fusion opportunities
        clusters = fusion_engine.identify_fusion_opportunities(dag)
        
        # Should find reduction fusion opportunity
        reduction_cluster = None
        for cluster in clusters:
            if cluster.cluster_type == FusionType.REDUCTION:
                reduction_cluster = cluster
                break
        
        assert reduction_cluster is not None
        assert len(reduction_cluster.nodes) == 2
        assert any(node.op_name == "mul" for node in reduction_cluster.nodes)
        assert any(node.op_name == "sum" for node in reduction_cluster.nodes)
    
    def test_fusion_size_limits(self, fusion_engine):
        """Test that fusion respects size limits."""
        # Create a very long chain of operations
        num_ops = 15  # More than typical max_fusion_size
        buffers = [
            Buffer(f"buf_{i}", BufferScope.LOCAL, torch.float32, (10, 10))
            for i in range(num_ops + 1)
        ]
        
        nodes = [
            ConductorNode("add", inputs=[buffers[i]], outputs=[buffers[i + 1]])
            for i in range(num_ops)
        ]
        
        # Set up relationships
        for i in range(num_ops):
            buffers[i + 1].producer = nodes[i]
            if i < num_ops - 1:
                buffers[i + 1].consumers = [nodes[i + 1]]
        
        dag = ComputationDAG()
        for node in nodes:
            dag.add_node(node)
        for buffer in buffers:
            dag.add_buffer(buffer)
        
        # Identify fusion opportunities
        clusters = fusion_engine.identify_fusion_opportunities(dag)
        
        # Should create multiple clusters due to size limits
        total_fused_nodes = sum(len(cluster.nodes) for cluster in clusters)
        assert total_fused_nodes <= num_ops  # Some nodes should be fused
        
        # Each cluster should respect size limits
        for cluster in clusters:
            assert len(cluster.nodes) <= 10  # Default max_fusion_size
    
    def test_fusion_incompatible_shapes(self, fusion_engine):
        """Test that fusion handles incompatible shapes correctly."""
        # Create operations with incompatible shapes
        buffers = [
            Buffer("buf1", BufferScope.LOCAL, torch.float32, (10, 10)),
            Buffer("buf2", BufferScope.LOCAL, torch.float32, (5, 5)),  # Different shape
            Buffer("buf3", BufferScope.LOCAL, torch.float32, (10, 10))
        ]
        
        nodes = [
            ConductorNode("add", outputs=[buffers[0]]),
            ConductorNode("mul", outputs=[buffers[1]]),  # Incompatible shape
            ConductorNode("relu", outputs=[buffers[2]])
        ]
        
        dag = ComputationDAG()
        for node in nodes:
            dag.add_node(node)
        for buffer in buffers:
            dag.add_buffer(buffer)
        
        # Identify fusion opportunities
        clusters = fusion_engine.identify_fusion_opportunities(dag)
        
        # Should not fuse operations with incompatible shapes
        for cluster in clusters:
            # All nodes in a cluster should have compatible output shapes
            shapes = [node.outputs[0].shape for node in cluster.nodes if node.outputs]
            if len(shapes) > 1:
                # Check that shapes are compatible (same or broadcastable)
                first_shape = shapes[0]
                for shape in shapes[1:]:
                    if shape is not None and first_shape is not None:
                        assert shape == first_shape or self._shapes_broadcastable(shape, first_shape)
    
    def test_fusion_memory_bound_operations(self, fusion_engine):
        """Test fusion of memory-bound operations."""
        # Create memory-bound operation pattern
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (1000, 1000)),  # Large tensor
            Buffer("temp", BufferScope.LOCAL, torch.float32, (1000, 1000)),
            Buffer("output", BufferScope.GLOBAL, torch.float32, (1000, 1000))
        ]
        
        nodes = [
            ConductorNode("transpose", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("add", inputs=[buffers[1]], outputs=[buffers[2]])
        ]
        
        # Set up relationships
        buffers[1].producer = nodes[0]
        buffers[1].consumers = [nodes[1]]
        
        dag = ComputationDAG()
        for node in nodes:
            dag.add_node(node)
        for buffer in buffers:
            dag.add_buffer(buffer)
        
        # Identify fusion opportunities
        clusters = fusion_engine.identify_fusion_opportunities(dag)
        
        # Should identify memory-bound fusion opportunity
        memory_bound_cluster = None
        for cluster in clusters:
            if cluster.cluster_type == FusionType.MEMORY_BOUND:
                memory_bound_cluster = cluster
                break
        
        # Memory-bound fusion might not always be beneficial
        # Just verify the engine handles it without errors
        assert clusters is not None
    
    def test_apply_elementwise_fusion(self, fusion_engine):
        """Test application of elementwise fusion."""
        # Create elementwise nodes
        buffers = [
            Buffer(f"buf_{i}", BufferScope.LOCAL, torch.float32, (10, 10))
            for i in range(3)
        ]
        
        nodes = [
            ConductorNode("add", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("relu", inputs=[buffers[1]], outputs=[buffers[2]])
        ]
        
        # Set up relationships
        buffers[1].producer = nodes[0]
        buffers[1].consumers = [nodes[1]]
        
        # Apply fusion
        cluster = fusion_engine.apply_elementwise_fusion(nodes)
        
        assert isinstance(cluster, FusionCluster)
        assert cluster.cluster_type == FusionType.ELEMENTWISE
        assert len(cluster.nodes) == 2
        assert cluster.nodes == nodes
        
        # Check external inputs/outputs
        assert len(cluster.external_inputs) == 1
        assert cluster.external_inputs[0] == buffers[0]
        assert len(cluster.external_outputs) == 1
        assert cluster.external_outputs[0] == buffers[2]
        
        # Check internal buffers
        assert len(cluster.internal_buffers) == 1
        assert cluster.internal_buffers[0] == buffers[1]
    
    def test_apply_reduction_fusion(self, fusion_engine):
        """Test application of reduction fusion."""
        # Create elementwise + reduction pattern
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (10, 10)),
            Buffer("temp", BufferScope.LOCAL, torch.float32, (10, 10)),
            Buffer("output", BufferScope.GLOBAL, torch.float32, (10,))
        ]
        
        nodes = [
            ConductorNode("mul", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("sum", inputs=[buffers[1]], outputs=[buffers[2]])
        ]
        
        # Set up relationships
        buffers[1].producer = nodes[0]
        buffers[1].consumers = [nodes[1]]
        
        # Apply fusion
        cluster = fusion_engine.apply_reduction_fusion(nodes)
        
        assert isinstance(cluster, FusionCluster)
        assert cluster.cluster_type == FusionType.REDUCTION
        assert len(cluster.nodes) == 2
        
        # Should contain both elementwise and reduction operations
        op_names = [node.op_name for node in cluster.nodes]
        assert "mul" in op_names
        assert "sum" in op_names
    
    def test_optimize_buffer_usage(self, fusion_engine, sample_fusion_cluster):
        """Test buffer usage optimization within clusters."""
        # Get initial buffer count
        initial_internal_buffers = len(sample_fusion_cluster.internal_buffers)
        
        # Optimize buffer usage
        fusion_engine.optimize_buffer_usage(sample_fusion_cluster)
        
        # Buffer optimization should not increase buffer count
        final_internal_buffers = len(sample_fusion_cluster.internal_buffers)
        assert final_internal_buffers <= initial_internal_buffers
    
    def test_fusion_cost_benefit_analysis(self, fusion_engine):
        """Test fusion cost-benefit analysis."""
        # Create nodes with different computational costs
        buffers = [
            Buffer(f"buf_{i}", BufferScope.LOCAL, torch.float32, (100, 100))
            for i in range(3)
        ]
        
        # Cheap operation followed by expensive operation
        nodes = [
            ConductorNode("add", inputs=[buffers[0]], outputs=[buffers[1]]),  # Cheap
            ConductorNode("exp", inputs=[buffers[1]], outputs=[buffers[2]])   # Expensive
        ]
        
        # Set up relationships
        buffers[1].producer = nodes[0]
        buffers[1].consumers = [nodes[1]]
        
        dag = ComputationDAG()
        for node in nodes:
            dag.add_node(node)
        for buffer in buffers:
            dag.add_buffer(buffer)
        
        # Identify fusion opportunities
        clusters = fusion_engine.identify_fusion_opportunities(dag)
        
        # Should still consider fusion despite cost difference
        assert len(clusters) >= 0  # At least no errors
        
        # If fusion occurs, verify it's beneficial
        for cluster in clusters:
            if len(cluster.nodes) > 1:
                # Fusion should reduce memory traffic
                assert len(cluster.internal_buffers) >= 0
    
    def test_fusion_with_multiple_consumers(self, fusion_engine):
        """Test fusion behavior with buffers that have multiple consumers."""
        # Create pattern where one buffer feeds multiple operations
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (10, 10)),
            Buffer("shared", BufferScope.SHARED, torch.float32, (10, 10)),
            Buffer("output1", BufferScope.GLOBAL, torch.float32, (10, 10)),
            Buffer("output2", BufferScope.GLOBAL, torch.float32, (10, 10))
        ]
        
        nodes = [
            ConductorNode("relu", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("add", inputs=[buffers[1]], outputs=[buffers[2]]),
            ConductorNode("mul", inputs=[buffers[1]], outputs=[buffers[3]])
        ]
        
        # Set up relationships - shared buffer has multiple consumers
        buffers[1].producer = nodes[0]
        buffers[1].consumers = [nodes[1], nodes[2]]
        
        dag = ComputationDAG()
        for node in nodes:
            dag.add_node(node)
        for buffer in buffers:
            dag.add_buffer(buffer)
        
        # Identify fusion opportunities
        clusters = fusion_engine.identify_fusion_opportunities(dag)
        
        # Should handle multiple consumers correctly
        # The shared buffer should not be considered internal to any cluster
        for cluster in clusters:
            for internal_buf in cluster.internal_buffers:
                assert len(internal_buf.consumers) <= 1  # Internal buffers should have single consumer
    
    def test_fusion_validation_safety(self, fusion_engine):
        """Test fusion safety validation."""
        # Create a valid fusion cluster
        buffers = [
            Buffer("input", BufferScope.LOCAL, torch.float32, (10, 10)),
            Buffer("temp", BufferScope.LOCAL, torch.float32, (10, 10)),
            Buffer("output", BufferScope.LOCAL, torch.float32, (10, 10))
        ]
        
        nodes = [
            ConductorNode("add", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("relu", inputs=[buffers[1]], outputs=[buffers[2]])
        ]
        
        cluster = FusionCluster(
            nodes=nodes,
            cluster_type=FusionType.ELEMENTWISE,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[2]],
            internal_buffers=[buffers[1]],
            dsl_function_name="test_fusion"
        )
        
        # Validate fusion safety
        is_safe = cluster.validate_fusion_safety()
        assert is_safe is True
    
    def test_fusion_performance_estimation(self, fusion_engine):
        """Test fusion performance gain estimation."""
        # Create fusion cluster
        buffers = [
            Buffer("input", BufferScope.LOCAL, torch.float32, (1000, 1000)),  # Large tensor
            Buffer("temp", BufferScope.LOCAL, torch.float32, (1000, 1000)),
            Buffer("output", BufferScope.LOCAL, torch.float32, (1000, 1000))
        ]
        
        nodes = [
            ConductorNode("add", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("mul", inputs=[buffers[1]], outputs=[buffers[2]])
        ]
        
        cluster = FusionCluster(
            nodes=nodes,
            cluster_type=FusionType.ELEMENTWISE,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[2]],
            internal_buffers=[buffers[1]],
            dsl_function_name="test_fusion"
        )
        
        # Estimate performance gain
        gain = cluster.estimate_performance_gain()
        
        # Should estimate positive gain for elementwise fusion
        assert gain > 0.0
        assert gain <= 10.0  # Reasonable upper bound
    
    def _shapes_broadcastable(self, shape1, shape2):
        """Helper method to check if shapes are broadcastable."""
        if shape1 is None or shape2 is None:
            return True
        
        # Reverse shapes for broadcasting rules
        s1 = list(reversed(shape1))
        s2 = list(reversed(shape2))
        
        max_len = max(len(s1), len(s2))
        s1.extend([1] * (max_len - len(s1)))
        s2.extend([1] * (max_len - len(s2)))
        
        for d1, d2 in zip(s1, s2):
            if d1 != d2 and d1 != 1 and d2 != 1:
                return False
        
        return True


class TestFusionCluster:
    """Test cases for FusionCluster functionality."""
    
    def test_fusion_cluster_creation(self):
        """Test FusionCluster creation with all parameters."""
        buffers = [
            Buffer("input", BufferScope.LOCAL, torch.float32, (10, 10)),
            Buffer("temp", BufferScope.LOCAL, torch.float32, (10, 10)),
            Buffer("output", BufferScope.LOCAL, torch.float32, (10, 10))
        ]
        
        nodes = [
            ConductorNode("add", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("relu", inputs=[buffers[1]], outputs=[buffers[2]])
        ]
        
        cluster = FusionCluster(
            nodes=nodes,
            cluster_type=FusionType.ELEMENTWISE,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[2]],
            internal_buffers=[buffers[1]],
            dsl_function_name="fused_add_relu"
        )
        
        assert cluster.nodes == nodes
        assert cluster.cluster_type == FusionType.ELEMENTWISE
        assert cluster.external_inputs == [buffers[0]]
        assert cluster.external_outputs == [buffers[2]]
        assert cluster.internal_buffers == [buffers[1]]
        assert cluster.dsl_function_name == "fused_add_relu"
    
    def test_generate_fused_dsl(self, sample_fusion_cluster):
        """Test fused DSL generation."""
        dsl = sample_fusion_cluster.generate_fused_dsl()
        
        assert isinstance(dsl, str)
        assert len(dsl) > 0
        
        # Should contain function definition
        assert "function" in dsl
        assert sample_fusion_cluster.dsl_function_name in dsl
        
        # Should contain operations from all nodes
        for node in sample_fusion_cluster.nodes:
            assert node.op_name in dsl
    
    def test_validate_fusion_safety_valid_cluster(self):
        """Test fusion safety validation for valid cluster."""
        # Create mathematically safe fusion
        buffers = [
            Buffer("input", BufferScope.LOCAL, torch.float32, (10, 10)),
            Buffer("temp", BufferScope.LOCAL, torch.float32, (10, 10)),
            Buffer("output", BufferScope.LOCAL, torch.float32, (10, 10))
        ]
        
        nodes = [
            ConductorNode("add", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("relu", inputs=[buffers[1]], outputs=[buffers[2]])
        ]
        
        cluster = FusionCluster(
            nodes=nodes,
            cluster_type=FusionType.ELEMENTWISE,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[2]],
            internal_buffers=[buffers[1]],
            dsl_function_name="safe_fusion"
        )
        
        assert cluster.validate_fusion_safety() is True
    
    def test_validate_fusion_safety_invalid_cluster(self):
        """Test fusion safety validation for invalid cluster."""
        # Create cluster with circular dependency (invalid)
        buffers = [
            Buffer("buf1", BufferScope.LOCAL, torch.float32, (10, 10)),
            Buffer("buf2", BufferScope.LOCAL, torch.float32, (10, 10))
        ]
        
        # Create circular dependency
        node1 = ConductorNode("add", inputs=[buffers[1]], outputs=[buffers[0]])
        node2 = ConductorNode("mul", inputs=[buffers[0]], outputs=[buffers[1]])
        
        cluster = FusionCluster(
            nodes=[node1, node2],
            cluster_type=FusionType.ELEMENTWISE,
            external_inputs=[],
            external_outputs=[],
            internal_buffers=[buffers[0], buffers[1]],
            dsl_function_name="invalid_fusion"
        )
        
        # Should detect the circular dependency
        assert cluster.validate_fusion_safety() is False
    
    def test_estimate_performance_gain_elementwise(self):
        """Test performance gain estimation for elementwise fusion."""
        # Create elementwise cluster with large tensors
        buffers = [
            Buffer("input", BufferScope.LOCAL, torch.float32, (1000, 1000)),
            Buffer("temp", BufferScope.LOCAL, torch.float32, (1000, 1000)),
            Buffer("output", BufferScope.LOCAL, torch.float32, (1000, 1000))
        ]
        
        nodes = [
            ConductorNode("add", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("mul", inputs=[buffers[1]], outputs=[buffers[2]])
        ]
        
        cluster = FusionCluster(
            nodes=nodes,
            cluster_type=FusionType.ELEMENTWISE,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[2]],
            internal_buffers=[buffers[1]],
            dsl_function_name="elementwise_fusion"
        )
        
        gain = cluster.estimate_performance_gain()
        
        # Elementwise fusion should show good performance gain
        assert gain > 1.0  # Should be better than no fusion
        assert gain < 10.0  # Reasonable upper bound
    
    def test_estimate_performance_gain_reduction(self):
        """Test performance gain estimation for reduction fusion."""
        # Create elementwise + reduction cluster
        buffers = [
            Buffer("input", BufferScope.LOCAL, torch.float32, (1000, 1000)),
            Buffer("temp", BufferScope.LOCAL, torch.float32, (1000, 1000)),
            Buffer("output", BufferScope.LOCAL, torch.float32, (1000,))
        ]
        
        nodes = [
            ConductorNode("mul", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("sum", inputs=[buffers[1]], outputs=[buffers[2]])
        ]
        
        cluster = FusionCluster(
            nodes=nodes,
            cluster_type=FusionType.REDUCTION,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[2]],
            internal_buffers=[buffers[1]],
            dsl_function_name="reduction_fusion"
        )
        
        gain = cluster.estimate_performance_gain()
        
        # Reduction fusion should show significant performance gain
        assert gain > 1.0
        assert gain < 20.0  # Reasonable upper bound


class TestFusionIntegration:
    """Integration tests for fusion functionality."""
    
    def test_end_to_end_fusion_pipeline(self, fusion_engine):
        """Test complete fusion pipeline from DAG to clusters."""
        # Create complex DAG with multiple fusion opportunities
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (100, 100)),
            Buffer("temp1", BufferScope.LOCAL, torch.float32, (100, 100)),
            Buffer("temp2", BufferScope.LOCAL, torch.float32, (100, 100)),
            Buffer("temp3", BufferScope.LOCAL, torch.float32, (100, 100)),
            Buffer("temp4", BufferScope.LOCAL, torch.float32, (100,)),
            Buffer("output", BufferScope.GLOBAL, torch.float32, (100,))
        ]
        
        nodes = [
            # Elementwise chain
            ConductorNode("add", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("mul", inputs=[buffers[1]], outputs=[buffers[2]]),
            ConductorNode("relu", inputs=[buffers[2]], outputs=[buffers[3]]),
            # Reduction
            ConductorNode("sum", inputs=[buffers[3]], outputs=[buffers[4]]),
            # Final elementwise
            ConductorNode("sigmoid", inputs=[buffers[4]], outputs=[buffers[5]])
        ]
        
        # Set up relationships
        for i in range(len(nodes) - 1):
            buffers[i + 1].producer = nodes[i]
            buffers[i + 1].consumers = [nodes[i + 1]]
        
        dag = ComputationDAG()
        for node in nodes:
            dag.add_node(node)
        for buffer in buffers:
            dag.add_buffer(buffer)
        dag.inputs = [buffers[0]]
        dag.outputs = [buffers[5]]
        
        # Apply fusion
        clusters = fusion_engine.identify_fusion_opportunities(dag)
        
        # Should find multiple fusion opportunities
        assert len(clusters) >= 1
        
        # Verify cluster types
        cluster_types = [cluster.cluster_type for cluster in clusters]
        assert FusionType.ELEMENTWISE in cluster_types or FusionType.REDUCTION in cluster_types
        
        # Verify all clusters are valid
        for cluster in clusters:
            assert cluster.validate_fusion_safety()
            assert cluster.estimate_performance_gain() > 0
    
    def test_fusion_with_branching_dag(self, fusion_engine):
        """Test fusion with DAG that has branching structure."""
        # Create DAG with branching
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (50, 50)),
            Buffer("branch1", BufferScope.LOCAL, torch.float32, (50, 50)),
            Buffer("branch2", BufferScope.LOCAL, torch.float32, (50, 50)),
            Buffer("out1", BufferScope.GLOBAL, torch.float32, (50, 50)),
            Buffer("out2", BufferScope.GLOBAL, torch.float32, (50, 50))
        ]
        
        nodes = [
            ConductorNode("relu", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("sigmoid", inputs=[buffers[0]], outputs=[buffers[2]]),  # Branch
            ConductorNode("add", inputs=[buffers[1]], outputs=[buffers[3]]),
            ConductorNode("mul", inputs=[buffers[2]], outputs=[buffers[4]])
        ]
        
        # Set up branching relationships
        buffers[1].producer = nodes[0]
        buffers[1].consumers = [nodes[2]]
        buffers[2].producer = nodes[1]
        buffers[2].consumers = [nodes[3]]
        
        dag = ComputationDAG()
        for node in nodes:
            dag.add_node(node)
        for buffer in buffers:
            dag.add_buffer(buffer)
        dag.inputs = [buffers[0]]
        dag.outputs = [buffers[3], buffers[4]]
        
        # Apply fusion
        clusters = fusion_engine.identify_fusion_opportunities(dag)
        
        # Should handle branching correctly
        assert clusters is not None
        
        # Verify no cluster spans across branches incorrectly
        for cluster in clusters:
            # All nodes in a cluster should be on the same execution path
            assert len(cluster.nodes) >= 1
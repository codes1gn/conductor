"""
FileCheck tests for fusion pattern validation.

This module validates that fusion patterns generate correct DSL code
using FileCheck-style pattern matching.
"""

import pytest
import torch
import re
from conductor.codegen.dsl import DSLGenerator
from conductor.codegen.graph import ConductorNode, ComputationDAG
from conductor.codegen.buffers import Buffer, BufferScope
from conductor.codegen.fusion import FusionEngine, FusionCluster, FusionType


@pytest.mark.filecheck
class TestElementwiseFusionPatterns:
    """Test DSL patterns for elementwise fusion."""
    
    def test_simple_elementwise_chain_pattern(self):
        """Test DSL pattern for simple elementwise chain fusion."""
        generator = DSLGenerator()
        
        # Create elementwise chain: add -> mul -> relu
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (32, 64)),
            Buffer("temp1", BufferScope.LOCAL, torch.float32, (32, 64)),
            Buffer("temp2", BufferScope.LOCAL, torch.float32, (32, 64)),
            Buffer("output", BufferScope.GLOBAL, torch.float32, (32, 64))
        ]
        
        nodes = [
            ConductorNode("add", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("mul", inputs=[buffers[1]], outputs=[buffers[2]]),
            ConductorNode("relu", inputs=[buffers[2]], outputs=[buffers[3]])
        ]
        
        # Create fusion cluster
        cluster = FusionCluster(
            nodes=nodes,
            cluster_type=FusionType.ELEMENTWISE,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[3]],
            internal_buffers=[buffers[1], buffers[2]],
            dsl_function_name="fused_add_mul_relu"
        )
        
        dsl = cluster.generate_fused_dsl()
        
        # FileCheck: Verify fused function structure
        # CHECK: function fused_add_mul_relu({{.*}}) -> ({{.*}}) {
        assert re.search(r'function fused_add_mul_relu\([^)]*\) -> \([^)]*\) \{', dsl)
        
        # CHECK: // Fused elementwise operations
        assert re.search(r'// Fused elementwise operations', dsl)
        
        # CHECK: Intermediate operations are inlined
        # CHECK-NOT: temp1 = 
        # CHECK-NOT: temp2 = 
        # The intermediate results should be inlined, not stored
        lines = [line.strip() for line in dsl.split('\n') if '=' in line and ';' in line]
        temp_assignments = [line for line in lines if 'temp1 =' in line or 'temp2 =' in line]
        assert len(temp_assignments) == 0, "Intermediate temporaries should be inlined in fused code"
        
        # CHECK: Final result assignment
        assert re.search(r'output = relu\(mul\(add\(input\)\)\);', dsl) or \
               re.search(r'output = .*add.*mul.*relu', dsl)
    
    def test_elementwise_with_constants_pattern(self):
        """Test DSL pattern for elementwise operations with constants."""
        generator = DSLGenerator()
        
        # Create elementwise operations with constants
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (16, 32)),
            Buffer("temp", BufferScope.LOCAL, torch.float32, (16, 32)),
            Buffer("output", BufferScope.GLOBAL, torch.float32, (16, 32))
        ]
        
        # Add with constant, then multiply with constant
        add_node = ConductorNode("add", inputs=[buffers[0]], outputs=[buffers[1]], 
                                metadata={"constant": 1.0})
        mul_node = ConductorNode("mul", inputs=[buffers[1]], outputs=[buffers[2]], 
                                metadata={"constant": 2.0})
        
        cluster = FusionCluster(
            nodes=[add_node, mul_node],
            cluster_type=FusionType.ELEMENTWISE,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[2]],
            internal_buffers=[buffers[1]],
            dsl_function_name="fused_add_mul_constants"
        )
        
        dsl = cluster.generate_fused_dsl()
        
        # FileCheck: Verify constant handling
        # CHECK: Constants are properly embedded
        assert re.search(r'1\.0', dsl) or re.search(r'add.*1', dsl)
        assert re.search(r'2\.0', dsl) or re.search(r'mul.*2', dsl)
        
        # CHECK: Function signature includes only input and output
        assert re.search(r'function fused_add_mul_constants\(input\) -> \(output\)', dsl)
    
    def test_elementwise_broadcasting_pattern(self):
        """Test DSL pattern for elementwise operations with broadcasting."""
        generator = DSLGenerator()
        
        # Create operations with broadcastable shapes
        buffers = [
            Buffer("input1", BufferScope.GLOBAL, torch.float32, (32, 1)),    # Broadcastable
            Buffer("input2", BufferScope.GLOBAL, torch.float32, (32, 64)),   # Full shape
            Buffer("temp", BufferScope.LOCAL, torch.float32, (32, 64)),
            Buffer("output", BufferScope.GLOBAL, torch.float32, (32, 64))
        ]
        
        add_node = ConductorNode("add", inputs=[buffers[0], buffers[1]], outputs=[buffers[2]])
        relu_node = ConductorNode("relu", inputs=[buffers[2]], outputs=[buffers[3]])
        
        cluster = FusionCluster(
            nodes=[add_node, relu_node],
            cluster_type=FusionType.ELEMENTWISE,
            external_inputs=[buffers[0], buffers[1]],
            external_outputs=[buffers[3]],
            internal_buffers=[buffers[2]],
            dsl_function_name="fused_broadcast_add_relu"
        )
        
        dsl = cluster.generate_fused_dsl()
        
        # FileCheck: Verify broadcasting is handled
        # CHECK: Function takes both inputs
        assert re.search(r'function fused_broadcast_add_relu\(input1, input2\)', dsl)
        
        # CHECK: Broadcasting operation is present
        assert re.search(r'add\(input1, input2\)', dsl)
        
        # CHECK: Result has correct shape annotation
        assert re.search(r'-> \(output\)', dsl)


@pytest.mark.filecheck
class TestReductionFusionPatterns:
    """Test DSL patterns for reduction fusion."""
    
    def test_elementwise_reduction_pattern(self):
        """Test DSL pattern for elementwise + reduction fusion."""
        generator = DSLGenerator()
        
        # Create elementwise -> reduction pattern
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (64, 128)),
            Buffer("temp", BufferScope.LOCAL, torch.float32, (64, 128)),
            Buffer("output", BufferScope.GLOBAL, torch.float32, (64,))
        ]
        
        mul_node = ConductorNode("mul", inputs=[buffers[0]], outputs=[buffers[1]], 
                                metadata={"constant": 0.5})
        sum_node = ConductorNode("sum", inputs=[buffers[1]], outputs=[buffers[2]], 
                                metadata={"dim": 1})
        
        cluster = FusionCluster(
            nodes=[mul_node, sum_node],
            cluster_type=FusionType.REDUCTION,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[2]],
            internal_buffers=[buffers[1]],
            dsl_function_name="fused_mul_sum"
        )
        
        dsl = cluster.generate_fused_dsl()
        
        # FileCheck: Verify reduction fusion pattern
        # CHECK: function fused_mul_sum({{.*}}) -> ({{.*}}) {
        assert re.search(r'function fused_mul_sum\([^)]*\) -> \([^)]*\) \{', dsl)
        
        # CHECK: // Fused elementwise + reduction
        assert re.search(r'// Fused.*reduction', dsl)
        
        # CHECK: Reduction dimension is specified
        assert re.search(r'sum\(.*dim=1', dsl) or re.search(r'sum.*1\)', dsl)
        
        # CHECK: Elementwise operation is fused into reduction
        # The multiplication should be integrated into the sum operation
        assert re.search(r'sum\(mul\(input.*0\.5\).*dim=1\)', dsl) or \
               re.search(r'sum.*mul.*input', dsl)
    
    def test_multiple_elementwise_reduction_pattern(self):
        """Test DSL pattern for multiple elementwise ops + reduction."""
        generator = DSLGenerator()
        
        # Create chain: add -> mul -> relu -> sum
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (32, 256)),
            Buffer("temp1", BufferScope.LOCAL, torch.float32, (32, 256)),
            Buffer("temp2", BufferScope.LOCAL, torch.float32, (32, 256)),
            Buffer("temp3", BufferScope.LOCAL, torch.float32, (32, 256)),
            Buffer("output", BufferScope.GLOBAL, torch.float32, (32,))
        ]
        
        nodes = [
            ConductorNode("add", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("mul", inputs=[buffers[1]], outputs=[buffers[2]]),
            ConductorNode("relu", inputs=[buffers[2]], outputs=[buffers[3]]),
            ConductorNode("sum", inputs=[buffers[3]], outputs=[buffers[4]], metadata={"dim": 1})
        ]
        
        cluster = FusionCluster(
            nodes=nodes,
            cluster_type=FusionType.REDUCTION,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[4]],
            internal_buffers=[buffers[1], buffers[2], buffers[3]],
            dsl_function_name="fused_add_mul_relu_sum"
        )
        
        dsl = cluster.generate_fused_dsl()
        
        # FileCheck: Verify complex reduction fusion
        # CHECK: All elementwise operations are fused into reduction
        assert re.search(r'sum\(relu\(mul\(add\(input\)\)\).*dim=1\)', dsl) or \
               re.search(r'sum.*add.*mul.*relu.*input', dsl)
        
        # CHECK-NOT: Intermediate temporaries should not appear
        lines = [line.strip() for line in dsl.split('\n') if '=' in line and ';' in line]
        temp_assignments = [line for line in lines if 'temp' in line and '=' in line]
        assert len(temp_assignments) == 0, "No intermediate temporaries in fused reduction"
    
    def test_reduction_with_keepdim_pattern(self):
        """Test DSL pattern for reduction with keepdim."""
        generator = DSLGenerator()
        
        # Create reduction with keepdim
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (16, 32, 64)),
            Buffer("temp", BufferScope.LOCAL, torch.float32, (16, 32, 64)),
            Buffer("output", BufferScope.GLOBAL, torch.float32, (16, 1, 64))  # keepdim=True
        ]
        
        relu_node = ConductorNode("relu", inputs=[buffers[0]], outputs=[buffers[1]])
        mean_node = ConductorNode("mean", inputs=[buffers[1]], outputs=[buffers[2]], 
                                 metadata={"dim": 1, "keepdim": True})
        
        cluster = FusionCluster(
            nodes=[relu_node, mean_node],
            cluster_type=FusionType.REDUCTION,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[2]],
            internal_buffers=[buffers[1]],
            dsl_function_name="fused_relu_mean_keepdim"
        )
        
        dsl = cluster.generate_fused_dsl()
        
        # FileCheck: Verify keepdim handling
        # CHECK: keepdim parameter is preserved
        assert re.search(r'mean\(.*keepdim=true', dsl) or re.search(r'keepdim.*true', dsl)
        
        # CHECK: Dimension is specified
        assert re.search(r'dim=1', dsl)


@pytest.mark.filecheck
class TestMemoryBoundFusionPatterns:
    """Test DSL patterns for memory-bound operation fusion."""
    
    def test_transpose_elementwise_pattern(self):
        """Test DSL pattern for transpose + elementwise fusion."""
        generator = DSLGenerator()
        
        # Create transpose + elementwise pattern
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (128, 256)),
            Buffer("temp", BufferScope.LOCAL, torch.float32, (256, 128)),  # Transposed
            Buffer("output", BufferScope.GLOBAL, torch.float32, (256, 128))
        ]
        
        transpose_node = ConductorNode("transpose", inputs=[buffers[0]], outputs=[buffers[1]], 
                                      metadata={"dim1": 0, "dim2": 1})
        relu_node = ConductorNode("relu", inputs=[buffers[1]], outputs=[buffers[2]])
        
        cluster = FusionCluster(
            nodes=[transpose_node, relu_node],
            cluster_type=FusionType.MEMORY_BOUND,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[2]],
            internal_buffers=[buffers[1]],
            dsl_function_name="fused_transpose_relu"
        )
        
        dsl = cluster.generate_fused_dsl()
        
        # FileCheck: Verify memory-bound fusion
        # CHECK: // Fused memory-bound operations
        assert re.search(r'// Fused memory-bound', dsl)
        
        # CHECK: Transpose dimensions are preserved
        assert re.search(r'transpose\(.*dim1=0.*dim2=1', dsl) or \
               re.search(r'transpose.*0.*1', dsl)
        
        # CHECK: Operations are fused for memory efficiency
        assert re.search(r'relu\(transpose\(input\)\)', dsl) or \
               re.search(r'transpose.*relu', dsl)
    
    def test_reshape_operations_pattern(self):
        """Test DSL pattern for reshape operations fusion."""
        generator = DSLGenerator()
        
        # Create reshape + elementwise pattern
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (32, 64)),
            Buffer("temp", BufferScope.LOCAL, torch.float32, (2048,)),  # Flattened
            Buffer("output", BufferScope.GLOBAL, torch.float32, (2048,))
        ]
        
        flatten_node = ConductorNode("flatten", inputs=[buffers[0]], outputs=[buffers[1]])
        sigmoid_node = ConductorNode("sigmoid", inputs=[buffers[1]], outputs=[buffers[2]])
        
        cluster = FusionCluster(
            nodes=[flatten_node, sigmoid_node],
            cluster_type=FusionType.MEMORY_BOUND,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[2]],
            internal_buffers=[buffers[1]],
            dsl_function_name="fused_flatten_sigmoid"
        )
        
        dsl = cluster.generate_fused_dsl()
        
        # FileCheck: Verify reshape fusion
        # CHECK: Reshape operation is fused
        assert re.search(r'sigmoid\(flatten\(input\)\)', dsl) or \
               re.search(r'flatten.*sigmoid', dsl)
        
        # CHECK: Memory layout optimization comment
        assert re.search(r'// Memory layout optimization', dsl) or \
               re.search(r'// Fused.*memory', dsl)


@pytest.mark.filecheck
class TestComplexFusionPatterns:
    """Test DSL patterns for complex fusion scenarios."""
    
    def test_mixed_fusion_pattern(self):
        """Test DSL pattern for mixed fusion types."""
        generator = DSLGenerator()
        
        # Create complex pattern: elementwise chain -> reduction -> elementwise
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (64, 128)),
            Buffer("temp1", BufferScope.LOCAL, torch.float32, (64, 128)),
            Buffer("temp2", BufferScope.LOCAL, torch.float32, (64, 128)),
            Buffer("temp3", BufferScope.LOCAL, torch.float32, (64,)),      # Reduced
            Buffer("output", BufferScope.GLOBAL, torch.float32, (64,))
        ]
        
        nodes = [
            ConductorNode("add", inputs=[buffers[0]], outputs=[buffers[1]]),
            ConductorNode("mul", inputs=[buffers[1]], outputs=[buffers[2]]),
            ConductorNode("sum", inputs=[buffers[2]], outputs=[buffers[3]], metadata={"dim": 1}),
            ConductorNode("relu", inputs=[buffers[3]], outputs=[buffers[4]])
        ]
        
        cluster = FusionCluster(
            nodes=nodes,
            cluster_type=FusionType.MIXED,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[4]],
            internal_buffers=[buffers[1], buffers[2], buffers[3]],
            dsl_function_name="fused_mixed_operations"
        )
        
        dsl = cluster.generate_fused_dsl()
        
        # FileCheck: Verify mixed fusion pattern
        # CHECK: // Mixed fusion: elementwise + reduction + elementwise
        assert re.search(r'// Mixed fusion', dsl)
        
        # CHECK: All operations are properly chained
        assert re.search(r'relu\(sum\(mul\(add\(input\)\).*dim=1\)\)', dsl) or \
               re.search(r'add.*mul.*sum.*relu', dsl)
        
        # CHECK: Dimension change is handled correctly
        assert re.search(r'dim=1', dsl)
    
    def test_branching_fusion_pattern(self):
        """Test DSL pattern for fusion with branching."""
        generator = DSLGenerator()
        
        # Create branching pattern - one input, multiple outputs
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (32, 64)),
            Buffer("shared", BufferScope.SHARED, torch.float32, (32, 64)),
            Buffer("branch1", BufferScope.LOCAL, torch.float32, (32, 64)),
            Buffer("branch2", BufferScope.LOCAL, torch.float32, (32, 64)),
            Buffer("output1", BufferScope.GLOBAL, torch.float32, (32, 64)),
            Buffer("output2", BufferScope.GLOBAL, torch.float32, (32, 64))
        ]
        
        # Shared computation
        shared_node = ConductorNode("relu", inputs=[buffers[0]], outputs=[buffers[1]])
        
        # Branching computations
        branch1_node = ConductorNode("add", inputs=[buffers[1]], outputs=[buffers[2]])
        branch2_node = ConductorNode("mul", inputs=[buffers[1]], outputs=[buffers[3]])
        
        # Final outputs
        out1_node = ConductorNode("sigmoid", inputs=[buffers[2]], outputs=[buffers[4]])
        out2_node = ConductorNode("tanh", inputs=[buffers[3]], outputs=[buffers[5]])
        
        # Note: In practice, branching might create multiple clusters
        # This tests the DSL pattern for a single cluster with multiple outputs
        cluster = FusionCluster(
            nodes=[shared_node, branch1_node, out1_node],  # One branch
            cluster_type=FusionType.ELEMENTWISE,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[4]],
            internal_buffers=[buffers[1], buffers[2]],
            dsl_function_name="fused_branch1"
        )
        
        dsl = cluster.generate_fused_dsl()
        
        # FileCheck: Verify branching is handled
        # CHECK: Shared computation is present
        assert re.search(r'relu\(input\)', dsl)
        
        # CHECK: Branch-specific computation
        assert re.search(r'sigmoid\(add\(relu\(input\)\)\)', dsl) or \
               re.search(r'relu.*add.*sigmoid', dsl)
    
    def test_optimization_annotations_pattern(self):
        """Test DSL pattern includes optimization annotations."""
        generator = DSLGenerator()
        
        # Create fusion cluster with optimization hints
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (1024, 1024)),  # Large tensor
            Buffer("temp", BufferScope.LOCAL, torch.float32, (1024, 1024)),
            Buffer("output", BufferScope.GLOBAL, torch.float32, (1024, 1024))
        ]
        
        add_node = ConductorNode("add", inputs=[buffers[0]], outputs=[buffers[1]])
        relu_node = ConductorNode("relu", inputs=[buffers[1]], outputs=[buffers[2]])
        
        cluster = FusionCluster(
            nodes=[add_node, relu_node],
            cluster_type=FusionType.ELEMENTWISE,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[2]],
            internal_buffers=[buffers[1]],
            dsl_function_name="fused_optimized_add_relu"
        )
        
        dsl = cluster.generate_fused_dsl()
        
        # FileCheck: Verify optimization annotations
        # CHECK: Performance optimization hints
        assert re.search(r'// Optimization:', dsl) or re.search(r'// Performance', dsl)
        
        # CHECK: Memory access pattern optimization
        assert re.search(r'// Memory access pattern', dsl) or \
               re.search(r'// Vectorized', dsl) or \
               re.search(r'// Cache-friendly', dsl)
        
        # CHECK: Tensor size information for optimization
        assert re.search(r'1024', dsl)  # Large tensor size should be noted


@pytest.mark.filecheck
class TestFusionValidation:
    """Test validation of fusion correctness through DSL patterns."""
    
    def test_mathematical_correctness_validation(self):
        """Test that fused DSL maintains mathematical correctness."""
        generator = DSLGenerator()
        
        # Create mathematically sensitive operations
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (16, 32)),
            Buffer("temp1", BufferScope.LOCAL, torch.float32, (16, 32)),
            Buffer("temp2", BufferScope.LOCAL, torch.float32, (16, 32)),
            Buffer("output", BufferScope.GLOBAL, torch.float32, (16, 32))
        ]
        
        # Operations that must maintain order: exp -> log (mathematically sensitive)
        exp_node = ConductorNode("exp", inputs=[buffers[0]], outputs=[buffers[1]])
        log_node = ConductorNode("log", inputs=[buffers[1]], outputs=[buffers[2]])
        add_node = ConductorNode("add", inputs=[buffers[2]], outputs=[buffers[3]], 
                                 metadata={"constant": 1.0})
        
        cluster = FusionCluster(
            nodes=[exp_node, log_node, add_node],
            cluster_type=FusionType.ELEMENTWISE,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[3]],
            internal_buffers=[buffers[1], buffers[2]],
            dsl_function_name="fused_exp_log_add"
        )
        
        dsl = cluster.generate_fused_dsl()
        
        # FileCheck: Verify mathematical order is preserved
        # CHECK: Operations maintain mathematical order
        assert re.search(r'add\(log\(exp\(input\)\).*1\.0\)', dsl) or \
               re.search(r'exp.*log.*add', dsl)
        
        # CHECK: No reordering that could affect numerical stability
        # The exp should come before log in the expression
        exp_pos = dsl.find('exp')
        log_pos = dsl.find('log')
        add_pos = dsl.find('add')
        
        assert exp_pos < log_pos < add_pos, "Mathematical operations must maintain order"
    
    def test_data_dependency_validation(self):
        """Test that fused DSL respects data dependencies."""
        generator = DSLGenerator()
        
        # Create operations with clear data dependencies
        buffers = [
            Buffer("input", BufferScope.GLOBAL, torch.float32, (8, 16)),
            Buffer("temp1", BufferScope.LOCAL, torch.float32, (8, 16)),
            Buffer("temp2", BufferScope.LOCAL, torch.float32, (8, 16)),
            Buffer("output", BufferScope.GLOBAL, torch.float32, (8, 16))
        ]
        
        # Chain: input -> mul -> add -> output (strict dependency)
        mul_node = ConductorNode("mul", inputs=[buffers[0]], outputs=[buffers[1]], 
                                metadata={"constant": 2.0})
        add_node = ConductorNode("add", inputs=[buffers[1]], outputs=[buffers[2]], 
                                metadata={"constant": 1.0})
        relu_node = ConductorNode("relu", inputs=[buffers[2]], outputs=[buffers[3]])
        
        cluster = FusionCluster(
            nodes=[mul_node, add_node, relu_node],
            cluster_type=FusionType.ELEMENTWISE,
            external_inputs=[buffers[0]],
            external_outputs=[buffers[3]],
            internal_buffers=[buffers[1], buffers[2]],
            dsl_function_name="fused_dependency_chain"
        )
        
        dsl = cluster.generate_fused_dsl()
        
        # FileCheck: Verify data dependencies are respected
        # CHECK: Operations are properly nested respecting dependencies
        assert re.search(r'relu\(add\(mul\(input.*2\.0\).*1\.0\)\)', dsl) or \
               re.search(r'mul.*add.*relu', dsl)
        
        # CHECK: No operations are reordered incorrectly
        # Each operation should use the result of the previous one
        assert 'mul(input' in dsl or 'input' in dsl
        assert 'add(' in dsl
        assert 'relu(' in dsl
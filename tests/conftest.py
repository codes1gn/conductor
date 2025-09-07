"""
Pytest configuration and shared fixtures for Conductor tests.

This module provides common test fixtures, configuration, and utilities
used across the test suite.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Optional

import conductor
from conductor.graph.buffers import Buffer, BufferScope, BufferManager
from conductor.graph.graph_analyzer import ComputationDAG, GraphAnalyzer
from conductor.graph.graph_nodes import ConductorNode
from conductor.graph.fusion import FusionCluster, FusionEngine
from conductor.codegen.dslgen import ChoreoDslGen as DSLGenerator
from conductor.compiler.jit_compiler import JITCompiler
from conductor.compiler.loader import CompiledArtifact, ExecutableKernel


# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        'temp_dir': None,
        'mock_conductor_cli': True,
        'enable_performance_tests': False,
        'enable_hardware_tests': False,
    }


@pytest.fixture(scope="session")
def temp_test_dir(test_config):
    """Create temporary directory for test artifacts."""
    temp_dir = tempfile.mkdtemp(prefix="conductor_test_")
    test_config['temp_dir'] = temp_dir
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# Buffer fixtures
@pytest.fixture
def sample_buffer():
    """Create a sample buffer for testing."""
    return Buffer(
        name="test_buffer",
        scope=BufferScope.LOCAL,
        dtype=torch.float32,
        shape=(10, 10)
    )


@pytest.fixture
def buffer_manager():
    """Create a fresh BufferManager instance."""
    return BufferManager()


@pytest.fixture
def sample_buffers():
    """Create a set of sample buffers with different properties."""
    return {
        'input': Buffer("input", BufferScope.GLOBAL, torch.float32, (32, 128)),
        'temp1': Buffer("temp1", BufferScope.LOCAL, torch.float32, (32, 128)),
        'temp2': Buffer("temp2", BufferScope.LOCAL, torch.float32, (32, 128)),
        'output': Buffer("output", BufferScope.GLOBAL, torch.float32, (32, 128)),
        'shared': Buffer("shared", BufferScope.SHARED, torch.float32, (32, 128)),
    }


# Node fixtures
@pytest.fixture
def sample_node(sample_buffers):
    """Create a sample ConductorNode for testing."""
    return ConductorNode(
        op_name="relu",
        inputs=[sample_buffers['input']],
        outputs=[sample_buffers['output']]
    )


@pytest.fixture
def elementwise_chain_nodes(sample_buffers):
    """Create a chain of elementwise nodes for fusion testing."""
    nodes = [
        ConductorNode("add", inputs=[sample_buffers['input']], outputs=[sample_buffers['temp1']]),
        ConductorNode("mul", inputs=[sample_buffers['temp1']], outputs=[sample_buffers['temp2']]),
        ConductorNode("relu", inputs=[sample_buffers['temp2']], outputs=[sample_buffers['output']])
    ]
    
    # Set up buffer relationships
    sample_buffers['temp1'].producer = nodes[0]
    sample_buffers['temp1'].consumers = [nodes[1]]
    sample_buffers['temp2'].producer = nodes[1]
    sample_buffers['temp2'].consumers = [nodes[2]]
    
    return nodes


# DAG fixtures
@pytest.fixture
def sample_dag(elementwise_chain_nodes, sample_buffers):
    """Create a sample ComputationDAG for testing."""
    dag = ComputationDAG()
    
    # Add nodes
    for node in elementwise_chain_nodes:
        dag.add_node(node)
    
    # Add buffers
    for buffer in sample_buffers.values():
        dag.add_buffer(buffer)
    
    # Set inputs and outputs
    dag.inputs = [sample_buffers['input']]
    dag.outputs = [sample_buffers['output']]
    
    return dag


@pytest.fixture
def simple_dag():
    """Create a simple DAG with one operation for basic testing."""
    dag = ComputationDAG()
    
    input_buf = Buffer("input", BufferScope.GLOBAL, torch.float32, (10, 10))
    output_buf = Buffer("output", BufferScope.GLOBAL, torch.float32, (10, 10))
    
    relu_node = ConductorNode("relu", inputs=[input_buf], outputs=[output_buf])
    
    dag.add_node(relu_node)
    dag.add_buffer(input_buf)
    dag.add_buffer(output_buf)
    dag.inputs = [input_buf]
    dag.outputs = [output_buf]
    
    return dag


# Fusion fixtures
@pytest.fixture
def sample_fusion_cluster(elementwise_chain_nodes):
    """Create a sample FusionCluster for testing."""
    return FusionCluster(
        nodes=elementwise_chain_nodes[:2],  # First two nodes
        cluster_type="elementwise",
        external_inputs=[elementwise_chain_nodes[0].inputs[0]],
        external_outputs=[elementwise_chain_nodes[1].outputs[0]],
        internal_buffers=[elementwise_chain_nodes[0].outputs[0]],
        dsl_function_name="fused_add_mul"
    )


# Component fixtures
@pytest.fixture
def graph_analyzer():
    """Create a GraphAnalyzer instance."""
    return GraphAnalyzer()


@pytest.fixture
def fusion_engine():
    """Create a FusionEngine instance."""
    return FusionEngine()


@pytest.fixture
def dsl_generator():
    """Create a DSLGenerator instance."""
    return DSLGenerator()


@pytest.fixture
def jit_compiler(temp_test_dir):
    """Create a JITCompiler instance with temporary cache directory."""
    from conductor.choreo_jit import JITCompiler
    return JITCompiler(cache_dir=temp_test_dir)


@pytest.fixture
def aot_manager():
    """Create an AOTManager instance."""
    return AOTManager()


# PyTorch model fixtures
@pytest.fixture
def simple_torch_model():
    """Create a simple PyTorch model for testing."""
    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x)
    
    return SimpleModel()


@pytest.fixture
def elementwise_torch_model():
    """Create a PyTorch model with elementwise operations."""
    class ElementwiseModel(torch.nn.Module):
        def forward(self, x):
            y = x + 1.0
            z = y * 2.0
            return torch.relu(z)
    
    return ElementwiseModel()


@pytest.fixture
def linear_torch_model():
    """Create a PyTorch model with linear layer."""
    class LinearModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 20)
            
        def forward(self, x):
            return self.linear(x)
    
    return LinearModel()


@pytest.fixture
def complex_torch_model():
    """Create a more complex PyTorch model for integration testing."""
    class ComplexModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(128, 256)
            self.linear2 = torch.nn.Linear(256, 128)
            
        def forward(self, x):
            # Elementwise operations
            y = x + 1.0
            z = torch.relu(y)
            
            # Linear transformations
            h1 = self.linear1(z)
            h2 = torch.relu(h1)
            
            # More elementwise
            h3 = h2 * 0.5
            output = self.linear2(h3)
            
            # Reduction
            return torch.sum(output, dim=-1)
    
    return ComplexModel()


# FX Graph fixtures
@pytest.fixture
def simple_fx_graph(simple_torch_model):
    """Create a simple FX graph for testing."""
    return torch.fx.symbolic_trace(simple_torch_model)


@pytest.fixture
def elementwise_fx_graph(elementwise_torch_model):
    """Create an FX graph with elementwise operations."""
    return torch.fx.symbolic_trace(elementwise_torch_model)


@pytest.fixture
def complex_fx_graph(complex_torch_model):
    """Create a complex FX graph for integration testing."""
    return torch.fx.symbolic_trace(complex_torch_model)


# Mock fixtures
@pytest.fixture
def mock_conductor_cli():
    """Mock the Conductor CLI compiler."""
    with patch('conductor.runtime.jit.subprocess.run') as mock_run:
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Compilation successful",
            stderr=""
        )
        yield mock_run


@pytest.fixture
def mock_executable_kernel():
    """Mock ExecutableKernel for testing."""
    mock_kernel = Mock(spec=ExecutableKernel)
    mock_kernel.execute.return_value = torch.randn(10, 10)
    return mock_kernel


@pytest.fixture
def mock_compiled_artifact(temp_test_dir):
    """Mock CompiledArtifact for testing."""
    artifact_path = Path(temp_test_dir) / "test_artifact.so"
    artifact_path.touch()  # Create empty file
    
    return CompiledArtifact(
        path=str(artifact_path),
        artifact_type="shared_library",
        entry_point="main",
        metadata={
            "graph_hash": "test_hash_123",
            "node_count": 3,
            "fusion_clusters": 1
        }
    )


# Test data fixtures
@pytest.fixture
def sample_tensor_data():
    """Create sample tensor data for testing."""
    return {
        'small': torch.randn(4, 4),
        'medium': torch.randn(32, 128),
        'large': torch.randn(256, 512),
        'batch': torch.randn(16, 32, 64),
        'int_tensor': torch.randint(0, 100, (10, 10)),
        'bool_tensor': torch.randint(0, 2, (5, 5)).bool(),
    }


# Performance testing fixtures
@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        'warmup_runs': 3,
        'benchmark_runs': 10,
        'timeout_seconds': 30,
        'memory_limit_mb': 1024,
    }


# Utility functions
def create_test_model(operations: List[str], input_shape: tuple = (10, 10)):
    """
    Create a test model with specified operations.
    
    Args:
        operations: List of operation names to chain together
        input_shape: Shape of input tensor
        
    Returns:
        torch.nn.Module: Test model
    """
    class TestModel(torch.nn.Module):
        def __init__(self, ops):
            super().__init__()
            self.ops = ops
            
        def forward(self, x):
            result = x
            for op in self.ops:
                if op == 'add':
                    result = result + 1.0
                elif op == 'mul':
                    result = result * 2.0
                elif op == 'relu':
                    result = torch.relu(result)
                elif op == 'sigmoid':
                    result = torch.sigmoid(result)
                elif op == 'sum':
                    result = torch.sum(result, dim=-1)
                elif op == 'mean':
                    result = torch.mean(result, dim=-1)
                else:
                    raise ValueError(f"Unknown operation: {op}")
            return result
    
    return TestModel(operations)


def assert_dsl_contains_pattern(dsl_code: str, pattern: str, description: str = ""):
    """
    Assert that DSL code contains a specific pattern.
    
    Args:
        dsl_code: Generated DSL code
        pattern: Regex pattern to match
        description: Description of what the pattern checks
    """
    import re
    if not re.search(pattern, dsl_code):
        pytest.fail(f"DSL pattern check failed: {description}\nPattern: {pattern}\nDSL:\n{dsl_code}")


def assert_dsl_not_contains_pattern(dsl_code: str, pattern: str, description: str = ""):
    """
    Assert that DSL code does NOT contain a specific pattern.
    
    Args:
        dsl_code: Generated DSL code
        pattern: Regex pattern that should not match
        description: Description of what the pattern checks
    """
    import re
    if re.search(pattern, dsl_code):
        pytest.fail(f"DSL negative pattern check failed: {description}\nPattern: {pattern}\nDSL:\n{dsl_code}")


# Pytest hooks for test collection and reporting
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""
    for item in items:
        # Add markers based on test path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "filecheck" in str(item.fspath):
            item.add_marker(pytest.mark.filecheck)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Skip tests that require external dependencies
        if "requires_conductor" in item.keywords:
            if not shutil.which("conductor"):
                item.add_marker(pytest.mark.skip(reason="Conductor CLI not available"))
        
        if "requires_gcu" in item.keywords:
            # Skip GCU tests if hardware not available
            item.add_marker(pytest.mark.skip(reason="GCU hardware not available"))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for complete workflows"
    )
    config.addinivalue_line(
        "markers", "filecheck: FileCheck-style DSL validation tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and benchmarking tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "requires_conductor: Tests that require Conductor CLI compiler"
    )
    config.addinivalue_line(
        "markers", "requires_gcu: Tests that require GCU hardware"
    )


# Test result collection for coverage reporting
@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Set up test session and collect results."""
    # Initialize test session
    yield
    # Cleanup after all tests complete
#!/usr/bin/env python3
"""
Simple tests for graph analysis.
"""

import torch
import torch.nn as nn
import torch.fx as fx
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from conductor.graph.graph_analyzer import GraphAnalyzer


class SimpleModel(nn.Module):
    def forward(self, x, y):
        return x + y


def test_basic_graph_analysis():
    """Test that graph analysis works for simple operations."""
    model = SimpleModel()
    x = torch.randn(4, 8)
    y = torch.randn(4, 8)
    
    # Trace and analyze
    traced = fx.symbolic_trace(model)
    analyzer = GraphAnalyzer()
    dag = analyzer.parse_fx_graph(traced, example_inputs=[x, y])
    
    # Basic checks
    assert len(dag.nodes) == 1  # One add operation
    assert len(dag.inputs) == 2  # Two inputs
    assert len(dag.outputs) == 1  # One output
    assert dag.nodes[0].op_name == 'add'


def test_shape_inference_fix():
    """Test that the shape inference fix works correctly."""
    class MultiOpModel(nn.Module):
        def forward(self, x, y, z):
            return (x + y) * z
    
    model = MultiOpModel()
    x = torch.randn(16, 32)
    y = torch.randn(16, 32) 
    z = torch.randn(16, 32)
    
    # Trace and analyze
    traced = fx.symbolic_trace(model)
    analyzer = GraphAnalyzer()
    dag = analyzer.parse_fx_graph(traced, example_inputs=[x, y, z])
    
    # Check that all buffers have valid shapes (no None shapes)
    none_shapes = [buf for buf in dag.buffers if buf.shape is None]
    assert len(none_shapes) == 0, f"Found buffers with None shapes: {[buf.name for buf in none_shapes]}"
    
    # Check that intermediate buffers have correct shapes
    for buf in dag.buffers:
        if buf.shape:
            assert buf.shape == (16, 32), f"Buffer {buf.name} has wrong shape: {buf.shape}"


if __name__ == "__main__":
    test_basic_graph_analysis()
    test_shape_inference_fix()
    print("✅ All graph tests passed!")

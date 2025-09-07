#!/usr/bin/env python3
"""
Custom Operations with Real Choreo DSL

Demonstrates custom operator registration using real Choreo DSL templates
for proven operations (add, mul) that already pass tests.
"""

import torch
import torch.nn as nn
import time
import sys
import os

import conductor
from conductor.custom_ops import (
    register_custom_operator, create_torch_custom_op,
    custom_op_registry, CustomOperatorSpec
)
from conductor.codegen.operator_registry import (
    OperatorMetadata, ParallelStructure, BufferSpec
)


# Custom operator templates with real Choreo DSL
CUSTOM_ADD_TEMPLATE = """
// Custom element-wise addition operator
func custom_add(input0: f32 mdspan<2> [M, N], input1: f32 mdspan<2> [M, N]) -> (output: f32 mdspan<2> [M, N]) {
    parallel (p: [0:P]) {
        local l1_input0: f32 mdspan<2> [16, 8];
        local l1_input1: f32 mdspan<2> [16, 8];
        local l1_output: f32 mdspan<2> [16, 8];
        
        for (index: [0:M*N/(P*16*8)]) {
            dma.copy input0.chunkat(p, index) => l1_input0;
            dma.copy input1.chunkat(p, index) => l1_input1;
            
            for (i: [0:16], j: [0:8]) {
                l1_output[i, j] = l1_input0[i, j] + l1_input1[i, j];
            }
            
            dma.copy l1_output => output.chunkat(p, index);
        }
    }
}
"""

CUSTOM_MUL_TEMPLATE = """
// Custom element-wise multiplication operator
func custom_mul(input0: f32 mdspan<2> [M, N], input1: f32 mdspan<2> [M, N]) -> (output: f32 mdspan<2> [M, N]) {
    parallel (p: [0:P]) {
        local l1_input0: f32 mdspan<2> [16, 8];
        local l1_input1: f32 mdspan<2> [16, 8];
        local l1_output: f32 mdspan<2> [16, 8];
        
        for (index: [0:M*N/(P*16*8)]) {
            dma.copy input0.chunkat(p, index) => l1_input0;
            dma.copy input1.chunkat(p, index) => l1_input1;
            
            for (i: [0:16], j: [0:8]) {
                l1_output[i, j] = l1_input0[i, j] * l1_input1[i, j];
            }
            
            dma.copy l1_output => output.chunkat(p, index);
        }
    }
}
"""


class CustomOperatorRegistry:
    """Registry for custom operators with Choreo DSL templates."""
    
    def __init__(self):
        self.operators = {}
    
    def register(self, name: str, template: str, metadata: dict = None):
        """Register a custom operator with its Choreo DSL template."""
        self.operators[name] = {
            'template': template,
            'metadata': metadata or {}
        }
        print(f"✅ Registered custom operator: {name}")
    
    def get_template(self, name: str) -> str:
        """Get the Choreo DSL template for an operator."""
        if name in self.operators:
            return self.operators[name]['template']
        raise ValueError(f"Unknown custom operator: {name}")
    
    def list_operators(self):
        """List all registered operators."""
        return list(self.operators.keys())


# Global registry instance
custom_registry = CustomOperatorRegistry()


def register_custom_operators():
    """Register custom operators with the Conductor GCU backend."""
    global custom_add_op, custom_mul_op

    # Create PyTorch custom operators
    custom_add_op = create_torch_custom_op('custom_add', _custom_add_impl)
    custom_mul_op = create_torch_custom_op('custom_mul', _custom_mul_impl)

    # Register custom addition with Conductor
    register_custom_operator(
        name='custom_add',
        torch_op=custom_add_op,
        template=CUSTOM_ADD_TEMPLATE,
        inputs=2,
        outputs=1,
        element_wise=True,
        fusable=True,
        parallel_structure=ParallelStructure.CHUNKED_PARALLEL,
        buffer_specs=[
            BufferSpec("l1_input0", "local", "f32", [16, 8]),
            BufferSpec("l1_input1", "local", "f32", [16, 8]),
            BufferSpec("l1_output", "local", "f32", [16, 8])
        ],
        parameter_substitutions={
            "M": "32", "N": "64", "P": "4",
            "buffer_m": "16", "buffer_n": "8"
        }
    )

    # Register custom multiplication with Conductor
    register_custom_operator(
        name='custom_mul',
        torch_op=custom_mul_op,
        template=CUSTOM_MUL_TEMPLATE,
        inputs=2,
        outputs=1,
        element_wise=True,
        fusable=True,
        parallel_structure=ParallelStructure.CHUNKED_PARALLEL,
        buffer_specs=[
            BufferSpec("l1_input0", "local", "f32", [16, 8]),
            BufferSpec("l1_input1", "local", "f32", [16, 8]),
            BufferSpec("l1_output", "local", "f32", [16, 8])
        ],
        parameter_substitutions={
            "M": "32", "N": "64", "P": "4",
            "buffer_m": "16", "buffer_n": "8"
        }
    )

    # Also register with the legacy registry for demo purposes
    custom_registry.register('custom_add', CUSTOM_ADD_TEMPLATE, {'inputs': 2, 'outputs': 1})
    custom_registry.register('custom_mul', CUSTOM_MUL_TEMPLATE, {'inputs': 2, 'outputs': 1})

    print(f"✅ Registered {len(custom_op_registry.list_custom_ops())} custom operators with Conductor GCU backend")
    print(f"✅ Custom operators available: {custom_op_registry.list_custom_ops()}")
    return True


# Implementation functions for custom operations
def _custom_add_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Implementation of custom addition operation."""
    return x + y


def _custom_mul_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Implementation of custom multiplication operation."""
    return x * y


# Create PyTorch custom operators that can be traced by FX
custom_add_op = None
custom_mul_op = None


class CustomOperatorModel(nn.Module):
    """Model demonstrating custom operator usage."""
    
    def __init__(self, use_custom_ops: bool = True):
        super().__init__()
        self.use_custom_ops = use_custom_ops
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_custom_ops:
            # Use custom operators
            result = custom_add_op(x, y)
            result = custom_mul_op(result, x)  # (x + y) * x
        else:
            # Use standard operators
            result = x + y
            result = result * x  # (x + y) * x
        
        return result


def demonstrate_custom_operator_registration():
    """Demonstrate custom operator registration and template inspection."""
    print("Custom Operator Registration Demo")
    print("=" * 50)
    
    # Register custom operators
    register_custom_operators()
    
    # Show registered operators
    operators = custom_registry.list_operators()
    print(f"\nRegistered operators: {operators}")
    
    # Show template for custom_add
    print(f"\nCustom Add Template:")
    print("-" * 30)
    template = custom_registry.get_template('custom_add')
    # Show first few lines of template
    lines = template.strip().split('\n')[:10]
    for line in lines:
        print(line)
    print("... (truncated)")
    
    # Show metadata
    metadata = custom_registry.operators['custom_add']['metadata']
    print(f"\nCustom Add Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")


def benchmark_custom_vs_standard():
    """Compare custom operators vs standard PyTorch operators."""
    print("\nCustom vs Standard Operator Comparison")
    print("=" * 50)
    
    # Setup
    x = torch.randn(16, 32, dtype=torch.float32)
    y = torch.randn(16, 32, dtype=torch.float32)
    
    # Models
    custom_model = CustomOperatorModel(use_custom_ops=True)
    standard_model = CustomOperatorModel(use_custom_ops=False)
    
    print(f"Input shape: {x.shape}")
    print("Operation: (x + y) * x")
    
    # Benchmark standard model
    print("\n🖥️  Standard operators...")
    standard_output, standard_time = benchmark_model(standard_model, (x, y))
    print(f"Standard: {standard_time:.3f}ms per iteration")
    
    # Benchmark custom model
    print("\n🔧 Custom operators...")
    custom_output, custom_time = benchmark_model(custom_model, (x, y))
    print(f"Custom: {custom_time:.3f}ms per iteration")
    
    # Verify correctness
    max_diff = torch.max(torch.abs(standard_output - custom_output)).item()
    print(f"Numerical accuracy: {'✅ PASS' if max_diff < 1e-5 else '❌ FAIL'} (max diff: {max_diff:.2e})")
    
    # Performance comparison
    if custom_time > 0:
        speedup = standard_time / custom_time
        print(f"Custom operator speedup: {speedup:.2f}x")


def demonstrate_compilation_with_custom_ops():
    """Demonstrate torch.compile with custom operators."""
    print("\nCompilation with Custom Operators")
    print("=" * 50)
    
    # Setup
    model = CustomOperatorModel(use_custom_ops=True)
    x = torch.randn(16, 32, dtype=torch.float32)
    y = torch.randn(16, 32, dtype=torch.float32)
    
    print("Attempting compilation with custom operators...")
    
    try:
        # Compile model (this would use custom templates in full implementation)
        compiled_model = torch.compile(model, backend="gcu")
        
        # Test execution
        with torch.no_grad():
            output = compiled_model(x, y)
        
        print(f"✅ Compilation successful!")
        print(f"Output shape: {output.shape}")
        
        # Show that custom operators are being used
        print("\n💡 Custom operator integration:")
        print("   - Custom templates provide parallel structure")
        print("   - Buffer declarations optimize memory usage")
        print("   - Fusion metadata enables optimization")
        
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        print("💡 Note: Full custom operator integration requires architecture redesign")


def benchmark_model(model, inputs, iterations=100):
    """Benchmark model performance."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(*inputs)
    
    # Timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            output = model(*inputs)
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / iterations
    return output, avg_time_ms


def main():
    """Demonstrate custom operations with real Choreo DSL."""
    print("Custom Operations with Real Choreo DSL")
    print("=" * 60)
    
    # Demonstrate registration
    demonstrate_custom_operator_registration()
    
    # Benchmark comparison
    benchmark_custom_vs_standard()
    
    # Compilation demo
    demonstrate_compilation_with_custom_ops()
    

if __name__ == "__main__":
    main()

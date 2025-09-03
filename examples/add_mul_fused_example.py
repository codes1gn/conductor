#!/usr/bin/env python3
"""
Fused Add-Multiply with GCU Acceleration

Demonstrates torch.compile with GCU backend for fused operations.
Shows kernel fusion optimization: (x + y) * z
"""

import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import conductor


class FusedAddMulModel(nn.Module):
    """Fused add-multiply model: (x + y) * z"""
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return (x + y) * z


def benchmark_model(model, inputs, name="Model", iterations=100):
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


def compile_and_benchmark(model, inputs):
    """Compile model with GCU backend and benchmark."""
    compile_start = time.time()
    compiled_model = torch.compile(model, backend="gcu")
    compile_time_ms = (time.time() - compile_start) * 1000
    
    output, avg_time_ms = benchmark_model(compiled_model, inputs, "GCU")
    return output, avg_time_ms, compile_time_ms


def verify_correctness(cpu_output, gcu_output, tolerance=1e-5):
    """Verify numerical correctness between CPU and GCU outputs."""
    max_diff = torch.max(torch.abs(cpu_output - gcu_output)).item()
    return max_diff <= tolerance, max_diff


def show_debug_artifacts():
    """Show generated debug artifacts."""
    debug_dir = os.path.join(os.path.dirname(__file__), '..', 'debug_dir')
    if os.path.exists(debug_dir):
        files = [f for f in os.listdir(debug_dir) if os.path.isfile(os.path.join(debug_dir, f))]
        print(f"Debug artifacts: {len(files)} files in {debug_dir}")
    else:
        print("No debug artifacts found")


def main():
    """Demonstrate fused add-multiply with GCU acceleration."""
    print("Fused Add-Multiply with GCU Acceleration")
    print("=" * 50)
    
    # Setup
    model = FusedAddMulModel()
    x = torch.randn(16, 32, dtype=torch.float32)
    y = torch.randn(16, 32, dtype=torch.float32)
    z = torch.randn(16, 32, dtype=torch.float32)
    inputs = (x, y, z)
    print(f"Input shape: {x.shape}")
    print("Operation: (x + y) * z")
    
    # CPU baseline
    print("\n🖥️  CPU baseline...")
    cpu_output, cpu_time = benchmark_model(model, inputs, "CPU")
    print(f"CPU: {cpu_time:.3f}ms per iteration")
    
    # GCU compilation and execution
    print("\n🚀 GCU acceleration...")
    try:
        gcu_output, gcu_time, compile_time = compile_and_benchmark(model, inputs)
        print(f"GCU: {gcu_time:.3f}ms per iteration")
        print(f"Compilation: {compile_time:.2f}ms")
        
        # Verify correctness
        is_correct, max_diff = verify_correctness(cpu_output, gcu_output)
        print(f"Numerical accuracy: {'✅ PASS' if is_correct else '❌ FAIL'} (max diff: {max_diff:.2e})")
        
        # Performance summary
        speedup = cpu_time / gcu_time if gcu_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        show_debug_artifacts()
        return is_correct
        
    except Exception as e:
        print(f"❌ GCU execution failed: {e}")
        print("Note: Fallback to CPU execution may occur for complex operations")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

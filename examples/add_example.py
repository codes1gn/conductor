#!/usr/bin/env python3
"""
Element-wise Addition Example with GCU Acceleration

This example demonstrates how to use torch.compile with the GCU backend
for element-wise addition operations. It shows the complete end-to-end
workflow from PyTorch model definition to GCU execution.

Key Features:
- Simple element-wise addition operation
- torch.compile integration with GCU backend
- Performance comparison between CPU and GCU
- Numerical correctness verification
- Debug artifact inspection

Usage:
    python3 add_example.py
"""

import torch
import torch.nn as nn
import time
import sys
import os

# Add conductor to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import conductor backend
import conductor


class AdditionModel(nn.Module):
    """
    Simple model that performs element-wise addition.
    
    This model demonstrates the most basic GCU-accelerated operation:
    element-wise tensor addition.
    """
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Perform element-wise addition.

        Args:
            x: First input tensor
            y: Second input tensor

        Returns:
            Element-wise sum of x and y
        """
        return x + y


def run_cpu_baseline(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple:
    """Run the model on CPU for baseline performance."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, y)
    
    # Timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            output = model(x, y)
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / 100
    return output, avg_time_ms


def run_gcu_accelerated(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple:
    """Run the model with GCU acceleration through torch.compile."""
    model.eval()
    
    # Compile model with GCU backend
    print("🔧 Compiling model with GCU backend...")
    compile_start = time.time()
    
    try:
        compiled_model = torch.compile(model, backend="gcu")
        compile_end = time.time()
        compilation_time_ms = (compile_end - compile_start) * 1000
        
        print(f"✅ Compilation successful ({compilation_time_ms:.2f}ms)")
        
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        raise
    
    # Warmup compiled model
    with torch.no_grad():
        for _ in range(10):
            _ = compiled_model(x, y)
    
    # Timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            output = compiled_model(x, y)
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / 100
    return output, avg_time_ms, compilation_time_ms


def verify_correctness(cpu_output: torch.Tensor, gcu_output: torch.Tensor) -> bool:
    """Verify that GCU output matches CPU output."""
    if cpu_output.shape != gcu_output.shape:
        print(f"❌ Shape mismatch: CPU {cpu_output.shape} vs GCU {gcu_output.shape}")
        return False
    
    max_diff = torch.max(torch.abs(cpu_output - gcu_output)).item()
    mean_diff = torch.mean(torch.abs(cpu_output - gcu_output)).item()
    
    print(f"📊 Numerical comparison:")
    print(f"   Max difference: {max_diff:.2e}")
    print(f"   Mean difference: {mean_diff:.2e}")
    
    tolerance = 1e-5
    if max_diff <= tolerance:
        print(f"✅ Numerical correctness verified (tolerance: {tolerance:.2e})")
        return True
    else:
        print(f"❌ Numerical difference exceeds tolerance ({tolerance:.2e})")
        return False


def inspect_debug_artifacts():
    """Inspect generated debug artifacts."""
    debug_dir = os.path.join(os.path.dirname(__file__), '..', 'debug_dir')
    
    if os.path.exists(debug_dir):
        print(f"🔍 Debug artifacts in {debug_dir}:")
        for file in os.listdir(debug_dir):
            file_path = os.path.join(debug_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"   📄 {file} ({size} bytes)")
    else:
        print(f"📁 No debug artifacts found (directory {debug_dir} doesn't exist)")


def main():
    """Main function demonstrating element-wise addition with GCU acceleration."""
    print("Element-wise Addition Example with GCU Acceleration")
    print("=" * 60)
    
    # Configuration
    batch_size = 16
    feature_size = 32
    
    print(f"Configuration:")
    print(f"  Tensor shape: [{batch_size}, {feature_size}]")
    print(f"  Operation: element-wise addition (x + y)")
    print()
    
    # Create model and inputs
    print("🏗️  Setting up model and data...")
    model = AdditionModel()
    x = torch.randn(batch_size, feature_size, dtype=torch.float32)
    y = torch.randn(batch_size, feature_size, dtype=torch.float32)
    
    print(f"✅ Model created: AdditionModel")
    print(f"✅ Input tensors: x{x.shape}, y{y.shape}")
    print()
    
    # Run CPU baseline
    print("🖥️  Running CPU baseline...")
    try:
        cpu_output, cpu_time = run_cpu_baseline(model, x, y)
        print(f"✅ CPU execution: {cpu_time:.3f}ms per iteration")
    except Exception as e:
        print(f"❌ CPU execution failed: {e}")
        return False
    
    print()
    
    # Run GCU accelerated
    print("🚀 Running GCU accelerated...")
    try:
        gcu_output, gcu_time, compile_time = run_gcu_accelerated(model, x, y)
        print(f"✅ GCU execution: {gcu_time:.3f}ms per iteration")
        print(f"📊 Compilation overhead: {compile_time:.2f}ms")
    except Exception as e:
        print(f"❌ GCU execution failed: {e}")
        print(f"💡 This might be expected if GCU hardware is not available")
        return False
    
    print()
    
    # Verify correctness
    print("🔍 Verifying numerical correctness...")
    is_correct = verify_correctness(cpu_output, gcu_output)
    
    print()
    
    # Performance summary
    print("📈 Performance Summary")
    print("=" * 40)
    print(f"CPU time:     {cpu_time:.3f}ms")
    print(f"GCU time:     {gcu_time:.3f}ms")
    
    if gcu_time > 0:
        speedup = cpu_time / gcu_time
        print(f"Speedup:      {speedup:.2f}x")
    
    print(f"Compile time: {compile_time:.2f}ms")
    print()
    
    # Inspect debug artifacts
    inspect_debug_artifacts()
    print()
    
    # Final status
    if is_correct:
        print("🎯 Success! Element-wise addition with GCU acceleration working correctly.")
        print()
        print("💡 Key takeaways:")
        print("   - Element-wise addition successfully compiled for GCU")
        print("   - Numerical correctness maintained")
        print("   - Debug artifacts available for inspection")
        return True
    else:
        print("❌ Numerical correctness issues detected.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

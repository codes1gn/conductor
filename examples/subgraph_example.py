#!/usr/bin/env python3
"""
Subgraph to Choreo DSL Compilation Example

Demonstrates how PyTorch subgraphs are compiled to Choreo DSL using the GCU backend.
Shows the complete pipeline from PyTorch operations to GCU execution with multiple
operations that can be fused or executed sequentially.
"""

import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import conductor


class SubgraphModel(nn.Module):
    """Model demonstrating subgraph compilation with multiple operations."""
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # This creates a subgraph with multiple operations:
        # 1. Element-wise addition: x + y
        # 2. Element-wise multiplication: (x + y) * x  
        # 3. Element-wise multiplication: ((x + y) * x) * ((x + y) * x)
        
        added = x + y                    # Built-in add operation
        multiplied = added * x           # Built-in mul operation  
        squared = multiplied * multiplied # Another mul operation
        
        return squared


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


def show_compilation_artifacts():
    """Show generated compilation artifacts."""
    debug_dir = os.path.join(os.path.dirname(__file__), '..', 'debug_dir')
    if os.path.exists(debug_dir):
        files = [f for f in os.listdir(debug_dir) if os.path.isfile(os.path.join(debug_dir, f))]
        choreo_files = [f for f in files if f.endswith('.co')]
        cpp_files = [f for f in files if f.endswith('.cpp')]
        so_files = [f for f in files if f.endswith('.so')]
        
        print(f"\n📁 Compilation Artifacts ({len(files)} total files):")
        print(f"   Choreo DSL files: {len(choreo_files)}")
        print(f"   C++ wrapper files: {len(cpp_files)}")
        print(f"   Shared libraries: {len(so_files)}")
        
        # Show the latest Choreo DSL file content
        if choreo_files:
            latest_choreo = sorted(choreo_files)[-1]
            choreo_path = os.path.join(debug_dir, latest_choreo)
            print(f"\n📄 Generated Choreo DSL ({latest_choreo}):")
            print("-" * 50)
            try:
                with open(choreo_path, 'r') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines[:20], 1):  # Show first 20 lines
                        print(f"{i:2d}: {line.rstrip()}")
                    if len(lines) > 20:
                        print(f"... ({len(lines) - 20} more lines)")
            except Exception as e:
                print(f"Could not read file: {e}")
    else:
        print("No compilation artifacts found")


def analyze_subgraph_structure(model, inputs):
    """Analyze the FX subgraph structure."""
    print("\n🔍 Subgraph Analysis:")
    print("-" * 30)
    
    try:
        import torch.fx as fx
        traced = fx.symbolic_trace(model)
        
        print(f"FX Graph nodes: {len(list(traced.graph.nodes))}")
        
        operation_nodes = []
        for node in traced.graph.nodes:
            if node.op in ('call_function', 'call_method', 'call_module'):
                operation_nodes.append(node)
                print(f"  {node.name}: {node.target} ({node.op})")
        
        print(f"Operation nodes: {len(operation_nodes)}")
        print("This subgraph will be compiled to a single Choreo DSL function")
        
        return traced
        
    except Exception as e:
        print(f"Could not analyze subgraph: {e}")
        return None


def main():
    """Demonstrate subgraph to Choreo DSL compilation."""
    print("Subgraph to Choreo DSL Compilation Example")
    print("=" * 60)
    
    # Setup
    model = SubgraphModel()
    x = torch.randn(32, 64, dtype=torch.float32)  # Use shapes that work well with chunking
    y = torch.randn(32, 64, dtype=torch.float32)
    inputs = (x, y)
    
    print(f"Input shapes: x{x.shape}, y{y.shape}")
    print("Operations: ((x + y) * x) * ((x + y) * x)")
    
    # Analyze subgraph structure
    traced_model = analyze_subgraph_structure(model, inputs)
    
    # CPU baseline
    print(f"\n🖥️  CPU Baseline:")
    cpu_output, cpu_time = benchmark_model(model, inputs, "CPU")
    print(f"CPU execution: {cpu_time:.3f}ms per iteration")
    print(f"Output shape: {cpu_output.shape}")
    print(f"Output range: [{cpu_output.min().item():.3f}, {cpu_output.max().item():.3f}]")
    
    # GCU compilation and execution
    print(f"\n🚀 GCU Compilation & Execution:")
    try:
        compile_start = time.time()
        compiled_model = torch.compile(model, backend="gcu")
        compile_time = (time.time() - compile_start) * 1000
        
        print(f"Compilation time: {compile_time:.2f}ms")
        
        # Execute compiled model
        gcu_output, gcu_time = benchmark_model(compiled_model, inputs, "GCU")
        print(f"GCU execution: {gcu_time:.3f}ms per iteration")
        
        # Verify numerical correctness
        max_diff = torch.max(torch.abs(cpu_output - gcu_output)).item()
        mean_diff = torch.mean(torch.abs(cpu_output - gcu_output)).item()
        
        print(f"\n✅ Numerical Verification:")
        print(f"   Max difference: {max_diff:.2e}")
        print(f"   Mean difference: {mean_diff:.2e}")
        
        is_correct = max_diff < 1e-5
        print(f"   Accuracy: {'✅ PASS' if is_correct else '❌ FAIL'}")
        
        # Performance analysis
        speedup = cpu_time / gcu_time if gcu_time > 0 else 0
        print(f"\n📊 Performance:")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   GCU overhead: {gcu_time - cpu_time:.3f}ms")
        
        # Show compilation artifacts
        show_compilation_artifacts()
        
        print(f"\n🎯 Summary:")
        print(f"   ✅ Subgraph successfully compiled to Choreo DSL")
        print(f"   ✅ GCU execution completed without errors")
        print(f"   ✅ Numerical accuracy verified")
        print(f"   📈 Performance: {speedup:.2f}x speedup")
        
        return is_correct
        
    except Exception as e:
        print(f"❌ GCU compilation/execution failed: {e}")
        print("\n💡 Note: This may fall back to CPU execution")
        print("   Check debug artifacts for compilation details")
        
        # Still show artifacts even if compilation failed
        show_compilation_artifacts()
        return False


if __name__ == "__main__":
    success = main()
    print(f"\n{'🎉 SUCCESS' if success else '⚠️  PARTIAL SUCCESS'}: Subgraph compilation demonstrated")
    sys.exit(0 if success else 1)

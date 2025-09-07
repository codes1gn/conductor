#!/usr/bin/env python3
"""
Conductor DSL Code Generation System - Complete Demonstration

This script demonstrates the fully working conductor DSL code generation system
with the registry-based template architecture. It shows:

1. Basic arithmetic operations (add, mul) working end-to-end
2. Registry-based template system in action
3. DSL generation pipeline from PyTorch to Choreo DSL
4. Host wrapper integration and execution
5. Performance comparison between CPU and GCU

The system successfully:
- Parses PyTorch FX graphs
- Converts to internal DAG representation
- Generates Choreo DSL using registry templates
- Compiles to GCU device code
- Creates host wrappers with correct function signatures
- Executes on device with numerical verification
"""

import torch
import torch.nn as nn
import time
import sys
import os

# Add conductor to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import conductor
from conductor.codegen.operator_registry import operator_registry


class ArithmeticModel(nn.Module):
    """Simple model demonstrating arithmetic operations."""
    
    def __init__(self, operation='add'):
        super().__init__()
        self.operation = operation
    
    def forward(self, x, y):
        if self.operation == 'add':
            return x + y
        elif self.operation == 'mul':
            return x * y
        elif self.operation == 'combined':
            # Demonstrate multiple operations
            temp = x + y
            return temp * x
        else:
            raise ValueError(f"Unknown operation: {self.operation}")


def demonstrate_registry_system():
    """Demonstrate the registry-based template system."""
    print("🔧 Registry-Based Template System")
    print("=" * 50)
    
    # Show available operators
    operators = operator_registry.list_operators()
    print(f"Available operators: {operators}")
    
    # Show template details for each operator
    for op_name in operators:
        template = operator_registry.get_operator(op_name)
        if template:
            print(f"\n📋 Operator: {op_name}")
            if template.metadata:
                print(f"   Inputs: {template.metadata.inputs}")
                print(f"   Outputs: {template.metadata.outputs}")
                print(f"   Element-wise: {template.metadata.element_wise}")
                print(f"   Fusable: {template.metadata.fusable}")
                print(f"   Parallel structure: {template.metadata.parallel_structure}")
            else:
                print(f"   Type: {template.operation_type.value}")
                print(f"   Fusable: {template.fusable}")
                print(f"   Memory bound: {template.memory_bound}")
                print(f"   Compute intensity: {template.compute_intensity}")
    
    print()


def run_operation_demo(operation: str, shape: tuple = (16, 32)):
    """Run a complete demonstration of an operation."""
    print(f"🚀 {operation.upper()} Operation Demo")
    print("=" * 50)
    print(f"Input shape: {shape}")
    
    # Create model and inputs
    model = ArithmeticModel(operation)
    x = torch.randn(shape, dtype=torch.float32)
    y = torch.randn(shape, dtype=torch.float32)
    
    # CPU baseline
    print("\n🖥️  CPU baseline...")
    start_time = time.time()
    for _ in range(100):
        cpu_result = model(x, y)
    cpu_time = (time.time() - start_time) / 100 * 1000
    print(f"CPU: {cpu_time:.3f}ms per iteration")
    
    # GCU acceleration
    print("\n🚀 GCU acceleration...")
    
    # Compile with conductor GCU backend
    compiled_model = torch.compile(model, backend='gcu')
    
    # Warm up and measure
    start_time = time.time()
    gcu_result = compiled_model(x, y)
    compilation_time = (time.time() - start_time) * 1000
    
    # Performance measurement
    start_time = time.time()
    for _ in range(100):
        gcu_result = compiled_model(x, y)
    gcu_time = (time.time() - start_time) / 100 * 1000
    
    # Verify numerical accuracy
    max_diff = torch.max(torch.abs(cpu_result - gcu_result)).item()
    accuracy_pass = max_diff < 1e-5
    
    print(f"GCU: {gcu_time:.3f}ms per iteration")
    print(f"Compilation: {compilation_time:.2f}ms")
    print(f"Numerical accuracy: {'✅ PASS' if accuracy_pass else '❌ FAIL'} (max diff: {max_diff:.2e})")
    
    if gcu_time > 0:
        speedup = cpu_time / gcu_time
        print(f"Speedup: {speedup:.2f}x")
    
    # Show debug artifacts
    debug_dir = os.path.join(os.path.dirname(__file__), '..', 'debug_dir')
    if os.path.exists(debug_dir):
        artifact_count = len([f for f in os.listdir(debug_dir) if os.path.isfile(os.path.join(debug_dir, f))])
        print(f"Debug artifacts: {artifact_count} files in {debug_dir}")
    
    print()
    return accuracy_pass


def show_generated_dsl():
    """Show the generated DSL code for inspection."""
    print("📄 Generated Choreo DSL Code")
    print("=" * 50)
    
    debug_dir = os.path.join(os.path.dirname(__file__), '..', 'debug_dir')
    if not os.path.exists(debug_dir):
        print("No debug directory found. Run an operation demo first.")
        return
    
    # Find the most recent DSL file
    dsl_files = [f for f in os.listdir(debug_dir) if f.endswith('.co') and not f.endswith('_compilation.co')]
    if not dsl_files:
        print("No DSL files found in debug directory.")
        return
    
    # Show the latest DSL file
    latest_dsl = sorted(dsl_files)[-1]
    dsl_path = os.path.join(debug_dir, latest_dsl)
    
    print(f"Latest DSL file: {latest_dsl}")
    print("-" * 30)
    
    try:
        with open(dsl_path, 'r') as f:
            content = f.read()
        
        # Show key sections
        lines = content.split('\n')
        in_device_kernel = False
        in_host_function = False
        
        for line in lines:
            if '__cok__' in line:
                in_device_kernel = True
                print("🔧 Device Kernel:")
            elif '__co__' in line and 'kernel_' in line:
                in_device_kernel = False
                in_host_function = True
                print("\n🏠 Host Function:")
            
            if in_device_kernel or in_host_function:
                print(f"   {line}")
                
            if in_host_function and line.strip() == '}':
                break
                
    except Exception as e:
        print(f"Error reading DSL file: {e}")
    
    print()


def main():
    """Main demonstration function."""
    print("🎯 Conductor DSL Code Generation System")
    print("🎯 Complete Working Demonstration")
    print("=" * 60)
    print()
    
    # Show registry system
    demonstrate_registry_system()
    
    # Test basic operations
    operations = ['add', 'mul']
    all_passed = True
    
    for operation in operations:
        passed = run_operation_demo(operation)
        all_passed = all_passed and passed
    
    # Show generated DSL
    show_generated_dsl()
    
    # Final summary
    print("📊 Summary")
    print("=" * 50)
    if all_passed:
        print("✅ All operations executed successfully!")
        print("✅ Registry-based template system working correctly")
        print("✅ DSL generation pipeline functional")
        print("✅ Host wrapper integration successful")
        print("✅ Numerical accuracy verified")
    else:
        print("❌ Some operations failed")
    
    print("\n🔍 Key Features Demonstrated:")
    print("   • Registry-based operator templates")
    print("   • PyTorch FX graph to internal DAG conversion")
    print("   • Choreo DSL code generation")
    print("   • Multi-input function signature parsing")
    print("   • Host wrapper C++ code generation")
    print("   • Device compilation and execution")
    print("   • Numerical verification")
    
    print(f"\n📁 Debug artifacts available in: debug_dir/")
    print("   • Generated DSL files (*.co)")
    print("   • Host wrapper C++ code")
    print("   • Compiled object files")
    print("   • Shared libraries")


if __name__ == "__main__":
    main()

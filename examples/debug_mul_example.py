#!/usr/bin/env python3
"""
Debug-enabled version of mul_example.py with comprehensive tracing.

This example demonstrates the complete debug tracing system showing:
1. Input FX Graph Module representation
2. Internal DAG representation  
3. Generated Choreo DSL code
4. Host wrapper integration details

Run with: CONDUCTOR_DEBUG=1 python examples/debug_mul_example.py
"""

import os
import sys
import torch
import time

# Enable debug tracing
os.environ['CONDUCTOR_DEBUG'] = '1'
os.environ['CONDUCTOR_DEBUG_FX'] = '1'
os.environ['CONDUCTOR_DEBUG_DAG'] = '1'
os.environ['CONDUCTOR_DEBUG_DSL'] = '1'
os.environ['CONDUCTOR_DEBUG_WRAPPER'] = '1'
os.environ['CONDUCTOR_DEBUG_META'] = '1'
os.environ['CONDUCTOR_DEBUG_FLOW'] = '1'

import conductor
from conductor.config.debug_tracer import get_debug_tracer, enable_debug_tracing, DebugTraceConfig

def simple_mul_model(x, y):
    """Simple multiplication model for testing."""
    return x * y

def main():
    print("Element-wise Multiplication with Comprehensive Debug Tracing")
    print("=" * 65)
    
    # Enable debug tracing with simplified configuration
    debug_config = DebugTraceConfig(
        enabled=True,
        max_tensor_elements=50,
        indent_size=2
    )
    enable_debug_tracing(debug_config)
    
    # Create test inputs with different shape to avoid cache
    input_shape = [8, 16]  # Different from add example
    print(f"Input shape: {input_shape}")
    
    # Create input tensors
    x = torch.randn(input_shape, dtype=torch.float32)
    y = torch.randn(input_shape, dtype=torch.float32)
    
    print(f"\n🖥️  CPU baseline...")
    # CPU baseline
    cpu_start = time.time()
    cpu_result = simple_mul_model(x, y)
    cpu_time = time.time() - cpu_start
    print(f"CPU: {cpu_time*1000:.3f}ms per iteration")
    
    print(f"\n🚀 GCU acceleration with debug tracing...")
    
    # Compile with GCU backend (this will trigger all debug tracing)
    gcu_start = time.time()
    compiled_model = torch.compile(simple_mul_model, backend="gcu")
    compilation_time = time.time() - gcu_start
    
    # Execute compiled model
    exec_start = time.time()
    gcu_result = compiled_model(x, y)
    exec_time = time.time() - exec_start
    
    print(f"GCU: {exec_time*1000:.3f}ms per iteration")
    print(f"Compilation: {compilation_time*1000:.2f}ms")
    
    # Verify numerical accuracy
    max_diff = torch.max(torch.abs(cpu_result - gcu_result)).item()
    print(f"Numerical accuracy: {'✅ PASS' if max_diff < 1e-5 else '❌ FAIL'} (max diff: {max_diff:.2e})")
    
    # Calculate speedup
    speedup = cpu_time / exec_time if exec_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")
    
    # Get debug tracer and show summary
    debug_tracer = get_debug_tracer()
    if debug_tracer.is_enabled():
        debug_tracer.print_section_header("Debug Tracing Summary", 1)
        print(f"Total debug sections printed: {debug_tracer.section_counter}")
        print(f"Trace data collected: {len(debug_tracer.trace_data)} items")
        
        # Save trace data to file
        debug_file = os.path.join(os.path.dirname(__file__), '..', 'debug_dir', 'debug_trace_mul.json')
        debug_tracer.save_trace_to_file(debug_file)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple debug tracing example that will successfully compile.

This example demonstrates the complete debug tracing system showing:
1. Input FX Graph Module representation
2. Internal DAG representation  
3. Generated Choreo DSL code
4. Host wrapper integration details

Run with: CONDUCTOR_DEBUG=1 python examples/debug_simple_add.py
"""

import os
import sys
import torch
import time
import tempfile
import shutil

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

def simple_add(x, y):
    """Simple addition model for testing."""
    return x + y

def main():
    print("Simple Addition with Comprehensive Debug Tracing")
    print("=" * 55)
    
    # Clear any existing cache
    cache_dirs = [d for d in os.listdir('/tmp') if d.startswith('conductor_cache_')]
    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(os.path.join('/tmp', cache_dir))
            print(f"Cleared cache: {cache_dir}")
        except:
            pass
    
    # Enable debug tracing with simplified configuration
    debug_config = DebugTraceConfig(
        enabled=True,
        max_tensor_elements=20,
        indent_size=2
    )
    enable_debug_tracing(debug_config)
    
    # Create test inputs with unique shape
    input_shape = [2, 4]  # Small unique shape
    print(f"Input shape: {input_shape}")
    
    # Create input tensors
    x = torch.randn(input_shape, dtype=torch.float32)
    y = torch.randn(input_shape, dtype=torch.float32)
    
    print(f"\n🖥️  CPU baseline...")
    # CPU baseline
    cpu_start = time.time()
    cpu_result = simple_add(x, y)
    cpu_time = time.time() - cpu_start
    print(f"CPU: {cpu_time*1000:.3f}ms per iteration")
    
    print(f"\n🚀 GCU acceleration with debug tracing...")
    
    # Compile with GCU backend (this will trigger all debug tracing)
    gcu_start = time.time()
    compiled_model = torch.compile(simple_add, backend="gcu")
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
        debug_file = os.path.join(os.path.dirname(__file__), '..', 'debug_dir', 'debug_trace_simple_add.json')
        debug_tracer.save_trace_to_file(debug_file)

if __name__ == "__main__":
    main()

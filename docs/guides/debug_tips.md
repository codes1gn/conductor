# Comprehensive Debug Tracing Guide for Conductor

This guide explains how to use the comprehensive debug tracing system implemented in Conductor to debug end-to-end compilation pipeline issues.

## Overview

The debug tracing system provides complete visibility into the compilation pipeline, showing:

1. **Input FX Graph Module** - Complete torch.fx.GraphModule representation
2. **Internal DAG Representation** - Conductor's internal computation graph
3. **Generated Choreo DSL Code** - Complete DSL with kernels and host functions
4. **Host Wrapper Integration** - C++ wrapper code and memory management

## Quick Start

### Enable Debug Tracing

Set environment variables to enable debug tracing:

```bash
export CONDUCTOR_DEBUG=1
export CONDUCTOR_DEBUG_FX=1
export CONDUCTOR_DEBUG_DAG=1
export CONDUCTOR_DEBUG_DSL=1
export CONDUCTOR_DEBUG_WRAPPER=1
export CONDUCTOR_DEBUG_META=1
export CONDUCTOR_DEBUG_FLOW=1
```

### Run Your Example

```bash
python examples/debug_simple_add.py
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CONDUCTOR_DEBUG` | Enable/disable debug tracing | `0` |
| `CONDUCTOR_DEBUG_FX` | Show FX graph module details | `1` |
| `CONDUCTOR_DEBUG_DAG` | Show internal DAG representation | `1` |
| `CONDUCTOR_DEBUG_DSL` | Show generated Choreo DSL code | `1` |
| `CONDUCTOR_DEBUG_WRAPPER` | Show host wrapper integration | `1` |
| `CONDUCTOR_DEBUG_META` | Show metadata and tensor info | `1` |
| `CONDUCTOR_DEBUG_FLOW` | Show data flow connections | `1` |
| `CONDUCTOR_DEBUG_MAX_ELEMENTS` | Max tensor elements to display | `100` |
| `CONDUCTOR_DEBUG_INDENT` | Indentation size for JSON output | `2` |

## Debug Output Sections

### 1. FX Graph Module

Shows the complete PyTorch FX graph representation:

```
================================================================================
=== FX GRAPH MODULE ===
================================================================================
Graph Structure:
graph():
    %l_x_ : torch.Tensor [num_users=1] = placeholder[target=L_x_]
    %l_y_ : torch.Tensor [num_users=1] = placeholder[target=L_y_]
    %add : [num_users=1] = call_function[target=operator.add](args = (%l_x_, %l_y_), kwargs = {})
    return (add,)

------------------------------------------------------------
--- Generated Code ---
------------------------------------------------------------
def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
    l_x_ = L_x_
    l_y_ = L_y_
    add = l_x_ + l_y_;  l_x_ = l_y_ = None
    return (add,)

------------------------------------------------------------
--- Node Details ---
------------------------------------------------------------
Node 0: l_x_
  Op: placeholder
  Target: L_x_
  Args: ()
  Kwargs: {}
  Meta: {...}

------------------------------------------------------------
--- Input Tensor Information ---
------------------------------------------------------------
Input 0:
  Shape: torch.Size([2, 4])
  Dtype: torch.float32
  Device: cpu
  Requires grad: False
  Data: tensor([...])
```

### 2. Internal DAG Representation

Shows Conductor's internal computation graph:

```
================================================================================
=== INTERNAL DAG REPRESENTATION ===
================================================================================
Total nodes: 1
Input buffers: 2
Output buffers: 1

------------------------------------------------------------
--- Input Buffers ---
------------------------------------------------------------
Input 0: l_x_
  Shape: (2, 4)
  Dtype: torch.float32
  Scope: BufferScope.GLOBAL

------------------------------------------------------------
--- Output Buffers ---
------------------------------------------------------------
Output 0: add
  Shape: (2, 4)
  Dtype: torch.float32
  Scope: BufferScope.GLOBAL

------------------------------------------------------------
--- Computation Nodes ---
------------------------------------------------------------
Node 0: add
  Inputs: [...]
  Outputs: [...]
  Metadata: {...}
  Type: Elementwise operation

------------------------------------------------------------
--- Data Flow Connections ---
------------------------------------------------------------
Data Flow Graph:
  Node_0_add -> (no connections)
```

### 3. Generated Choreo DSL Code

Shows the complete generated DSL implementation:

```
================================================================================
=== GENERATED CHOREO DSL CODE ===
================================================================================

------------------------------------------------------------
--- Complete DSL Implementation ---
------------------------------------------------------------
// Generated Choreo DSL
// Auto-generated from PyTorch FX Graph via Conductor

#include "choreo.h"

__co__ auto kernel_6788ff95(f32 [2, 4] l_x_, f32 [2, 4] l_y_) {
  f32 [2, 4] add;
  parallel p by 4 {
    foreach index in [4] {
      lf = dma.copy.async l_x_.chunkat(p, index) => local;
      rf = dma.copy.async l_y_.chunkat(p, index) => local;
      wait lf, rf;
      
      local f32 [lf.span] l1_out;
      
      foreach i in [l1_out.span] {
        l1_out.at(i) = lf.at(i) + rf.at(i);
      }
      
      dma.copy l1_out => add.chunkat(p, index);
    }
  }
  return add;
}

------------------------------------------------------------
--- Device Kernel Code ---
------------------------------------------------------------
__cok__ {
// Device kernel for relu
template<int size>
__co_device__ void relu_kernel(float* input, float* output) {
  for (int i = 0; i < size; ++i) {
    output[i] = input[i] > 0.0f ? input[i] : 0.0f;
  }
}
}
```

### 4. Host Wrapper Integration

Shows the C++ host wrapper and memory management:

```
================================================================================
=== HOST WRAPPER INTEGRATION ===
================================================================================

------------------------------------------------------------
--- Function Signature ---
------------------------------------------------------------
Generated signature: choreo::spanned_data<choreo::f32, 2> kernel_6788ff95(const choreo::spanned_view<choreo::f32, 2>& input_0, const choreo::spanned_view<choreo::f32, 2>& input_1)

------------------------------------------------------------
--- Buffer Allocation & Memory Management ---
------------------------------------------------------------
Buffer: input_0
  Size: 32 bytes
  Alignment: default
  Memory type: device
  Dtype: f32
  Shape: [2, 4]

Buffer: input_1
  Size: 32 bytes
  Alignment: default
  Memory type: device
  Dtype: f32
  Shape: [2, 4]

Buffer: output
  Size: 32 bytes
  Alignment: default
  Memory type: device
  Dtype: f32
  Shape: [2, 4]

------------------------------------------------------------
--- Critical Host Wrapper Code ---
------------------------------------------------------------
Data Marshalling & Execution Code:
   45: // Convert PyTorch tensors to GCU buffers
   46: auto gcu_input_0 = tensor_to_gcu_buffer(input_0);
   47: auto gcu_input_1 = tensor_to_gcu_buffer(input_1);
   48: auto gcu_output = allocate_buffer(output_shape, output_dtype);
   49: 
   50: // Launch kernel
   51: auto result = kernel_6788ff95(gcu_input_0, gcu_input_1);
   52: 
   53: // Convert result back to PyTorch tensor
   54: return gcu_buffer_to_tensor(result);
```

## Programmatic Usage

You can also enable debug tracing programmatically:

```python
import os
from conductor.config.debug_tracer import enable_debug_tracing, DebugTraceConfig

# Enable with custom configuration
config = DebugTraceConfig(
    enabled=True,
    max_tensor_elements=50,
    indent_size=2
)
enable_debug_tracing(config)

# Your torch.compile code here
compiled_model = torch.compile(model, backend="gcu")
result = compiled_model(x, y)
```

## Debug Files

Debug tracing also saves files to the `debug_dir/` directory:

- `kernel_*.co` - Generated Choreo DSL source
- `host_wrapper_*.cpp` - Generated C++ host wrapper
- `kernel_*.o` - Compiled object file
- `kernel_*.so` - Compiled shared library
- `debug_trace_*.json` - Complete debug trace data

## Common Use Cases

### 1. Debugging Compilation Failures

When compilation fails, the debug output shows exactly where the issue occurs:

```bash
CONDUCTOR_DEBUG=1 python your_example.py
```

Look for error messages in the DSL generation or compilation sections.

### 2. Debugging Runtime Shape Mismatches

The debug output shows the exact shapes at each stage:

- FX graph shows PyTorch tensor shapes
- Internal DAG shows buffer shapes
- Host wrapper shows GCU buffer shapes

### 3. Debugging Performance Issues

The debug output shows:

- Parallel factors and chunking strategies
- Memory allocation patterns
- DMA operation sequences

### 4. Debugging Numerical Accuracy

Compare the generated DSL operations with expected computations to identify numerical issues.

## Tips and Best Practices

1. **Start with a simple example** - Use small tensor shapes for easier debugging
2. **Clear cache when debugging** - Set `CONDUCTOR_DEBUG=1` to disable caching
3. **Focus on specific sections** - Disable sections you don't need with environment variables
4. **Save debug output** - Redirect output to files for detailed analysis
5. **Use programmatic configuration** - For fine-grained control over debug output

## Integration with Existing Examples

All existing examples can be run with debug tracing:

```bash
# Debug the add example
CONDUCTOR_DEBUG=1 python examples/add_example.py

# Debug the mul example  
CONDUCTOR_DEBUG=1 python examples/mul_example.py

# Debug with custom configuration
CONDUCTOR_DEBUG=1 CONDUCTOR_DEBUG_MAX_ELEMENTS=20 python examples/add_example.py
```

The debug tracing system is designed to be non-intrusive and can be enabled/disabled without modifying your code.

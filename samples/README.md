# Conductor DSL Samples

This directory contains sample Conductor DSL (.co) files that demonstrate the output of the Conductor PyTorch Backend Integration. These files show how different PyTorch operations and fusion patterns are converted to Conductor DSL.

## Directory Structure

```
samples/
├── elementwise/          # Elementwise operation examples
│   ├── add.co           # Addition operation
│   ├── mul.co           # Multiplication operation
│   ├── relu.co          # ReLU activation
│   └── gelu.co          # GELU activation
├── reduction/           # Reduction operation examples
│   ├── sum.co           # Sum reduction
│   ├── mean.co          # Mean reduction
│   └── max.co           # Max reduction
├── fused/              # Fused operation examples
│   ├── add_relu.co     # Fused addition + ReLU
│   ├── linear_gelu.co  # Fused linear + GELU
│   └── conv_bn_relu.co # Fused convolution + batch norm + ReLU
├── complex/            # Complex operation patterns
│   ├── attention.co    # Multi-head attention
│   ├── layer_norm.co   # Layer normalization
│   └── transformer_block.co # Complete transformer block
└── optimized/          # Optimized patterns
    ├── buffer_reuse.co # Buffer reuse optimization
    ├── memory_layout.co # Memory layout optimization
    └── kernel_fusion.co # Advanced kernel fusion
```

## DSL Syntax Overview

Conductor DSL uses a simple, readable syntax for describing computations:

### Basic Operations
```
// Elementwise operations
result = add(input1, input2);
result = mul(input, 2.0);
result = relu(input);

// Reductions
result = sum(input, dim=1);
result = mean(input, dim=-1);

// Matrix operations
result = matmul(input1, input2);
result = transpose(input, dim1=0, dim2=1);
```

### Buffer Declarations
```
// Buffer scope declarations
local temp1: float32[batch_size, hidden_size];
shared intermediate: float32[batch_size, seq_len, hidden_size];
global weights: float32[hidden_size, output_size];
```

### Function Definitions
```
function fused_linear_relu(input: float32[N, D], weight: float32[D, H], bias: float32[H]) -> float32[N, H] {
    temp = matmul(input, weight);
    temp_biased = add(temp, bias);
    result = relu(temp_biased);
    return result;
}
```

## Sample Files

### Elementwise Operations

These samples show how basic elementwise operations are represented in Conductor DSL.

**Key Features:**
- Simple operation mapping
- Type preservation
- Shape inference
- Memory layout optimization

### Reduction Operations

Examples of reduction operations that aggregate data along specific dimensions.

**Key Features:**
- Dimension specification
- Keepdim handling
- Numerical stability
- Parallel reduction strategies

### Fused Operations

Demonstrate how multiple operations are combined into single kernels for better performance.

**Key Features:**
- Operation clustering
- Intermediate buffer elimination
- Kernel launch reduction
- Memory bandwidth optimization

### Complex Patterns

Show how high-level PyTorch operations are decomposed and optimized.

**Key Features:**
- Multi-step decomposition
- Advanced fusion heuristics
- Memory scope management
- Performance optimization

### Optimized Patterns

Examples of advanced optimizations applied by the Conductor backend.

**Key Features:**
- Buffer reuse strategies
- Memory layout transformations
- Loop fusion and tiling
- Hardware-specific optimizations

## Usage

These sample files serve multiple purposes:

1. **Reference**: Understand how PyTorch operations map to Conductor DSL
2. **Testing**: Validate DSL generation correctness using FileCheck
3. **Documentation**: Learn Conductor DSL syntax and patterns
4. **Debugging**: Compare generated DSL against expected patterns

### Generating Your Own Samples

You can generate DSL samples from your PyTorch models:

```python
import torch
import conductor
from conductor.codegen import DSLGenerator

# Create a simple model
model = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 1)
)

# Trace the model
example_input = torch.randn(5, 10)
traced_model = torch.fx.symbolic_trace(model)

# Generate DSL (this would be done internally by the backend)
# This is for illustration - actual API may differ
try:
    from conductor.codegen import GraphAnalyzer, FusionEngine, DSLGenerator
    
    analyzer = GraphAnalyzer()
    fusion_engine = FusionEngine()
    generator = DSLGenerator()
    
    # Process graph
    dag = analyzer.parse_fx_graph(traced_model)
    clusters = fusion_engine.identify_fusion_opportunities(dag)
    dsl_code = generator.generate_dsl_file(dag)
    
    # Save to file
    with open('my_model.co', 'w') as f:
        f.write(dsl_code)
        
    print("DSL generated successfully!")
    
except Exception as e:
    print(f"DSL generation not yet implemented: {e}")
```

### Validating DSL Files

Use FileCheck to validate DSL structure:

```bash
# Install LLVM FileCheck (if available)
# FileCheck < samples/elementwise/add.co

# Or use Python-based validation
python -c "
import re
with open('samples/elementwise/add.co', 'r') as f:
    content = f.read()
    
# Check for expected patterns
assert 'function' in content
assert 'add(' in content
assert 'return' in content
print('✓ DSL validation passed')
"
```

## Contributing Samples

When contributing new sample files:

1. **Follow naming conventions**: Use descriptive names that match the operation or pattern
2. **Include comments**: Explain complex patterns and optimizations
3. **Add metadata**: Include information about the source PyTorch code
4. **Validate syntax**: Ensure DSL is syntactically correct
5. **Document patterns**: Explain the optimization strategy used

### Sample File Template

```
// File: samples/category/operation_name.co
// Description: Brief description of the operation or pattern
// Source: PyTorch code that generates this DSL
// Optimizations: List of optimizations applied

// Buffer declarations
local input: float32[batch_size, input_size];
local output: float32[batch_size, output_size];

// Function definition
function operation_name(input: float32[N, D]) -> float32[N, H] {
    // Operation implementation
    result = operation(input);
    return result;
}

// Main execution
output = operation_name(input);
```

## Learning Resources

- [Conductor DSL Specification](../docs/dsl_specification.md) - Complete DSL syntax reference
- [Fusion Patterns Guide](../docs/guides/fusion_patterns.md) - Understanding fusion optimization
- [Performance Optimization](../docs/guides/performance.md) - DSL performance best practices
- [FileCheck Testing](../docs/development/testing.md#filecheck-tests) - Validating DSL generation

## Notes

- These samples represent the target output of the Conductor backend
- Actual DSL generation is implemented in the codegen module
- Files are provided for reference and testing purposes
- DSL syntax may evolve as the project develops
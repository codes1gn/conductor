# GCU Acceleration Examples

This directory contains clear, operator-based examples demonstrating how to use PyTorch with GCU acceleration through the Conductor framework.

## Example Files

### Basic Operations
- **`add_example.py`** - Element-wise addition (x + y)
- **`mul_example.py`** - Element-wise multiplication (x * y)

### Fused Operations
- **`add_mul_fused_example.py`** - Fused add-multiply ((x + y) * z)

### Complete Integration
- **`torch_compile_complete_example.py`** ⭐ **RECOMMENDED** - Comprehensive neural network example

## Requirements

### Hardware & Software
- **GCU Hardware**: Physical GCU device with runtime libraries (for full execution)
- **Choreo Compiler**: Available in PATH (`choreo` command)
- **Python**: 3.8+ with PyTorch 2.0+
- **Conductor**: Properly installed and configured

### Environment Setup
```bash
# Verify choreo compiler
choreo --help

# Install conductor
pip install -e .
```

## Usage

Each example can be run independently:

```bash
# Run basic addition example
python3 examples/add_example.py

# Run multiplication example
python3 examples/mul_example.py

# Run fused operations example
python3 examples/add_mul_fused_example.py

# Run complete integration example
python3 examples/torch_compile_complete_example.py
```

## What Each Example Demonstrates

### End-to-End Functionality
All examples demonstrate complete end-to-end functionality:
1. **Model Definition** - Simple PyTorch models with specific operations
2. **torch.compile Integration** - Using `torch.compile(model, backend="gcu")`
3. **Performance Comparison** - CPU vs GCU execution timing
4. **Numerical Correctness** - Verification that results match
5. **Debug Artifacts** - Inspection of generated compilation artifacts

### Key Features
- **Clear Operation Focus** - Each example focuses on specific operations
- **Production-Ready Patterns** - Proper error handling and logging
- **Debug Visibility** - Shows how to inspect generated artifacts
- **Performance Analysis** - Timing and speedup calculations

## Usage Patterns

### torch.compile Integration (Recommended)
```python
import torch
import conductor  # Registers 'gcu' backend

# Compile model for GCU
compiled_model = torch.compile(model, backend='gcu')

# Execute (automatically uses GCU)
result = compiled_model(x, y)
```

### Production Error Handling
```python
def safe_gcu_execution(model, inputs):
    try:
        # Try GCU execution
        compiled_model = torch.compile(model, backend='gcu')
        return compiled_model(*inputs)
    except Exception as e:
        if "undefined symbol" in str(e):
            # GCU runtime not available - fallback to CPU
            return model(*inputs)
        else:
            raise  # Re-raise unexpected errors
```

## Debug Artifacts

All examples generate debug artifacts in the `debug_dir/` directory:
- **`.choreo` files** - Generated Choreo DSL source code
- **`.cpp` files** - Host wrapper C++ code
- **`.o` files** - Compiled object files
- **`.so` files** - Final shared libraries

## Expected Behavior

### With GCU Hardware
- Compilation succeeds
- Execution succeeds with potential speedup
- Numerical correctness maintained

### Without GCU Hardware
- Compilation succeeds (host wrapper compilation works)
- Execution may fail with runtime errors (expected)
- Debug artifacts still generated for inspection

## Troubleshooting

### Common Issues

#### 1. "Choreo compiler not found"
```bash
# Install choreo compiler and add to PATH
export PATH="/path/to/choreo/bin:$PATH"
```

#### 2. "undefined symbol" errors
```
This indicates GCU runtime libraries are not available.
The compilation pipeline is working correctly.
Would succeed on actual GCU hardware.
```

#### 3. "GCU backend not registered"
```python
# Ensure conductor is imported to register backend
import conductor
```

### Debug Mode
```bash
# Enable debug logging
export CONDUCTOR_LOG_LEVEL=DEBUG
python3 examples/add_example.py
```

## Naming Convention

Examples follow a clear naming convention:
- **Single operations**: `{operator}_example.py` (e.g., `add_example.py`)
- **Fused operations**: `{op1}_{op2}_fused_example.py` (e.g., `add_mul_fused_example.py`)
- **Complex examples**: Descriptive names (e.g., `torch_compile_complete_example.py`)

---

**Ready to accelerate your PyTorch models on GCU hardware!** 🚀

# GCU Acceleration Examples

Concise examples demonstrating PyTorch with GCU acceleration through the Conductor framework.

## Examples

- **`add_example.py`** - Element-wise addition (x + y)
- **`mul_example.py`** - Element-wise multiplication (x * y)
- **`add_mul_fused_example.py`** - Fused operations ((x + y) * z)
- **`custom_operations.py`** - Custom operator registration and usage

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

```bash
python3 examples/add_example.py
python3 examples/mul_example.py
python3 examples/add_mul_fused_example.py
python3 examples/custom_operations.py
```

## Features

Each example demonstrates:
- **torch.compile Integration** - `torch.compile(model, backend="gcu")`
- **Performance Comparison** - CPU vs GCU timing
- **Numerical Correctness** - Result verification
- **Debug Artifacts** - Compilation artifact inspection

## Basic Usage

```python
import torch
import conductor

# Compile and run
model = MyModel()
compiled_model = torch.compile(model, backend='gcu')
result = compiled_model(inputs)
```

## Troubleshooting

- **"Choreo compiler not found"**: Add choreo to PATH
- **"undefined symbol"**: GCU runtime not available (expected without hardware)
- **Debug mode**: `export CONDUCTOR_LOG_LEVEL=DEBUG`

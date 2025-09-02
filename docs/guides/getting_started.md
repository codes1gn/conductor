# Getting Started with Conductor

This guide will help you get up and running with the Conductor PyTorch Backend Integration quickly.

## What is Conductor?

Conductor is a PyTorch backend integration that enables seamless execution of ML models on custom 'gcu' hardware through the Conductor compiler. It provides:

- **Zero Learning Curve**: Drop-in replacement using standard `torch.compile` API
- **Performance First**: Intelligent operation fusion and optimized memory management
- **Developer Friendly**: Minimalist design with clear, readable code
- **Production Ready**: Robust JIT/AOT modes with fallback mechanisms

## Installation

### Prerequisites

- Python ≥3.8
- PyTorch ≥2.0
- Conductor CLI compiler (for compilation)

### Install from PyPI

```bash
pip install conductor
```

### Install from Source

```bash
git clone https://github.com/conductor/conductor-pytorch.git
cd conductor-pytorch
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/conductor/conductor-pytorch.git
cd conductor-pytorch
pip install -e ".[dev]"
pre-commit install
```

## Basic Usage

### 1. Import and Register Backend

```python
import torch
import conductor  # Automatically registers 'gcu' backend
```

The backend is automatically registered when you import the conductor package.

### 2. Compile Your Model

```python
# Define your model
model = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 1)
)

# Compile for GCU hardware
compiled_model = torch.compile(model, backend='gcu')
```

### 3. Use as Normal

```python
# Create input data
x = torch.randn(5, 10)

# Execute compiled model
output = compiled_model(x)
```

## Compilation Modes

### JIT Mode (Default)

JIT (Just-In-Time) mode compiles models dynamically during execution:

```python
# JIT compilation (default)
compiled_model = torch.compile(model, backend='gcu')
```

**Pros:**
- No pre-compilation step required
- Great for development and experimentation
- Automatic caching of compilation results

**Cons:**
- First execution includes compilation overhead
- Compilation happens at runtime

### AOT Mode

AOT (Ahead-Of-Time) mode uses precompiled artifacts:

```python
# AOT compilation
compiled_model = torch.compile(model, backend='gcu', mode='aot')
```

**Pros:**
- Minimal startup overhead
- Predictable performance
- Ideal for production deployment

**Cons:**
- Requires pre-compilation step
- Less flexible for dynamic models

## Checking Backend Status

### Verify Backend Registration

```python
import torch
import conductor

# Check if backend is available
available_backends = torch._dynamo.list_backends()
print('gcu' in available_backends)  # Should print True

# Get backend information
info = conductor.get_backend_info()
print(f"Backend version: {info['version']}")
print(f"Supported operations: {len(info['supported_ops'])}")
```

### List Supported Operations

```python
import conductor

# Get list of supported operations
supported_ops = conductor.list_supported_operations()
print(f"Supported operations: {supported_ops}")
```

## Configuration

### Backend Configuration

```python
import conductor

# Configure backend settings
conductor.configure_backend({
    'fusion_enabled': True,
    'cache_enabled': True,
    'fallback_enabled': True,
    'log_level': 'INFO'
})
```

### Environment Variables

You can also configure Conductor using environment variables:

```bash
export CONDUCTOR_FUSION_ENABLED=true
export CONDUCTOR_CACHE_DIR=/tmp/conductor_cache
export CONDUCTOR_LOG_LEVEL=DEBUG
export CONDUCTOR_FALLBACK_ENABLED=true
```

## Performance Tips

### 1. Enable Fusion

Fusion combines multiple operations into single kernels for better performance:

```python
# Fusion is enabled by default, but you can configure it
conductor.configure_backend({'fusion_enabled': True})
```

### 2. Use Caching

Compilation results are cached automatically to avoid recompilation:

```python
# Caching is enabled by default
conductor.configure_backend({'cache_enabled': True})
```

### 3. Warm Up Models

For production use, warm up your models to avoid first-run compilation overhead:

```python
# Warm up the model
with torch.no_grad():
    dummy_input = torch.randn(1, 10)
    _ = compiled_model(dummy_input)  # Triggers compilation
    
# Now subsequent calls will be fast
for batch in dataloader:
    output = compiled_model(batch)
```

## Error Handling and Fallback

Conductor automatically falls back to the Inductor backend when operations are unsupported:

```python
import torch
import conductor

model = MyComplexModel()
compiled_model = torch.compile(model, backend='gcu')

# If unsupported operations are encountered,
# Conductor will automatically fallback to Inductor
try:
    output = compiled_model(input_data)
    print("Execution successful")
except Exception as e:
    print(f"Error: {e}")
    # Check logs for fallback information
```

## Next Steps

- Read the [Advanced Usage Guide](advanced_usage.md) for more features
- Check out [Examples](../../examples/) for complete working examples
- See [API Reference](../api/) for detailed API documentation
- Visit [Troubleshooting](troubleshooting.md) if you encounter issues

## Common Patterns

### Model Training

```python
import torch
import conductor

model = MyModel()
compiled_model = torch.compile(model, backend='gcu')

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        output = compiled_model(batch.x)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()
```

### Model Inference

```python
import torch
import conductor

# Load pretrained model
model = torch.load('model.pth')
model.eval()

# Compile for inference
compiled_model = torch.compile(model, backend='gcu')

# Inference loop
with torch.no_grad():
    for batch in test_dataloader:
        predictions = compiled_model(batch)
        # Process predictions...
```
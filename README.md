# Conductor = Choreo + Inductor 

A minimalist Torch inductor backend that enables Choreo DSL for GCU hardware.

## Features

- **Zero Learning Curve**: Drop-in replacement using standard `torch.compile` API
- **Performance First**: Intelligent operation fusion and optimized memory management  
- **Developer Friendly**: Minimalist design with clear, readable code and comprehensive documentation
- **Production Ready**: Robust JIT/AOT modes with fallback mechanisms

## Quick Start

### Installation

```bash
pip install conductor
```

### Basic Usage

```python
import torch
import conductor  # Automatically registers 'gcu' backend

# Define your model
model = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 1)
)

# Compile for GCU hardware
compiled_model = torch.compile(model, backend='gcu')

# Use as normal
x = torch.randn(5, 10)
output = compiled_model(x)
```

## Architecture

Conductor provides a clean pipeline from PyTorch FX Graph to executable GCU code:

1. **Graph Analysis**: Parse FX Graph into internal DAG representation
2. **Fusion Optimization**: Apply heuristics to cluster compatible operations  
3. **Buffer Management**: Intelligent scoping (local/shared/global) for memory optimization
4. **DSL Generation**: Emit Conductor DSL (.co files) with topological ordering
5. **Compilation**: Invoke Conductor CLI to generate executable artifacts
6. **Runtime Integration**: Load and execute compiled kernels on GCU hardware

## Development

### Setup Development Environment

```bash
git clone https://github.com/codes1gn/conductor.git
cd conductor
pip install -e ".[dev]"
pre-commit install # optional for pre-commit hooks
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=conductor

# Run specific test categories
pytest -m unit
pytest -m integration
```

### Code Quality

```bash
# Format code
black conductor/
isort conductor/

# Type checking
mypy conductor/

# Linting
flake8 conductor/
```

## Requirements

- Python >=3.8
- PyTorch >=2.0
- Conductor CLI compiler (for compilation)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development/contributing.md) for details.

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Getting Started Guide](docs/guides/getting_started.md)** - Quick start for new users
- **[Advanced Usage](docs/guides/advanced_usage.md)** - Advanced features and customization
- **[API Reference](docs/api/)** - Complete API documentation
- **[Examples](examples/)** - Working code examples and demonstrations
- **[Troubleshooting](docs/guides/troubleshooting.md)** - Common issues and solutions
- **[Contributing Guide](docs/development/contributing.md)** - How to contribute to the project

## Examples

The `examples/` directory contains comprehensive examples:

- **[Basic Usage](examples/basic_usage.py)** - Simple torch.compile integration
- **[Advanced Fusion](examples/advanced_fusion.py)** - Fusion optimization strategies
- **[Benchmarking](examples/benchmarking.py)** - Performance comparison and analysis
- **[Custom Operations](examples/custom_operations.py)** - Extending with custom operations

## Sample DSL Files

The `samples/` directory contains example Choreo DSL files showing how PyTorch operations are impl'd by 
Choreo Language:

- **Elementwise Operations** - Basic operations like add, mul, relu
- **Fused Operations** - Optimized fusion patterns
- **Complex Patterns** - Advanced operations like attention mechanisms

## Support

- **Documentation**: [docs/](docs/) directory
- **Issues**: [GitHub Issues](https://github.com/conductor/conductor-pytorch/issues)
- **Discussions**: [GitHub Discussions](https://github.com/conductor/conductor-pytorch/discussions)

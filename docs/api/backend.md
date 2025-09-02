# Backend API Reference

The backend module provides the main interface for registering and configuring the Conductor PyTorch backend.

## Functions

### register_backend()

Registers the Conductor 'gcu' backend with PyTorch's compilation system.

```python
def register_backend() -> None
```

**Description:**
Automatically called when importing the conductor package. Registers the 'gcu' backend so it can be used with `torch.compile(backend='gcu')`.

**Raises:**
- `RuntimeError`: If PyTorch version is incompatible
- `ImportError`: If required dependencies are missing

**Example:**
```python
import conductor  # Automatically calls register_backend()

# Or call manually
conductor.register_backend()
```

### is_backend_registered()

Checks if the Conductor backend is registered with PyTorch.

```python
def is_backend_registered() -> bool
```

**Returns:**
- `bool`: True if backend is registered, False otherwise

**Example:**
```python
import conductor

if conductor.is_backend_registered():
    print("Backend is ready to use")
else:
    print("Backend registration failed")
```

### get_backend_info()

Returns information about the Conductor backend.

```python
def get_backend_info() -> Dict[str, Any]
```

**Returns:**
- `Dict[str, Any]`: Backend information including:
  - `version`: Conductor version
  - `supported_ops`: List of supported operations
  - `compilation_modes`: Available compilation modes
  - `device_support`: Supported device types

**Example:**
```python
import conductor

info = conductor.get_backend_info()
print(f"Version: {info['version']}")
print(f"Supported operations: {len(info['supported_ops'])}")
```

### list_supported_operations()

Returns a list of operations supported by the Conductor backend.

```python
def list_supported_operations() -> List[str]
```

**Returns:**
- `List[str]`: List of supported operation names

**Example:**
```python
import conductor

ops = conductor.list_supported_operations()
print(f"Supported operations: {ops}")

# Check if specific operation is supported
if 'conv2d' in ops:
    print("Convolution operations are supported")
```

### configure_backend()

Configures the Conductor backend with custom settings.

```python
def configure_backend(config: Dict[str, Any]) -> None
```

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary

**Configuration Options:**

#### Fusion Configuration
```python
{
    'fusion_config': {
        'elementwise_fusion': bool,      # Enable elementwise fusion (default: True)
        'reduction_fusion': bool,        # Enable reduction fusion (default: True)
        'memory_bound_fusion': bool,     # Enable memory-bound fusion (default: True)
        'max_fusion_size': int,          # Maximum operations per fusion (default: 10)
        'fusion_threshold': float        # Fusion benefit threshold (default: 0.8)
    }
}
```

#### Buffer Configuration
```python
{
    'buffer_config': {
        'auto_scope_promotion': bool,    # Auto promote buffer scopes (default: True)
        'buffer_reuse_enabled': bool,    # Enable buffer reuse (default: True)
        'memory_pool_size': str,         # Memory pool size (default: '1GB')
        'temp_buffer_limit': int         # Max temporary buffers (default: 100)
    }
}
```

#### Compilation Configuration
```python
{
    'compilation_config': {
        'optimization_level': str,       # O0, O1, O2, O3 (default: 'O2')
        'debug_symbols': bool,           # Include debug symbols (default: False)
        'parallel_compilation': bool,    # Parallel compilation (default: True)
        'compilation_timeout': int       # Timeout in seconds (default: 300)
    }
}
```

#### Cache Configuration
```python
{
    'cache_config': {
        'cache_enabled': bool,           # Enable caching (default: True)
        'cache_dir': str,                # Cache directory (default: auto)
        'max_cache_size': str,           # Max cache size (default: '5GB')
        'eviction_policy': str           # 'lru', 'fifo' (default: 'lru')
    }
}
```

#### Debug Configuration
```python
{
    'debug_mode': bool,                  # Enable debug mode (default: False)
    'log_level': str,                    # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'save_intermediate_files': bool,     # Save debug files (default: False)
    'debug_output_dir': str              # Debug output directory
}
```

**Example:**
```python
import conductor

# Configure for development
conductor.configure_backend({
    'debug_mode': True,
    'log_level': 'DEBUG',
    'fusion_config': {
        'max_fusion_size': 5  # Smaller fusions for debugging
    },
    'compilation_config': {
        'optimization_level': 'O1',  # Faster compilation
        'debug_symbols': True
    }
})

# Configure for production
conductor.configure_backend({
    'debug_mode': False,
    'log_level': 'WARNING',
    'fusion_config': {
        'max_fusion_size': 15  # Aggressive fusion
    },
    'compilation_config': {
        'optimization_level': 'O3',  # Maximum optimization
        'debug_symbols': False
    }
})
```

## Classes

### ConductorBackend

The main backend implementation class.

```python
class ConductorBackend:
    """Conductor PyTorch backend implementation."""
```

#### Methods

##### __call__(graph_module, example_inputs)

Compiles a PyTorch FX graph for execution on GCU hardware.

```python
def __call__(
    self, 
    graph_module: torch.fx.GraphModule, 
    example_inputs: List[torch.Tensor]
) -> Callable
```

**Parameters:**
- `graph_module` (torch.fx.GraphModule): FX graph to compile
- `example_inputs` (List[torch.Tensor]): Example input tensors

**Returns:**
- `Callable`: Compiled function for execution

**Raises:**
- `UnsupportedOperationError`: If graph contains unsupported operations
- `CompilationError`: If compilation fails
- `DeviceError`: If GCU device is not available

**Example:**
```python
import torch
import conductor

# This is typically called automatically by torch.compile
backend = conductor.ConductorBackend()

# Manual compilation (not recommended for normal use)
compiled_fn = backend(graph_module, example_inputs)
result = compiled_fn(*example_inputs)
```

## Environment Variables

The backend can also be configured using environment variables:

- `CONDUCTOR_FUSION_ENABLED`: Enable/disable fusion (true/false)
- `CONDUCTOR_CACHE_DIR`: Cache directory path
- `CONDUCTOR_LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `CONDUCTOR_DEBUG_MODE`: Enable debug mode (true/false)
- `CONDUCTOR_FALLBACK_ENABLED`: Enable fallback to Inductor (true/false)
- `CONDUCTOR_OPTIMIZATION_LEVEL`: Compilation optimization level (O0/O1/O2/O3)

**Example:**
```bash
export CONDUCTOR_DEBUG_MODE=true
export CONDUCTOR_LOG_LEVEL=DEBUG
export CONDUCTOR_CACHE_DIR=/tmp/conductor_cache
python my_script.py
```

## Usage Examples

### Basic Usage

```python
import torch
import conductor

# Define model
model = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 1)
)

# Compile with Conductor backend
compiled_model = torch.compile(model, backend='gcu')

# Use normally
x = torch.randn(5, 10)
output = compiled_model(x)
```

### Advanced Configuration

```python
import torch
import conductor

# Configure for specific workload
conductor.configure_backend({
    'fusion_config': {
        'elementwise_fusion': True,
        'max_fusion_size': 8
    },
    'buffer_config': {
        'buffer_reuse_enabled': True,
        'memory_pool_size': '2GB'
    },
    'compilation_config': {
        'optimization_level': 'O3',
        'parallel_compilation': True
    }
})

# Compile model
compiled_model = torch.compile(model, backend='gcu')
```

### Error Handling

```python
import torch
import conductor

try:
    compiled_model = torch.compile(model, backend='gcu')
    output = compiled_model(input_data)
except conductor.UnsupportedOperationError as e:
    print(f"Unsupported operation: {e.operation}")
    # Fallback will be automatic if enabled
except conductor.CompilationError as e:
    print(f"Compilation failed: {e}")
    # Check logs for details
except Exception as e:
    print(f"Unexpected error: {e}")
```
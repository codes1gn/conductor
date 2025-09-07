# Conductor Configuration System

The Conductor framework uses a simplified, unified configuration system that replaces the previous complex environment variable approach with easy-to-use JSON or YAML configuration files.

## Configuration Files

The configuration system supports both JSON and YAML formats:

- **Default location**: `conductor/config/conductor_config.yaml` (preferred) or `conductor/config/conductor_config.json`
- **Custom location**: Specify a custom path when loading configuration

## Configuration Sections

### Debug Configuration
```yaml
debug:
  enabled: false              # Enable/disable debug tracing
  max_tensor_elements: 100    # Max elements to show in tensor dumps
  indent_size: 2              # Indentation for debug output
```

### Cache Configuration
```yaml
cache:
  enabled: true               # Enable/disable compilation caching
  max_size_mb: 1024          # Maximum cache size in megabytes
  cache_dir: null            # Custom cache directory (null = system temp)
```

### Compilation Configuration
```yaml
compilation:
  timeout_seconds: 300        # Compilation timeout
  optimization_level: "O2"    # Optimization level
  enable_fusion: true         # Enable operation fusion
```

### Runtime Configuration
```yaml
runtime:
  device: "gcu"              # Target device
  memory_pool_size_mb: 512   # Memory pool size
  enable_profiling: false    # Enable runtime profiling
```

### Logging Configuration
```yaml
logging:
  level: "INFO"              # Log level
  enable_file_logging: false # Enable file logging
  log_file: "conductor.log"  # Log file name
```

## Usage

### Basic Usage
```python
import conductor

# Get the global configuration
config = conductor.get_config()

# Check if debug is enabled
if config.is_debug_enabled():
    print("Debug mode is active")

# Check if caching is enabled
if config.is_cache_enabled():
    print("Caching is enabled")
```

### Custom Configuration
```python
from conductor.config import load_config

# Load from custom file
config = load_config("my_config.yaml")

# Use the custom configuration
conductor.set_config(config)
```

### Environment Variable Overrides

Some settings can be overridden with environment variables:

- `CONDUCTOR_DEBUG=1` - Enable debug mode
- `CONDUCTOR_DISABLE_CACHE=1` - Disable caching

## Migration from Old System

The new system replaces these old environment variables:

| Old Environment Variable | New Configuration |
|--------------------------|-------------------|
| `CONDUCTOR_DEBUG` | `debug.enabled` |
| `CONDUCTOR_DEBUG_FX` | Removed (unified debug) |
| `CONDUCTOR_DEBUG_DAG` | Removed (unified debug) |
| `CONDUCTOR_DEBUG_DSL` | Removed (unified debug) |
| `CONDUCTOR_DEBUG_WRAPPER` | Removed (unified debug) |
| `CONDUCTOR_DEBUG_META` | Removed (unified debug) |
| `CONDUCTOR_DEBUG_FLOW` | Removed (unified debug) |

## Benefits

1. **Simplified Interface**: Single unified debug switch instead of multiple granular options
2. **Easy Cache Control**: Clear enable/disable option for caching
3. **Standard Formats**: JSON/YAML support for easy editing
4. **Predictable Location**: Fixed location in `conductor/config/` directory
5. **Developer Friendly**: Easy to read, modify, and version control

# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Conductor PyTorch Backend Integration.

## Common Issues

### Backend Registration Issues

#### Problem: Backend not found
```
ERROR: Conductor 'gcu' backend not found!
```

**Causes and Solutions:**

1. **Package not imported**
   ```python
   # ❌ Wrong
   import torch
   torch.compile(model, backend='gcu')  # Backend not registered
   
   # ✅ Correct
   import torch
   import conductor  # This registers the backend
   torch.compile(model, backend='gcu')
   ```

2. **Import error during registration**
   ```python
   # Check for import errors
   try:
       import conductor
       print("✓ Conductor imported successfully")
   except ImportError as e:
       print(f"✗ Import failed: {e}")
   ```

3. **PyTorch version incompatibility**
   ```bash
   # Check PyTorch version
   python -c "import torch; print(torch.__version__)"
   
   # Conductor requires PyTorch >= 2.0
   pip install "torch>=2.0"
   ```

#### Problem: Registration fails silently
```python
import conductor
available_backends = torch._dynamo.list_backends()
print('gcu' in available_backends)  # Returns False
```

**Solution:**
```python
# Check registration status
import conductor

if not conductor.is_backend_registered():
    print("Backend not registered, attempting manual registration...")
    try:
        conductor.register_backend()
        print("✓ Backend registered successfully")
    except Exception as e:
        print(f"✗ Registration failed: {e}")
```

### Compilation Issues

#### Problem: Unsupported operation error
```
UnsupportedOperationError: Unsupported operation 'custom_op': Not implemented in Conductor DSL
```

**Solutions:**

1. **Check supported operations**
   ```python
   import conductor
   
   supported_ops = conductor.list_supported_operations()
   print(f"Supported operations: {supported_ops}")
   
   # Check if your operation is supported
   if 'custom_op' not in supported_ops:
       print("Operation not supported, will fallback to Inductor")
   ```

2. **Enable fallback mechanism**
   ```python
   # Ensure fallback is enabled (default)
   conductor.configure_backend({'fallback_enabled': True})
   
   # Compilation will automatically fallback for unsupported ops
   compiled_model = torch.compile(model, backend='gcu')
   ```

3. **Add custom operation support**
   ```python
   from conductor.codegen import register_custom_operation
   
   @register_custom_operation('custom_op')
   def custom_op_converter(node, inputs, outputs, metadata):
       return f"custom_op({', '.join(inputs)}) -> {outputs[0]}"
   ```

#### Problem: Compilation timeout
```
CompilationError: Compilation timed out after 300 seconds
```

**Solutions:**

1. **Increase timeout**
   ```python
   conductor.configure_backend({
       'compilation_config': {
           'compilation_timeout': 600  # 10 minutes
       }
   })
   ```

2. **Reduce model complexity**
   ```python
   # Split large models into smaller parts
   class LargeModel(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.part1 = ModelPart1()
           self.part2 = ModelPart2()
       
       def forward(self, x):
           # Compile parts separately
           x = self.compiled_part1(x)
           x = self.compiled_part2(x)
           return x
   
   model = LargeModel()
   model.compiled_part1 = torch.compile(model.part1, backend='gcu')
   model.compiled_part2 = torch.compile(model.part2, backend='gcu')
   ```

#### Problem: Out of memory during compilation
```
CompilationError: Out of memory during DSL generation
```

**Solutions:**

1. **Reduce fusion aggressiveness**
   ```python
   conductor.configure_backend({
       'fusion_config': {
           'max_fusion_size': 5,  # Reduce from default 10
           'fusion_threshold': 0.9  # Increase threshold
       }
   })
   ```

2. **Limit buffer usage**
   ```python
   conductor.configure_backend({
       'buffer_config': {
           'temp_buffer_limit': 50,  # Reduce from default 100
           'memory_pool_size': '512MB'  # Reduce from default 1GB
       }
   })
   ```

### Runtime Issues

#### Problem: Execution fails after successful compilation
```
RuntimeError: Failed to execute compiled kernel
```

**Solutions:**

1. **Check device availability**
   ```python
   from conductor.device import get_gcu_interface
   
   try:
       gcu = get_gcu_interface()
       devices = gcu.list_devices()
       if not devices:
           print("No GCU devices available")
       else:
           print(f"Available devices: {devices}")
   except Exception as e:
       print(f"Device interface error: {e}")
   ```

2. **Verify tensor placement**
   ```python
   # Ensure tensors are on correct device
   x = x.to('gcu:0')  # Move to GCU device
   model = model.to('gcu:0')
   
   compiled_model = torch.compile(model, backend='gcu')
   output = compiled_model(x)
   ```

3. **Enable debug mode**
   ```python
   conductor.configure_backend({
       'debug_mode': True,
       'log_level': 'DEBUG'
   })
   
   # This will provide detailed execution logs
   output = compiled_model(input_data)
   ```

#### Problem: Performance degradation
```
Model runs slower with Conductor than with default backend
```

**Solutions:**

1. **Check fusion effectiveness**
   ```python
   from conductor.utils import ConductorProfiler
   
   profiler = ConductorProfiler()
   with profiler:
       output = compiled_model(input_data)
   
   stats = profiler.get_stats()
   print(f"Fusion ratio: {stats['fusion_ratio']:.2%}")
   
   if stats['fusion_ratio'] < 0.5:
       print("Low fusion ratio, check fusion configuration")
   ```

2. **Warm up the model**
   ```python
   # First execution includes compilation overhead
   with torch.no_grad():
       dummy_input = torch.randn_like(input_data)
       _ = compiled_model(dummy_input)  # Warm up
   
   # Now measure actual performance
   import time
   start = time.time()
   output = compiled_model(input_data)
   end = time.time()
   print(f"Execution time: {end - start:.4f}s")
   ```

3. **Optimize for your workload**
   ```python
   # For small models, disable fusion
   if model_size < threshold:
       conductor.configure_backend({'fusion_enabled': False})
   
   # For memory-bound models, enable aggressive buffer reuse
   if is_memory_bound(model):
       conductor.configure_backend({
           'buffer_config': {'buffer_reuse_enabled': True}
       })
   ```

### Memory Issues

#### Problem: GPU out of memory
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size**
   ```python
   # Use smaller batches
   batch_size = 16  # Reduce from 32
   dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
   ```

2. **Enable gradient checkpointing**
   ```python
   from torch.utils.checkpoint import checkpoint
   
   class CheckpointedModel(torch.nn.Module):
       def forward(self, x):
           # Use checkpointing for memory-intensive layers
           x = checkpoint(self.layer1, x)
           x = checkpoint(self.layer2, x)
           return x
   ```

3. **Optimize buffer management**
   ```python
   conductor.configure_backend({
       'buffer_config': {
           'buffer_reuse_enabled': True,
           'auto_scope_promotion': False,  # Prevent unnecessary promotions
           'memory_pool_size': '512MB'     # Limit memory pool
       }
   })
   ```

### Cache Issues

#### Problem: Cache corruption or invalid cache entries
```
CacheError: Invalid cache entry for graph signature
```

**Solutions:**

1. **Clear cache**
   ```python
   from conductor.utils import CacheManager
   
   cache_manager = CacheManager()
   cache_manager.clear_cache()
   print("Cache cleared")
   ```

2. **Disable caching temporarily**
   ```python
   conductor.configure_backend({'cache_enabled': False})
   
   # Compile without caching
   compiled_model = torch.compile(model, backend='gcu')
   
   # Re-enable caching
   conductor.configure_backend({'cache_enabled': True})
   ```

3. **Validate cache integrity**
   ```python
   cache_manager = CacheManager()
   
   # Check cache health
   health_report = cache_manager.check_cache_health()
   print(f"Cache health: {health_report}")
   
   if health_report['corrupted_entries'] > 0:
       cache_manager.repair_cache()
   ```

## Debugging Techniques

### Enable Verbose Logging

```python
import logging
import conductor

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
conductor.configure_backend({'log_level': 'DEBUG'})

# This will show detailed information about:
# - Backend registration
# - Graph analysis
# - Fusion decisions
# - DSL generation
# - Compilation process
# - Execution details
```

### Save Intermediate Files

```python
conductor.configure_backend({
    'debug_mode': True,
    'save_intermediate_files': True,
    'debug_output_dir': './debug_output'
})

# This will save:
# - Original FX graph (fx_graph.txt)
# - Conductor DAG (conductor_dag.txt)
# - Generated DSL (generated.co)
# - Compilation logs (compilation.log)
```

### Profile Performance

```python
from conductor.utils import ConductorProfiler, MemoryTracker

# Comprehensive profiling
profiler = ConductorProfiler()
memory_tracker = MemoryTracker()

with profiler, memory_tracker:
    compiled_model = torch.compile(model, backend='gcu')
    output = compiled_model(input_data)

# Analyze results
perf_stats = profiler.get_stats()
memory_stats = memory_tracker.get_stats()

print("Performance Stats:")
print(f"  Compilation time: {perf_stats['compilation_time']:.2f}s")
print(f"  Execution time: {perf_stats['execution_time']:.2f}s")
print(f"  Fusion ratio: {perf_stats['fusion_ratio']:.2%}")

print("Memory Stats:")
print(f"  Peak memory: {memory_stats['peak_memory']:.2f} MB")
print(f"  Buffer reuse ratio: {memory_stats['reuse_ratio']:.2%}")
```

### Inspect Generated Code

```python
from conductor.codegen import DSLGenerator

# Generate DSL without compilation
generator = DSLGenerator()
dsl_code = generator.generate_dsl_file(fx_graph)

print("Generated DSL:")
print(dsl_code)

# Check for issues:
# - Missing operations
# - Incorrect buffer scoping
# - Suboptimal fusion
# - Memory layout problems
```

## Getting Help

### Check System Information

```python
import conductor
import torch
import sys

print("System Information:")
print(f"  Python version: {sys.version}")
print(f"  PyTorch version: {torch.__version__}")
print(f"  Conductor version: {conductor.__version__}")

# Backend information
info = conductor.get_backend_info()
print(f"  Backend status: {info}")

# Device information
try:
    from conductor.device import get_gcu_interface
    gcu = get_gcu_interface()
    devices = gcu.list_devices()
    print(f"  GCU devices: {devices}")
except Exception as e:
    print(f"  GCU device error: {e}")
```

### Report Issues

When reporting issues, please include:

1. **System information** (from above)
2. **Minimal reproducible example**
3. **Error messages and stack traces**
4. **Debug logs** (with `log_level='DEBUG'`)
5. **Generated DSL code** (if applicable)

### Community Resources

- **Documentation**: [https://conductor-pytorch.readthedocs.io](https://conductor-pytorch.readthedocs.io)
- **GitHub Issues**: [https://github.com/conductor/conductor-pytorch/issues](https://github.com/conductor/conductor-pytorch/issues)
- **Discussions**: [https://github.com/conductor/conductor-pytorch/discussions](https://github.com/conductor/conductor-pytorch/discussions)
- **Stack Overflow**: Tag questions with `conductor-pytorch`

### Performance Optimization Checklist

- [ ] Backend registered successfully
- [ ] Fusion enabled and effective (>50% fusion ratio)
- [ ] Caching enabled and working
- [ ] Model warmed up before performance measurement
- [ ] Appropriate batch size for your hardware
- [ ] Memory usage optimized
- [ ] Debug mode disabled in production
- [ ] Fallback mechanism working for unsupported operations
# Technical Guidelines for Conductor Integration

## Core Technology Stack
### Runtime Requirements
- **Python**: >=3.8 (for typing support and modern features)
- **PyTorch**: >=2.0 (for FX Graph API and torch.compile compatibility)
- **Platform**: Linux/macOS/Windows with cross-platform compatibility

### Package Architecture
- **Distribution**: Independent Python package 'conductor' via PyPI
- **Installation**: Standard pip/conda installation with automatic dependency resolution
- **Backend Registration**: Python API with optional C++ extension using pybind11 for performance-critical paths

### Core Dependencies
- **Essential**: PyTorch (FX Graph processing), subprocess (compiler integration), ctypes (artifact loading)
- **Build Tools**: setuptools (setup.py), build (pyproject.toml), wheel (distribution)
- **Optional**: numpy (testing utilities), pybind11 (C++ extensions)
- **Development**: pytest (testing), black (formatting), mypy (type checking)

## Compilation Pipeline
### DSL Generation
- **Input**: PyTorch FX Graph (torch.fx.GraphModule)
- **Processing**: Pure Python conversion to Conductor DSL (.co files)
- **Output**: Optimized DSL with fusion clusters and buffer management
- **Validation**: LLVM FileCheck for correctness verification

### Compiler Integration
- **Method**: Subprocess calls to Conductor CLI compiler
- **Input Format**: Conductor DSL (.co files)
- **Output Artifacts**: Shared libraries (.so) for JIT, object files (.o) for AOT
- **Error Handling**: Comprehensive error parsing and user-friendly messages

### Runtime Loading
- **JIT Mode**: Dynamic loading via ctypes with caching mechanisms
- **AOT Mode**: Static linking integration with PyTorch's compilation system
- **Memory Management**: Efficient buffer allocation and cleanup

## Architecture Patterns
### Graph Representation
```python
class ConductorNode:
    """Represents a single operation in the computation graph."""
    op: str                    # Operation identifier
    inputs: List[Buffer]       # Input buffers with scope information
    outputs: List[Buffer]      # Output buffers with scope information
    metadata: Dict[str, Any]   # Operation-specific parameters
    
class Buffer:
    """Represents data flow between operations."""
    name: str                  # Unique identifier
    scope: BufferScope         # Memory scope (local/shared/global)
    dtype: torch.dtype         # Data type information
    shape: Optional[List[int]] # Shape information when available
    
class FusionCluster:
    """Groups compatible operations for optimization."""
    nodes: List[ConductorNode] # Operations to fuse
    fusion_type: FusionType    # Elementwise, reduction, etc.
    dsl_function: str          # Generated DSL function name
```

### Fusion Heuristics
- **Elementwise Chains**: Fuse consecutive add, mul, relu, etc.
- **Reduction Patterns**: Combine elementwise + reduction (sum, max, etc.)
- **Memory Locality**: Prioritize operations sharing buffer access patterns
- **Kernel Launch Overhead**: Balance fusion benefits vs. compilation complexity

### Buffer Scope Management
- **Local**: Temporary variables within single kernel execution
- **Shared**: Inter-kernel communication within single model execution
- **Global**: Persistent data across multiple model invocations
- **Automatic Promotion**: Smart scope elevation based on usage patterns

## Code Quality Standards
### Naming Conventions
- **Functions/Variables**: `snake_case` with descriptive names
- **Classes**: `CamelCase` with clear purpose indication
- **Constants**: `UPPER_SNAKE_CASE` for configuration values
- **Private Members**: Leading underscore for internal APIs

### Documentation Requirements
- **Public APIs**: Complete docstrings with parameters, returns, examples, and type hints
- **Internal Functions**: Inline comments explaining complex logic
- **Architecture Decisions**: Design rationale in module-level docstrings
- **Examples**: Working code samples for all major features

### Error Handling Strategy
- **Custom Exceptions**: Specific exception types for different failure modes
- **Graceful Degradation**: Fallback to Inductor for unsupported operations
- **User-Friendly Messages**: Clear error descriptions with suggested solutions
- **Debug Information**: Detailed logging for development and troubleshooting

### Testing Framework
- **Unit Tests**: pytest for individual component testing
- **Integration Tests**: End-to-end pipeline validation
- **Performance Tests**: Benchmarking against baseline implementations
- **Correctness Validation**: FileCheck for DSL output verification

## Security and Reliability
### Compilation Safety
- **Input Validation**: Sanitize all user inputs and FX Graph content
- **Sandboxed Execution**: Isolate compiler subprocess execution
- **Resource Limits**: Prevent excessive memory/CPU usage during compilation
- **Code Injection Prevention**: No dynamic code execution from untrusted sources

### Platform Compatibility
- **Cross-Platform**: Support Linux, macOS, Windows with consistent behavior
- **Python Version**: Maintain compatibility across supported Python versions
- **Dependency Management**: Pin critical dependencies, allow flexibility for others
- **Graceful Fallbacks**: Handle missing optional dependencies appropriately

### Performance Considerations
- **Compilation Caching**: Intelligent caching of compiled artifacts
- **Memory Efficiency**: Minimize memory footprint during graph processing
- **Lazy Loading**: Load components only when needed
- **Parallel Processing**: Utilize multiple cores where beneficial and safe
# Conductor: PyTorch Backend Integration Product

## Product Purpose
Conductor is a PyTorch backend integration that enables seamless execution of ML models on custom 'gcu' hardware through the Conductor compiler. It provides a minimalist, humanized approach to custom backend development with full torch.compile compatibility.

## Target Users
- ML engineers using PyTorch for model training and inference on custom hardware
- Hardware teams developing 'gcu' devices requiring PyTorch ecosystem integration
- Researchers needing custom backend extensions for specialized AI workflows
- Organizations seeking performance optimization through hardware acceleration

## Core Value Proposition
- **Zero Learning Curve**: Drop-in replacement using standard torch.compile API
- **Performance First**: Intelligent operation fusion and optimized memory management
- **Developer Friendly**: Minimalist design with clear, readable code and comprehensive documentation
- **Production Ready**: Robust JIT/AOT modes with fallback mechanisms

## Key Features
- **Minimalist Architecture**: Clean DAG + Buffer AST for efficient graph representation
- **JIT Mode**: Dynamic FX Graph → Conductor DSL → compilation → execution pipeline
- **AOT Mode**: Precompiled artifact loading and integration
- **Smart Fusion**: Automatic clustering of compatible operations for performance
- **Buffer Management**: Intelligent scoping (local/shared/global) for memory optimization
- **Fallback Support**: Seamless fallback to Inductor for unsupported operations

## Technical Architecture
### FX Graph → Conductor DSL Pipeline
1. **Graph Analysis**: Parse FX Graph into ConductorNode DAG representation
2. **Fusion Optimization**: Apply heuristics to cluster elementwise and reduction operations
3. **Buffer Allocation**: Manage memory scopes and temporary variable lifecycle
4. **DSL Generation**: Emit Conductor DSL (.co files) with topological ordering
5. **Compilation**: Invoke Conductor CLI to generate executable artifacts
6. **Runtime Integration**: Load and execute compiled kernels on GCU hardware

### Deployment Strategy
- **Package Distribution**: Independent Python package via PyPI
- **Dependency Management**: Minimal dependencies (PyTorch 2.0+, Python 3.8+)
- **Installation**: Simple `pip install conductor` with automatic backend registration
- **Integration**: API-based or monkey-patch backend registration options

## Success Criteria
### Functional Requirements
- Complete FX Graph operation coverage for common ML workloads
- Successful JIT and AOT mode execution on target hardware
- Robust fallback mechanism for unsupported operations
- Full compatibility with torch.compile API

### Performance Targets
- JIT compilation overhead < 10% of model execution time
- AOT mode performance within 5% of native Conductor compiler
- Memory usage optimization through intelligent buffer management
- Fusion effectiveness: >80% reduction in kernel launches for fusible operations

### Quality Metrics
- Test coverage > 90% for core functionality
- Zero breaking changes to PyTorch user workflows
- Comprehensive documentation with examples and best practices
- Active community engagement and contribution guidelines
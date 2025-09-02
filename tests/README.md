# Conductor Test Suite

This directory contains the comprehensive test suite for the Conductor PyTorch Backend Integration. The test suite is designed to validate correctness, performance, and reliability of all components.

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── pytest.ini                    # Pytest configuration
├── test_runner.py                 # Comprehensive test runner
├── unit/                          # Unit tests for individual components
│   ├── test_buffers.py           # Buffer management tests
│   ├── test_graph.py             # Graph representation tests
│   ├── test_fusion.py            # Fusion logic tests (original)
│   ├── test_fusion_enhanced.py   # Enhanced fusion tests
│   ├── test_dsl.py               # DSL generation tests
│   ├── test_backend_enhanced.py  # Backend registration tests
│   ├── test_runtime_*.py         # Runtime component tests
│   ├── test_device_*.py          # Device interface tests
│   └── test_utils_*.py           # Utility function tests
├── integration/                   # Integration tests for complete workflows
│   ├── test_jit_pipeline.py      # JIT compilation pipeline tests
│   ├── test_aot_pipeline.py      # AOT compilation pipeline tests
│   ├── test_end_to_end_pipeline.py # Complete end-to-end tests
│   ├── test_backend_registration.py # Backend integration tests
│   └── test_fallback_mechanism.py # Fallback mechanism tests
├── filecheck/                     # FileCheck-style DSL validation tests
│   ├── test_dsl_output.py        # DSL structure validation
│   └── test_fusion_patterns.py   # Fusion pattern validation
├── performance/                   # Performance and benchmarking tests
│   └── test_compilation_performance.py # Compilation performance tests
└── test_package_structure.py     # Package structure validation
```

## Test Categories

### Unit Tests (`tests/unit/`)

Unit tests validate individual components in isolation:

- **Buffer Management**: Tests for `Buffer`, `BufferScope`, and `BufferManager` classes
- **Graph Representation**: Tests for `ConductorNode`, `ComputationDAG`, and `GraphAnalyzer`
- **Fusion Logic**: Tests for `FusionEngine`, `FusionCluster`, and fusion heuristics
- **DSL Generation**: Tests for `DSLGenerator` and DSL output correctness
- **Backend Integration**: Tests for backend registration and PyTorch integration
- **Runtime Components**: Tests for JIT/AOT compilation and artifact loading
- **Device Interface**: Tests for GCU device interface and hardware abstraction
- **Utilities**: Tests for caching, logging, error handling, and helper functions

**Coverage Target**: >90% code coverage for all core functionality

### Integration Tests (`tests/integration/`)

Integration tests validate complete workflows and component interactions:

- **JIT Pipeline**: End-to-end JIT compilation from FX Graph to execution
- **AOT Pipeline**: Precompiled artifact loading and execution
- **Backend Registration**: PyTorch backend integration and torch.compile compatibility
- **Fallback Mechanisms**: Graceful fallback to Inductor when operations are unsupported
- **Error Recovery**: Robust error handling and recovery strategies
- **Performance Integration**: End-to-end performance validation

### FileCheck Tests (`tests/filecheck/`)

FileCheck-style tests validate DSL generation correctness using pattern matching:

- **DSL Structure**: Function definitions, buffer declarations, operation sequences
- **Fusion Patterns**: Elementwise fusion, reduction fusion, memory-bound fusion
- **Optimization Correctness**: Buffer scope assignment, topological ordering
- **Mathematical Correctness**: Preservation of mathematical semantics in fused code

### Performance Tests (`tests/performance/`)

Performance tests validate performance requirements and detect regressions:

- **Compilation Performance**: Compilation time for different model sizes
- **Memory Usage**: Memory consumption during compilation and execution
- **Scalability**: Performance with large models and batch sizes
- **Cache Performance**: Impact of compilation result caching
- **Regression Detection**: Baseline metrics for detecting performance regressions

## Running Tests

### Using the Test Runner

The comprehensive test runner provides easy access to different test categories:

```bash
# Run all tests with coverage
python tests/test_runner.py --category all --verbose

# Run specific test categories
python tests/test_runner.py --category unit
python tests/test_runner.py --category integration
python tests/test_runner.py --category filecheck
python tests/test_runner.py --category performance

# Run tests matching a pattern
python tests/test_runner.py --pattern "test_fusion"

# Check test environment
python tests/test_runner.py --check-env
```

### Using pytest Directly

```bash
# Run all tests
pytest

# Run specific test categories using markers
pytest -m unit
pytest -m integration
pytest -m filecheck
pytest -m performance

# Run with coverage
pytest --cov=conductor --cov-report=html

# Run specific test files
pytest tests/unit/test_buffers.py
pytest tests/integration/test_jit_pipeline.py

# Run with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_fusion"
```

### Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit`: Unit tests for individual components
- `@pytest.mark.integration`: Integration tests for complete workflows
- `@pytest.mark.filecheck`: FileCheck-style DSL validation tests
- `@pytest.mark.performance`: Performance and benchmarking tests
- `@pytest.mark.slow`: Tests that take a long time to run
- `@pytest.mark.requires_conductor`: Tests requiring Conductor CLI compiler
- `@pytest.mark.requires_gcu`: Tests requiring GCU hardware

## Test Configuration

### pytest.ini

The `pytest.ini` file configures:
- Test discovery patterns
- Coverage reporting
- Marker definitions
- Warning filters
- Minimum coverage thresholds

### conftest.py

The `conftest.py` file provides:
- Shared fixtures for common test objects
- Mock configurations for external dependencies
- Test environment setup and teardown
- Utility functions for test validation

## Test Fixtures

### Core Component Fixtures

- `sample_buffer`: Basic buffer for testing
- `buffer_manager`: Fresh BufferManager instance
- `sample_buffers`: Set of buffers with different properties
- `sample_node`: Basic ConductorNode for testing
- `elementwise_chain_nodes`: Chain of fusible nodes
- `sample_dag`: Complete ComputationDAG for testing
- `sample_fusion_cluster`: FusionCluster for testing

### PyTorch Model Fixtures

- `simple_torch_model`: Basic ReLU model
- `elementwise_torch_model`: Model with elementwise operations
- `linear_torch_model`: Model with linear layers
- `complex_torch_model`: Complex model for integration testing

### Mock Fixtures

- `mock_conductor_cli`: Mock Conductor CLI compiler
- `mock_executable_kernel`: Mock ExecutableKernel
- `mock_compiled_artifact`: Mock CompiledArtifact

## Writing Tests

### Unit Test Example

```python
import pytest
from conductor.codegen.buffers import Buffer, BufferScope

class TestBuffer:
    def test_buffer_creation(self):
        """Test basic buffer creation."""
        buffer = Buffer("test", BufferScope.LOCAL, torch.float32, (10, 10))
        
        assert buffer.name == "test"
        assert buffer.scope == BufferScope.LOCAL
        assert buffer.dtype == torch.float32
        assert buffer.shape == (10, 10)
    
    def test_scope_promotion(self):
        """Test buffer scope promotion."""
        buffer = Buffer("test", BufferScope.LOCAL, torch.float32)
        buffer.promote_scope(BufferScope.SHARED)
        
        assert buffer.scope == BufferScope.SHARED
```

### Integration Test Example

```python
@pytest.mark.integration
class TestJITPipeline:
    def test_end_to_end_compilation(self, simple_torch_model, mock_conductor_cli):
        """Test complete JIT compilation pipeline."""
        x = torch.randn(5, 10)
        traced_model = torch.fx.symbolic_trace(simple_torch_model)
        
        # Mock compilation pipeline
        with patch('os.path.exists', return_value=True):
            with patch('conductor.runtime.loader.ExecutableKernel.load_from_file'):
                backend = ConductorBackend()
                compiled_fn = backend(traced_model, [x])
                result = compiled_fn(x)
                
                assert isinstance(result, torch.Tensor)
```

### FileCheck Test Example

```python
@pytest.mark.filecheck
class TestDSLPatterns:
    def test_fusion_pattern(self):
        """Test DSL pattern for fusion."""
        # Generate DSL
        dsl = generate_fusion_dsl(cluster)
        
        # FileCheck: Verify pattern
        assert re.search(r'function fused_.*\(.*\) -> \(.*\) \{', dsl)
        assert re.search(r'// Fused operations', dsl)
        assert 'temp =' not in dsl  # Intermediates should be inlined
```

### Performance Test Example

```python
@pytest.mark.performance
class TestPerformance:
    def test_compilation_time(self, performance_config):
        """Test compilation performance."""
        start_time = time.perf_counter()
        artifact = compiler.compile_graph(traced_model)
        compilation_time = time.perf_counter() - start_time
        
        assert compilation_time < performance_config['timeout_seconds']
```

## Coverage Reporting

### Generating Coverage Reports

```bash
# HTML coverage report
pytest --cov=conductor --cov-report=html
# Open htmlcov/index.html in browser

# Terminal coverage report
pytest --cov=conductor --cov-report=term-missing

# XML coverage report (for CI)
pytest --cov=conductor --cov-report=xml
```

### Coverage Requirements

- **Overall Coverage**: >90% for all core functionality
- **Unit Test Coverage**: >95% for individual components
- **Integration Coverage**: >80% for workflow validation
- **Critical Path Coverage**: 100% for error handling and fallback mechanisms

## Continuous Integration

### Test Automation

Tests are automatically run in CI/CD pipelines:

1. **Pre-commit Hooks**: Run fast unit tests and linting
2. **Pull Request Validation**: Run full test suite with coverage
3. **Nightly Builds**: Run performance tests and regression detection
4. **Release Validation**: Run complete test suite including slow tests

### Test Environment Matrix

Tests are run across multiple environments:

- **Python Versions**: 3.8, 3.9, 3.10, 3.11
- **PyTorch Versions**: 2.0, 2.1, latest
- **Operating Systems**: Linux, macOS, Windows
- **Hardware**: CPU-only, GPU (when available)

## Debugging Tests

### Running Individual Tests

```bash
# Run single test with verbose output
pytest tests/unit/test_buffers.py::TestBuffer::test_buffer_creation -v

# Run with debugger
pytest --pdb tests/unit/test_buffers.py::TestBuffer::test_buffer_creation

# Run with print statements
pytest -s tests/unit/test_buffers.py
```

### Test Debugging Tips

1. **Use fixtures**: Leverage shared fixtures for consistent test setup
2. **Mock external dependencies**: Use mocks to isolate components under test
3. **Check test isolation**: Ensure tests don't depend on each other
4. **Validate assumptions**: Use assertions to check intermediate states
5. **Use descriptive names**: Make test names clearly describe what they test

## Contributing Tests

### Test Guidelines

1. **Write tests first**: Follow TDD practices when adding new features
2. **Test edge cases**: Include boundary conditions and error cases
3. **Use descriptive names**: Test names should clearly describe the scenario
4. **Keep tests focused**: Each test should validate one specific behavior
5. **Use appropriate markers**: Mark tests with correct categories
6. **Document complex tests**: Add docstrings explaining test purpose
7. **Mock external dependencies**: Don't rely on external services or hardware

### Test Review Checklist

- [ ] Tests cover new functionality completely
- [ ] Edge cases and error conditions are tested
- [ ] Tests are properly categorized with markers
- [ ] Mock objects are used appropriately
- [ ] Test names are descriptive and clear
- [ ] Tests run independently and don't affect each other
- [ ] Performance tests include appropriate thresholds
- [ ] FileCheck tests validate DSL correctness

## Troubleshooting

### Common Test Issues

1. **Import Errors**: Ensure Conductor package is installed in development mode
2. **Mock Failures**: Check that mocks match actual interface signatures
3. **Timeout Issues**: Increase timeout values for slow operations
4. **Coverage Issues**: Check that all code paths are exercised
5. **Flaky Tests**: Identify and fix non-deterministic behavior

### Getting Help

- Check test output for specific error messages
- Review test logs for debugging information
- Use pytest's built-in debugging features
- Consult the troubleshooting guide in the main documentation
- Ask for help in project discussions or issues
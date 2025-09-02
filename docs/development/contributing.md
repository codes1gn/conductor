# Contributing to Conductor

We welcome contributions to the Conductor PyTorch Backend Integration! This guide will help you get started with contributing to the project.

## Getting Started

### Prerequisites

- Python ≥3.8
- PyTorch ≥2.0
- Git
- Basic understanding of PyTorch compilation and FX graphs

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/conductor-pytorch.git
   cd conductor-pytorch
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv conductor-dev
   source conductor-dev/bin/activate  # On Windows: conductor-dev\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Verify Installation**
   ```bash
   python -c "import conductor; print('✓ Conductor installed successfully')"
   pytest --version
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 2. Make Your Changes

Follow the project structure and coding standards:

- **Code Style**: We use Black for formatting and isort for import sorting
- **Type Hints**: All public APIs must have type hints
- **Documentation**: Add docstrings for all public functions and classes
- **Tests**: Write tests for new functionality

### 3. Run Tests and Checks

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=conductor --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration

# Format code
black conductor/ tests/ examples/
isort conductor/ tests/ examples/

# Type checking
mypy conductor/

# Linting
flake8 conductor/
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add support for custom operations"
```

**Commit Message Format:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or modifications
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Link to related issues
- Test results and coverage information

## Code Standards

### Python Code Style

We follow PEP 8 with some modifications:

```python
# Good: Clear, descriptive names
def parse_fx_graph(graph_module: torch.fx.GraphModule) -> ComputationDAG:
    """Parse FX Graph into internal DAG representation."""
    pass

# Good: Type hints for all public APIs
class ConductorNode:
    def __init__(
        self, 
        op_name: str, 
        inputs: List[Buffer], 
        outputs: List[Buffer]
    ) -> None:
        self.op_name = op_name
        self.inputs = inputs
        self.outputs = outputs

# Good: Comprehensive docstrings
def apply_fusion_optimization(
    nodes: List[ConductorNode], 
    config: FusionConfig
) -> List[FusionCluster]:
    """
    Apply fusion optimization to a list of nodes.
    
    Args:
        nodes: List of nodes to optimize
        config: Fusion configuration parameters
        
    Returns:
        List of fusion clusters created
        
    Raises:
        FusionError: If fusion validation fails
        
    Example:
        >>> nodes = [node1, node2, node3]
        >>> config = FusionConfig(max_size=10)
        >>> clusters = apply_fusion_optimization(nodes, config)
    """
    pass
```

### Documentation Standards

#### Docstring Format

Use Google-style docstrings:

```python
def compile_graph(
    graph_module: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    config: Optional[CompilationConfig] = None
) -> CompiledArtifact:
    """
    Compile FX Graph to executable artifact.
    
    This function performs the complete compilation pipeline from FX Graph
    analysis through DSL generation to final artifact creation.
    
    Args:
        graph_module: PyTorch FX graph to compile
        example_inputs: Example input tensors for shape inference
        config: Optional compilation configuration. If None, uses defaults.
        
    Returns:
        Compiled artifact ready for execution
        
    Raises:
        UnsupportedOperationError: If graph contains unsupported operations
        CompilationError: If compilation process fails
        
    Example:
        >>> model = torch.nn.Linear(10, 1)
        >>> graph_module = torch.fx.symbolic_trace(model)
        >>> inputs = [torch.randn(5, 10)]
        >>> artifact = compile_graph(graph_module, inputs)
    """
```

#### API Documentation

- All public APIs must have complete docstrings
- Include parameter types, return types, and exceptions
- Provide usage examples for complex APIs
- Document any side effects or state changes

### Testing Standards

#### Test Structure

```python
import pytest
import torch
from conductor.codegen import GraphAnalyzer, ConductorNode

class TestGraphAnalyzer:
    """Test suite for GraphAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = GraphAnalyzer()
        self.simple_model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU()
        )
    
    def test_parse_simple_graph(self):
        """Test parsing of simple linear model."""
        # Arrange
        graph_module = torch.fx.symbolic_trace(self.simple_model)
        
        # Act
        dag = self.analyzer.parse_fx_graph(graph_module)
        
        # Assert
        assert len(dag.nodes) == 2
        assert dag.nodes[0].op_name == 'linear'
        assert dag.nodes[1].op_name == 'relu'
    
    def test_unsupported_operation_raises_error(self):
        """Test that unsupported operations raise appropriate error."""
        # This test would use a model with unsupported operations
        pass
    
    @pytest.mark.parametrize("input_shape,expected_nodes", [
        ((10,), 2),
        ((5, 10), 2),
        ((2, 5, 10), 2),
    ])
    def test_different_input_shapes(self, input_shape, expected_nodes):
        """Test graph parsing with different input shapes."""
        pass
```

#### Test Categories

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_buffer_creation():
    """Unit test for buffer creation."""
    pass

@pytest.mark.integration
def test_end_to_end_compilation():
    """Integration test for complete compilation pipeline."""
    pass

@pytest.mark.performance
def test_compilation_performance():
    """Performance test for compilation speed."""
    pass

@pytest.mark.slow
def test_large_model_compilation():
    """Test compilation of large models (may take time)."""
    pass
```

#### Test Coverage

- Aim for >90% code coverage
- Test both success and failure paths
- Include edge cases and boundary conditions
- Test error handling and recovery

## Project Structure

### Adding New Features

When adding new features, follow this structure:

```
conductor/
├── codegen/           # Code generation components
│   ├── new_feature.py # Your new feature implementation
│   └── __init__.py    # Export public APIs
├── runtime/           # Runtime components
├── device/            # Device-specific code
└── utils/             # Shared utilities

tests/
├── unit/
│   └── test_new_feature.py  # Unit tests
├── integration/
│   └── test_new_feature_integration.py  # Integration tests
└── filecheck/
    └── test_new_feature_dsl.py  # DSL validation tests

examples/
└── new_feature_example.py  # Usage example

docs/
├── api/
│   └── new_feature.md      # API documentation
└── guides/
    └── new_feature_guide.md # User guide
```

### Module Organization

- **Keep modules focused**: Each module should have a single responsibility
- **Use clear interfaces**: Define clear APIs between modules
- **Minimize dependencies**: Avoid circular imports and excessive coupling
- **Document interfaces**: Clearly document module boundaries and contracts

## Specific Contribution Areas

### 1. Operation Support

Adding support for new PyTorch operations:

```python
# In conductor/codegen/operations.py
@register_operation('new_op')
def convert_new_op(node: torch.fx.Node, context: ConversionContext) -> ConductorNode:
    """Convert new_op to Conductor representation."""
    # Implementation here
    pass

# Add corresponding test
def test_new_op_conversion():
    """Test conversion of new_op."""
    pass
```

### 2. Fusion Heuristics

Improving fusion optimization:

```python
# In conductor/codegen/fusion.py
class NewFusionHeuristic(FusionHeuristic):
    """New fusion strategy for specific operation patterns."""
    
    def can_fuse(self, nodes: List[ConductorNode]) -> bool:
        """Determine if nodes can be fused with this heuristic."""
        pass
    
    def create_cluster(self, nodes: List[ConductorNode]) -> FusionCluster:
        """Create fusion cluster from nodes."""
        pass
```

### 3. Backend Features

Adding new backend capabilities:

```python
# In conductor/backend.py
def new_backend_feature(config: Dict[str, Any]) -> None:
    """Implement new backend feature."""
    pass

# Update configuration schema
BACKEND_CONFIG_SCHEMA = {
    'new_feature_enabled': bool,
    'new_feature_params': dict,
    # ...
}
```

### 4. Device Support

Adding support for new devices:

```python
# In conductor/device/new_device.py
class NewDeviceInterface(DeviceInterface):
    """Interface for new device type."""
    
    def list_devices(self) -> List[str]:
        """List available devices."""
        pass
    
    def get_device_info(self, device_id: str) -> Dict[str, Any]:
        """Get device information."""
        pass
```

## Testing Guidelines

### Unit Tests

- Test individual functions and classes in isolation
- Use mocks for external dependencies
- Focus on edge cases and error conditions
- Keep tests fast and deterministic

### Integration Tests

- Test complete workflows end-to-end
- Use real PyTorch models and data
- Verify correctness against reference implementations
- Test fallback mechanisms

### Performance Tests

- Benchmark critical paths
- Compare against baseline implementations
- Monitor for performance regressions
- Test with realistic workloads

### FileCheck Tests

For DSL generation validation:

```python
def test_fusion_dsl_generation():
    """Test DSL generation for fusion clusters."""
    # Generate DSL
    dsl_code = generate_fusion_dsl(cluster)
    
    # Validate with FileCheck patterns
    assert "// Fused operation cluster" in dsl_code
    assert re.search(r"temp\d+ = add\(", dsl_code)
    assert re.search(r"temp\d+ = relu\(", dsl_code)
```

## Documentation Guidelines

### API Documentation

- Document all public APIs
- Include examples for complex functions
- Explain parameters, return values, and exceptions
- Keep documentation up-to-date with code changes

### User Guides

- Write from user perspective
- Include complete working examples
- Cover common use cases and patterns
- Provide troubleshooting information

### Developer Documentation

- Explain design decisions and trade-offs
- Document internal APIs and interfaces
- Provide architecture overviews
- Include contribution guidelines

## Review Process

### Pull Request Guidelines

1. **Clear Description**: Explain what changes you made and why
2. **Link Issues**: Reference related GitHub issues
3. **Test Coverage**: Include test results and coverage information
4. **Documentation**: Update relevant documentation
5. **Breaking Changes**: Clearly mark any breaking changes

### Review Checklist

Reviewers will check:

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Code coverage is maintained or improved
- [ ] Documentation is updated
- [ ] No breaking changes without justification
- [ ] Performance impact is considered
- [ ] Security implications are addressed

### Getting Reviews

- Request reviews from relevant maintainers
- Be responsive to feedback
- Make requested changes promptly
- Ask questions if feedback is unclear

## Release Process

### Version Numbering

We use semantic versioning (SemVer):

- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version numbers are bumped
- [ ] Release notes are prepared

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Request Comments**: Code-specific discussions

### Maintainer Contact

- Check the MAINTAINERS.md file for current maintainers
- Tag relevant maintainers in issues or PRs
- Be patient and respectful in communications

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please read and follow our Code of Conduct.

### Expected Behavior

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Acknowledge different perspectives and experiences

### Unacceptable Behavior

- Harassment or discrimination of any kind
- Trolling, insulting, or derogatory comments
- Personal attacks or political discussions
- Publishing private information without consent

## Recognition

Contributors will be recognized in:

- CONTRIBUTORS.md file
- Release notes for significant contributions
- GitHub contributor statistics
- Special recognition for major features or fixes

Thank you for contributing to Conductor! Your efforts help make PyTorch compilation better for everyone.
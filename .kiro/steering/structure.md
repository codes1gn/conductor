# Project Structure and Organization

## Repository Layout
```
conductor/
├── .kiro/                          # Kiro development configuration
│   ├── steering/                   # Development guidelines and standards
│   │   ├── product.md             # Product vision and requirements
│   │   ├── tech.md                # Technical guidelines and constraints
│   │   └── structure.md           # Project organization standards
│   └── specs/                     # Feature specifications
│       └── conductor-mvp-dev/     # MVP development specification
│           ├── requirements.md    # Functional and non-functional requirements
│           ├── design.md          # Architecture and design decisions
│           └── tasks.md           # Implementation task breakdown
├── conductor/                      # Main package source code
│   ├── __init__.py                # Package initialization and public API
│   ├── backend.py                 # PyTorch backend registration and integration
│   ├── codegen/                   # Code generation subsystem
│   │   ├── __init__.py           # Codegen module initialization
│   │   ├── graph.py              # FX Graph analysis and representation
│   │   ├── fusion.py             # Operation fusion logic and heuristics
│   │   ├── dsl.py                # Conductor DSL generation
│   │   └── buffers.py            # Buffer management and scoping
│   ├── runtime/                   # Runtime execution subsystem
│   │   ├── __init__.py           # Runtime module initialization
│   │   ├── jit.py                # Just-in-time compilation pipeline
│   │   ├── aot.py                # Ahead-of-time compilation support
│   │   └── loader.py             # Artifact loading and execution
│   ├── device/                    # Device-specific implementations
│   │   ├── __init__.py           # Device module initialization
│   │   ├── gcu.py                # GCU device Python interface
│   │   ├── gcu.cpp               # GCU device C++ extension (optional)
│   │   └── setup.py              # C++ extension build configuration
│   └── utils/                     # Shared utilities and helpers
│       ├── __init__.py           # Utils module initialization
│       ├── logging.py            # Logging configuration and utilities
│       ├── caching.py            # Compilation result caching
│       └── exceptions.py         # Custom exception definitions
├── examples/                       # Usage examples and demonstrations
│   ├── basic_usage.py             # Simple torch.compile integration example
│   ├── advanced_fusion.py         # Complex fusion optimization example
│   └── benchmarks/                # Performance benchmarking scripts
│       ├── jit_performance.py    # JIT mode performance tests
│       └── aot_performance.py    # AOT mode performance tests
├── samples/                        # Sample Conductor DSL files
│   ├── elementwise/               # Elementwise operation examples
│   │   ├── add.co                # Addition operation DSL
│   │   ├── mul.co                # Multiplication operation DSL
│   │   └── relu.co               # ReLU activation DSL
│   ├── reduction/                 # Reduction operation examples
│   │   ├── sum.co                # Sum reduction DSL
│   │   └── max.co                # Max reduction DSL
│   └── fused/                     # Fused operation examples
│       ├── add_relu.co           # Fused addition + ReLU DSL
│       └── matmul_bias.co        # Fused matrix multiplication + bias DSL
├── tests/                          # Comprehensive test suite
│   ├── unit/                      # Unit tests for individual components
│   │   ├── test_graph.py         # Graph representation tests
│   │   ├── test_fusion.py        # Fusion logic tests
│   │   ├── test_buffers.py       # Buffer management tests
│   │   └── test_dsl.py           # DSL generation tests
│   ├── integration/               # Integration tests for complete workflows
│   │   ├── test_jit_pipeline.py  # JIT mode end-to-end tests
│   │   ├── test_aot_pipeline.py  # AOT mode end-to-end tests
│   │   └── test_fallback.py      # Fallback mechanism tests
│   ├── performance/               # Performance and benchmarking tests
│   │   ├── test_compilation_time.py # Compilation performance tests
│   │   └── test_execution_time.py   # Runtime performance tests
│   └── filecheck/                 # FileCheck validation tests
│       ├── dsl_output/           # DSL generation correctness tests
│       └── fusion_patterns/      # Fusion pattern validation tests
├── docs/                          # Documentation and guides
│   ├── api/                      # API reference documentation
│   ├── guides/                   # User guides and tutorials
│   │   ├── getting_started.md   # Quick start guide
│   │   ├── advanced_usage.md    # Advanced features and customization
│   │   └── troubleshooting.md   # Common issues and solutions
│   └── development/              # Development and contribution guides
│       ├── contributing.md      # Contribution guidelines
│       └── architecture.md      # Detailed architecture documentation
├── .github/                       # GitHub configuration
│   ├── workflows/                # CI/CD pipeline definitions
│   │   ├── test.yml             # Automated testing workflow
│   │   ├── build.yml            # Package building workflow
│   │   └── release.yml          # Release automation workflow
│   └── ISSUE_TEMPLATE/           # Issue templates for bug reports and features
├── setup.py                       # Package installation configuration
├── pyproject.toml                 # Modern Python project configuration
├── requirements.txt               # Runtime dependencies
├── requirements-dev.txt           # Development dependencies
├── .gitignore                     # Git ignore patterns
├── .pre-commit-config.yaml        # Pre-commit hooks configuration
├── LICENSE                        # Project license
└── README.md                      # Project overview and quick start
```

## Module Organization Principles

### Separation of Concerns
- **Backend Integration** (`backend.py`): PyTorch interface and registration
- **Code Generation** (`codegen/`): FX Graph processing and DSL generation
- **Runtime Execution** (`runtime/`): Compilation and execution pipelines
- **Device Support** (`device/`): Hardware-specific implementations
- **Utilities** (`utils/`): Shared functionality and helpers

### Dependency Management
- **Core Dependencies**: Minimal runtime requirements (PyTorch, standard library)
- **Optional Dependencies**: Performance enhancements and development tools
- **Development Dependencies**: Testing, linting, documentation generation
- **Build Dependencies**: Package building and C++ extension compilation

### Testing Strategy
- **Unit Tests**: Individual component validation with high coverage
- **Integration Tests**: End-to-end workflow validation
- **Performance Tests**: Benchmarking and regression detection
- **Correctness Tests**: FileCheck validation for generated code

## Naming Conventions and Standards

### File and Directory Naming
- **Packages/Modules**: `lowercase_with_underscores`
- **Classes**: `CamelCase` with descriptive names
- **Functions/Variables**: `snake_case` with clear purpose
- **Constants**: `UPPER_SNAKE_CASE` for configuration values
- **Private Members**: Leading underscore for internal APIs

### Code Organization
- **Public APIs**: Clearly defined in `__init__.py` files
- **Internal APIs**: Prefixed with underscore, documented for maintainers
- **Configuration**: Centralized in dedicated configuration modules
- **Documentation**: Comprehensive docstrings for all public interfaces

### Version Control Strategy
- **Main Branch**: Stable, production-ready code
- **Development Branch**: Integration branch for new features
- **Feature Branches**: Individual feature development
- **Release Tags**: Semantic versioning for releases
- **Ignore Patterns**: Build artifacts, cache files, IDE configurations

## Extensibility and Customization

### Plugin Architecture
- **Fusion Heuristics**: Pluggable fusion strategy implementations
- **Buffer Scopes**: Extensible buffer management strategies
- **Device Backends**: Modular device-specific implementations
- **Optimization Passes**: Configurable graph optimization pipeline

### Configuration Management
- **Environment Variables**: Runtime configuration options
- **Configuration Files**: User-customizable settings
- **API Parameters**: Programmatic configuration interface
- **Default Values**: Sensible defaults for all configuration options

### Development Workflow
- **Code Quality**: Automated linting, formatting, and type checking
- **Testing**: Comprehensive test suite with CI/CD integration
- **Documentation**: Automated documentation generation and validation
- **Release Process**: Automated building, testing, and deployment
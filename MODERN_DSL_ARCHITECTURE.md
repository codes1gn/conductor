# Modern DSL Code Generation Architecture

## **Design Principles**

### **1. Single Responsibility Principle**
- Each class has one clear purpose
- Separation of concerns between data, logic, and presentation
- No god classes or monolithic components

### **2. Type Safety First**
- Comprehensive type hints with generics
- Dataclasses for structured data
- Protocol-based interfaces for flexibility
- Runtime type validation where needed

### **3. Dependency Injection**
- Constructor injection for dependencies
- Interface-based design for testability
- No global state or singletons

### **4. Immutable Data Structures**
- Frozen dataclasses for configuration
- Immutable context objects
- Functional programming patterns where appropriate

### **5. Composable Architecture**
- Small, focused components
- Builder pattern for complex objects
- Strategy pattern for different generation approaches

## **Core Architecture Components**

### **1. Data Layer - Structured Types**

```python
@dataclass(frozen=True)
class DSLGenerationConfig:
    """Configuration for DSL generation."""
    indent_size: int = 2
    parallel_factor: int = 1
    memory_level: MemoryLevel = MemoryLevel.L1
    enable_fusion: bool = True
    target_dialect: str = "choreo"

@dataclass(frozen=True)
class CodeGenerationContext:
    """Context for code generation operations."""
    function_name: str
    input_buffers: tuple[BufferInfo, ...]
    output_buffers: tuple[BufferInfo, ...]
    parallel_config: ParallelConfig
    naming_strategy: NamingStrategy

@dataclass(frozen=True)
class BufferInfo:
    """Information about a buffer in the computation."""
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    scope: BufferScope
    memory_level: MemoryLevel

@dataclass(frozen=True)
class OperationInfo:
    """Information about an operation to generate."""
    op_name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: dict[str, Any]
    metadata: dict[str, Any]
```

### **2. Interface Layer - Protocols**

```python
from typing import Protocol

class CodeGenerator(Protocol):
    """Protocol for code generation components."""
    
    def generate(self, context: CodeGenerationContext) -> CodeFragment: ...

class TemplateRenderer(Protocol):
    """Protocol for template rendering."""
    
    def render(self, template: str, context: dict[str, Any]) -> str: ...

class NamingStrategy(Protocol):
    """Protocol for naming strategies."""
    
    def get_buffer_name(self, buffer_info: BufferInfo) -> str: ...
    def get_variable_name(self, operation: OperationInfo) -> str: ...

class SyntaxValidator(Protocol):
    """Protocol for syntax validation."""
    
    def validate(self, code: str) -> ValidationResult: ...
```

### **3. Core Generation Engine**

```python
@dataclass(frozen=True)
class CodeFragment:
    """A fragment of generated code with metadata."""
    content: str
    dependencies: tuple[str, ...]
    declarations: tuple[str, ...]
    metadata: dict[str, Any]

class ModernDSLGenerator:
    """Modern DSL generator with clean architecture."""
    
    def __init__(
        self,
        config: DSLGenerationConfig,
        template_renderer: TemplateRenderer,
        naming_strategy: NamingStrategy,
        syntax_validator: SyntaxValidator,
    ):
        self._config = config
        self._template_renderer = template_renderer
        self._naming_strategy = naming_strategy
        self._syntax_validator = syntax_validator
    
    def generate_dsl(self, dag: ComputationDAG) -> DSLResult:
        """Generate complete DSL from computation DAG."""
        context = self._build_generation_context(dag)
        
        # Generate components
        header = self._generate_header(context)
        declarations = self._generate_declarations(context)
        function_body = self._generate_function_body(context)
        
        # Combine and validate
        complete_dsl = self._combine_fragments([header, declarations, function_body])
        validation_result = self._syntax_validator.validate(complete_dsl.content)
        
        return DSLResult(
            content=complete_dsl.content,
            validation=validation_result,
            metadata=complete_dsl.metadata
        )
```

### **4. Specialized Generators**

```python
class HeaderGenerator:
    """Generates DSL headers and includes."""
    
    def generate(self, context: CodeGenerationContext) -> CodeFragment:
        template = self._get_header_template()
        content = self._template_renderer.render(template, {
            'includes': context.required_includes,
            'function_name': context.function_name
        })
        return CodeFragment(content=content, dependencies=(), declarations=())

class ParallelLoopGenerator:
    """Generates parallel loop structures."""
    
    def generate(self, context: CodeGenerationContext) -> CodeFragment:
        template = self._get_parallel_template()
        content = self._template_renderer.render(template, {
            'parallel_factor': context.parallel_config.factor,
            'loop_variable': context.parallel_config.variable,
            'body': context.loop_body
        })
        return CodeFragment(content=content, dependencies=(), declarations=())

class OperationGenerator:
    """Generates individual operation code."""
    
    def __init__(self, operation_registry: OperationRegistry):
        self._operation_registry = operation_registry
    
    def generate(self, operation: OperationInfo, context: CodeGenerationContext) -> CodeFragment:
        template = self._operation_registry.get_template(operation.op_name)
        content = template.render(operation, context)
        return CodeFragment(content=content, dependencies=(), declarations=())
```

## **Key Improvements Over Current Architecture**

### **1. Type Safety**
- All data structures are typed with dataclasses
- Protocol-based interfaces for flexibility
- Generic types where appropriate
- Runtime validation for critical paths

### **2. Testability**
- Dependency injection enables easy mocking
- Small, focused components are easy to unit test
- Immutable data structures prevent test interference
- Clear interfaces make integration testing straightforward

### **3. Maintainability**
- Single responsibility for each component
- Clear separation between data, logic, and presentation
- Consistent error handling patterns
- Comprehensive documentation and type hints

### **4. Extensibility**
- Protocol-based design allows easy extension
- Strategy pattern for different generation approaches
- Builder pattern for complex configuration
- Plugin architecture for custom operations

### **5. Performance**
- Immutable data structures enable caching
- Lazy evaluation where appropriate
- Efficient template rendering
- Minimal object creation in hot paths

## **Implementation Strategy**

### **Phase 1: Core Data Structures**
1. Create typed data structures (`DSLGenerationConfig`, `CodeGenerationContext`, etc.)
2. Define protocols for all interfaces
3. Implement basic validation and error types

### **Phase 2: Template System**
1. Create modern template renderer with Jinja2 or similar
2. Implement template-based code generation
3. Add syntax validation for generated code

### **Phase 3: Specialized Generators**
1. Implement `HeaderGenerator`, `ParallelLoopGenerator`, `OperationGenerator`
2. Create composable generation pipeline
3. Add comprehensive error handling

### **Phase 4: Integration**
1. Create `ModernDSLGenerator` main class
2. Integrate with existing DAG structures
3. Add backward compatibility layer

### **Phase 5: Migration and Cleanup**
1. Migrate existing functionality to new architecture
2. Remove legacy `ChoreoDslGen` class
3. Update all integration points

## **File Structure**

```
conductor/codegen/modern/
├── __init__.py
├── types.py              # Data structures and protocols
├── config.py             # Configuration management
├── context.py            # Generation context
├── generators/
│   ├── __init__.py
│   ├── base.py           # Base generator classes
│   ├── header.py         # Header generation
│   ├── parallel.py       # Parallel loop generation
│   ├── operation.py      # Operation generation
│   └── validation.py     # Syntax validation
├── templates/
│   ├── __init__.py
│   ├── renderer.py       # Template rendering engine
│   └── choreo/           # Choreo-specific templates
│       ├── header.j2
│       ├── parallel.j2
│       └── operations.j2
├── naming/
│   ├── __init__.py
│   ├── strategies.py     # Naming strategies
│   └── validators.py     # Name validation
└── main.py               # Main DSL generator class
```

## **Migration Benefits**

### **Immediate Benefits**
- Type safety catches errors at development time
- Clear interfaces make testing straightforward
- Modular design enables parallel development
- Better error messages and debugging

### **Long-term Benefits**
- Easy to add new target languages (CUDA, OpenCL, etc.)
- Simple to extend with new optimization passes
- Maintainable codebase that scales with team growth
- Performance optimizations through caching and lazy evaluation

### **Risk Mitigation**
- Incremental migration reduces deployment risk
- Backward compatibility layer ensures existing code works
- Comprehensive test suite validates functionality
- Clear rollback strategy if issues arise

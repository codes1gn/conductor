# Migration Roadmap: ChoreoDslGen → Modern DSL Architecture

## **Overview**

This roadmap outlines the step-by-step migration from the current monolithic `ChoreoDslGen` class (1805 lines) to a modern, composable DSL generation architecture.

## **Migration Strategy: Incremental Replacement**

### **Principle: Parallel Development**
- Build new architecture alongside existing system
- Maintain backward compatibility during transition
- Enable rollback at any stage
- Validate functionality at each step

## **Phase 1: Foundation (Estimated: 2-3 days)**

### **Step 1.1: Create Modern Module Structure**
```bash
mkdir -p conductor/codegen/modern/{generators,templates,naming}
```

**Files to create:**
- `conductor/codegen/modern/__init__.py`
- `conductor/codegen/modern/types.py` - Core data structures
- `conductor/codegen/modern/config.py` - Configuration management
- `conductor/codegen/modern/context.py` - Generation context

**Success Criteria:**
- ✅ All new modules import without errors
- ✅ Type checking passes with mypy
- ✅ Basic data structures are functional

### **Step 1.2: Implement Core Data Structures**
**Priority: HIGH** - Foundation for everything else

```python
# types.py - Core types and protocols
@dataclass(frozen=True)
class DSLGenerationConfig: ...

@dataclass(frozen=True) 
class CodeGenerationContext: ...

@dataclass(frozen=True)
class BufferInfo: ...

# Protocols for interfaces
class CodeGenerator(Protocol): ...
class TemplateRenderer(Protocol): ...
```

**Success Criteria:**
- ✅ All data structures are immutable and typed
- ✅ Protocols define clear interfaces
- ✅ Validation functions work correctly

### **Step 1.3: Basic Template System**
**Files:**
- `conductor/codegen/modern/templates/renderer.py`
- `conductor/codegen/modern/templates/choreo/header.j2`

**Success Criteria:**
- ✅ Template rendering works for simple cases
- ✅ Jinja2 integration is functional
- ✅ Template validation catches syntax errors

## **Phase 2: Core Generators (Estimated: 3-4 days)**

### **Step 2.1: Header Generator**
**File:** `conductor/codegen/modern/generators/header.py`

**Functionality:**
- Generate DSL file headers
- Handle include statements
- Function signature generation

**Migration Target:** Replace `_generate_header()` method

**Success Criteria:**
- ✅ Generates identical headers to current system
- ✅ Handles all include scenarios
- ✅ Type-safe and well-tested

### **Step 2.2: Operation Generator**
**File:** `conductor/codegen/modern/generators/operation.py`

**Functionality:**
- Generate individual operation code
- Handle different operation types (add, mul, etc.)
- Template-based code generation

**Migration Target:** Replace `_generate_single_operation_choreo()` method

**Success Criteria:**
- ✅ Generates correct code for add/mul operations
- ✅ Handles all operation types from registry
- ✅ Produces syntactically valid Choreo DSL

### **Step 2.3: Parallel Loop Generator**
**File:** `conductor/codegen/modern/generators/parallel.py`

**Functionality:**
- Generate parallel loop structures
- Handle different parallelization strategies
- DMA operation integration

**Migration Target:** Replace `_generate_parallel_computation_with_annotation()` method

**Success Criteria:**
- ✅ Generates correct parallel loops
- ✅ Handles different tensor shapes
- ✅ Integrates with DMA operations

## **Phase 3: Integration Layer (Estimated: 2-3 days)**

### **Step 3.1: Modern DSL Generator Main Class**
**File:** `conductor/codegen/modern/main.py`

```python
class ModernDSLGenerator:
    def __init__(self, config: DSLGenerationConfig): ...
    def generate_dsl(self, dag: ComputationDAG) -> DSLResult: ...
```

**Success Criteria:**
- ✅ Generates complete DSL files
- ✅ Handles all current test cases
- ✅ Produces syntactically correct output

### **Step 3.2: Backward Compatibility Layer**
**File:** `conductor/codegen/modern/compat.py`

```python
class ChoreoDslGenCompat:
    """Backward compatibility wrapper for ChoreoDslGen."""
    def __init__(self, config=None):
        self._modern_generator = ModernDSLGenerator(...)
    
    def generate_dsl_from_dag(self, dag, function_name):
        # Delegate to modern implementation
        return self._modern_generator.generate_dsl(dag).content
```

**Success Criteria:**
- ✅ Drop-in replacement for ChoreoDslGen
- ✅ All existing tests pass
- ✅ No breaking changes to public API

### **Step 3.3: Integration Testing**
**Test all examples with modern generator:**
- ✅ `add_example.py` works with `max diff: 0.00e+00`
- ✅ `mul_example.py` works with `max diff: 0.00e+00`
- ✅ `add_mul_fused_example.py` works with `max diff: 0.00e+00`

## **Phase 4: Migration and Cleanup (Estimated: 1-2 days)**

### **Step 4.1: Switch Integration Points**
**Files to update:**
- `conductor/compiler/jit_compiler.py` - Use modern generator
- `conductor/codegen/__init__.py` - Export modern classes
- Any other integration points

**Success Criteria:**
- ✅ All integration points use modern generator
- ✅ No references to old ChoreoDslGen remain
- ✅ All tests pass with new system

### **Step 4.2: Remove Legacy Code**
**Files to remove:**
- Most of `conductor/codegen/dslgen.py` (keep only compatibility wrapper)
- Any other legacy DSL generation code

**Success Criteria:**
- ✅ Codebase is significantly smaller and cleaner
- ✅ No dead code remains
- ✅ All functionality preserved

### **Step 4.3: Final Validation**
**Comprehensive testing:**
- ✅ All examples work perfectly
- ✅ Performance is equal or better
- ✅ Generated DSL compiles successfully
- ✅ No regressions in functionality

## **Risk Mitigation**

### **Rollback Strategy**
1. **Phase 1-2**: Easy rollback - just don't use new modules
2. **Phase 3**: Compatibility layer allows instant rollback
3. **Phase 4**: Git revert if issues discovered

### **Validation at Each Step**
- Run all examples after each phase
- Maintain test coverage above 90%
- Performance benchmarking at each milestone
- Code review for each major component

### **Parallel Development**
- New architecture developed in separate modules
- No changes to existing code until Phase 4
- Continuous integration validates both systems

## **Success Metrics**

### **Code Quality**
- ✅ Reduce DSL generation code from 1805 lines to <500 lines
- ✅ Achieve 100% type coverage with mypy
- ✅ Maintain test coverage above 95%
- ✅ Zero code duplication in DSL generation

### **Functionality**
- ✅ All examples work with perfect numerical accuracy
- ✅ Generated DSL compiles without errors
- ✅ Performance equal or better than current system
- ✅ Support for all current operation types

### **Maintainability**
- ✅ Clear separation of concerns
- ✅ Easy to add new operation types
- ✅ Simple to extend for new target languages
- ✅ Comprehensive documentation and examples

## **Timeline Summary**

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1 | 2-3 days | Foundation, data structures, basic templates |
| Phase 2 | 3-4 days | Core generators (header, operation, parallel) |
| Phase 3 | 2-3 days | Integration layer, compatibility wrapper |
| Phase 4 | 1-2 days | Migration, cleanup, final validation |
| **Total** | **8-12 days** | **Complete modern DSL architecture** |

## **Next Steps**

1. **Approve architecture design** - Review MODERN_DSL_ARCHITECTURE.md
2. **Begin Phase 1** - Create foundation modules
3. **Incremental development** - Build and validate each component
4. **Continuous testing** - Ensure no regressions at any step

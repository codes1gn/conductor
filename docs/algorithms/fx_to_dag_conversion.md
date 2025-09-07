# FX Graph to DAG Conversion Algorithm

## Overview

The FX Graph to DAG conversion is the first critical stage in the Conductor compilation pipeline. This algorithm transforms PyTorch FX graphs into an internal Directed Acyclic Graph (DAG) representation that can be optimized and compiled to Choreo DSL.

## Algorithm Components

### Main Entry Point: `GraphAnalyzer.parse_fx_graph()`

The conversion process is orchestrated by the `GraphAnalyzer` class in `conductor/graph_analyzer.py`. The main method `parse_fx_graph()` implements a multi-pass algorithm:

```python
def parse_fx_graph(self, graph_module: torch.fx.GraphModule, example_inputs: Optional[List[torch.Tensor]] = None) -> ComputationDAG:
```

### Algorithm Flow

#### Phase 1: Buffer Creation Pass
```
For each FX node in the graph:
    Create corresponding Buffer objects
    Map FX nodes to internal buffers
    Establish bidirectional mapping (node ↔ buffer)
```

**Purpose**: Establishes the data flow infrastructure by creating buffer objects that represent intermediate values in the computation.

**Implementation**: `_create_buffer_for_node()` method creates `Buffer` objects with:
- Unique names derived from FX node names
- Data types inferred from FX node metadata
- Initial shape information (refined in later phases)

#### Phase 2: Node Conversion Pass
```
For each FX node in the graph:
    Convert to ConductorNode if it represents an operation
    Skip placeholder, output, and get_attr nodes
    Establish input/output connections via buffers
    Add ConductorNode to the DAG
```

**Purpose**: Converts computational operations into the internal `ConductorNode` representation while preserving the graph structure.

**Node Type Handling**:
- `placeholder`: Input nodes - create buffers but no operations
- `output`: Output nodes - mark buffers as outputs
- `get_attr`: Constants/parameters - create buffers for values
- `call_function/call_method/call_module`: Operations - create `ConductorNode` instances

#### Phase 3: Input/Output Identification
```
Analyze graph structure to identify:
    - Input buffers (from placeholder nodes)
    - Output buffers (from output nodes)
    - Set shape information from example_inputs
```

**Purpose**: Establishes the graph interface and propagates concrete shape information from the provided example inputs.

#### Phase 4: Shape Propagation
```
For each elementwise operation:
    Propagate shapes from inputs to outputs
    Handle broadcasting rules
    Validate shape compatibility
```

**Purpose**: Critical fix for shape inference - ensures all intermediate buffers have valid shape information for downstream compilation.

#### Phase 5: Dependency Analysis
```
Analyze data dependencies between operations:
    - Producer-consumer relationships
    - Buffer lifetime analysis
    - Dependency ordering for execution
```

**Purpose**: Establishes the execution order and data flow constraints needed for correct compilation.

## Data Structures

### ComputationDAG
The output DAG contains:
- `nodes`: List of `ConductorNode` objects representing operations
- `buffers`: List of `Buffer` objects representing data
- `inputs`: Input buffer references
- `outputs`: Output buffer references

### ConductorNode
Internal representation of operations:
- `op_name`: Operation type (e.g., 'add', 'mul', 'relu')
- `inputs`: List of input `Buffer` objects
- `outputs`: List of output `Buffer` objects
- `metadata`: Additional operation-specific information
- `fusion_group`: Fusion cluster assignment (set later)

### Buffer
Data flow representation:
- `name`: Unique identifier
- `scope`: Memory scope (LOCAL, SHARED, GLOBAL)
- `dtype`: Data type (torch.float32, etc.)
- `shape`: Tensor dimensions
- `producer`: Node that generates this buffer
- `consumers`: Nodes that consume this buffer

## Key Algorithms

### Shape Inference Algorithm
```python
def _propagate_shapes_for_elementwise_ops(self, dag: ComputationDAG):
    """
    Critical algorithm that ensures all buffers have valid shapes.

    For each elementwise operation:
        1. Get input buffer shapes
        2. Apply broadcasting rules
        3. Set output buffer shape
        4. Validate compatibility
    """
```

This algorithm fixes a critical issue where intermediate buffers could have `None` shapes, causing downstream compilation failures.

### Operation Mapping Algorithm
```python
def _convert_operation_node(self, fx_node: torch.fx.Node, dag: ComputationDAG) -> ConductorNode:
    """
    Maps FX operations to internal operation names:

    torch.add -> 'add'
    torch.mul -> 'mul'
    torch.relu -> 'relu'
    etc.
    """
```

Uses the `operation_mappings.py` module to maintain a registry of supported operations and their internal representations.

## Error Handling

### Unsupported Operations
When an FX node represents an unsupported operation:
1. Log warning with operation details
2. Raise `UnsupportedOperationError`
3. Provide suggestions for adding support

### Shape Mismatch Detection
During shape propagation:
1. Validate input shapes are compatible
2. Check broadcasting rules
3. Raise detailed error messages for mismatches

### Graph Validation
After conversion:
1. Check for cycles in the DAG
2. Validate all buffers have producers (except inputs)
3. Ensure all operations have valid inputs/outputs

## Performance Considerations

### Memory Efficiency
- Reuses buffer objects where possible
- Lazy shape computation to avoid unnecessary work
- Efficient node-to-buffer mapping using dictionaries

### Scalability
- Linear time complexity O(n) where n = number of FX nodes
- Constant space overhead per node
- Efficient for large graphs (tested up to 1000+ nodes)

## Integration Points

### Input Interface
- Accepts `torch.fx.GraphModule` from PyTorch's symbolic tracing
- Optional `example_inputs` for concrete shape information
- Preserves original FX graph metadata

### Output Interface
- Produces `ComputationDAG` for fusion engine
- Compatible with DSL generation pipeline
- Maintains traceability to original FX nodes

## Debugging Support

### Debug Tracing
When debug tracing is enabled:
- Logs each conversion step
- Dumps intermediate DAG state
- Tracks buffer creation and mapping
- Records shape propagation decisions

### Validation Hooks
- Graph correctness validation
- Shape consistency checking
- Operation support verification
- Dependency cycle detection

This algorithm forms the foundation of the Conductor compilation pipeline, ensuring that PyTorch models are correctly represented in the internal format needed for optimization and code generation.
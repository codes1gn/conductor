# Codegen API Reference

The codegen module handles FX Graph analysis, operation fusion, and Conductor DSL generation.

## Core Classes

### Buffer

Represents data flow and memory management in the computation graph.

```python
@dataclass
class Buffer:
    name: str
    scope: BufferScope
    dtype: torch.dtype
    shape: Optional[Tuple[int, ...]]
    producer: Optional['ConductorNode']
    consumers: List['ConductorNode']
    is_temporary: bool = False
```

#### Attributes

- `name` (str): Unique identifier for the buffer
- `scope` (BufferScope): Memory scope (LOCAL, SHARED, GLOBAL)
- `dtype` (torch.dtype): Data type information
- `shape` (Optional[Tuple[int, ...]]): Shape when statically known
- `producer` (Optional[ConductorNode]): Node that produces this buffer
- `consumers` (List[ConductorNode]): Nodes that consume this buffer
- `is_temporary` (bool): Whether this is a temporary intermediate buffer

#### Methods

##### promote_scope(new_scope)

Promote buffer to higher scope when needed for sharing.

```python
def promote_scope(self, new_scope: BufferScope) -> None
```

**Parameters:**
- `new_scope` (BufferScope): Target scope to promote to

**Example:**
```python
buffer = Buffer("temp1", BufferScope.LOCAL, torch.float32, (32, 128))
buffer.promote_scope(BufferScope.SHARED)  # Promote to shared scope
```

##### get_memory_footprint()

Calculate memory requirements for this buffer.

```python
def get_memory_footprint(self) -> int
```

**Returns:**
- `int`: Memory footprint in bytes

### ConductorNode

Represents a single operation in the computation DAG.

```python
@dataclass
class ConductorNode:
    op_name: str
    inputs: List[Buffer]
    outputs: List[Buffer]
    metadata: Dict[str, Any]
    fusion_group: Optional['FusionCluster'] = None
```

#### Attributes

- `op_name` (str): Operation identifier (e.g., 'add', 'mul', 'relu')
- `inputs` (List[Buffer]): Input buffers with dependency information
- `outputs` (List[Buffer]): Output buffers produced by this operation
- `metadata` (Dict[str, Any]): Operation-specific parameters and attributes
- `fusion_group` (Optional[FusionCluster]): Fusion cluster membership

#### Methods

##### can_fuse_with(other)

Determine if this node can be fused with another node.

```python
def can_fuse_with(self, other: 'ConductorNode') -> bool
```

**Parameters:**
- `other` (ConductorNode): Node to check fusion compatibility with

**Returns:**
- `bool`: True if nodes can be fused, False otherwise

##### generate_dsl()

Generate Conductor DSL code for this operation.

```python
def generate_dsl(self) -> str
```

**Returns:**
- `str`: DSL code for this operation

##### estimate_cost()

Estimate computational cost for scheduling decisions.

```python
def estimate_cost(self) -> float
```

**Returns:**
- `float`: Estimated computational cost

### FusionCluster

Groups compatible operations for optimization.

```python
@dataclass
class FusionCluster:
    nodes: List[ConductorNode]
    cluster_type: FusionType
    external_inputs: List[Buffer]
    external_outputs: List[Buffer]
    internal_buffers: List[Buffer]
    dsl_function_name: str
```

#### Attributes

- `nodes` (List[ConductorNode]): Operations included in this cluster
- `cluster_type` (FusionType): Type of fusion (ELEMENTWISE, REDUCTION, MIXED)
- `external_inputs` (List[Buffer]): Inputs from outside the cluster
- `external_outputs` (List[Buffer]): Outputs consumed outside the cluster
- `internal_buffers` (List[Buffer]): Temporary buffers within the cluster
- `dsl_function_name` (str): Generated DSL function identifier

#### Methods

##### validate_fusion_safety()

Verify that fusion preserves mathematical correctness.

```python
def validate_fusion_safety(self) -> bool
```

**Returns:**
- `bool`: True if fusion is mathematically safe

##### generate_fused_dsl()

Generate optimized DSL code for the entire cluster.

```python
def generate_fused_dsl(self) -> str
```

**Returns:**
- `str`: Optimized DSL code for the fusion cluster

##### estimate_performance_gain()

Estimate performance improvement from fusion.

```python
def estimate_performance_gain(self) -> float
```

**Returns:**
- `float`: Estimated performance gain ratio

## Analysis Classes

### GraphAnalyzer

Analyzes FX Graph and builds internal representation.

```python
class GraphAnalyzer:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
```

#### Methods

##### parse_fx_graph(graph_module)

Convert FX Graph to internal DAG representation.

```python
def parse_fx_graph(self, graph_module: torch.fx.GraphModule) -> 'ComputationDAG'
```

**Parameters:**
- `graph_module` (torch.fx.GraphModule): PyTorch FX graph to analyze

**Returns:**
- `ComputationDAG`: Internal DAG representation

**Raises:**
- `UnsupportedOperationError`: If graph contains unsupported operations
- `GraphAnalysisError`: If graph structure is invalid

##### identify_data_dependencies(dag)

Analyze data flow and establish buffer dependencies.

```python
def identify_data_dependencies(self, dag: 'ComputationDAG') -> None
```

**Parameters:**
- `dag` (ComputationDAG): DAG to analyze

##### validate_graph_correctness(dag)

Verify graph integrity and detect potential issues.

```python
def validate_graph_correctness(self, dag: 'ComputationDAG') -> bool
```

**Parameters:**
- `dag` (ComputationDAG): DAG to validate

**Returns:**
- `bool`: True if graph is valid

### FusionEngine

Implements operation fusion heuristics and optimization.

```python
class FusionEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
```

#### Methods

##### identify_fusion_opportunities(dag)

Find groups of operations that can be safely fused.

```python
def identify_fusion_opportunities(self, dag: 'ComputationDAG') -> List[FusionCluster]
```

**Parameters:**
- `dag` (ComputationDAG): DAG to analyze for fusion opportunities

**Returns:**
- `List[FusionCluster]`: List of identified fusion clusters

##### apply_elementwise_fusion(nodes)

Fuse consecutive elementwise operations.

```python
def apply_elementwise_fusion(self, nodes: List[ConductorNode]) -> FusionCluster
```

**Parameters:**
- `nodes` (List[ConductorNode]): Nodes to fuse

**Returns:**
- `FusionCluster`: Created fusion cluster

##### apply_reduction_fusion(nodes)

Fuse elementwise operations with following reductions.

```python
def apply_reduction_fusion(self, nodes: List[ConductorNode]) -> FusionCluster
```

**Parameters:**
- `nodes` (List[ConductorNode]): Nodes to fuse

**Returns:**
- `FusionCluster`: Created fusion cluster

##### optimize_buffer_usage(cluster)

Optimize memory usage within fusion clusters.

```python
def optimize_buffer_usage(self, cluster: FusionCluster) -> None
```

**Parameters:**
- `cluster` (FusionCluster): Cluster to optimize

### DSLGenerator

Generates Conductor DSL code from processed graph.

```python
class DSLGenerator:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
```

#### Methods

##### generate_dsl_file(dag)

Generate complete DSL file for the computation graph.

```python
def generate_dsl_file(self, dag: 'ComputationDAG') -> str
```

**Parameters:**
- `dag` (ComputationDAG): DAG to generate DSL for

**Returns:**
- `str`: Complete DSL file content

##### emit_buffer_declarations(buffers)

Generate buffer declarations with appropriate scoping.

```python
def emit_buffer_declarations(self, buffers: List[Buffer]) -> str
```

**Parameters:**
- `buffers` (List[Buffer]): Buffers to declare

**Returns:**
- `str`: DSL buffer declarations

##### emit_operation_sequence(nodes)

Generate operation sequence maintaining topological order.

```python
def emit_operation_sequence(self, nodes: List[ConductorNode]) -> str
```

**Parameters:**
- `nodes` (List[ConductorNode]): Nodes to emit

**Returns:**
- `str`: DSL operation sequence

##### optimize_temporary_variables(dsl_code)

Optimize temporary variable usage in generated DSL.

```python
def optimize_temporary_variables(self, dsl_code: str) -> str
```

**Parameters:**
- `dsl_code` (str): DSL code to optimize

**Returns:**
- `str`: Optimized DSL code

## Enums and Constants

### BufferScope

Defines memory scope hierarchy for buffer management.

```python
class BufferScope(Enum):
    LOCAL = "local"      # Temporary variables within single kernel
    SHARED = "shared"    # Inter-kernel communication within model execution
    GLOBAL = "global"    # Persistent data across multiple model invocations
```

### FusionType

Categorizes different types of operation fusion.

```python
class FusionType(Enum):
    ELEMENTWISE = "elementwise"      # Pure elementwise operation chains
    REDUCTION = "reduction"          # Elementwise followed by reduction
    MIXED = "mixed"                  # Complex fusion patterns
    MEMORY_BOUND = "memory_bound"    # Memory bandwidth limited operations
    COMPUTE_BOUND = "compute_bound"  # Computation intensive operations
```

## Utility Functions

### register_custom_operation(op_name)

Decorator to register custom operation converters.

```python
def register_custom_operation(op_name: str) -> Callable
```

**Parameters:**
- `op_name` (str): Name of the custom operation

**Returns:**
- `Callable`: Decorator function

**Example:**
```python
@register_custom_operation('my_custom_op')
def convert_my_op(node, inputs, outputs, metadata):
    return f"my_custom_op({', '.join(inputs)}) -> {outputs[0]}"
```

### get_supported_operations()

Get list of operations supported by the codegen system.

```python
def get_supported_operations() -> List[str]
```

**Returns:**
- `List[str]`: List of supported operation names

## Usage Examples

### Basic Graph Analysis

```python
from conductor.codegen import GraphAnalyzer

# Analyze FX graph
analyzer = GraphAnalyzer()
dag = analyzer.parse_fx_graph(fx_graph_module)

# Validate graph
if analyzer.validate_graph_correctness(dag):
    print("Graph is valid")
```

### Fusion Optimization

```python
from conductor.codegen import FusionEngine

# Apply fusion optimization
fusion_engine = FusionEngine({
    'max_fusion_size': 10,
    'fusion_threshold': 0.8
})

clusters = fusion_engine.identify_fusion_opportunities(dag)
print(f"Found {len(clusters)} fusion opportunities")
```

### DSL Generation

```python
from conductor.codegen import DSLGenerator

# Generate DSL code
generator = DSLGenerator()
dsl_code = generator.generate_dsl_file(dag)

# Save to file
with open('output.co', 'w') as f:
    f.write(dsl_code)
```

### Custom Operation Registration

```python
from conductor.codegen import register_custom_operation

@register_custom_operation('custom_relu')
def convert_custom_relu(node, inputs, outputs, metadata):
    x = inputs[0]
    out = outputs[0]
    threshold = metadata.get('threshold', 0.0)
    return f"{out} = max({x}, {threshold});"

# Now custom_relu operations will be supported
```

## Error Handling

### Common Exceptions

- `UnsupportedOperationError`: Operation not supported by codegen
- `GraphAnalysisError`: Invalid graph structure
- `FusionError`: Fusion validation failed
- `DSLGenerationError`: DSL generation failed

### Error Recovery

```python
from conductor.codegen import GraphAnalyzer, UnsupportedOperationError

analyzer = GraphAnalyzer()

try:
    dag = analyzer.parse_fx_graph(fx_graph)
except UnsupportedOperationError as e:
    print(f"Unsupported operation: {e.operation}")
    # Handle fallback or skip operation
```
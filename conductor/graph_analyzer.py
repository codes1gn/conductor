"""
FX Graph analysis and internal representation.

This module provides classes and functions for parsing PyTorch FX Graphs
and converting them to Conductor's internal DAG representation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch
import logging
from .buffers import Buffer
from .fusion_rules import get_fusion_rules

logger = logging.getLogger(__name__)


@dataclass
class ConductorNode:
    """
    Represents a single operation in the computation DAG.
    
    This class encapsulates all information needed to represent a PyTorch
    operation in Conductor's internal format, including inputs, outputs,
    and operation-specific metadata.
    """
    op_name: str                                    # Operation identifier (e.g., 'add', 'mul', 'relu')
    inputs: List[Buffer] = field(default_factory=list)     # Input buffers with dependency information
    outputs: List[Buffer] = field(default_factory=list)    # Output buffers produced by this operation
    metadata: Dict[str, Any] = field(default_factory=dict) # Operation-specific parameters and attributes
    fusion_group: Optional['FusionCluster'] = None         # Fusion cluster membership
    
    def can_fuse_with(self, other: 'ConductorNode') -> bool:
        """
        Determine if this node can be fused with another node.
        
        Args:
            other: Another ConductorNode to check fusion compatibility
            
        Returns:
            True if nodes can be safely fused, False otherwise
        """
        # Use centralized fusion rules configuration
        fusion_rules = get_fusion_rules()

        # Check if operations can fuse based on rules
        if not fusion_rules.can_operations_fuse(self.op_name, other.op_name):
            return False

        # Get the appropriate compatibility check method
        check_method = fusion_rules.get_fusion_compatibility_check(self.op_name, other.op_name)

        if check_method == 'check_elementwise_elementwise':
            return self._check_shape_compatibility(other)
        elif check_method == 'check_elementwise_reduction':
            return self._check_elementwise_reduction_compatibility(other)

        return False
    
    def _check_shape_compatibility(self, other: 'ConductorNode') -> bool:
        """
        Check if two nodes have compatible shapes for fusion.
        
        Args:
            other: Another ConductorNode to check compatibility
            
        Returns:
            True if shapes are compatible, False otherwise
        """
        # For elementwise operations, shapes must be broadcastable
        if not self.outputs or not other.outputs:
            return False
            
        self_shape = self.outputs[0].shape
        other_shape = other.outputs[0].shape
        
        # If shapes are unknown, assume compatible
        if self_shape is None or other_shape is None:
            return True
            
        # Check if shapes are identical or broadcastable
        return self._shapes_broadcastable(self_shape, other_shape)
    
    def _check_elementwise_reduction_compatibility(self, other: 'ConductorNode') -> bool:
        """
        Check if elementwise operation can be fused with reduction.
        
        Args:
            other: Reduction operation to check compatibility
            
        Returns:
            True if compatible, False otherwise
        """
        # Check if this node's output is consumed by the reduction
        if not self.outputs or not other.inputs:
            return False
            
        # Simple check: if any output buffer matches any input buffer
        self_output_names = {buf.name for buf in self.outputs}
        other_input_names = {buf.name for buf in other.inputs}
        
        return bool(self_output_names.intersection(other_input_names))
    
    def _shapes_broadcastable(self, shape1: tuple, shape2: tuple) -> bool:
        """
        Check if two shapes are broadcastable according to PyTorch rules.
        
        Args:
            shape1: First shape tuple
            shape2: Second shape tuple
            
        Returns:
            True if shapes are broadcastable, False otherwise
        """
        # Reverse iterate through dimensions
        for dim1, dim2 in zip(reversed(shape1), reversed(shape2)):
            if dim1 != dim2 and dim1 != 1 and dim2 != 1:
                return False
        return True
        
    def generate_dsl(self) -> str:
        """
        Generate Conductor DSL code for this operation.
        
        Returns:
            DSL code string representing this operation
        """
        # Generate input and output buffer names
        input_names = [buf.name for buf in self.inputs]
        output_names = [buf.name for buf in self.outputs]
        
        # Generate operation-specific DSL
        if self.op_name == 'add':
            if len(input_names) == 2 and len(output_names) == 1:
                return f"{output_names[0]} = add({input_names[0]}, {input_names[1]})"
        
        elif self.op_name == 'mul':
            if len(input_names) == 2 and len(output_names) == 1:
                return f"{output_names[0]} = mul({input_names[0]}, {input_names[1]})"
        
        elif self.op_name == 'relu':
            if len(input_names) == 1 and len(output_names) == 1:
                return f"{output_names[0]} = relu({input_names[0]})"
        
        elif self.op_name == 'sum':
            if len(input_names) == 1 and len(output_names) == 1:
                # Check for reduction dimensions in metadata
                dim = self.metadata.get('dim', None)
                if dim is not None:
                    return f"{output_names[0]} = sum({input_names[0]}, dim={dim})"
                else:
                    return f"{output_names[0]} = sum({input_names[0]})"
        
        elif self.op_name == 'matmul':
            if len(input_names) == 2 and len(output_names) == 1:
                return f"{output_names[0]} = matmul({input_names[0]}, {input_names[1]})"
        
        # Generic fallback for other operations
        input_str = ", ".join(input_names)
        output_str = ", ".join(output_names)
        
        if len(output_names) == 1:
            return f"{output_str} = {self.op_name}({input_str})"
        else:
            return f"{output_str} = {self.op_name}({input_str})"
        
    def estimate_cost(self) -> float:
        """
        Estimate computational cost for scheduling decisions.
        
        Returns:
            Estimated cost metric for this operation
        """
        # Base cost by operation type
        operation_costs = {
            # Elementwise operations (low cost)
            'add': 1.0, 'sub': 1.0, 'mul': 1.0, 'div': 2.0,
            'relu': 0.5, 'sigmoid': 3.0, 'tanh': 3.0,
            'abs': 0.5, 'neg': 0.5, 'exp': 4.0, 'log': 4.0,
            'sqrt': 2.0, 'sin': 3.0, 'cos': 3.0,
            
            # Reduction operations (medium cost)
            'sum': 2.0, 'mean': 2.5, 'max': 2.0, 'min': 2.0,
            'argmax': 3.0, 'argmin': 3.0,
            
            # Matrix operations (high cost)
            'matmul': 10.0, 'conv2d': 15.0, 'linear': 8.0,
            
            # Memory operations
            'reshape': 0.1, 'transpose': 1.0, 'permute': 1.0
        }
        
        base_cost = operation_costs.get(self.op_name, 5.0)  # Default medium cost
        
        # Scale by output tensor size if known
        if self.outputs and self.outputs[0].shape is not None:
            shape = self.outputs[0].shape
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            
            # Scale cost by number of elements (logarithmically)
            import math
            size_factor = math.log10(max(num_elements, 1)) / 6.0  # Normalize to ~1.0 for 1M elements
            base_cost *= (1.0 + size_factor)
        
        return base_cost
    
    def __hash__(self):
        """Make ConductorNode hashable for use in sets and dictionaries."""
        # Use id() to ensure unique hash for each instance
        return hash(id(self))
    
    def __eq__(self, other):
        """Define equality based on object identity."""
        return self is other


@dataclass
class ComputationDAG:
    """
    Represents the complete computation graph as a directed acyclic graph.
    
    This class maintains the full graph structure with nodes, edges,
    and metadata needed for optimization and code generation.
    """
    nodes: List[ConductorNode] = field(default_factory=list)
    buffers: List[Buffer] = field(default_factory=list)
    inputs: List[Buffer] = field(default_factory=list)
    outputs: List[Buffer] = field(default_factory=list)
    
    def add_node(self, node: ConductorNode) -> None:
        """Add a node to the computation graph."""
        self.nodes.append(node)
        
    def add_buffer(self, buffer: Buffer) -> None:
        """Add a buffer to the computation graph."""
        self.buffers.append(buffer)
        
    def validate_graph_correctness(self) -> bool:
        """
        Verify graph integrity and detect potential issues.
        
        Returns:
            True if graph is valid, False otherwise
        """
        # Basic structural validation
        try:
            # Check that all nodes have valid inputs/outputs
            for node in self.nodes:
                # All input buffers should exist in the DAG
                for input_buffer in node.inputs:
                    if input_buffer not in self.buffers:
                        return False
                
                # All output buffers should exist in the DAG
                for output_buffer in node.outputs:
                    if output_buffer not in self.buffers:
                        return False
            
            # Check that input/output lists contain valid buffers
            for input_buffer in self.inputs:
                if input_buffer not in self.buffers:
                    return False
            
            for output_buffer in self.outputs:
                if output_buffer not in self.buffers:
                    return False
            
            return True
        
        except Exception:
            return False


class GraphAnalyzer:
    """
    Analyzes FX Graph and builds internal representation.
    
    This class handles the conversion from PyTorch's FX Graph format
    to Conductor's internal DAG representation, including data dependency
    analysis and graph validation.
    """
    
    def __init__(self):
        """Initialize the GraphAnalyzer with buffer management."""
        from .buffers import BufferManager
        self.buffer_manager = BufferManager()
        self._node_to_buffer = {}  # Maps FX nodes to output buffers
        self._buffer_to_node = {}  # Maps buffers to producing nodes
    
    def parse_fx_graph(self, graph_module: torch.fx.GraphModule, example_inputs: Optional[List[torch.Tensor]] = None) -> ComputationDAG:
        """
        Convert FX Graph to internal DAG representation.
        
        Args:
            graph_module: PyTorch FX Graph to convert
            
        Returns:
            ComputationDAG representing the input graph
        """
        dag = ComputationDAG()
        
        # Reset internal state
        self._node_to_buffer.clear()
        self._buffer_to_node.clear()
        
        # First pass: create buffers for all FX nodes
        for node in graph_module.graph.nodes:
            self._create_buffer_for_node(node, dag)
        
        # Second pass: create ConductorNodes and establish connections
        for node in graph_module.graph.nodes:
            conductor_node = self._convert_fx_node(node, dag)
            if conductor_node:
                dag.add_node(conductor_node)
        
        # Identify inputs and outputs with shape information from example inputs
        self._identify_graph_inputs_outputs(graph_module.graph, dag, example_inputs)

        # CRITICAL FIX: Third pass - propagate shapes after input shapes are set
        self._propagate_shapes_for_elementwise_ops(dag)

        # Analyze data dependencies
        self.identify_data_dependencies(dag)
        
        return dag

    def _propagate_shapes_for_elementwise_ops(self, dag: ComputationDAG) -> None:
        """
        Propagate shapes from input buffers to output buffers for element-wise operations.
        This runs after input shapes are set from example_inputs.
        """
        for node in dag.nodes:
            if node.op_name in ['add', 'mul', 'custom_add', 'custom_mul']:
                # For element-wise operations, output shape should match input shape
                if node.inputs and node.outputs:
                    input_buffer = node.inputs[0]
                    output_buffer = node.outputs[0]

                    if input_buffer.shape and not output_buffer.shape:
                        output_buffer.shape = input_buffer.shape
                        output_buffer.dtype = input_buffer.dtype
                        logger.debug(f"Propagated shape for {output_buffer.name}: {output_buffer.shape}")

    def _create_buffer_for_node(self, fx_node: torch.fx.Node, dag: ComputationDAG) -> None:
        """
        Create a buffer for an FX node's output.
        
        Args:
            fx_node: FX Graph node
            dag: Computation DAG being built
        """
        if fx_node.op in ('placeholder', 'get_attr', 'call_function', 'call_method', 'call_module'):
            # Determine buffer properties
            buffer_name = self._generate_buffer_name(fx_node)
            dtype, shape = self._infer_buffer_properties(fx_node)
            
            # Create buffer
            buffer = self.buffer_manager.allocate_buffer(buffer_name, dtype, shape)
            
            # Add to DAG and tracking
            dag.add_buffer(buffer)
            self._node_to_buffer[fx_node] = buffer
            self._buffer_to_node[buffer] = fx_node
    
    def _convert_fx_node(self, fx_node: torch.fx.Node, dag: ComputationDAG) -> Optional[ConductorNode]:
        """
        Convert an FX node to a ConductorNode.
        
        Args:
            fx_node: FX Graph node to convert
            dag: Computation DAG being built
            
        Returns:
            ConductorNode if conversion successful, None otherwise
        """
        if fx_node.op == 'placeholder':
            # Input nodes don't create operations
            return None
        
        elif fx_node.op == 'output':
            # Output nodes don't create operations
            return None
        
        elif fx_node.op == 'get_attr':
            # Attribute access (constants, parameters) - no operation needed
            return None
        
        elif fx_node.op in ('call_function', 'call_method', 'call_module'):
            return self._convert_operation_node(fx_node, dag)
        
        return None
    
    def _convert_operation_node(self, fx_node: torch.fx.Node, dag: ComputationDAG) -> Optional[ConductorNode]:
        """
        Convert an FX operation node to ConductorNode.

        Args:
            fx_node: FX operation node
            dag: Computation DAG being built

        Returns:
            ConductorNode representing the operation
        """
        # Get operation name
        op_name = self._extract_operation_name(fx_node)

        # Get input buffers
        input_buffers = []
        for arg in fx_node.args:
            if isinstance(arg, torch.fx.Node) and arg in self._node_to_buffer:
                input_buffers.append(self._node_to_buffer[arg])

        # Handle keyword arguments
        for key, value in fx_node.kwargs.items():
            if isinstance(value, torch.fx.Node) and value in self._node_to_buffer:
                input_buffers.append(self._node_to_buffer[value])

        # Get output buffer
        output_buffer = self._node_to_buffer.get(fx_node)
        output_buffers = [output_buffer] if output_buffer else []

        # Note: Shape inference is now handled in _propagate_shapes_for_elementwise_ops

        # Extract metadata
        metadata = self._extract_node_metadata(fx_node)

        # Create ConductorNode
        conductor_node = ConductorNode(
            op_name=op_name,
            inputs=input_buffers,
            outputs=output_buffers,
            metadata=metadata
        )
        
        # Update buffer producer/consumer relationships
        for buf in input_buffers:
            buf.consumers.append(conductor_node)
        
        for buf in output_buffers:
            buf.producer = conductor_node
        
        return conductor_node
    
    def _extract_operation_name(self, fx_node: torch.fx.Node) -> str:
        """
        Extract operation name from FX node.
        
        Args:
            fx_node: FX Graph node
            
        Returns:
            Operation name string
        """
        if fx_node.op == 'call_function':
            # Function calls like torch.add, torch.relu
            target = fx_node.target
            if hasattr(target, '__name__'):
                name = target.__name__
            else:
                name = str(target).split('.')[-1]
            
            # Map common PyTorch functions to simplified names
            name_mapping = {
                'add': 'add',
                'mul': 'mul', 
                'sub': 'sub',
                'div': 'div',
                'relu': 'relu',
                'sigmoid': 'sigmoid',
                'tanh': 'tanh',
                'sum': 'sum',
                'mean': 'mean',
                'max': 'max',
                'min': 'min',
                'matmul': 'matmul',
                'mm': 'matmul',  # Matrix multiply alias
                'bmm': 'matmul', # Batch matrix multiply
                'addmm': 'addmm', # Add matrix multiply
                'linear': 'linear',
                'conv2d': 'conv2d',
                'reshape': 'reshape',
                'view': 'reshape',  # View is essentially reshape
                'transpose': 'transpose',
                'permute': 'permute',
                'abs': 'abs',
                'neg': 'neg',
                'exp': 'exp',
                'log': 'log',
                'sqrt': 'sqrt',
                'sin': 'sin',
                'cos': 'cos',
            }
            
            return name_mapping.get(name, name)
        
        elif fx_node.op == 'call_method':
            # Method calls like tensor.add(), tensor.relu()
            method_name = fx_node.target
            
            # Map common tensor methods
            method_mapping = {
                'add': 'add',
                'add_': 'add',  # In-place version
                'mul': 'mul',
                'mul_': 'mul',
                'sub': 'sub', 
                'sub_': 'sub',
                'div': 'div',
                'div_': 'div',
                'relu': 'relu',
                'relu_': 'relu',
                'sigmoid': 'sigmoid',
                'sigmoid_': 'sigmoid',
                'tanh': 'tanh',
                'tanh_': 'tanh',
                'sum': 'sum',
                'mean': 'mean',
                'max': 'max',
                'min': 'min',
                'matmul': 'matmul',
                'mm': 'matmul',
                'reshape': 'reshape',
                'view': 'reshape',
                'transpose': 'transpose',
                'permute': 'permute',
                'abs': 'abs',
                'abs_': 'abs',
                'neg': 'neg',
                'neg_': 'neg',
                'exp': 'exp',
                'exp_': 'exp',
                'log': 'log',
                'log_': 'log',
                'sqrt': 'sqrt',
                'sqrt_': 'sqrt',
            }
            
            return method_mapping.get(method_name, method_name)
        
        elif fx_node.op == 'call_module':
            # Module calls like nn.Linear, nn.Conv2d
            # For now, use the module class name
            return fx_node.target.split('.')[-1].lower()
        
        return 'unknown'
    
    def _extract_node_metadata(self, fx_node: torch.fx.Node) -> Dict[str, Any]:
        """
        Extract metadata from FX node.
        
        Args:
            fx_node: FX Graph node
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Add keyword arguments as metadata
        if fx_node.kwargs:
            metadata.update(fx_node.kwargs)
        
        # Add positional arguments that are not nodes (constants, etc.)
        non_node_args = []
        for arg in fx_node.args:
            if not isinstance(arg, torch.fx.Node):
                non_node_args.append(arg)
        
        if non_node_args:
            metadata['args'] = non_node_args
        
        # Add node-specific metadata
        if hasattr(fx_node, 'meta'):
            metadata.update(fx_node.meta)
        
        return metadata
    
    def _generate_buffer_name(self, fx_node: torch.fx.Node) -> str:
        """
        Generate a unique buffer name for an FX node.
        
        Args:
            fx_node: FX Graph node
            
        Returns:
            Unique buffer name
        """
        # Use the FX node name as base, ensuring uniqueness
        base_name = fx_node.name if fx_node.name else f"node_{id(fx_node)}"
        
        # Clean up the name (remove special characters)
        import re
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)
        
        return clean_name
    
    def _infer_buffer_properties(self, fx_node: torch.fx.Node) -> tuple:
        """
        Infer buffer dtype and shape from FX node.
        
        Args:
            fx_node: FX Graph node
            
        Returns:
            Tuple of (dtype, shape)
        """
        dtype = torch.float32  # Default dtype
        shape = None  # Unknown shape by default
        
        # Try to get type information from node metadata
        if hasattr(fx_node, 'meta') and 'tensor_meta' in fx_node.meta:
            tensor_meta = fx_node.meta['tensor_meta']
            if hasattr(tensor_meta, 'dtype'):
                dtype = tensor_meta.dtype
            if hasattr(tensor_meta, 'shape'):
                shape = tuple(tensor_meta.shape)
        
        # Try to get type from 'val' metadata (common in some FX graphs)
        elif hasattr(fx_node, 'meta') and 'val' in fx_node.meta:
            val = fx_node.meta['val']
            if hasattr(val, 'dtype'):
                dtype = val.dtype
            if hasattr(val, 'shape'):
                shape = tuple(val.shape)
        
        return dtype, shape
    
    def _identify_graph_inputs_outputs(self, fx_graph: torch.fx.Graph, dag: ComputationDAG, example_inputs: Optional[List[torch.Tensor]] = None) -> None:
        """
        Identify input and output buffers of the graph.

        Args:
            fx_graph: FX Graph
            dag: Computation DAG being built
            example_inputs: Example input tensors for shape inference
        """
        # Find input nodes (placeholders) and update their shapes from example inputs
        input_index = 0
        for node in fx_graph.nodes:
            if node.op == 'placeholder' and node in self._node_to_buffer:
                buffer = self._node_to_buffer[node]

                # Update buffer shape from example inputs if available
                if example_inputs and input_index < len(example_inputs):
                    example_tensor = example_inputs[input_index]
                    buffer.shape = tuple(example_tensor.shape)
                    buffer.dtype = example_tensor.dtype
                    logger.debug(f"Updated input buffer {buffer.name} shape to {buffer.shape} from example input")
                    input_index += 1

                dag.inputs.append(buffer)
        
        # Find output nodes
        for node in fx_graph.nodes:
            if node.op == 'output':
                # Output node args contain the actual output values
                for arg in node.args:
                    if isinstance(arg, (list, tuple)):
                        # Multiple outputs
                        for output_node in arg:
                            if isinstance(output_node, torch.fx.Node) and output_node in self._node_to_buffer:
                                buffer = self._node_to_buffer[output_node]
                                dag.outputs.append(buffer)
                    elif isinstance(arg, torch.fx.Node) and arg in self._node_to_buffer:
                        # Single output
                        buffer = self._node_to_buffer[arg]

                        # Infer output shape from input shapes for element-wise operations
                        if dag.inputs and not buffer.shape:
                            # For element-wise operations, output shape matches input shape
                            buffer.shape = dag.inputs[0].shape
                            buffer.dtype = dag.inputs[0].dtype
                            logger.debug(f"Inferred output buffer {buffer.name} shape to {buffer.shape} from input")

                        dag.outputs.append(buffer)
        
    def identify_data_dependencies(self, dag: ComputationDAG) -> None:
        """
        Analyze data flow and establish buffer dependencies.
        
        Args:
            dag: Computation DAG to analyze
        """
        # Data dependencies are already established during node conversion
        # through the producer/consumer relationships in buffers
        
        # Additional validation: ensure all buffers have proper connections
        for buffer in dag.buffers:
            # Input buffers should have no producer
            if buffer in dag.inputs:
                if buffer.producer is not None:
                    raise ValueError(f"Input buffer {buffer.name} should not have a producer")
            
            # Non-input buffers should have exactly one producer
            elif buffer.producer is None:
                # This might be a constant or parameter buffer, which is okay
                pass
            
            # Output buffers should have at least one consumer or be graph outputs
            if not buffer.consumers and buffer not in dag.outputs:
                # Mark as temporary buffer
                buffer.is_temporary = True
        
        # Promote buffer scopes based on usage patterns
        self._optimize_buffer_scopes(dag)
    
    def _optimize_buffer_scopes(self, dag: ComputationDAG) -> None:
        """
        Optimize buffer scopes based on usage patterns.
        
        Args:
            dag: Computation DAG to optimize
        """
        from .buffers import BufferScope
        
        for buffer in dag.buffers:
            # Input and output buffers should be GLOBAL scope
            if buffer in dag.inputs or buffer in dag.outputs:
                buffer.promote_scope(BufferScope.GLOBAL)
            
            # Buffers with multiple consumers should be SHARED scope
            elif len(buffer.consumers) > 1:
                buffer.promote_scope(BufferScope.SHARED)
            
            # Single-consumer buffers can remain LOCAL
            # (already set by default in BufferManager)
        
    def validate_graph_correctness(self, dag: ComputationDAG) -> bool:
        """
        Verify graph integrity and detect potential issues.
        
        Args:
            dag: Computation DAG to validate
            
        Returns:
            True if graph is valid, False otherwise
        """
        try:
            # Check for cycles (DAG property)
            if self._has_cycles(dag):
                return False
            
            # Check buffer consistency
            if not self._validate_buffer_consistency(dag):
                return False
            
            # Check node connectivity
            if not self._validate_node_connectivity(dag):
                return False
            
            return True
        
        except Exception:
            return False
    
    def _has_cycles(self, dag: ComputationDAG) -> bool:
        """
        Check if the graph has cycles using DFS.
        
        Args:
            dag: Computation DAG to check
            
        Returns:
            True if cycles exist, False otherwise
        """
        # Use DFS with coloring: white (0), gray (1), black (2)
        color = {node: 0 for node in dag.nodes}
        
        def dfs(node):
            if color[node] == 1:  # Gray - back edge found (cycle)
                return True
            if color[node] == 2:  # Black - already processed
                return False
            
            color[node] = 1  # Mark as gray (visiting)
            
            # Visit all successors
            for output_buffer in node.outputs:
                for consumer in output_buffer.consumers:
                    if dfs(consumer):
                        return True
            
            color[node] = 2  # Mark as black (processed)
            return False
        
        # Check all nodes
        for node in dag.nodes:
            if color[node] == 0 and dfs(node):
                return True
        
        return False
    
    def _validate_buffer_consistency(self, dag: ComputationDAG) -> bool:
        """
        Validate buffer producer/consumer consistency.
        
        Args:
            dag: Computation DAG to validate
            
        Returns:
            True if consistent, False otherwise
        """
        for buffer in dag.buffers:
            # Check producer consistency
            if buffer.producer:
                if buffer not in buffer.producer.outputs:
                    return False
            
            # Check consumer consistency
            for consumer in buffer.consumers:
                if buffer not in consumer.inputs:
                    return False
        
        return True
    
    def _validate_node_connectivity(self, dag: ComputationDAG) -> bool:
        """
        Validate node input/output connectivity.
        
        Args:
            dag: Computation DAG to validate
            
        Returns:
            True if properly connected, False otherwise
        """
        for node in dag.nodes:
            # Check input buffer consistency
            for input_buffer in node.inputs:
                if node not in input_buffer.consumers:
                    return False
            
            # Check output buffer consistency
            for output_buffer in node.outputs:
                if output_buffer.producer != node:
                    return False
        
        return True
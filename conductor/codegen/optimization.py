"""
Advanced optimization algorithms for Conductor compilation.

This module implements sophisticated optimization techniques including
buffer reuse optimization, memory layout optimization, and advanced
fusion heuristics for complex operation patterns.
"""

import math
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from .buffers import Buffer, BufferScope, BufferManager
from .graph import ConductorNode, ComputationDAG
from .fusion import FusionCluster, FusionType
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MemoryLayoutStrategy(Enum):
    """Memory layout optimization strategies."""
    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"
    BLOCKED = "blocked"
    CACHE_FRIENDLY = "cache_friendly"
    VECTORIZED = "vectorized"


@dataclass
class OptimizationHint:
    """Optimization hint for specific operations or patterns."""
    operation: str
    hint_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    
    def __post_init__(self):
        """Validate optimization hint."""
        if self.priority < 1 or self.priority > 10:
            raise ValueError("Priority must be between 1 and 10")


class BufferReuseOptimizer:
    """
    Advanced buffer reuse optimization.
    
    This class implements sophisticated algorithms to maximize buffer
    reuse and minimize memory allocation overhead.
    """
    
    def __init__(self):
        """Initialize buffer reuse optimizer."""
        self.reuse_graph = {}
        self.interference_graph = {}
        self.liveness_intervals = {}
    
    def optimize_buffer_reuse(self, dag: ComputationDAG) -> Dict[str, str]:
        """
        Optimize buffer reuse across the computation graph.
        
        Args:
            dag: Computation DAG to optimize
            
        Returns:
            Dictionary mapping original buffer names to reused buffer names
        """
        logger.info("Starting advanced buffer reuse optimization")
        
        # Build liveness analysis
        self._analyze_buffer_liveness(dag)
        
        # Build interference graph
        self._build_interference_graph(dag)
        
        # Apply graph coloring for buffer allocation
        reuse_mapping = self._apply_graph_coloring()
        
        # Validate reuse mapping
        self._validate_reuse_mapping(reuse_mapping, dag)
        
        logger.info(f"Buffer reuse optimization completed: {len(reuse_mapping)} reuses")
        return reuse_mapping
    
    def _analyze_buffer_liveness(self, dag: ComputationDAG):
        """Analyze buffer liveness intervals."""
        self.liveness_intervals = {}
        
        # Topological ordering of nodes
        ordered_nodes = self._topological_sort(dag.nodes)
        
        for i, node in enumerate(ordered_nodes):
            # Input buffers are live at this point
            for input_buf in node.inputs:
                if input_buf.name not in self.liveness_intervals:
                    self.liveness_intervals[input_buf.name] = {'start': i, 'end': i}
                else:
                    self.liveness_intervals[input_buf.name]['end'] = i
            
            # Output buffers become live
            for output_buf in node.outputs:
                if output_buf.name not in self.liveness_intervals:
                    self.liveness_intervals[output_buf.name] = {'start': i, 'end': i}
                
                # Extend liveness to last consumer
                last_consumer_idx = self._find_last_consumer(output_buf, ordered_nodes)
                if last_consumer_idx is not None:
                    self.liveness_intervals[output_buf.name]['end'] = last_consumer_idx
    
    def _build_interference_graph(self, dag: ComputationDAG):
        """Build interference graph for buffer allocation."""
        self.interference_graph = {}
        
        for buf1_name, interval1 in self.liveness_intervals.items():
            self.interference_graph[buf1_name] = set()
            
            for buf2_name, interval2 in self.liveness_intervals.items():
                if buf1_name != buf2_name:
                    # Check if intervals overlap
                    if self._intervals_overlap(interval1, interval2):
                        # Check if buffers are compatible for reuse
                        buf1 = self._find_buffer_by_name(dag, buf1_name)
                        buf2 = self._find_buffer_by_name(dag, buf2_name)
                        
                        if buf1 and buf2 and not self._buffers_compatible(buf1, buf2):
                            self.interference_graph[buf1_name].add(buf2_name)
    
    def _apply_graph_coloring(self) -> Dict[str, str]:
        """Apply graph coloring algorithm for buffer allocation."""
        reuse_mapping = {}
        color_to_buffer = {}
        buffer_to_color = {}
        
        # Sort buffers by degree (most constrained first)
        buffers_by_degree = sorted(
            self.interference_graph.keys(),
            key=lambda b: len(self.interference_graph[b]),
            reverse=True
        )
        
        for buffer_name in buffers_by_degree:
            # Find available colors (reusable buffers)
            used_colors = set()
            for neighbor in self.interference_graph[buffer_name]:
                if neighbor in buffer_to_color:
                    used_colors.add(buffer_to_color[neighbor])
            
            # Find first available color
            color = 0
            while color in used_colors:
                color += 1
            
            buffer_to_color[buffer_name] = color
            
            if color in color_to_buffer:
                # Reuse existing buffer
                reuse_mapping[buffer_name] = color_to_buffer[color]
            else:
                # Create new buffer group
                color_to_buffer[color] = buffer_name
        
        return reuse_mapping
    
    def _intervals_overlap(self, interval1: Dict, interval2: Dict) -> bool:
        """Check if two liveness intervals overlap."""
        return not (interval1['end'] < interval2['start'] or interval2['end'] < interval1['start'])
    
    def _buffers_compatible(self, buf1: Buffer, buf2: Buffer) -> bool:
        """Check if two buffers are compatible for reuse."""
        # Same data type and shape
        if buf1.dtype != buf2.dtype:
            return False
        
        if buf1.shape != buf2.shape:
            return False
        
        # Compatible scopes
        if buf1.scope != buf2.scope:
            return False
        
        # Both must be temporary
        if not (buf1.is_temporary and buf2.is_temporary):
            return False
        
        return True
    
    def _topological_sort(self, nodes: List[ConductorNode]) -> List[ConductorNode]:
        """Topological sort of computation nodes."""
        # Simple topological sort implementation
        # In practice, this would use the actual dependency graph
        return nodes
    
    def _find_last_consumer(self, buffer: Buffer, ordered_nodes: List[ConductorNode]) -> Optional[int]:
        """Find the index of the last consumer of a buffer."""
        last_idx = None
        for i, node in enumerate(ordered_nodes):
            if buffer in node.inputs:
                last_idx = i
        return last_idx
    
    def _find_buffer_by_name(self, dag: ComputationDAG, name: str) -> Optional[Buffer]:
        """Find buffer by name in DAG."""
        for buffer in dag.buffers:
            if buffer.name == name:
                return buffer
        return None
    
    def _validate_reuse_mapping(self, reuse_mapping: Dict[str, str], dag: ComputationDAG):
        """Validate that buffer reuse mapping is correct."""
        for original, reused in reuse_mapping.items():
            original_buf = self._find_buffer_by_name(dag, original)
            reused_buf = self._find_buffer_by_name(dag, reused)
            
            if original_buf and reused_buf:
                assert self._buffers_compatible(original_buf, reused_buf), \
                    f"Incompatible buffer reuse: {original} -> {reused}"


class MemoryLayoutOptimizer:
    """
    Memory layout optimization for cache efficiency.
    
    This class implements algorithms to optimize memory access patterns
    and data layout for better cache performance.
    """
    
    def __init__(self):
        """Initialize memory layout optimizer."""
        self.cache_line_size = 64  # bytes
        self.l1_cache_size = 32 * 1024  # 32KB
        self.l2_cache_size = 256 * 1024  # 256KB
    
    def optimize_memory_layout(self, dag: ComputationDAG) -> Dict[str, OptimizationHint]:
        """
        Optimize memory layout for cache efficiency.
        
        Args:
            dag: Computation DAG to optimize
            
        Returns:
            Dictionary of optimization hints for buffers
        """
        logger.info("Starting memory layout optimization")
        
        optimization_hints = {}
        
        # Analyze access patterns
        access_patterns = self._analyze_access_patterns(dag)
        
        # Optimize layout for each buffer
        for buffer in dag.buffers:
            if buffer.shape and len(buffer.shape) >= 2:
                hint = self._optimize_buffer_layout(buffer, access_patterns.get(buffer.name, {}))
                if hint:
                    optimization_hints[buffer.name] = hint
        
        # Optimize for vectorization
        vectorization_hints = self._optimize_for_vectorization(dag)
        optimization_hints.update(vectorization_hints)
        
        logger.info(f"Memory layout optimization completed: {len(optimization_hints)} hints")
        return optimization_hints
    
    def _analyze_access_patterns(self, dag: ComputationDAG) -> Dict[str, Dict[str, Any]]:
        """Analyze memory access patterns for each buffer."""
        access_patterns = {}
        
        for node in dag.nodes:
            # Analyze input access patterns
            for input_buf in node.inputs:
                pattern = self._get_operation_access_pattern(node.op_name, 'input')
                if input_buf.name not in access_patterns:
                    access_patterns[input_buf.name] = {}
                access_patterns[input_buf.name].update(pattern)
            
            # Analyze output access patterns
            for output_buf in node.outputs:
                pattern = self._get_operation_access_pattern(node.op_name, 'output')
                if output_buf.name not in access_patterns:
                    access_patterns[output_buf.name] = {}
                access_patterns[output_buf.name].update(pattern)
        
        return access_patterns
    
    def _get_operation_access_pattern(self, op_name: str, buffer_type: str) -> Dict[str, Any]:
        """Get access pattern for specific operation and buffer type."""
        patterns = {
            'matmul': {
                'input': {'pattern': 'row_major', 'stride': 'unit', 'locality': 'high'},
                'output': {'pattern': 'row_major', 'stride': 'unit', 'locality': 'high'}
            },
            'conv2d': {
                'input': {'pattern': 'blocked', 'stride': 'strided', 'locality': 'medium'},
                'output': {'pattern': 'blocked', 'stride': 'unit', 'locality': 'high'}
            },
            'transpose': {
                'input': {'pattern': 'column_major', 'stride': 'strided', 'locality': 'low'},
                'output': {'pattern': 'row_major', 'stride': 'unit', 'locality': 'medium'}
            },
            'add': {
                'input': {'pattern': 'row_major', 'stride': 'unit', 'locality': 'high'},
                'output': {'pattern': 'row_major', 'stride': 'unit', 'locality': 'high'}
            },
            'relu': {
                'input': {'pattern': 'row_major', 'stride': 'unit', 'locality': 'high'},
                'output': {'pattern': 'row_major', 'stride': 'unit', 'locality': 'high'}
            }
        }
        
        return patterns.get(op_name, {}).get(buffer_type, {
            'pattern': 'row_major', 'stride': 'unit', 'locality': 'medium'
        })
    
    def _optimize_buffer_layout(self, buffer: Buffer, access_pattern: Dict[str, Any]) -> Optional[OptimizationHint]:
        """Optimize layout for a specific buffer."""
        if not buffer.shape or len(buffer.shape) < 2:
            return None
        
        # Calculate memory footprint
        footprint = buffer.get_memory_footprint()
        
        # Choose layout strategy based on access pattern and size
        if footprint > self.l2_cache_size:
            # Large buffers: use blocked layout
            strategy = MemoryLayoutStrategy.BLOCKED
            block_size = int(math.sqrt(self.l1_cache_size // 4))  # Assume float32
        elif access_pattern.get('pattern') == 'column_major':
            strategy = MemoryLayoutStrategy.COLUMN_MAJOR
            block_size = None
        else:
            strategy = MemoryLayoutStrategy.ROW_MAJOR
            block_size = None
        
        return OptimizationHint(
            operation='memory_layout',
            hint_type='layout_strategy',
            parameters={
                'strategy': strategy.value,
                'block_size': block_size,
                'alignment': self.cache_line_size
            },
            priority=5
        )
    
    def _optimize_for_vectorization(self, dag: ComputationDAG) -> Dict[str, OptimizationHint]:
        """Optimize buffers for vectorization."""
        hints = {}
        
        for buffer in dag.buffers:
            if buffer.shape and buffer.shape[-1] % 4 == 0:  # Vectorizable
                hints[f"{buffer.name}_vectorization"] = OptimizationHint(
                    operation='vectorization',
                    hint_type='simd_optimization',
                    parameters={
                        'vector_width': 4,
                        'alignment': 16,
                        'prefetch': True
                    },
                    priority=7
                )
        
        return hints


class AdvancedFusionHeuristics:
    """
    Advanced fusion heuristics for complex operation patterns.
    
    This class implements sophisticated fusion strategies that go beyond
    simple elementwise and reduction fusion.
    """
    
    def __init__(self):
        """Initialize advanced fusion heuristics."""
        self.fusion_patterns = self._initialize_fusion_patterns()
    
    def _initialize_fusion_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize known fusion patterns."""
        return {
            'conv_bn_relu': {
                'pattern': ['conv2d', 'batch_norm', 'relu'],
                'type': FusionType.COMPUTE_BOUND,
                'benefit': 8.0,
                'constraints': ['same_spatial_dims']
            },
            'linear_gelu': {
                'pattern': ['linear', 'gelu'],
                'type': FusionType.COMPUTE_BOUND,
                'benefit': 6.0,
                'constraints': ['compatible_shapes']
            },
            'attention_pattern': {
                'pattern': ['matmul', 'softmax', 'matmul'],
                'type': FusionType.MEMORY_BOUND,
                'benefit': 10.0,
                'constraints': ['attention_semantics']
            },
            'layer_norm_pattern': {
                'pattern': ['mean', 'sub', 'pow', 'mean', 'add', 'sqrt', 'div', 'mul', 'add'],
                'type': FusionType.REDUCTION,
                'benefit': 12.0,
                'constraints': ['reduction_dims']
            }
        }
    
    def identify_advanced_fusion_opportunities(self, dag: ComputationDAG) -> List[FusionCluster]:
        """
        Identify advanced fusion opportunities in the computation graph.
        
        Args:
            dag: Computation DAG to analyze
            
        Returns:
            List of advanced fusion clusters
        """
        logger.info("Analyzing advanced fusion opportunities")
        
        clusters = []
        
        # Pattern matching for known fusion patterns
        for pattern_name, pattern_info in self.fusion_patterns.items():
            pattern_clusters = self._find_pattern_matches(dag, pattern_name, pattern_info)
            clusters.extend(pattern_clusters)
        
        # Data flow analysis for custom patterns
        dataflow_clusters = self._analyze_dataflow_patterns(dag)
        clusters.extend(dataflow_clusters)
        
        # Memory access pattern fusion
        memory_clusters = self._analyze_memory_access_patterns(dag)
        clusters.extend(memory_clusters)
        
        # Validate and rank clusters
        validated_clusters = self._validate_and_rank_clusters(clusters)
        
        logger.info(f"Found {len(validated_clusters)} advanced fusion opportunities")
        return validated_clusters
    
    def _find_pattern_matches(self, dag: ComputationDAG, pattern_name: str, pattern_info: Dict[str, Any]) -> List[FusionCluster]:
        """Find matches for a specific fusion pattern."""
        clusters = []
        pattern_ops = pattern_info['pattern']
        
        # Simple pattern matching (in practice, this would be more sophisticated)
        for i in range(len(dag.nodes) - len(pattern_ops) + 1):
            candidate_nodes = dag.nodes[i:i + len(pattern_ops)]
            
            if self._matches_pattern(candidate_nodes, pattern_ops, pattern_info['constraints']):
                cluster = self._create_pattern_cluster(
                    candidate_nodes, 
                    pattern_name, 
                    pattern_info['type'],
                    pattern_info['benefit']
                )
                clusters.append(cluster)
        
        return clusters
    
    def _matches_pattern(self, nodes: List[ConductorNode], pattern: List[str], constraints: List[str]) -> bool:
        """Check if nodes match a fusion pattern."""
        if len(nodes) != len(pattern):
            return False
        
        # Check operation sequence
        for node, expected_op in zip(nodes, pattern):
            if node.op_name != expected_op:
                return False
        
        # Check constraints
        for constraint in constraints:
            if not self._check_constraint(nodes, constraint):
                return False
        
        return True
    
    def _check_constraint(self, nodes: List[ConductorNode], constraint: str) -> bool:
        """Check a specific fusion constraint."""
        if constraint == 'same_spatial_dims':
            # Check that spatial dimensions are preserved
            return True  # Simplified check
        elif constraint == 'compatible_shapes':
            # Check shape compatibility
            return True  # Simplified check
        elif constraint == 'attention_semantics':
            # Check attention pattern semantics
            return True  # Simplified check
        elif constraint == 'reduction_dims':
            # Check reduction dimension consistency
            return True  # Simplified check
        
        return True
    
    def _create_pattern_cluster(self, nodes: List[ConductorNode], pattern_name: str, fusion_type: FusionType, benefit: float) -> FusionCluster:
        """Create fusion cluster for matched pattern."""
        # Collect external inputs and outputs
        external_inputs = []
        external_outputs = []
        internal_buffers = []
        
        all_inputs = set()
        all_outputs = set()
        
        for node in nodes:
            all_inputs.update(node.inputs)
            all_outputs.update(node.outputs)
        
        # External inputs: inputs not produced by nodes in cluster
        for inp in all_inputs:
            if inp.producer not in nodes:
                external_inputs.append(inp)
        
        # External outputs: outputs consumed outside cluster
        for out in all_outputs:
            if any(consumer not in nodes for consumer in out.consumers):
                external_outputs.append(out)
            else:
                internal_buffers.append(out)
        
        return FusionCluster(
            nodes=nodes,
            cluster_type=fusion_type,
            external_inputs=external_inputs,
            external_outputs=external_outputs,
            internal_buffers=internal_buffers,
            dsl_function_name=f"fused_{pattern_name}"
        )
    
    def _analyze_dataflow_patterns(self, dag: ComputationDAG) -> List[FusionCluster]:
        """Analyze data flow patterns for fusion opportunities."""
        clusters = []
        
        # Find producer-consumer chains
        chains = self._find_producer_consumer_chains(dag)
        
        for chain in chains:
            if len(chain) >= 2 and self._is_fusible_chain(chain):
                cluster = self._create_dataflow_cluster(chain)
                clusters.append(cluster)
        
        return clusters
    
    def _find_producer_consumer_chains(self, dag: ComputationDAG) -> List[List[ConductorNode]]:
        """Find chains of producer-consumer relationships."""
        chains = []
        visited = set()
        
        for node in dag.nodes:
            if node in visited:
                continue
            
            chain = self._build_chain_from_node(node, dag, visited)
            if len(chain) > 1:
                chains.append(chain)
        
        return chains
    
    def _build_chain_from_node(self, start_node: ConductorNode, dag: ComputationDAG, visited: set) -> List[ConductorNode]:
        """Build a chain starting from a specific node."""
        chain = [start_node]
        visited.add(start_node)
        
        current_node = start_node
        while True:
            # Find next node in chain
            next_node = None
            for output_buf in current_node.outputs:
                if len(output_buf.consumers) == 1:  # Single consumer
                    consumer = output_buf.consumers[0]
                    if consumer not in visited and len(consumer.inputs) == 1:  # Single input
                        next_node = consumer
                        break
            
            if next_node is None:
                break
            
            chain.append(next_node)
            visited.add(next_node)
            current_node = next_node
        
        return chain
    
    def _is_fusible_chain(self, chain: List[ConductorNode]) -> bool:
        """Check if a chain of nodes is fusible."""
        # Check that all operations are compatible
        for node in chain:
            if not self._is_fusible_operation(node.op_name):
                return False
        
        # Check memory constraints
        total_memory = sum(
            sum(buf.get_memory_footprint() for buf in node.outputs if buf.shape)
            for node in chain
        )
        
        # Don't fuse if it would use too much memory
        if total_memory > 100 * 1024 * 1024:  # 100MB limit
            return False
        
        return True
    
    def _is_fusible_operation(self, op_name: str) -> bool:
        """Check if an operation is fusible."""
        fusible_ops = {
            'add', 'mul', 'sub', 'div', 'relu', 'sigmoid', 'tanh', 'gelu',
            'exp', 'log', 'sqrt', 'pow', 'abs', 'neg', 'sum', 'mean'
        }
        return op_name in fusible_ops
    
    def _create_dataflow_cluster(self, chain: List[ConductorNode]) -> FusionCluster:
        """Create fusion cluster from dataflow chain."""
        # Similar to pattern cluster creation
        external_inputs = []
        external_outputs = []
        internal_buffers = []
        
        # First node inputs are external
        external_inputs.extend(chain[0].inputs)
        
        # Last node outputs are external
        external_outputs.extend(chain[-1].outputs)
        
        # Intermediate buffers are internal
        for i in range(len(chain) - 1):
            internal_buffers.extend(chain[i].outputs)
        
        return FusionCluster(
            nodes=chain,
            cluster_type=FusionType.ELEMENTWISE,
            external_inputs=external_inputs,
            external_outputs=external_outputs,
            internal_buffers=internal_buffers,
            dsl_function_name=f"fused_chain_{len(chain)}_ops"
        )
    
    def _analyze_memory_access_patterns(self, dag: ComputationDAG) -> List[FusionCluster]:
        """Analyze memory access patterns for fusion opportunities."""
        # This would implement memory access pattern analysis
        # For now, return empty list
        return []
    
    def _validate_and_rank_clusters(self, clusters: List[FusionCluster]) -> List[FusionCluster]:
        """Validate and rank fusion clusters by benefit."""
        validated_clusters = []
        
        for cluster in clusters:
            if cluster.validate_fusion_safety():
                validated_clusters.append(cluster)
        
        # Sort by estimated performance gain
        validated_clusters.sort(
            key=lambda c: c.estimate_performance_gain(),
            reverse=True
        )
        
        return validated_clusters


class OptimizationPipeline:
    """
    Comprehensive optimization pipeline.
    
    This class coordinates all optimization passes to maximize
    performance while maintaining correctness.
    """
    
    def __init__(self):
        """Initialize optimization pipeline."""
        self.buffer_optimizer = BufferReuseOptimizer()
        self.layout_optimizer = MemoryLayoutOptimizer()
        self.fusion_heuristics = AdvancedFusionHeuristics()
    
    def optimize(self, dag: ComputationDAG) -> Dict[str, Any]:
        """
        Apply comprehensive optimization to computation graph.
        
        Args:
            dag: Computation DAG to optimize
            
        Returns:
            Dictionary with optimization results and statistics
        """
        logger.info("Starting comprehensive optimization pipeline")
        
        optimization_results = {
            'buffer_reuse_mapping': {},
            'memory_layout_hints': {},
            'fusion_clusters': [],
            'optimization_stats': {}
        }
        
        # Phase 1: Advanced fusion analysis
        fusion_clusters = self.fusion_heuristics.identify_advanced_fusion_opportunities(dag)
        optimization_results['fusion_clusters'] = fusion_clusters
        
        # Phase 2: Buffer reuse optimization
        buffer_reuse = self.buffer_optimizer.optimize_buffer_reuse(dag)
        optimization_results['buffer_reuse_mapping'] = buffer_reuse
        
        # Phase 3: Memory layout optimization
        layout_hints = self.layout_optimizer.optimize_memory_layout(dag)
        optimization_results['memory_layout_hints'] = layout_hints
        
        # Calculate optimization statistics
        stats = self._calculate_optimization_stats(dag, optimization_results)
        optimization_results['optimization_stats'] = stats
        
        logger.info("Optimization pipeline completed")
        return optimization_results
    
    def _calculate_optimization_stats(self, dag: ComputationDAG, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimization statistics."""
        total_nodes = len(dag.nodes)
        fused_nodes = sum(len(cluster.nodes) for cluster in results['fusion_clusters'])
        
        total_buffers = len(dag.buffers)
        reused_buffers = len(results['buffer_reuse_mapping'])
        
        return {
            'total_nodes': total_nodes,
            'fused_nodes': fused_nodes,
            'fusion_ratio': fused_nodes / total_nodes if total_nodes > 0 else 0.0,
            'total_buffers': total_buffers,
            'reused_buffers': reused_buffers,
            'buffer_reuse_ratio': reused_buffers / total_buffers if total_buffers > 0 else 0.0,
            'memory_layout_optimizations': len(results['memory_layout_hints']),
            'fusion_clusters': len(results['fusion_clusters'])
        }
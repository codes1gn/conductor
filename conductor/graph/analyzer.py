"""
Modern Graph Analysis Module for Conductor.

This module provides comprehensive analysis of computation graphs using
modern patterns and clean architecture principles.
"""

from __future__ import annotations

import torch
from typing import List, Dict, Set, Optional, Tuple, Any, Protocol
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod

from .graph_nodes import ConductorNode
from .graph_analyzer import ComputationDAG
from ..utils.constants import OpType, FusionType
from ..utils.logging import get_logger

logger = get_logger(__name__)


class GraphAnalyzer(Protocol):
    """Protocol for graph analyzers."""
    
    def analyze(self, dag: ComputationDAG) -> 'AnalysisResult':
        """Analyze the computation graph."""
        ...


@dataclass
class AnalysisResult:
    """Result of graph analysis."""
    
    # Topology information
    execution_order: List[ConductorNode] = field(default_factory=list)
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    
    # Fusion opportunities
    fusion_groups: List[List[ConductorNode]] = field(default_factory=list)
    fusion_types: Dict[str, FusionType] = field(default_factory=dict)
    
    # Memory analysis
    memory_requirements: Dict[str, int] = field(default_factory=dict)
    buffer_lifetimes: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    
    # Performance hints
    parallelization_opportunities: List[str] = field(default_factory=list)
    optimization_hints: Dict[str, Any] = field(default_factory=dict)


class TopologyAnalyzer:
    """Analyzes graph topology and dependencies."""
    
    def analyze_topology(self, dag: ComputationDAG) -> Tuple[List[ConductorNode], Dict[str, List[str]]]:
        """
        Analyze graph topology and return execution order and dependencies.
        
        Returns:
            Tuple of (execution_order, dependency_graph)
        """
        # Build dependency graph
        dependency_graph = {}
        node_map = {id(node): node for node in dag.nodes}
        
        for node in dag.nodes:
            node_id = str(id(node))
            dependency_graph[node_id] = []
            
            # Find dependencies based on input buffers
            for input_buffer in getattr(node, 'input_buffers', []):
                for other_node in dag.nodes:
                    if other_node != node:
                        for output_buffer in getattr(other_node, 'output_buffers', []):
                            if input_buffer.name == output_buffer.name:
                                dependency_graph[node_id].append(str(id(other_node)))
        
        # Topological sort
        execution_order = self._topological_sort(dag.nodes, dependency_graph)
        
        return execution_order, dependency_graph
    
    def _topological_sort(self, nodes: List[ConductorNode], deps: Dict[str, List[str]]) -> List[ConductorNode]:
        """Perform topological sort on the nodes."""
        # Kahn's algorithm
        in_degree = {str(id(node)): 0 for node in nodes}
        node_map = {str(id(node)): node for node in nodes}
        
        # Calculate in-degrees
        for node_id, dependencies in deps.items():
            for dep in dependencies:
                if dep in in_degree:
                    in_degree[node_id] += 1
        
        # Find nodes with no incoming edges
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(node_map[current])
            
            # Remove edges from current node
            for node_id, dependencies in deps.items():
                if current in dependencies:
                    in_degree[node_id] -= 1
                    if in_degree[node_id] == 0:
                        queue.append(node_id)
        
        if len(result) != len(nodes):
            raise ValueError("Graph contains cycles")
        
        return result


class FusionAnalyzer:
    """Analyzes fusion opportunities in the graph."""
    
    def analyze_fusion(self, dag: ComputationDAG) -> Tuple[List[List[ConductorNode]], Dict[str, FusionType]]:
        """
        Analyze fusion opportunities.
        
        Returns:
            Tuple of (fusion_groups, fusion_types)
        """
        fusion_groups = []
        fusion_types = {}
        
        # Group consecutive elementwise operations
        current_group = []
        
        for node in dag.nodes:
            op_type = self._classify_operation(node.op_name)
            
            if op_type == OpType.ELEMENTWISE:
                current_group.append(node)
            else:
                if len(current_group) > 1:
                    fusion_groups.append(current_group.copy())
                    group_id = f"elementwise_group_{len(fusion_groups)}"
                    fusion_types[group_id] = FusionType.ELEMENTWISE
                current_group = []
        
        # Handle final group
        if len(current_group) > 1:
            fusion_groups.append(current_group)
            group_id = f"elementwise_group_{len(fusion_groups)}"
            fusion_types[group_id] = FusionType.ELEMENTWISE
        
        return fusion_groups, fusion_types
    
    def _classify_operation(self, op_name: str) -> OpType:
        """Classify operation type."""
        elementwise_ops = {'add', 'mul', 'sub', 'div', 'relu', 'sigmoid', 'tanh'}
        reduction_ops = {'sum', 'mean', 'max', 'min'}
        matmul_ops = {'mm', 'bmm', 'addmm', 'matmul'}
        
        if op_name in elementwise_ops:
            return OpType.ELEMENTWISE
        elif op_name in reduction_ops:
            return OpType.REDUCTION
        elif op_name in matmul_ops:
            return OpType.MATMUL
        else:
            return OpType.CUSTOM


class MemoryAnalyzer:
    """Analyzes memory requirements and buffer lifetimes."""
    
    def analyze_memory(self, dag: ComputationDAG, execution_order: List[ConductorNode]) -> Tuple[Dict[str, int], Dict[str, Tuple[int, int]]]:
        """
        Analyze memory requirements and buffer lifetimes.
        
        Returns:
            Tuple of (memory_requirements, buffer_lifetimes)
        """
        memory_requirements = {}
        buffer_lifetimes = {}
        
        # Calculate memory requirements for each buffer
        for buffer in dag.buffers:
            if hasattr(buffer, 'shape') and buffer.shape:
                # Assume float32 (4 bytes per element)
                size = 4
                for dim in buffer.shape:
                    size *= dim
                memory_requirements[buffer.name] = size
        
        # Calculate buffer lifetimes based on execution order
        buffer_first_use = {}
        buffer_last_use = {}
        
        for i, node in enumerate(execution_order):
            # Check input buffers
            for input_buffer in getattr(node, 'input_buffers', []):
                if input_buffer.name not in buffer_first_use:
                    buffer_first_use[input_buffer.name] = i
                buffer_last_use[input_buffer.name] = i
            
            # Check output buffers
            for output_buffer in getattr(node, 'output_buffers', []):
                if output_buffer.name not in buffer_first_use:
                    buffer_first_use[output_buffer.name] = i
                buffer_last_use[output_buffer.name] = i
        
        # Create lifetime tuples
        for buffer_name in buffer_first_use:
            first = buffer_first_use[buffer_name]
            last = buffer_last_use.get(buffer_name, first)
            buffer_lifetimes[buffer_name] = (first, last)
        
        return memory_requirements, buffer_lifetimes


class ModernGraphAnalyzer:
    """
    Modern graph analyzer using composition of specialized analyzers.
    
    This class coordinates multiple analysis passes to provide comprehensive
    graph analysis using clean, modular architecture.
    """
    
    def __init__(self):
        """Initialize the analyzer with specialized components."""
        self.topology_analyzer = TopologyAnalyzer()
        self.fusion_analyzer = FusionAnalyzer()
        self.memory_analyzer = MemoryAnalyzer()
    
    def analyze(self, dag: ComputationDAG) -> AnalysisResult:
        """
        Perform comprehensive graph analysis.
        
        Args:
            dag: The computation DAG to analyze
            
        Returns:
            Complete analysis result
        """
        logger.info(f"Starting comprehensive analysis of DAG with {len(dag.nodes)} nodes")
        
        result = AnalysisResult()
        
        # Topology analysis
        logger.debug("Performing topology analysis")
        result.execution_order, result.dependency_graph = self.topology_analyzer.analyze_topology(dag)
        
        # Fusion analysis
        logger.debug("Performing fusion analysis")
        result.fusion_groups, result.fusion_types = self.fusion_analyzer.analyze_fusion(dag)
        
        # Memory analysis
        logger.debug("Performing memory analysis")
        result.memory_requirements, result.buffer_lifetimes = self.memory_analyzer.analyze_memory(
            dag, result.execution_order
        )
        
        # Generate optimization hints
        result.optimization_hints = self._generate_optimization_hints(result)
        
        logger.info(f"Analysis complete: {len(result.fusion_groups)} fusion groups, "
                   f"{sum(result.memory_requirements.values())} bytes total memory")
        
        return result
    
    def _generate_optimization_hints(self, result: AnalysisResult) -> Dict[str, Any]:
        """Generate optimization hints based on analysis results."""
        hints = {}
        
        # Parallelization hints
        if len(result.execution_order) > 4:
            hints['parallel_factor'] = min(8, len(result.execution_order) // 2)
        
        # Memory optimization hints
        total_memory = sum(result.memory_requirements.values())
        if total_memory > 1024 * 1024:  # > 1MB
            hints['use_memory_pooling'] = True
        
        # Fusion hints
        if len(result.fusion_groups) > 0:
            hints['enable_fusion'] = True
            hints['fusion_groups'] = len(result.fusion_groups)
        
        return hints


# Global instance
graph_analyzer = ModernGraphAnalyzer()

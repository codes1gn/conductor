"""
Code generation subsystem for Conductor.

This module handles the conversion from PyTorch FX Graph representation
to Conductor DSL, including graph analysis, operation fusion, and
buffer management.
"""

# Public API exports for code generation components
from .graph import GraphAnalyzer, ComputationDAG, ConductorNode
from .fusion import FusionEngine, FusionCluster, FusionType
from .dsl import DSLGenerator
from .buffers import Buffer, BufferScope, BufferManager

__all__ = [
    "GraphAnalyzer",
    "ComputationDAG", 
    "ConductorNode",
    "FusionEngine",
    "FusionCluster",
    "FusionType",
    "DSLGenerator",
    "Buffer",
    "BufferScope",
    "BufferManager",
]
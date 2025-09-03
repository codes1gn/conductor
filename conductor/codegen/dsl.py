"""
Choreo DSL generation (Legacy Interface).

This module handles the generation of Choreo DSL (.co files) from
the internal DAG representation. This is a legacy interface that
now delegates to the real ChoreoDSLGenerator.
"""

from typing import List
from .graph import ComputationDAG, ConductorNode
from .buffers import Buffer
from .choreo_dsl import ChoreoDSLGenerator

# TODO: refactor this to remove legacy interface, also remove the legacy file
class DSLGenerator:
    """
    Generates Choreo DSL code from processed graph (Legacy Interface).

    This class now delegates to ChoreoDSLGenerator to generate real
    Choreo DSL syntax instead of fake conductor DSL.
    """
    
    def __init__(self):
        """Initialize DSL generator with real Choreo DSL generator."""
        self._choreo_generator = ChoreoDSLGenerator()

    def generate_dsl_file(self, dag: ComputationDAG, function_name: str = "conductor_kernel") -> str:
        """
        Generate complete Choreo DSL file for the computation graph.

        This method now delegates to the real ChoreoDSLGenerator to produce
        actual Choreo DSL instead of fake conductor DSL.

        Args:
            dag: ComputationDAG to convert to DSL
            function_name: Name of the generated __co__ function

        Returns:
            Complete Choreo DSL file content as string
        """
        # Delegate to the real Choreo DSL generator
        return self._choreo_generator.generate_dsl_file(dag, function_name)

    # Legacy methods for backward compatibility - delegate to choreo generator
    def emit_buffer_declarations(self, buffers: List[Buffer]) -> str:
        """Legacy method - now delegates to ChoreoDSLGenerator."""
        # This method is kept for backward compatibility but is no longer used
        # in the main DSL generation pipeline
        return ""

    def emit_operation_sequence(self, nodes: List[ConductorNode]) -> str:
        """Legacy method - now delegates to ChoreoDSLGenerator."""
        # This method is kept for backward compatibility but is no longer used
        return ""

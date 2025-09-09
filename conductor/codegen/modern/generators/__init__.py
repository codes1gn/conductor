"""
Specialized Code Generators.

This module contains specialized generators for different aspects of DSL code:
- HeaderGenerator: DSL file headers and includes
- OperationGenerator: Individual operation code
- ParallelLoopGenerator: Parallel loop structures
- ValidationGenerator: Syntax validation and error checking

Each generator follows the CodeGenerator protocol and produces CodeFragment
objects that can be composed into complete DSL files.
"""

# Generators will be implemented in subsequent phases
# from .header import HeaderGenerator
# from .operation import OperationGenerator  
# from .parallel import ParallelLoopGenerator
# from .validation import SyntaxValidator

__all__ = [
    # Will be added as generators are implemented
    # "HeaderGenerator",
    # "OperationGenerator", 
    # "ParallelLoopGenerator",
    # "SyntaxValidator",
]

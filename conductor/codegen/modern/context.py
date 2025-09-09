"""
Generation Context Management.

This module provides utilities for creating and managing code generation
contexts. It includes builders for complex context creation and validation
of context consistency.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import torch

from .types import (
    CodeGenerationContext,
    BufferInfo,
    ParallelConfig,
    DSLGenerationConfig,
    BufferScope,
    validate_generation_context,
)
from ...utils.constants import MemoryLevel


class ContextBuilder:
    """Builder for creating CodeGenerationContext objects."""
    
    def __init__(self):
        self._function_name: Optional[str] = None
        self._input_buffers: List[BufferInfo] = []
        self._output_buffers: List[BufferInfo] = []
        self._parallel_config: Optional[ParallelConfig] = None
        self._config: Optional[DSLGenerationConfig] = None
        self._metadata: Dict[str, Any] = {}
    
    def with_function_name(self, name: str) -> 'ContextBuilder':
        """Set the function name."""
        self._function_name = name
        return self
    
    def with_input_buffer(self, buffer: BufferInfo) -> 'ContextBuilder':
        """Add an input buffer."""
        self._input_buffers.append(buffer)
        return self
    
    def with_output_buffer(self, buffer: BufferInfo) -> 'ContextBuilder':
        """Add an output buffer."""
        self._output_buffers.append(buffer)
        return self
    
    def with_parallel_config(self, config: ParallelConfig) -> 'ContextBuilder':
        """Set the parallel configuration."""
        self._parallel_config = config
        return self
    
    def with_config(self, config: DSLGenerationConfig) -> 'ContextBuilder':
        """Set the DSL generation configuration."""
        self._config = config
        return self
    
    def with_metadata(self, key: str, value: Any) -> 'ContextBuilder':
        """Add metadata."""
        self._metadata[key] = value
        return self
    
    def build(self) -> CodeGenerationContext:
        """Build the context."""
        if self._function_name is None:
            raise ValueError("Function name is required")
        if not self._input_buffers:
            raise ValueError("At least one input buffer is required")
        if not self._output_buffers:
            raise ValueError("At least one output buffer is required")
        if self._parallel_config is None:
            raise ValueError("Parallel configuration is required")
        if self._config is None:
            raise ValueError("DSL generation configuration is required")
        
        context = CodeGenerationContext(
            function_name=self._function_name,
            input_buffers=tuple(self._input_buffers),
            output_buffers=tuple(self._output_buffers),
            parallel_config=self._parallel_config,
            config=self._config,
            metadata=self._metadata.copy(),
        )
        
        validate_generation_context(context)
        return context


def create_generation_context(
    function_name: str,
    input_buffers: List[BufferInfo],
    output_buffers: List[BufferInfo],
    config: DSLGenerationConfig,
    parallel_factor: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> CodeGenerationContext:
    """Create a generation context with sensible defaults."""
    parallel_config = ParallelConfig(
        factor=parallel_factor,
        variable="p",
        chunk_size=4,
        strategy="simple",
    )
    
    context = CodeGenerationContext(
        function_name=function_name,
        input_buffers=tuple(input_buffers),
        output_buffers=tuple(output_buffers),
        parallel_config=parallel_config,
        config=config,
        metadata=metadata or {},
    )
    
    validate_generation_context(context)
    return context


def create_buffer_info(
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    scope: BufferScope = BufferScope.LOCAL,
    memory_level: MemoryLevel = MemoryLevel.L1,
    is_input: bool = False,
    is_output: bool = False,
) -> BufferInfo:
    """Create a BufferInfo with validation."""
    buffer = BufferInfo(
        name=name,
        shape=shape,
        dtype=dtype,
        scope=scope,
        memory_level=memory_level,
        is_input=is_input,
        is_output=is_output,
    )
    
    from .types import validate_buffer_info
    validate_buffer_info(buffer)
    return buffer


def create_parallel_config(
    factor: int = 1,
    variable: str = "p",
    chunk_size: int = 4,
    strategy: str = "simple",
) -> ParallelConfig:
    """Create a ParallelConfig with validation."""
    if factor < 1:
        raise ValueError("Parallel factor must be at least 1")
    if not variable:
        raise ValueError("Parallel variable name cannot be empty")
    if chunk_size < 1:
        raise ValueError("Chunk size must be at least 1")
    
    return ParallelConfig(
        factor=factor,
        variable=variable,
        chunk_size=chunk_size,
        strategy=strategy,
    )


def context_from_dag(dag, function_name: str, config: DSLGenerationConfig) -> CodeGenerationContext:
    """Create a context from a computation DAG."""
    # Import here to avoid circular imports
    from ...optimization.graph_analyzer import ComputationDAG
    
    if not isinstance(dag, ComputationDAG):
        raise TypeError("Expected ComputationDAG")
    
    # Convert DAG buffers to BufferInfo
    input_buffers = []
    for buffer in dag.inputs:
        buffer_info = create_buffer_info(
            name=buffer.name,
            shape=tuple(buffer.shape) if buffer.shape else (1,),
            dtype=buffer.dtype,
            scope=BufferScope.GLOBAL,
            memory_level=MemoryLevel.GLOBAL,
            is_input=True,
        )
        input_buffers.append(buffer_info)
    
    output_buffers = []
    for buffer in dag.outputs:
        buffer_info = create_buffer_info(
            name=buffer.name,
            shape=tuple(buffer.shape) if buffer.shape else (1,),
            dtype=buffer.dtype,
            scope=BufferScope.GLOBAL,
            memory_level=MemoryLevel.GLOBAL,
            is_output=True,
        )
        output_buffers.append(buffer_info)
    
    return create_generation_context(
        function_name=function_name,
        input_buffers=input_buffers,
        output_buffers=output_buffers,
        config=config,
        parallel_factor=config.parallel_factor,
        metadata={"dag_nodes": len(dag.nodes), "dag_buffers": len(dag.buffers)},
    )

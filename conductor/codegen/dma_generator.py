"""
DMA Generation Utilities for Conductor.

This module provides reusable DMA generation patterns for Choreo DSL,
eliminating hardcoded DMA operations and making them configurable.
"""

from __future__ import annotations

from typing import Optional, Any
from dataclasses import dataclass
from enum import Enum

from .dsl_constants import DSLKeywords, MemoryLevel


class DMADirection(Enum):
    """DMA transfer direction."""

    LOAD = "load"  # From global/shared to local
    STORE = "store"  # From local to global/shared


@dataclass
class DMAPattern:
    """Configuration for DMA operation patterns."""

    direction: DMADirection
    source_memory: MemoryLevel
    target_memory: MemoryLevel
    is_async: bool = True
    chunk_pattern: str = "chunkat"  # chunkat, slice, etc.


@dataclass
class DMAOperation:
    """Represents a single DMA operation."""

    source_var: str
    target_var: str
    pattern: DMAPattern
    parallel_var: Optional[str] = None
    index_var: Optional[str] = None

    def generate_dsl(self) -> str:
        """Generate DSL code for this DMA operation."""
        if self.pattern.is_async:
            dma_op = DSLKeywords.DMA_COPY_ASYNC.value
        else:
            dma_op = DSLKeywords.DMA_COPY.value

        if self.pattern.direction == DMADirection.LOAD:
            if self.parallel_var and self.index_var:
                source_expr = f"{self.source_var}.{self.pattern.chunk_pattern}({self.parallel_var}, {self.index_var})"
            else:
                source_expr = self.source_var

            target_expr = f"{self.target_memory_keyword()}"
            return f"{self.target_var} = {dma_op} {source_expr} => {target_expr};"

        else:  # STORE
            source_expr = self.source_var
            if self.parallel_var and self.index_var:
                target_expr = f"{self.target_var}.{self.pattern.chunk_pattern}({self.parallel_var}, {self.index_var})"
            else:
                target_expr = self.target_var

            return f"{dma_op} {source_expr} => {target_expr};"

    def target_memory_keyword(self) -> str:
        """Get target memory keyword."""
        if self.pattern.target_memory == MemoryLevel.L1:
            return DSLKeywords.LOCAL.value
        elif self.pattern.target_memory == MemoryLevel.L2:
            return "shared"
        else:
            return "global"


class DMAGenerator:
    """Generates reusable DMA operation patterns."""

    def __init__(self):
        """Initialize the DMA generator."""
        self.default_patterns = self._create_default_patterns()

    def _create_default_patterns(self) -> dict[str, DMAPattern]:
        """Create default DMA patterns."""
        return {
            "global_to_l1_async": DMAPattern(
                direction=DMADirection.LOAD,
                source_memory=MemoryLevel.GLOBAL,
                target_memory=MemoryLevel.L1,
                is_async=True,
            ),
            "l1_to_global": DMAPattern(
                direction=DMADirection.STORE,
                source_memory=MemoryLevel.L1,
                target_memory=MemoryLevel.GLOBAL,
                is_async=False,
            ),
            "l2_to_l1_async": DMAPattern(
                direction=DMADirection.LOAD,
                source_memory=MemoryLevel.L2,
                target_memory=MemoryLevel.L1,
                is_async=True,
            ),
            "l1_to_l2": DMAPattern(
                direction=DMADirection.STORE,
                source_memory=MemoryLevel.L1,
                target_memory=MemoryLevel.L2,
                is_async=False,
            ),
        }

    def generate_load_sequence(
        self,
        source_vars: list[str],
        target_vars: list[str],
        parallel_var: str,
        index_var: str,
        pattern_name: str = "global_to_l1_async",
    ) -> list[str]:
        """
        Generate a sequence of DMA load operations.

        Args:
            source_vars: Source variable names
            target_vars: Target variable names
            parallel_var: Parallel loop variable
            index_var: Index variable
            pattern_name: DMA pattern to use

        Returns:
            List of DSL lines for the load sequence
        """
        lines = []
        pattern = self.default_patterns.get(pattern_name)
        if not pattern:
            raise ValueError(f"Unknown DMA pattern: {pattern_name}")

        # Generate individual load operations
        for source_var, target_var in zip(source_vars, target_vars):
            dma_op = DMAOperation(
                source_var=source_var,
                target_var=target_var,
                pattern=pattern,
                parallel_var=parallel_var,
                index_var=index_var,
            )
            lines.append(dma_op.generate_dsl())

        # Generate wait statement for async operations
        if pattern.is_async and target_vars:
            wait_vars = ", ".join(target_vars)
            lines.append(f"{DSLKeywords.WAIT.value} {wait_vars};")

        return lines

    def generate_store_sequence(
        self,
        source_vars: list[str],
        target_vars: list[str],
        parallel_var: str,
        index_var: str,
        pattern_name: str = "l1_to_global",
    ) -> list[str]:
        """
        Generate a sequence of DMA store operations.

        Args:
            source_vars: Source variable names
            target_vars: Target variable names
            parallel_var: Parallel loop variable
            index_var: Index variable
            pattern_name: DMA pattern to use

        Returns:
            List of DSL lines for the store sequence
        """
        lines = []
        pattern = self.default_patterns.get(pattern_name)
        if not pattern:
            raise ValueError(f"Unknown DMA pattern: {pattern_name}")

        # Generate individual store operations
        for source_var, target_var in zip(source_vars, target_vars):
            dma_op = DMAOperation(
                source_var=source_var,
                target_var=target_var,
                pattern=pattern,
                parallel_var=parallel_var,
                index_var=index_var,
            )
            lines.append(dma_op.generate_dsl())

        return lines

    def generate_buffer_declaration(
        self, var_name: str, dtype: str, memory_level: MemoryLevel, shape_expr: str
    ) -> str:
        """
        Generate buffer declaration.

        Args:
            var_name: Variable name
            dtype: Data type (f32, s32, etc.)
            memory_level: Memory level
            shape_expr: Shape expression (e.g., "[lf.span]", "[4, 4]")

        Returns:
            DSL buffer declaration
        """
        if memory_level == MemoryLevel.L1:
            memory_keyword = DSLKeywords.LOCAL.value
        elif memory_level == MemoryLevel.L2:
            memory_keyword = "shared"
        else:
            memory_keyword = "global"

        return f"{memory_keyword} {dtype} {shape_expr} {var_name};"

    def generate_parallel_loop(
        self,
        parallel_var: str,
        parallel_factor: int,
        index_var: str,
        chunk_size: int,
        body_lines: list[str],
    ) -> list[str]:
        """
        Generate parallel loop structure.

        Args:
            parallel_var: Parallel variable name
            parallel_factor: Parallel factor
            index_var: Index variable name
            chunk_size: Chunk size for foreach loop
            body_lines: Lines inside the loop body

        Returns:
            Complete parallel loop structure
        """
        lines = []

        # Parallel declaration
        lines.append(f"{DSLKeywords.PARALLEL.value} {parallel_var} by {parallel_factor}")

        # Foreach loop
        lines.append(f"  {DSLKeywords.FOREACH.value} {index_var} in [{chunk_size}] {{")

        # Body with proper indentation
        for body_line in body_lines:
            lines.append(f"    {body_line}")

        # Close foreach
        lines.append("  }")

        return lines

    def add_custom_pattern(self, name: str, pattern: DMAPattern):
        """Add a custom DMA pattern."""
        self.default_patterns[name] = pattern

    def get_pattern(self, name: str) -> Optional[DMAPattern]:
        """Get a DMA pattern by name."""
        return self.default_patterns.get(name)


# Global instance for use across DSL generation
dma_generator = DMAGenerator()

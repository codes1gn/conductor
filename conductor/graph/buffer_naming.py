"""
Buffer Naming Utilities for Conductor.

This module provides utilities for generating unique and consistent buffer names
across DSL generation, addressing the TODO in dslgen.py about hardcoded identifiers.

Note: This module is being deprecated in favor of conductor.utils.naming
"""

from __future__ import annotations

# Import from new utils location for backward compatibility
from ..utils.constants import MemoryLevel, BufferType
from ..utils.naming import (
    BufferNamingContext,
    BufferNamingManager,
    get_output_buffer_name,
    get_load_buffer_name,
    get_intermediate_buffer_name,
    reset_naming_state as reset_buffer_naming,
    default_buffer_naming_manager as buffer_naming_manager,
)

# All functionality is now provided by the imports above

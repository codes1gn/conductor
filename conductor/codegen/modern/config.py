"""
Configuration Management for Modern DSL Generation.

This module provides utilities for creating, validating, and managing
DSL generation configurations. It supports both programmatic configuration
and loading from external sources.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import torch

from .types import DSLGenerationConfig, ValidationLevel
from ...utils.constants import MemoryLevel


def create_default_config() -> DSLGenerationConfig:
    """Create a default DSL generation configuration."""
    return DSLGenerationConfig(
        indent_size=2,
        parallel_factor=1,
        memory_level=MemoryLevel.L1,
        enable_fusion=True,
        target_dialect="choreo",
        validation_level=ValidationLevel.SYNTAX,
        enable_optimization=True,
        debug_mode=False,
    )


def create_config_from_dict(config_dict: Dict[str, Any]) -> DSLGenerationConfig:
    """Create a configuration from a dictionary."""
    # Start with defaults
    config = create_default_config()
    
    # Override with provided values
    kwargs = {}
    
    if "indent_size" in config_dict:
        kwargs["indent_size"] = int(config_dict["indent_size"])
    
    if "parallel_factor" in config_dict:
        kwargs["parallel_factor"] = int(config_dict["parallel_factor"])
    
    if "memory_level" in config_dict:
        level_str = config_dict["memory_level"]
        if isinstance(level_str, str):
            kwargs["memory_level"] = MemoryLevel(level_str.upper())
        else:
            kwargs["memory_level"] = level_str
    
    if "enable_fusion" in config_dict:
        kwargs["enable_fusion"] = bool(config_dict["enable_fusion"])
    
    if "target_dialect" in config_dict:
        kwargs["target_dialect"] = str(config_dict["target_dialect"])
    
    if "validation_level" in config_dict:
        level_str = config_dict["validation_level"]
        if isinstance(level_str, str):
            kwargs["validation_level"] = ValidationLevel(level_str.upper())
        else:
            kwargs["validation_level"] = level_str
    
    if "enable_optimization" in config_dict:
        kwargs["enable_optimization"] = bool(config_dict["enable_optimization"])
    
    if "debug_mode" in config_dict:
        kwargs["debug_mode"] = bool(config_dict["debug_mode"])
    
    # Create new config with overrides
    return DSLGenerationConfig(**{
        **config.__dict__,
        **kwargs
    })


def validate_config(config: DSLGenerationConfig) -> None:
    """Validate a DSL generation configuration."""
    if config.indent_size < 0:
        raise ValueError("Indent size must be non-negative")
    
    if config.parallel_factor < 1:
        raise ValueError("Parallel factor must be at least 1")
    
    if config.target_dialect not in ["choreo", "cuda", "opencl"]:
        raise ValueError(f"Unsupported target dialect: {config.target_dialect}")
    
    if not isinstance(config.memory_level, MemoryLevel):
        raise TypeError("Memory level must be a MemoryLevel enum")
    
    if not isinstance(config.validation_level, ValidationLevel):
        raise TypeError("Validation level must be a ValidationLevel enum")


def create_config_for_testing() -> DSLGenerationConfig:
    """Create a configuration optimized for testing."""
    return DSLGenerationConfig(
        indent_size=2,
        parallel_factor=1,
        memory_level=MemoryLevel.L1,
        enable_fusion=False,  # Disable fusion for simpler testing
        target_dialect="choreo",
        validation_level=ValidationLevel.FULL,
        enable_optimization=False,  # Disable optimization for predictable output
        debug_mode=True,
    )


def create_config_for_performance() -> DSLGenerationConfig:
    """Create a configuration optimized for performance."""
    return DSLGenerationConfig(
        indent_size=2,
        parallel_factor=4,  # Higher parallelism
        memory_level=MemoryLevel.L1,
        enable_fusion=True,
        target_dialect="choreo",
        validation_level=ValidationLevel.SYNTAX,  # Minimal validation for speed
        enable_optimization=True,
        debug_mode=False,
    )


def merge_configs(base: DSLGenerationConfig, override: DSLGenerationConfig) -> DSLGenerationConfig:
    """Merge two configurations, with override taking precedence."""
    return DSLGenerationConfig(
        indent_size=override.indent_size,
        parallel_factor=override.parallel_factor,
        memory_level=override.memory_level,
        enable_fusion=override.enable_fusion,
        target_dialect=override.target_dialect,
        validation_level=override.validation_level,
        enable_optimization=override.enable_optimization,
        debug_mode=override.debug_mode,
    )


def config_to_dict(config: DSLGenerationConfig) -> Dict[str, Any]:
    """Convert a configuration to a dictionary."""
    return {
        "indent_size": config.indent_size,
        "parallel_factor": config.parallel_factor,
        "memory_level": config.memory_level.value,
        "enable_fusion": config.enable_fusion,
        "target_dialect": config.target_dialect,
        "validation_level": config.validation_level.value,
        "enable_optimization": config.enable_optimization,
        "debug_mode": config.debug_mode,
    }

"""
Simplified Configuration System for Conductor Framework.

This module provides a clean, unified configuration interface that replaces
the previous complex granular options with simple, developer-friendly controls.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from .logging import get_logger

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class DebugConfig:
    """Debug and tracing configuration."""

    enabled: bool = False
    max_tensor_elements: int = 100
    indent_size: int = 2

    # Tracing configuration
    trace_dag: bool = True
    trace_dsl: bool = True
    trace_compilation: bool = True
    trace_execution: bool = False
    verbose_level: int = 1


@dataclass
class CacheConfig:
    """Cache configuration."""

    enabled: bool = True
    max_size_mb: int = 1024
    cache_dir: Optional[str] = None


@dataclass
class CompilationConfig:
    """Compilation configuration."""

    timeout_seconds: int = 300
    optimization_level: str = "O2"
    enable_fusion: bool = True
    dsl_indent_size: int = 2

    # Graph analysis switches
    enable_graph_validation: bool = True
    enable_shape_inference: bool = True
    enable_buffer_optimization: bool = True


@dataclass
class RuntimeConfig:
    """Runtime configuration."""

    device: str = "gcu"
    memory_pool_size_mb: int = 512
    enable_profiling: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    enable_file_logging: bool = False
    log_file: str = "conductor.log"


class ConductorConfig:
    """
    Unified configuration manager for the Conductor framework.

    This class provides a simple interface to manage all configuration
    options through a single JSON file, replacing the previous complex
    environment variable system.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_file: Path to configuration file. If None, uses default location.
        """
        self.config_file = self._get_config_file_path(config_file)
        self._config_data = self._load_config()

        # Initialize configuration sections
        self.debug = self._create_debug_config()
        self.cache = self._create_cache_config()
        self.compilation = self._create_compilation_config()
        self.runtime = self._create_runtime_config()
        self.logging = self._create_logging_config()

    def _get_config_file_path(self, config_file: Optional[str]) -> Path:
        """Get the configuration file path."""
        if config_file:
            return Path(config_file)

        # Default location: try YAML first, then JSON
        config_dir = Path(__file__).parent
        yaml_config = config_dir / "conductor_config.yaml"
        json_config = config_dir / "conductor_config.json"

        if YAML_AVAILABLE and yaml_config.exists():
            return yaml_config
        else:
            return json_config

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file (JSON or YAML)."""
        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    if self.config_file.suffix.lower() in [".yaml", ".yml"] and YAML_AVAILABLE:
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return config_data
            else:
                logger.warning(f"Configuration file {self.config_file} not found, using defaults")
                return {}
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}

    def _create_debug_config(self) -> DebugConfig:
        """Create debug configuration from loaded data."""
        debug_data = self._config_data.get("debug", {})

        # Check environment variable override
        env_enabled = os.getenv("CONDUCTOR_DEBUG", "").lower() in ("1", "true", "yes")
        enabled = env_enabled or debug_data.get("enabled", False)

        return DebugConfig(
            enabled=enabled,
            max_tensor_elements=debug_data.get("max_tensor_elements", 100),
            indent_size=debug_data.get("indent_size", 2),
            trace_dag=debug_data.get("trace_dag", True),
            trace_dsl=debug_data.get("trace_dsl", True),
            trace_compilation=debug_data.get("trace_compilation", True),
            trace_execution=debug_data.get("trace_execution", False),
            verbose_level=debug_data.get("verbose_level", 1),
        )

    def _create_cache_config(self) -> CacheConfig:
        """Create cache configuration from loaded data."""
        cache_data = self._config_data.get("cache", {})

        # Check environment variable override
        env_disabled = os.getenv("CONDUCTOR_DISABLE_CACHE", "").lower() in ("1", "true", "yes")
        enabled = not env_disabled and cache_data.get("enabled", True)

        return CacheConfig(
            enabled=enabled,
            max_size_mb=cache_data.get("max_size_mb", 1024),
            cache_dir=cache_data.get("cache_dir"),
        )

    def _create_compilation_config(self) -> CompilationConfig:
        """Create compilation configuration from loaded data."""
        comp_data = self._config_data.get("compilation", {})

        return CompilationConfig(
            timeout_seconds=comp_data.get("timeout_seconds", 300),
            optimization_level=comp_data.get("optimization_level", "O2"),
            enable_fusion=comp_data.get("enable_fusion", True),
            dsl_indent_size=comp_data.get("dsl_indent_size", 2),
            enable_graph_validation=comp_data.get("enable_graph_validation", True),
            enable_shape_inference=comp_data.get("enable_shape_inference", True),
            enable_buffer_optimization=comp_data.get("enable_buffer_optimization", True),
        )

    def _create_runtime_config(self) -> RuntimeConfig:
        """Create runtime configuration from loaded data."""
        runtime_data = self._config_data.get("runtime", {})

        return RuntimeConfig(
            device=runtime_data.get("device", "gcu"),
            memory_pool_size_mb=runtime_data.get("memory_pool_size_mb", 512),
            enable_profiling=runtime_data.get("enable_profiling", False),
        )

    def _create_logging_config(self) -> LoggingConfig:
        """Create logging configuration from loaded data."""
        log_data = self._config_data.get("logging", {})

        return LoggingConfig(
            level=log_data.get("level", "INFO"),
            enable_file_logging=log_data.get("enable_file_logging", False),
            log_file=log_data.get("log_file", "conductor.log"),
        )

    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self.debug.enabled

    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.cache.enabled

    def save_config(self) -> None:
        """Save current configuration to file."""
        config_data = {
            "version": "1.0",
            "description": "Conductor Framework Configuration",
            "debug": {
                "enabled": self.debug.enabled,
                "max_tensor_elements": self.debug.max_tensor_elements,
                "indent_size": self.debug.indent_size,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "max_size_mb": self.cache.max_size_mb,
                "cache_dir": self.cache.cache_dir,
            },
            "compilation": {
                "timeout_seconds": self.compilation.timeout_seconds,
                "optimization_level": self.compilation.optimization_level,
                "enable_fusion": self.compilation.enable_fusion,
            },
            "runtime": {
                "device": self.runtime.device,
                "memory_pool_size_mb": self.runtime.memory_pool_size_mb,
                "enable_profiling": self.runtime.enable_profiling,
            },
            "logging": {
                "level": self.logging.level,
                "enable_file_logging": self.logging.enable_file_logging,
                "log_file": self.logging.log_file,
            },
        }

        try:
            with open(self.config_file, "w") as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")


# Global configuration instance
_global_config: Optional[ConductorConfig] = None


def get_config() -> ConductorConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = ConductorConfig()
    return _global_config


def set_config(config: ConductorConfig) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def load_config(config_file: str) -> ConductorConfig:
    """Load configuration from a specific file."""
    return ConductorConfig(config_file)

"""
Shared utilities and helpers for Conductor.

This module provides common utilities used throughout the Conductor
package, including logging, caching, performance monitoring, and exception handling.
"""

# Public API exports for utility components
from .logging import setup_logging, get_logger
from .caching import CompilationCache
from .exceptions import (
    ConductorError,
    CompilationError,
    UnsupportedOperationError,
    FallbackHandler
)
from .profiler import (
    ConductorProfiler,
    PerformanceMetrics,
    PerformanceBenchmark,
    MemoryTracker,
    profile_operation,
    get_last_compilation_stats
)
from .regression import (
    PerformanceRegressionDetector,
    ContinuousPerformanceMonitor,
    PerformanceBaseline,
    RegressionResult,
    create_performance_monitoring_system
)

__all__ = [
    "setup_logging",
    "get_logger",
    "CompilationCache",
    "ConductorError",
    "CompilationError", 
    "UnsupportedOperationError",
    "FallbackHandler",
    "ConductorProfiler",
    "PerformanceMetrics",
    "PerformanceBenchmark",
    "MemoryTracker",
    "profile_operation",
    "get_last_compilation_stats",
    "PerformanceRegressionDetector",
    "ContinuousPerformanceMonitor",
    "PerformanceBaseline",
    "RegressionResult",
    "create_performance_monitoring_system"
]
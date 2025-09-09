"""
Logging configuration and utilities.

This module provides centralized logging configuration for the
Conductor package with appropriate formatting and levels.
"""

import logging
import os
from typing import Optional


def setup_logging(level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """
    Configure logging for the Conductor package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
    """
    # Determine log level
    if level is None:
        level = os.environ.get("CONDUCTOR_LOG_LEVEL", "INFO")

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger for conductor
    logger = logging.getLogger("conductor")
    logger.setLevel(log_level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"conductor.{name}")


class ConductorLogger:
    """
    Centralized logging for debugging and monitoring.

    This class provides specialized logging methods for different
    aspects of the Conductor compilation and execution pipeline.
    """

    def __init__(self, name: str):
        """
        Initialize logger for specific component.

        Args:
            name: Component name for logging context
        """
        self.logger = get_logger(name)

    def log_compilation_start(self, graph_hash: str, node_count: int) -> None:
        """
        Log beginning of compilation process.

        Args:
            graph_hash: Unique identifier for the graph
            node_count: Number of nodes in the graph
        """
        self.logger.info(f"Starting compilation for graph {graph_hash[:8]}... ({node_count} nodes)")

    def log_fusion_decision(self, nodes: list, decision: bool, reason: str) -> None:
        """
        Log fusion decisions for debugging optimization.

        Args:
            nodes: List of node names being considered for fusion
            decision: Whether fusion was applied
            reason: Explanation for the decision
        """
        node_names = [str(node) for node in nodes]
        action = "Applied" if decision else "Skipped"
        self.logger.debug(f"{action} fusion for nodes {node_names}: {reason}")

    def log_performance_metrics(self, compilation_time: float, execution_time: float) -> None:
        """
        Log performance metrics for monitoring.

        Args:
            compilation_time: Time spent in compilation (seconds)
            execution_time: Time spent in execution (seconds)
        """
        self.logger.info(
            f"Performance: compilation={compilation_time:.3f}s, execution={execution_time:.3f}s"
        )

    def log_fallback(self, operation: str, reason: str) -> None:
        """
        Log fallback to alternative backend.

        Args:
            operation: Operation that triggered fallback
            reason: Reason for fallback
        """
        self.logger.warning(f"Falling back for operation '{operation}': {reason}")

    def log_cache_hit(self, graph_hash: str) -> None:
        """
        Log cache hit for compiled artifacts.

        Args:
            graph_hash: Graph identifier that was found in cache
        """
        self.logger.debug(f"Cache hit for graph {graph_hash[:8]}...")

    def log_cache_miss(self, graph_hash: str) -> None:
        """
        Log cache miss requiring new compilation.

        Args:
            graph_hash: Graph identifier that was not found in cache
        """
        self.logger.debug(f"Cache miss for graph {graph_hash[:8]}...")


# Initialize logging on module import
setup_logging()

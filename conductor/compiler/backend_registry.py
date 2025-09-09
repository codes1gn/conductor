"""
Simple backend registration for Conductor.

Provides minimal backend registration functionality without over-engineering.
"""

import torch
from typing import Callable
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Track registered backends
_registered_backends = set()


def register_backend(name: str, compiler_fn: Callable) -> bool:
    """
    Register a backend with PyTorch's torch.compile system.

    Args:
        name: Backend name
        compiler_fn: Compiler function

    Returns:
        True if registration successful, False otherwise
    """
    if name in _registered_backends:
        logger.info(f"Backend '{name}' already registered")
        return True

    try:
        if hasattr(torch._dynamo, "register_backend"):
            torch._dynamo.register_backend(name=name, compiler_fn=compiler_fn)
            _registered_backends.add(name)
            logger.info(f"Successfully registered '{name}' backend for torch.compile")
            return True
        else:
            logger.error("PyTorch does not support backend registration")
            return False
    except Exception as e:
        logger.error(f"Failed to register backend '{name}': {e}")
        return False


def is_backend_registered(name: str) -> bool:
    """Check if a backend is registered."""
    return name in _registered_backends

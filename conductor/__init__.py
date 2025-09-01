"""
Conductor: PyTorch Backend Integration for GCU Hardware

A minimalist PyTorch backend that enables seamless execution of ML models
on custom 'gcu' hardware through the Conductor compiler.

Key Features:
- Drop-in replacement using standard torch.compile API
- Intelligent operation fusion and optimized memory management
- JIT and AOT compilation modes with fallback mechanisms
- Zero learning curve for PyTorch developers
"""

__version__ = "0.1.0"
__author__ = "Conductor Team"
__email__ = "conductor@example.com"

# Public API exports
from .backend import register_backend, ConductorBackend

__all__ = [
    "register_backend",
    "ConductorBackend",
]

# Automatically register the backend when the package is imported
try:
    register_backend()
except Exception as e:
    import warnings
    warnings.warn(f"Failed to register Conductor backend: {e}", UserWarning)
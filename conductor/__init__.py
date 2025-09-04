"""
Conductor: PyTorch torch.compile Backend for GCU Hardware

A streamlined PyTorch backend that enables seamless execution of ML models
on GCU hardware through PyTorch's torch.compile system.

Key Features:
- Seamless torch.compile integration with 'gcu' backend
- GCU hardware acceleration through Conductor framework
- Automatic fallback to PyTorch eager execution
- Zero learning curve for PyTorch developers

Usage:
    import torch
    import conductor  # Registers 'gcu' backend

    compiled_model = torch.compile(model, backend='gcu')
    result = compiled_model(inputs)
"""

__version__ = "0.1.0"
__author__ = "Conductor Team"
__email__ = "conductor@example.com"

# Public API exports
from .gcu_backend import (
    register_backend,
    GCUInductorBackend,
    is_backend_registered,
    get_backend_info,
    list_supported_operations,
    get_global_gcu_backend
)

__all__ = [
    "register_backend",
    "GCUInductorBackend",
    "is_backend_registered",
    "get_backend_info",
    "list_supported_operations",
    "get_global_gcu_backend",
]

# Automatically register the backend when the package is imported
try:
    register_backend()
except Exception as e:
    import warnings
    warnings.warn(f"Failed to register Conductor backend: {e}", UserWarning)

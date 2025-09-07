"""
Package information utility.

This module provides a command-line utility for displaying
information about the Conductor installation and environment.
"""

import sys
import platform
from typing import Dict, Any
import conductor


def get_system_info() -> Dict[str, Any]:
    """
    Get system information relevant to Conductor.
    
    Returns:
        Dictionary containing system information
    """
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture(),
        'processor': platform.processor(),
    }
    
    # Try to get PyTorch information
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['torch_cuda_available'] = torch.cuda.is_available()
        info['torch_backends'] = getattr(torch._dynamo, 'list_backends', lambda: [])()
    except ImportError:
        info['torch_version'] = 'Not installed'
        info['torch_cuda_available'] = False
        info['torch_backends'] = []
    
    return info


def get_conductor_info() -> Dict[str, Any]:
    """
    Get Conductor-specific information.
    
    Returns:
        Dictionary containing Conductor information
    """
    info = {
        'version': conductor.__version__,
        'author': conductor.__author__,
        'backend_registered': False,
    }
    
    # Check if backend is registered
    try:
        from conductor.gcu_backend import is_backend_registered
        info['backend_registered'] = is_backend_registered()
    except Exception as e:
        info['backend_registration_error'] = str(e)
    
    # Get device information
    try:
        from conductor.device import get_gcu_interface
        gcu_interface = get_gcu_interface()
        info['available_devices'] = gcu_interface.list_devices()
        
        default_device = gcu_interface.get_default_device()
        if default_device:
            info['default_device'] = default_device.get_device_info()
    except Exception as e:
        info['device_error'] = str(e)
    
    return info


def print_info() -> None:
    """Print formatted information about Conductor and the system."""
    print("Conductor PyTorch Backend Integration")
    print("=" * 40)
    
    # Conductor information
    conductor_info = get_conductor_info()
    print(f"\nConductor Version: {conductor_info['version']}")
    print(f"Author: {conductor_info['author']}")
    print(f"Backend Registered: {conductor_info['backend_registered']}")
    
    if 'backend_registration_error' in conductor_info:
        print(f"Registration Error: {conductor_info['backend_registration_error']}")
    
    if 'available_devices' in conductor_info:
        print(f"Available Devices: {conductor_info['available_devices']}")
    
    if 'device_error' in conductor_info:
        print(f"Device Error: {conductor_info['device_error']}")
    
    # System information
    system_info = get_system_info()
    print(f"\nPython Version: {system_info['python_version'].split()[0]}")
    print(f"Platform: {system_info['platform']}")
    print(f"Architecture: {system_info['architecture'][0]}")
    
    if system_info['torch_version'] != 'Not installed':
        print(f"PyTorch Version: {system_info['torch_version']}")
        print(f"CUDA Available: {system_info['torch_cuda_available']}")
        print(f"Available Backends: {system_info['torch_backends']}")
    else:
        print("PyTorch: Not installed")


def main() -> None:
    """Main entry point for the conductor-info command."""
    try:
        print_info()
    except Exception as e:
        print(f"Error getting system information: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
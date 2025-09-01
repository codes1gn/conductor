#!/usr/bin/env python3
"""
Basic usage example for Conductor PyTorch Backend Integration.

This example demonstrates how to use the Conductor backend with torch.compile
for basic model compilation and execution.
"""

import torch
import conductor  # Automatically registers 'gcu' backend

def main():
    """Demonstrate basic Conductor backend usage."""
    print("Conductor PyTorch Backend Integration - Basic Usage Example")
    print("=" * 60)
    
    # Check if backend is registered
    available_backends = torch._dynamo.list_backends()
    print(f"Available backends: {available_backends}")
    
    if 'gcu' not in available_backends:
        print("ERROR: Conductor 'gcu' backend not found!")
        return
    
    print("✓ Conductor 'gcu' backend is available")
    
    # Define a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 20)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(20, 1)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    # Create model and sample input
    model = SimpleModel()
    x = torch.randn(5, 10)
    
    print(f"Model: {model}")
    print(f"Input shape: {x.shape}")
    
    # Run model normally first
    print("\n1. Running model normally...")
    with torch.no_grad():
        normal_output = model(x)
    print(f"Normal output shape: {normal_output.shape}")
    
    # Compile model with Conductor backend
    print("\n2. Compiling model with Conductor backend...")
    try:
        compiled_model = torch.compile(model, backend='gcu')
        print("✓ Model compiled successfully")
        
        # Note: Actual execution will fail since we haven't implemented
        # the full compilation pipeline yet, but compilation should succeed
        print("✓ Compilation completed (execution not yet implemented)")
        
    except Exception as e:
        print(f"Expected error during compilation: {e}")
        print("This is expected since the full pipeline is not yet implemented")
    
    print("\n3. Backend information:")
    print(f"Conductor version: {conductor.__version__}")
    
    # Show device information
    try:
        from conductor.device import get_gcu_interface
        gcu_interface = get_gcu_interface()
        devices = gcu_interface.list_devices()
        print(f"Available GCU devices: {devices}")
    except Exception as e:
        print(f"Device info error: {e}")


if __name__ == '__main__':
    main()
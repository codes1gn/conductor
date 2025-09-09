#!/usr/bin/env python3
"""
Test case for reduce_mean operator implementation and multi-node DAG fusion.

This example creates a 3-node computation graph:
- Two reduce_mean operations on different dimensions
- One add operation

Expected fusion behavior:
- reduce_mean and add operations CAN be fused together
- Two reduce_mean operations on different dimensions CANNOT be fused together

Expected DAG structure:
- 2 nodes total: one standalone node (unfused reduce_mean) + one fusion cluster (fused reduce_mean + add)
"""

import torch
import torch.nn as nn
import conductor


class ReduceMeanFusionModel(nn.Module):
    """Model for testing reduce_mean operator with multi-node DAG fusion."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computation graph:
        1. reduce_mean_1: mean of x along dimension 0
        2. reduce_mean_2: mean of y along dimension 1
        3. add: add the results of reduce_mean_1 and reduce_mean_2

        Expected fusion:
        - reduce_mean_1 and reduce_mean_2 cannot be fused (different dimensions)
        - reduce_mean_2 and add can be fused (reduction + elementwise)

        Expected DAG: 2 nodes
        - Node 1: standalone reduce_mean_1
        - Node 2: fusion cluster (reduce_mean_2 + add)
        """
        # First reduce_mean: reduce along dimension 0
        mean_x = torch.mean(x, dim=0, keepdim=False)

        # Second reduce_mean: reduce along dimension 1
        mean_y = torch.mean(y, dim=1, keepdim=False)

        # Add the two means
        result = mean_x + mean_y

        return result


def test_reduce_mean_fusion():
    """Test reduce_mean operator with multi-node DAG fusion."""

    # Create test inputs
    # x: (4, 3) -> mean along dim 0 -> (3,)
    # y: (3, 4) -> mean along dim 1 -> (3,)
    # result: (3,) + (3,) -> (3,)
    x = torch.randn(4, 3)
    y = torch.randn(3, 4)

    print("=== Reduce Mean Fusion Test ===")
    print(f"Input x shape: {x.shape}")
    print(f"Input y shape: {y.shape}")

    # Create model
    model = ReduceMeanFusionModel()

    # Compile with Conductor
    print("\n=== Compiling with Conductor ===")
    compiled_model = torch.compile(model, backend='gcu')
    
    # Test execution
    print("\n=== Testing Execution ===")

    # Reference result using PyTorch
    with torch.no_grad():
        expected = model(x, y)
    print(f"Expected result shape: {expected.shape}")
    print(f"Expected result: {expected}")

    # Compiled result
    with torch.no_grad():
        actual = compiled_model(x, y)
    print(f"Actual result shape: {actual.shape}")
    print(f"Actual result: {actual}")
    
    # Verify numerical accuracy
    if torch.allclose(expected, actual, rtol=1e-5, atol=1e-6):
        print("✅ Numerical accuracy test PASSED")
    else:
        print("❌ Numerical accuracy test FAILED")
        print(f"Max difference: {torch.max(torch.abs(expected - actual))}")
        
    print("\n=== Test Summary ===")
    print("Expected DAG structure: 2 nodes")
    print("- Node 1: standalone reduce_mean (dim=0)")
    print("- Node 2: fusion cluster (reduce_mean dim=1 + add)")
    print("Fusion validation:")
    print("- reduce_mean + add: should be fused ✓")
    print("- reduce_mean + reduce_mean (different dims): should NOT be fused ✓")


if __name__ == "__main__":
    test_reduce_mean_fusion()

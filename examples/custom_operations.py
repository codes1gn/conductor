#!/usr/bin/env python3
"""
Custom operations example for Conductor PyTorch Backend Integration.

This example demonstrates how to extend Conductor to support custom operations
and integrate them seamlessly with the compilation pipeline.
"""

import torch
import torch.nn as nn
import conductor
from typing import List, Dict, Any, Optional
import math


# Example 1: Simple custom operation
def custom_gelu_approx(x: torch.Tensor) -> torch.Tensor:
    """
    Custom GELU approximation using tanh.
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


# Example 2: Complex custom operation with multiple inputs
def custom_attention_score(query: torch.Tensor, key: torch.Tensor, scale: float = None) -> torch.Tensor:
    """
    Custom attention score computation with optional scaling.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores * scale
    
    # Apply custom normalization
    scores = scores - torch.max(scores, dim=-1, keepdim=True)[0]
    
    return torch.softmax(scores, dim=-1)


# Example 3: Custom operation with learnable parameters
class CustomLayerNorm(nn.Module):
    """
    Custom layer normalization with additional bias term.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias_scale: float = 0.1):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.bias_scale = bias_scale
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.extra_bias = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard layer norm
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply weights and biases
        output = self.weight * normalized + self.bias
        
        # Add custom bias term
        output = output + self.bias_scale * self.extra_bias
        
        return output


# Register custom operations with Conductor
def register_custom_operations():
    """Register custom operations with the Conductor backend."""
    
    try:
        from conductor.codegen import register_custom_operation
        
        # Register custom GELU approximation
        @register_custom_operation('custom_gelu_approx')
        def convert_custom_gelu(node, inputs: List[str], outputs: List[str], metadata: Dict[str, Any]) -> str:
            """Convert custom GELU to Conductor DSL."""
            x = inputs[0]
            out = outputs[0]
            
            # Generate DSL for GELU approximation
            dsl_code = f"""
            // Custom GELU approximation
            temp1 = pow({x}, 3.0);
            temp2 = 0.044715 * temp1;
            temp3 = {x} + temp2;
            temp4 = sqrt(2.0 / 3.14159265359) * temp3;
            temp5 = tanh(temp4);
            temp6 = 1.0 + temp5;
            temp7 = 0.5 * {x};
            {out} = temp7 * temp6;
            """
            return dsl_code.strip()
        
        # Register custom attention score
        @register_custom_operation('custom_attention_score')
        def convert_custom_attention(node, inputs: List[str], outputs: List[str], metadata: Dict[str, Any]) -> str:
            """Convert custom attention to Conductor DSL."""
            query, key = inputs[:2]
            out = outputs[0]
            scale = metadata.get('scale', 1.0)
            
            dsl_code = f"""
            // Custom attention score computation
            temp_scores = matmul({query}, transpose({key}, -2, -1));
            scaled_scores = {scale} * temp_scores;
            max_scores = max(scaled_scores, dim=-1, keepdim=true);
            normalized_scores = scaled_scores - max_scores;
            {out} = softmax(normalized_scores, dim=-1);
            """
            return dsl_code.strip()
        
        # Register custom layer norm
        @register_custom_operation('custom_layer_norm')
        def convert_custom_layer_norm(node, inputs: List[str], outputs: List[str], metadata: Dict[str, Any]) -> str:
            """Convert custom layer norm to Conductor DSL."""
            x, weight, bias, extra_bias = inputs
            out = outputs[0]
            eps = metadata.get('eps', 1e-5)
            bias_scale = metadata.get('bias_scale', 0.1)
            
            dsl_code = f"""
            // Custom layer normalization
            mean_val = mean({x}, dim=-1, keepdim=true);
            centered = {x} - mean_val;
            var_val = mean(centered * centered, dim=-1, keepdim=true);
            std_val = sqrt(var_val + {eps});
            normalized = centered / std_val;
            weighted = {weight} * normalized;
            biased = weighted + {bias};
            {out} = biased + {bias_scale} * {extra_bias};
            """
            return dsl_code.strip()
        
        print("✓ Custom operations registered successfully")
        return True
        
    except ImportError:
        print("✗ Custom operation registration not available (expected in current implementation)")
        return False


# Example models using custom operations
class ModelWithCustomOps(nn.Module):
    """Model demonstrating custom operations usage."""
    
    def __init__(self, input_size: int = 512, hidden_size: int = 1024):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.custom_norm = CustomLayerNorm(hidden_size, bias_scale=0.1)
        self.output_proj = nn.Linear(hidden_size, input_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard linear projection
        x = self.input_proj(x)
        
        # Custom GELU activation
        x = custom_gelu_approx(x)
        
        # Custom layer normalization
        x = self.custom_norm(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class CustomAttentionModel(nn.Module):
    """Model with custom attention mechanism."""
    
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Project to query, key, value
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Custom attention scores
        attention_weights = custom_attention_score(query, key, scale=1.0 / math.sqrt(self.head_dim))
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, value)
        
        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.output_proj(attended)
        
        return output


def demonstrate_custom_operations():
    """Demonstrate custom operations with Conductor backend."""
    
    print("Conductor Custom Operations Demo")
    print("=" * 40)
    
    # Check backend availability
    if not conductor.is_backend_registered():
        print("ERROR: Conductor backend not registered!")
        return
    
    print("✓ Conductor backend is available")
    
    # Register custom operations
    custom_ops_registered = register_custom_operations()
    
    # Configure Conductor for custom operations
    conductor.configure_backend({
        'debug_mode': True,
        'log_level': 'INFO',
        'save_intermediate_files': True,
        'fusion_config': {
            'elementwise_fusion': True,
            'custom_op_fusion': True  # Enable fusion with custom ops
        }
    })
    
    # Test 1: Model with custom operations
    print("\n1. Testing model with custom operations...")
    
    model1 = ModelWithCustomOps(input_size=256, hidden_size=512)
    input1 = torch.randn(16, 128, 256)
    
    print(f"Model parameters: {sum(p.numel() for p in model1.parameters()):,}")
    print(f"Input shape: {input1.shape}")
    
    # Run with eager mode first
    model1.eval()
    with torch.no_grad():
        eager_output = model1(input1)
    print(f"✓ Eager mode output shape: {eager_output.shape}")
    
    # Try compilation with Conductor
    try:
        compiled_model1 = torch.compile(model1, backend='gcu')
        
        with torch.no_grad():
            conductor_output = compiled_model1(input1)
        
        print(f"✓ Conductor output shape: {conductor_output.shape}")
        
        # Check numerical accuracy
        max_diff = torch.max(torch.abs(eager_output - conductor_output)).item()
        print(f"✓ Max difference: {max_diff:.2e}")
        
        if max_diff < 1e-5:
            print("✓ Numerical accuracy verified")
        else:
            print("⚠ Large numerical difference detected")
            
    except Exception as e:
        print(f"✗ Compilation failed (expected): {e}")
    
    # Test 2: Custom attention model
    print("\n2. Testing custom attention model...")
    
    model2 = CustomAttentionModel(d_model=512, num_heads=8)
    input2 = torch.randn(8, 64, 512)
    
    print(f"Model parameters: {sum(p.numel() for p in model2.parameters()):,}")
    print(f"Input shape: {input2.shape}")
    
    # Run with eager mode
    model2.eval()
    with torch.no_grad():
        eager_output2 = model2(input2)
    print(f"✓ Eager mode output shape: {eager_output2.shape}")
    
    # Try compilation
    try:
        compiled_model2 = torch.compile(model2, backend='gcu')
        
        with torch.no_grad():
            conductor_output2 = compiled_model2(input2)
        
        print(f"✓ Conductor output shape: {conductor_output2.shape}")
        
        # Check accuracy
        max_diff2 = torch.max(torch.abs(eager_output2 - conductor_output2)).item()
        print(f"✓ Max difference: {max_diff2:.2e}")
        
    except Exception as e:
        print(f"✗ Compilation failed (expected): {e}")


def demonstrate_operation_analysis():
    """Analyze custom operations for optimization opportunities."""
    
    print("\n" + "="*50)
    print("CUSTOM OPERATION ANALYSIS")
    print("="*50)
    
    # Analyze different custom operation patterns
    operations = {
        'custom_gelu': lambda x: custom_gelu_approx(x),
        'standard_gelu': lambda x: torch.nn.functional.gelu(x),
        'custom_attention': lambda q, k: custom_attention_score(q, k),
        'standard_attention': lambda q, k: torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1)), dim=-1)
    }
    
    # Test inputs
    x = torch.randn(32, 512)
    q = torch.randn(8, 64, 512)
    k = torch.randn(8, 64, 512)
    
    print("\nOperation Performance Comparison:")
    print(f"{'Operation':<20} {'Time (ms)':<12} {'Memory (MB)':<12}")
    print("-" * 50)
    
    for op_name, op_func in operations.items():
        try:
            # Determine inputs based on operation
            if 'attention' in op_name:
                inputs = (q, k)
            else:
                inputs = (x,)
            
            # Benchmark operation
            import time
            
            # Warmup
            for _ in range(10):
                _ = op_func(*inputs)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(100):
                output = op_func(*inputs)
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / 100 * 1000  # Convert to ms
            
            # Estimate memory usage
            memory_mb = sum(t.numel() * t.element_size() for t in inputs) / 1024**2
            if hasattr(output, 'numel'):
                memory_mb += output.numel() * output.element_size() / 1024**2
            
            print(f"{op_name:<20} {avg_time:<12.2f} {memory_mb:<12.1f}")
            
        except Exception as e:
            print(f"{op_name:<20} {'ERROR':<12} {str(e)[:10]:<12}")


def demonstrate_fusion_with_custom_ops():
    """Demonstrate fusion opportunities with custom operations."""
    
    print("\n" + "="*50)
    print("FUSION WITH CUSTOM OPERATIONS")
    print("="*50)
    
    # Create models with different fusion patterns
    class FusionTestModel(nn.Module):
        def __init__(self, pattern: str):
            super().__init__()
            self.pattern = pattern
            self.linear = nn.Linear(512, 512)
        
        def forward(self, x):
            if self.pattern == 'custom_then_standard':
                # Custom op followed by standard ops
                x = custom_gelu_approx(x)
                x = torch.relu(x)
                x = torch.sigmoid(x)
                return x
            
            elif self.pattern == 'standard_then_custom':
                # Standard ops followed by custom op
                x = torch.relu(x)
                x = torch.sigmoid(x)
                x = custom_gelu_approx(x)
                return x
            
            elif self.pattern == 'mixed_chain':
                # Mixed chain of operations
                x = self.linear(x)
                x = custom_gelu_approx(x)
                x = torch.layer_norm(x, (512,))
                x = torch.relu(x)
                return x
            
            else:
                return x
    
    patterns = ['custom_then_standard', 'standard_then_custom', 'mixed_chain']
    input_tensor = torch.randn(32, 512)
    
    for pattern in patterns:
        print(f"\nTesting fusion pattern: {pattern}")
        
        model = FusionTestModel(pattern)
        
        # Configure for fusion analysis
        conductor.configure_backend({
            'debug_mode': True,
            'fusion_config': {
                'elementwise_fusion': True,
                'custom_op_fusion': True,
                'max_fusion_size': 5
            }
        })
        
        try:
            compiled_model = torch.compile(model, backend='gcu')
            
            # Test execution
            with torch.no_grad():
                output = compiled_model(input_tensor)
            
            print(f"  ✓ Compilation successful, output shape: {output.shape}")
            print(f"  ✓ Check debug logs for fusion decisions")
            
        except Exception as e:
            print(f"  ✗ Compilation failed: {e}")


def main():
    """Main demonstration function."""
    
    print("Conductor Custom Operations Example")
    print("=" * 50)
    
    # Run demonstrations
    demonstrate_custom_operations()
    demonstrate_operation_analysis()
    demonstrate_fusion_with_custom_ops()
    
    print("\n" + "="*60)
    print("CUSTOM OPERATIONS SUMMARY")
    print("="*60)
    
    print("""
This example demonstrated:

1. Custom Operation Registration:
   - Simple mathematical operations (custom GELU)
   - Multi-input operations (custom attention)
   - Operations with learnable parameters (custom layer norm)

2. DSL Code Generation:
   - Converting custom operations to Conductor DSL
   - Handling operation metadata and parameters
   - Maintaining numerical accuracy

3. Fusion Opportunities:
   - Fusing custom operations with standard operations
   - Analyzing fusion patterns and benefits
   - Optimizing mixed operation chains

4. Performance Analysis:
   - Comparing custom vs standard implementations
   - Memory usage optimization
   - Execution time benchmarking

Best Practices for Custom Operations:

1. Registration:
   - Register operations early in your application
   - Provide clear DSL conversion logic
   - Handle edge cases and error conditions

2. Implementation:
   - Maintain numerical accuracy with reference implementation
   - Consider memory layout and access patterns
   - Optimize for target hardware characteristics

3. Testing:
   - Verify correctness against reference implementation
   - Test fusion compatibility with other operations
   - Benchmark performance improvements

4. Documentation:
   - Document operation semantics and parameters
   - Provide usage examples and best practices
   - Explain fusion opportunities and limitations

Note: Custom operation registration is a planned feature.
Current implementation focuses on standard PyTorch operations.
    """)


if __name__ == '__main__':
    main()
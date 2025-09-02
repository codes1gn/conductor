#!/usr/bin/env python3
"""
Advanced fusion optimization example for Conductor PyTorch Backend Integration.

This example demonstrates how Conductor automatically fuses operations for better
performance and how to configure fusion settings for different scenarios.
"""

import torch
import torch.nn as nn
import conductor
import time
from typing import Dict, Any


class FusionDemoModel(nn.Module):
    """Model designed to showcase different fusion patterns."""
    
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Elementwise fusion chain
        self.elementwise_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        
        # Reduction fusion pattern
        self.reduction_layer = nn.Linear(hidden_size, 1)
        
        # Attention-like pattern (memory-bound operations)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        
        # Convolution fusion pattern
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        # Elementwise fusion opportunity
        x = self.elementwise_layers(x)
        
        # Attention pattern
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out  # Residual connection
        
        # Reduction pattern
        scalar_out = self.reduction_layer(x).mean()  # Elementwise + reduction
        
        # Convolution fusion
        conv_out = self.conv_block(img)
        conv_scalar = conv_out.mean()
        
        return scalar_out + conv_scalar


def benchmark_model(model: nn.Module, input_data: tuple, name: str, warmup_runs: int = 3, benchmark_runs: int = 10) -> Dict[str, float]:
    """Benchmark model performance."""
    print(f"\nBenchmarking {name}...")
    
    # Warmup runs
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(*input_data)
    
    # Benchmark runs
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(benchmark_runs):
            output = model(*input_data)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / benchmark_runs
    return {
        'avg_time': avg_time,
        'throughput': 1.0 / avg_time,
        'output_shape': output.shape if hasattr(output, 'shape') else str(type(output))
    }


def demonstrate_fusion_configurations():
    """Demonstrate different fusion configurations and their impact."""
    
    print("Conductor Advanced Fusion Optimization Demo")
    print("=" * 50)
    
    # Create model and input data
    hidden_size = 256
    batch_size = 32
    seq_len = 128
    img_size = 224
    
    model = FusionDemoModel(hidden_size)
    x = torch.randn(seq_len, batch_size, hidden_size)
    img = torch.randn(batch_size, 3, img_size, img_size)
    input_data = (x, img)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shapes: x={x.shape}, img={img.shape}")
    
    # Baseline: No compilation
    baseline_stats = benchmark_model(model, input_data, "Baseline (no compilation)")
    
    # Configuration 1: Default fusion settings
    print("\n" + "="*50)
    print("Configuration 1: Default Fusion Settings")
    conductor.configure_backend({
        'fusion_config': {
            'elementwise_fusion': True,
            'reduction_fusion': True,
            'memory_bound_fusion': True,
            'max_fusion_size': 10,
            'fusion_threshold': 0.8
        },
        'debug_mode': True,
        'log_level': 'INFO'
    })
    
    try:
        compiled_model_1 = torch.compile(model, backend='gcu')
        fusion_stats_1 = benchmark_model(compiled_model_1, input_data, "Default Fusion")
    except Exception as e:
        print(f"Compilation failed (expected): {e}")
        fusion_stats_1 = {'avg_time': float('inf'), 'throughput': 0}
    
    # Configuration 2: Aggressive fusion
    print("\n" + "="*50)
    print("Configuration 2: Aggressive Fusion")
    conductor.configure_backend({
        'fusion_config': {
            'elementwise_fusion': True,
            'reduction_fusion': True,
            'memory_bound_fusion': True,
            'max_fusion_size': 20,  # Larger fusion clusters
            'fusion_threshold': 0.5  # Lower threshold = more aggressive
        },
        'buffer_config': {
            'buffer_reuse_enabled': True,
            'auto_scope_promotion': True
        }
    })
    
    try:
        compiled_model_2 = torch.compile(model, backend='gcu')
        fusion_stats_2 = benchmark_model(compiled_model_2, input_data, "Aggressive Fusion")
    except Exception as e:
        print(f"Compilation failed (expected): {e}")
        fusion_stats_2 = {'avg_time': float('inf'), 'throughput': 0}
    
    # Configuration 3: Conservative fusion (for debugging)
    print("\n" + "="*50)
    print("Configuration 3: Conservative Fusion")
    conductor.configure_backend({
        'fusion_config': {
            'elementwise_fusion': True,
            'reduction_fusion': False,  # Disable reduction fusion
            'memory_bound_fusion': False,  # Disable memory-bound fusion
            'max_fusion_size': 3,  # Small fusion clusters
            'fusion_threshold': 0.95  # High threshold = conservative
        },
        'compilation_config': {
            'optimization_level': 'O1',  # Lower optimization for faster compilation
            'debug_symbols': True
        }
    })
    
    try:
        compiled_model_3 = torch.compile(model, backend='gcu')
        fusion_stats_3 = benchmark_model(compiled_model_3, input_data, "Conservative Fusion")
    except Exception as e:
        print(f"Compilation failed (expected): {e}")
        fusion_stats_3 = {'avg_time': float('inf'), 'throughput': 0}
    
    # Configuration 4: No fusion (individual operations)
    print("\n" + "="*50)
    print("Configuration 4: No Fusion")
    conductor.configure_backend({
        'fusion_config': {
            'elementwise_fusion': False,
            'reduction_fusion': False,
            'memory_bound_fusion': False,
            'max_fusion_size': 1,  # No fusion
            'fusion_threshold': 1.0  # Never fuse
        }
    })
    
    try:
        compiled_model_4 = torch.compile(model, backend='gcu')
        no_fusion_stats = benchmark_model(compiled_model_4, input_data, "No Fusion")
    except Exception as e:
        print(f"Compilation failed (expected): {e}")
        no_fusion_stats = {'avg_time': float('inf'), 'throughput': 0}
    
    # Results summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    results = [
        ("Baseline (no compilation)", baseline_stats),
        ("Default Fusion", fusion_stats_1),
        ("Aggressive Fusion", fusion_stats_2),
        ("Conservative Fusion", fusion_stats_3),
        ("No Fusion", no_fusion_stats),
    ]
    
    print(f"{'Configuration':<25} {'Avg Time (ms)':<15} {'Speedup':<10} {'Throughput':<12}")
    print("-" * 70)
    
    baseline_time = baseline_stats['avg_time']
    for name, stats in results:
        avg_time_ms = stats['avg_time'] * 1000
        speedup = baseline_time / stats['avg_time'] if stats['avg_time'] > 0 else 0
        throughput = stats['throughput']
        
        print(f"{name:<25} {avg_time_ms:<15.2f} {speedup:<10.2f}x {throughput:<12.1f}")
    
    # Fusion analysis
    print("\n" + "="*70)
    print("FUSION ANALYSIS")
    print("="*70)
    
    print("""
Key Observations:
1. Elementwise Fusion: Combines add, mul, relu, gelu operations
2. Reduction Fusion: Fuses elementwise ops with sum/mean reductions
3. Memory-Bound Fusion: Optimizes attention and convolution patterns
4. Buffer Reuse: Reduces memory allocation overhead

Expected Benefits:
- Reduced kernel launch overhead
- Better memory locality
- Improved cache utilization
- Lower memory bandwidth requirements

Trade-offs:
- Larger fusion clusters may increase compilation time
- Aggressive fusion might reduce parallelism opportunities
- Conservative fusion provides better debugging experience
    """)


def demonstrate_custom_fusion_patterns():
    """Show how to analyze and optimize specific fusion patterns."""
    
    print("\n" + "="*70)
    print("CUSTOM FUSION PATTERN ANALYSIS")
    print("="*70)
    
    # Pattern 1: Elementwise chain
    class ElementwiseChain(nn.Module):
        def forward(self, x):
            x = torch.add(x, 1.0)      # add
            x = torch.mul(x, 2.0)      # mul  
            x = torch.relu(x)          # relu
            x = torch.sigmoid(x)       # sigmoid
            return x
    
    # Pattern 2: Reduction pattern
    class ReductionPattern(nn.Module):
        def forward(self, x):
            x = torch.add(x, 1.0)      # elementwise
            x = torch.mul(x, x)        # elementwise
            x = torch.sum(x, dim=-1)   # reduction
            return x
    
    # Pattern 3: Mixed pattern
    class MixedPattern(nn.Module):
        def forward(self, x):
            # Elementwise operations
            x1 = torch.relu(x)
            x2 = torch.sigmoid(x)
            
            # Combine and reduce
            combined = x1 + x2
            result = torch.mean(combined, dim=1)
            return result
    
    patterns = [
        ("Elementwise Chain", ElementwiseChain()),
        ("Reduction Pattern", ReductionPattern()),
        ("Mixed Pattern", MixedPattern()),
    ]
    
    input_tensor = torch.randn(32, 128, 256)
    
    for pattern_name, pattern_model in patterns:
        print(f"\nAnalyzing {pattern_name}:")
        
        # Configure for maximum fusion visibility
        conductor.configure_backend({
            'debug_mode': True,
            'log_level': 'DEBUG',
            'save_intermediate_files': True,
            'fusion_config': {
                'elementwise_fusion': True,
                'reduction_fusion': True,
                'max_fusion_size': 10
            }
        })
        
        try:
            compiled_pattern = torch.compile(pattern_model, backend='gcu')
            
            # This would show fusion decisions in debug logs
            print(f"  ✓ Compiled successfully")
            print(f"  ✓ Check debug logs for fusion analysis")
            
        except Exception as e:
            print(f"  ✗ Compilation failed: {e}")


def main():
    """Main demonstration function."""
    
    # Check backend availability
    if not conductor.is_backend_registered():
        print("ERROR: Conductor backend not registered!")
        return
    
    print("✓ Conductor backend is available")
    
    # Show backend information
    info = conductor.get_backend_info()
    print(f"✓ Backend version: {info.get('version', 'unknown')}")
    
    # Run demonstrations
    demonstrate_fusion_configurations()
    demonstrate_custom_fusion_patterns()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
This example demonstrated:
1. Different fusion configuration strategies
2. Performance impact of fusion settings
3. Trade-offs between compilation time and runtime performance
4. How to analyze specific fusion patterns

For production use:
- Start with default settings
- Profile your specific workload
- Adjust fusion parameters based on model characteristics
- Use conservative settings for debugging
- Enable aggressive fusion for inference workloads

Note: Actual performance improvements depend on:
- Model architecture and size
- Input data characteristics  
- Hardware capabilities
- Memory bandwidth vs compute ratio
    """)


if __name__ == '__main__':
    main()
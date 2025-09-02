#!/usr/bin/env python3
"""
Benchmarking and performance validation example for Conductor PyTorch Backend.

This example demonstrates how to benchmark Conductor against other backends
and validate performance improvements from fusion and optimization.
"""

import torch
import torch.nn as nn
import conductor
import time
import statistics
import json
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    backend_name: str
    model_name: str
    avg_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput: float
    memory_usage: float
    compilation_time: float = 0.0
    fusion_ratio: float = 0.0
    error: str = ""


class BenchmarkSuite:
    """Comprehensive benchmarking suite for Conductor backend."""
    
    def __init__(self, warmup_runs: int = 5, benchmark_runs: int = 20):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: List[BenchmarkResult] = []
    
    def benchmark_model(
        self, 
        model: nn.Module, 
        input_data: tuple, 
        backend_name: str,
        model_name: str,
        compile_fn: Callable = None
    ) -> BenchmarkResult:
        """Benchmark a single model with specified backend."""
        
        print(f"Benchmarking {model_name} with {backend_name}...")
        
        try:
            # Compile model if compile function provided
            compilation_start = time.time()
            if compile_fn:
                compiled_model = compile_fn(model)
            else:
                compiled_model = model
            compilation_time = time.time() - compilation_start
            
            # Warmup runs
            compiled_model.eval()
            with torch.no_grad():
                for _ in range(self.warmup_runs):
                    _ = compiled_model(*input_data)
            
            # Benchmark runs
            times = []
            memory_usage = 0
            
            for i in range(self.benchmark_runs):
                # Memory tracking
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    output = compiled_model(*input_data)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                
                # Track peak memory usage
                if torch.cuda.is_available():
                    memory_usage = max(memory_usage, torch.cuda.max_memory_allocated() / 1024**2)
            
            # Calculate statistics
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            min_time = min(times)
            max_time = max(times)
            throughput = 1.0 / avg_time
            
            # Get fusion ratio for Conductor backend
            fusion_ratio = 0.0
            if backend_name == 'conductor':
                try:
                    from conductor.utils import get_last_compilation_stats
                    stats = get_last_compilation_stats()
                    fusion_ratio = stats.get('fusion_ratio', 0.0)
                except:
                    pass
            
            result = BenchmarkResult(
                backend_name=backend_name,
                model_name=model_name,
                avg_time=avg_time,
                std_time=std_time,
                min_time=min_time,
                max_time=max_time,
                throughput=throughput,
                memory_usage=memory_usage,
                compilation_time=compilation_time,
                fusion_ratio=fusion_ratio
            )
            
            print(f"  ✓ Avg time: {avg_time*1000:.2f}ms, Throughput: {throughput:.1f} iter/s")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            result = BenchmarkResult(
                backend_name=backend_name,
                model_name=model_name,
                avg_time=float('inf'),
                std_time=0,
                min_time=float('inf'),
                max_time=float('inf'),
                throughput=0,
                memory_usage=0,
                error=str(e)
            )
        
        self.results.append(result)
        return result
    
    def compare_backends(self, model: nn.Module, input_data: tuple, model_name: str):
        """Compare model performance across different backends."""
        
        print(f"\nComparing backends for {model_name}")
        print("-" * 60)
        
        # Baseline: No compilation (eager mode)
        self.benchmark_model(
            model, input_data, "eager", model_name,
            compile_fn=None
        )
        
        # Inductor backend (PyTorch default)
        self.benchmark_model(
            model, input_data, "inductor", model_name,
            compile_fn=lambda m: torch.compile(m, backend='inductor')
        )
        
        # Conductor backend (our implementation)
        self.benchmark_model(
            model, input_data, "conductor", model_name,
            compile_fn=lambda m: torch.compile(m, backend='gcu')
        )
    
    def print_summary(self):
        """Print benchmark summary."""
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Group results by model
        models = {}
        for result in self.results:
            if result.model_name not in models:
                models[result.model_name] = []
            models[result.model_name].append(result)
        
        for model_name, model_results in models.items():
            print(f"\n{model_name}:")
            print(f"{'Backend':<12} {'Time (ms)':<12} {'Speedup':<10} {'Memory (MB)':<12} {'Fusion %':<10}")
            print("-" * 70)
            
            # Find baseline (eager) time for speedup calculation
            baseline_time = None
            for result in model_results:
                if result.backend_name == 'eager' and result.avg_time != float('inf'):
                    baseline_time = result.avg_time
                    break
            
            for result in model_results:
                if result.error:
                    print(f"{result.backend_name:<12} {'ERROR':<12} {'-':<10} {'-':<12} {'-':<10}")
                    continue
                
                time_ms = result.avg_time * 1000
                speedup = baseline_time / result.avg_time if baseline_time and result.avg_time > 0 else 1.0
                memory_mb = result.memory_usage
                fusion_pct = result.fusion_ratio * 100
                
                print(f"{result.backend_name:<12} {time_ms:<12.2f} {speedup:<10.2f}x {memory_mb:<12.1f} {fusion_pct:<10.1f}")
    
    def save_results(self, filename: str):
        """Save benchmark results to JSON file."""
        
        results_data = []
        for result in self.results:
            results_data.append({
                'backend_name': result.backend_name,
                'model_name': result.model_name,
                'avg_time': result.avg_time,
                'std_time': result.std_time,
                'min_time': result.min_time,
                'max_time': result.max_time,
                'throughput': result.throughput,
                'memory_usage': result.memory_usage,
                'compilation_time': result.compilation_time,
                'fusion_ratio': result.fusion_ratio,
                'error': result.error
            })
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to {filename}")


# Test Models
class SimpleLinear(nn.Module):
    """Simple linear model for basic benchmarking."""
    
    def __init__(self, input_size: int = 512, hidden_size: int = 1024, num_layers: int = 4):
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, 10))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ConvNet(nn.Module):
    """Convolutional network for vision benchmarking."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block for sequence modeling benchmarking."""
    
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        
        self.embedding = nn.Embedding(10000, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, 10000)
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x)
        x = x + self.pos_encoding[:seq_len]
        
        # Transformer
        x = self.transformer(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


def run_comprehensive_benchmark():
    """Run comprehensive benchmark across different model types."""
    
    print("Conductor PyTorch Backend - Comprehensive Benchmark")
    print("=" * 60)
    
    # Check backend availability
    if not conductor.is_backend_registered():
        print("ERROR: Conductor backend not registered!")
        return
    
    print("✓ Conductor backend is available")
    
    # Configure Conductor for benchmarking
    conductor.configure_backend({
        'fusion_config': {
            'elementwise_fusion': True,
            'reduction_fusion': True,
            'memory_bound_fusion': True,
            'max_fusion_size': 10
        },
        'compilation_config': {
            'optimization_level': 'O2',
            'parallel_compilation': True
        },
        'cache_config': {
            'cache_enabled': True
        },
        'debug_mode': False,
        'log_level': 'WARNING'
    })
    
    # Initialize benchmark suite
    suite = BenchmarkSuite(warmup_runs=3, benchmark_runs=10)
    
    # Test models and inputs
    test_cases = [
        {
            'name': 'SimpleLinear',
            'model': SimpleLinear(input_size=512, hidden_size=1024, num_layers=4),
            'input': (torch.randn(32, 512),)
        },
        {
            'name': 'ConvNet',
            'model': ConvNet(num_classes=10),
            'input': (torch.randn(16, 3, 224, 224),)
        },
        {
            'name': 'TransformerBlock',
            'model': TransformerBlock(d_model=512, nhead=8, num_layers=6),
            'input': (torch.randint(0, 10000, (8, 128)),)
        }
    ]
    
    # Run benchmarks
    for test_case in test_cases:
        suite.compare_backends(
            test_case['model'],
            test_case['input'],
            test_case['name']
        )
    
    # Print results
    suite.print_summary()
    
    # Save results
    results_file = Path('benchmark_results.json')
    suite.save_results(str(results_file))
    
    # Performance analysis
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    print("""
Key Metrics Analyzed:
1. Execution Time: Average time per forward pass
2. Throughput: Iterations per second
3. Memory Usage: Peak GPU memory consumption
4. Compilation Time: Time to compile the model
5. Fusion Ratio: Percentage of operations successfully fused

Expected Conductor Benefits:
- Reduced kernel launch overhead through fusion
- Better memory locality and cache utilization
- Optimized buffer management and reuse
- Hardware-specific optimizations for GCU

Factors Affecting Performance:
- Model architecture (compute vs memory bound)
- Batch size and input dimensions
- Fusion opportunities in the computation graph
- Hardware characteristics and memory bandwidth

Interpretation Guidelines:
- Speedup > 1.0x indicates performance improvement
- Higher fusion ratio typically correlates with better performance
- Memory usage should be similar or lower than baseline
- Compilation time is one-time cost (amortized over many runs)
    """)


def run_fusion_analysis():
    """Analyze fusion effectiveness for different operation patterns."""
    
    print("\n" + "="*80)
    print("FUSION EFFECTIVENESS ANALYSIS")
    print("="*80)
    
    # Test different fusion patterns
    fusion_patterns = {
        'elementwise_chain': lambda x: torch.relu(torch.add(torch.mul(x, 2.0), 1.0)),
        'reduction_pattern': lambda x: torch.sum(torch.relu(torch.add(x, 1.0)), dim=-1),
        'mixed_operations': lambda x: torch.mean(torch.sigmoid(torch.tanh(x)) + torch.relu(x), dim=1),
        'attention_like': lambda x: torch.softmax(torch.matmul(x, x.transpose(-2, -1)), dim=-1)
    }
    
    input_tensor = torch.randn(32, 128, 256)
    
    for pattern_name, pattern_fn in fusion_patterns.items():
        print(f"\nAnalyzing {pattern_name}:")
        
        # Create simple model wrapper
        class PatternModel(nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn
            
            def forward(self, x):
                return self.fn(x)
        
        model = PatternModel(pattern_fn)
        
        # Benchmark with different fusion settings
        for fusion_enabled in [False, True]:
            conductor.configure_backend({
                'fusion_config': {
                    'elementwise_fusion': fusion_enabled,
                    'reduction_fusion': fusion_enabled,
                    'memory_bound_fusion': fusion_enabled
                },
                'debug_mode': True
            })
            
            try:
                compiled_model = torch.compile(model, backend='gcu')
                
                # Quick benchmark
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(10):
                        _ = compiled_model(input_tensor)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                fusion_status = "enabled" if fusion_enabled else "disabled"
                print(f"  {fusion_status:>8}: {avg_time*1000:.2f}ms")
                
            except Exception as e:
                print(f"  Error with fusion {fusion_enabled}: {e}")


def main():
    """Main benchmark execution."""
    
    # Run comprehensive benchmark
    run_comprehensive_benchmark()
    
    # Run fusion analysis
    run_fusion_analysis()
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print("""
Benchmark Results Summary:
- Detailed results saved to benchmark_results.json
- Performance comparison across eager, inductor, and conductor backends
- Fusion effectiveness analysis for different operation patterns

Next Steps:
1. Analyze results to identify optimization opportunities
2. Tune fusion parameters for your specific workloads
3. Profile memory usage and compilation overhead
4. Test with your production models and data

For production deployment:
- Use AOT compilation for stable models
- Enable aggressive caching for repeated compilations
- Monitor performance regressions with CI/CD integration
- Profile end-to-end application performance
    """)


if __name__ == '__main__':
    main()
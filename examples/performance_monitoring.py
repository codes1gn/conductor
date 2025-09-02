#!/usr/bin/env python3
"""
Performance monitoring and optimization example for Conductor PyTorch Backend.

This example demonstrates how to use Conductor's performance monitoring,
profiling, and regression detection capabilities to optimize model performance.
"""

import torch
import torch.nn as nn
import conductor
import time
import tempfile
from pathlib import Path
from typing import Dict, Any

from conductor.utils.profiler import ConductorProfiler, PerformanceBenchmark
from conductor.utils.regression import PerformanceRegressionDetector, create_performance_monitoring_system
from conductor.codegen.optimization import OptimizationPipeline


class PerformanceTestModel(nn.Module):
    """Model designed for performance testing with various optimization opportunities."""
    
    def __init__(self, hidden_size: int = 256, num_layers: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create layers with fusion opportunities
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),  # Fusible with linear
                nn.Dropout(0.1),
            ])
        
        # Final projection
        self.output_proj = nn.Linear(hidden_size, 10)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        # Additional elementwise operations for fusion
        x = x + 0.1  # Bias addition
        x = x * 0.9  # Scaling
        
        return self.output_proj(x)


def demonstrate_performance_profiling():
    """Demonstrate comprehensive performance profiling."""
    print("Performance Profiling Demonstration")
    print("=" * 50)
    
    # Create model and input
    model = PerformanceTestModel(hidden_size=512, num_layers=6)
    x = torch.randn(32, 512)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {x.shape}")
    
    # Trace model
    traced_model = torch.fx.symbolic_trace(model)
    
    # Create profiler
    profiler = ConductorProfiler(enable_memory_tracking=True)
    
    print("\n1. Profiling Compilation Process...")
    
    with profiler:
        try:
            # This would normally compile with Conductor
            # For demo purposes, we'll simulate the compilation phases
            
            # Simulate graph analysis
            time.sleep(0.02)
            profiler.record_phase_time('graph_analysis', 0.02)
            
            # Simulate fusion analysis
            time.sleep(0.05)
            profiler.record_phase_time('fusion_analysis', 0.05)
            profiler.record_fusion_metrics(total_nodes=20, fused_nodes=16)
            
            # Simulate DSL generation
            time.sleep(0.03)
            profiler.record_phase_time('dsl_generation', 0.03)
            
            # Simulate compilation
            compilation_start = time.perf_counter()
            time.sleep(0.1)  # Simulate compilation time
            compilation_time = time.perf_counter() - compilation_start
            profiler.record_compilation_time(compilation_time)
            
            # Simulate execution
            execution_start = time.perf_counter()
            with torch.no_grad():
                output = model(x)  # Use original model for demo
            execution_time = time.perf_counter() - execution_start
            profiler.record_execution_time(execution_time)
            
            # Record additional metrics
            profiler.record_cache_metrics(hits=0, total=1)  # Cache miss
            profiler.record_buffer_reuse(reused_buffers=8, total_buffers=12)
            profiler.record_kernel_launches(4)  # After fusion
            
            print("✓ Compilation completed successfully")
            
        except Exception as e:
            print(f"✗ Compilation failed: {e}")
    
    # Display profiling results
    print("\n2. Performance Metrics:")
    stats = profiler.get_stats()
    
    for metric, value in stats.items():
        if isinstance(value, float):
            if 'time' in metric:
                print(f"  {metric}: {value:.3f}s")
            elif 'ratio' in metric:
                print(f"  {metric}: {value:.1%}")
            elif 'mb' in metric:
                print(f"  {metric}: {value:.1f}MB")
            else:
                print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
    
    # Display summary
    print("\n3. Performance Summary:")
    print(profiler.get_summary())
    
    # Validate performance requirements
    print("\n4. Performance Requirements Validation:")
    
    # JIT compilation overhead < 10% requirement
    if stats['execution_time'] > 0:
        overhead_ratio = stats['compilation_time'] / stats['execution_time']
        jit_requirement_met = overhead_ratio < 0.1
        print(f"  JIT Overhead: {overhead_ratio:.1%} ({'✓' if jit_requirement_met else '✗'} < 10%)")
    
    # Fusion effectiveness > 80% requirement
    fusion_requirement_met = stats['fusion_ratio'] > 0.8
    print(f"  Fusion Effectiveness: {stats['fusion_ratio']:.1%} ({'✓' if fusion_requirement_met else '✗'} > 80%)")
    
    # Buffer reuse effectiveness
    buffer_reuse_good = stats['buffer_reuse_ratio'] > 0.5
    print(f"  Buffer Reuse: {stats['buffer_reuse_ratio']:.1%} ({'✓' if buffer_reuse_good else '✗'} > 50%)")
    
    return profiler


def demonstrate_performance_benchmarking():
    """Demonstrate performance benchmarking framework."""
    print("\n" + "=" * 60)
    print("Performance Benchmarking Demonstration")
    print("=" * 60)
    
    # Create benchmark framework
    benchmark = PerformanceBenchmark(warmup_runs=2, benchmark_runs=5)
    
    # Test different model sizes
    model_configs = [
        (128, 2, "small"),
        (256, 4, "medium"),
        (512, 6, "large")
    ]
    
    print("\n1. Benchmarking Different Model Sizes:")
    
    for hidden_size, num_layers, size_name in model_configs:
        model = PerformanceTestModel(hidden_size, num_layers)
        x = torch.randn(16, hidden_size)
        
        def compile_and_execute():
            # Simulate compilation and execution
            traced = torch.fx.symbolic_trace(model)
            
            # Simulate compilation time based on model size
            compile_time = (hidden_size * num_layers) / 100000  # Heuristic
            time.sleep(compile_time)
            
            # Execute model
            with torch.no_grad():
                output = model(x)
            
            return output
        
        # Run benchmark
        results = benchmark.benchmark_function(
            compile_and_execute,
            name=f"{size_name}_model"
        )
        
        print(f"  {size_name.capitalize()} Model ({hidden_size}x{num_layers}):")
        print(f"    Average Time: {results['avg_time']:.3f}s")
        print(f"    Throughput: {results['throughput']:.1f} ops/sec")
        print(f"    Memory Usage: {results['peak_memory_mb']:.1f}MB")
        print(f"    Success Rate: {results['successful_runs']}/{results['total_runs']}")
    
    # Generate comprehensive report
    print("\n2. Benchmark Report:")
    report = benchmark.get_performance_report()
    print(report)
    
    # Validate performance requirements
    print("\n3. Performance Requirements Validation:")
    
    for result in benchmark.results:
        validation = benchmark.validate_performance_requirements({
            'compilation_time': result['avg_time'] * 0.1,  # Assume 10% is compilation
            'execution_time': result['avg_time'] * 0.9,    # Assume 90% is execution
            'peak_memory_mb': result['peak_memory_mb'],
            'fusion_ratio': 0.85  # Assume good fusion
        })
        
        print(f"  {result['name']}:")
        for req, passed in validation.items():
            status = "✓" if passed else "✗"
            print(f"    {req}: {status}")


def demonstrate_regression_detection():
    """Demonstrate performance regression detection."""
    print("\n" + "=" * 60)
    print("Performance Regression Detection Demonstration")
    print("=" * 60)
    
    # Create temporary directory for baselines
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create regression detector
        detector = PerformanceRegressionDetector(baseline_dir=temp_dir)
        
        print("\n1. Creating Performance Baseline:")
        
        # Create baseline performance metrics
        from conductor.utils.profiler import PerformanceMetrics
        
        baseline_metrics = PerformanceMetrics(
            compilation_time=0.15,
            execution_time=0.05,
            memory_peak_mb=256.0,
            fusion_ratio=0.85,
            cache_hit_ratio=0.0,  # First run
            buffer_reuse_ratio=0.75
        )
        
        model_info = {
            'type': 'PerformanceTestModel',
            'hidden_size': 512,
            'num_layers': 6,
            'input_shape': (32, 512),
            'parameters': sum(p.numel() for p in PerformanceTestModel(512, 6).parameters())
        }
        
        # Create baseline
        baseline = detector.create_baseline(
            name="performance_test_model",
            metrics=baseline_metrics,
            model_info=model_info
        )
        
        print(f"✓ Created baseline: {baseline.name}")
        print(f"  Compilation time: {baseline.metrics.compilation_time:.3f}s")
        print(f"  Execution time: {baseline.metrics.execution_time:.3f}s")
        print(f"  Memory usage: {baseline.metrics.memory_peak_mb:.1f}MB")
        print(f"  Fusion ratio: {baseline.metrics.fusion_ratio:.1%}")
        
        print("\n2. Testing Current Performance (No Regression):")
        
        # Test with similar performance (no regression)
        current_metrics_good = PerformanceMetrics(
            compilation_time=0.14,  # Slightly better
            execution_time=0.048,   # Slightly better
            memory_peak_mb=250.0,   # Slightly better
            fusion_ratio=0.87,      # Better fusion
            cache_hit_ratio=0.8,    # Cache hit
            buffer_reuse_ratio=0.78 # Better reuse
        )
        
        regressions_good = detector.detect_regressions(
            current_metrics=current_metrics_good,
            baseline_name="performance_test_model",
            model_info=model_info
        )
        
        print(f"  Regressions detected: {len([r for r in regressions_good if r.is_regression])}")
        
        for result in regressions_good:
            if result.is_regression:
                print(f"    {result}")
            elif abs(result.change_percent) > 1.0:  # Show improvements
                print(f"    {result} (Improvement)")
        
        print("\n3. Testing Current Performance (With Regressions):")
        
        # Test with worse performance (regressions)
        current_metrics_bad = PerformanceMetrics(
            compilation_time=0.22,  # 47% slower - major regression
            execution_time=0.06,    # 20% slower - major regression
            memory_peak_mb=350.0,   # 37% more memory - critical regression
            fusion_ratio=0.70,      # 18% worse fusion - major regression
            cache_hit_ratio=0.3,    # Poor cache performance
            buffer_reuse_ratio=0.60 # Worse reuse
        )
        
        regressions_bad = detector.detect_regressions(
            current_metrics=current_metrics_bad,
            baseline_name="performance_test_model",
            model_info=model_info
        )
        
        print(f"  Regressions detected: {len([r for r in regressions_bad if r.is_regression])}")
        
        for result in regressions_bad:
            if result.is_regression:
                severity_icon = {"minor": "⚠️", "major": "🔶", "critical": "🚨"}.get(result.severity, "❓")
                print(f"    {severity_icon} {result}")
        
        print("\n4. Regression Report:")
        
        # Generate comprehensive regression report
        report = detector.generate_regression_report(regressions_bad, "performance_test_model")
        print(report)


def demonstrate_continuous_monitoring():
    """Demonstrate continuous performance monitoring."""
    print("\n" + "=" * 60)
    print("Continuous Performance Monitoring Demonstration")
    print("=" * 60)
    
    # Create monitoring system
    with tempfile.TemporaryDirectory() as temp_dir:
        detector, monitor = create_performance_monitoring_system(baseline_dir=temp_dir)
        
        # Set up alert callback
        alerts_received = []
        
        def alert_callback(alert_data):
            alerts_received.append(alert_data)
            print(f"🚨 PERFORMANCE ALERT: {alert_data['model_name']}")
            print(f"   Timestamp: {alert_data['timestamp']}")
            print(f"   Regressions: {len(alert_data['regressions'])}")
        
        monitor.add_alert_callback(alert_callback)
        
        print("\n1. Starting Continuous Monitoring:")
        monitor.start_monitoring()
        print("✓ Monitoring system active")
        
        # Create baseline first
        from conductor.utils.profiler import PerformanceMetrics
        
        baseline_metrics = PerformanceMetrics(
            compilation_time=0.1,
            execution_time=0.05,
            memory_peak_mb=200.0,
            fusion_ratio=0.8
        )
        
        model_info = {'type': 'test_model', 'input_shape': (16, 128)}
        
        detector.create_baseline(
            name="continuous_test_model",
            metrics=baseline_metrics,
            model_info=model_info
        )
        
        print("\n2. Simulating Model Compilations:")
        
        # Simulate good performance (no alerts)
        good_profiler = ConductorProfiler()
        good_profiler.metrics = PerformanceMetrics(
            compilation_time=0.09,
            execution_time=0.048,
            memory_peak_mb=195.0,
            fusion_ratio=0.82
        )
        
        print("  Testing good performance...")
        monitor.monitor_compilation("continuous_test_model", good_profiler, model_info)
        print(f"  Alerts triggered: {len(alerts_received)}")
        
        # Simulate bad performance (should trigger alerts)
        bad_profiler = ConductorProfiler()
        bad_profiler.metrics = PerformanceMetrics(
            compilation_time=0.18,  # 80% slower - critical
            execution_time=0.065,   # 30% slower - critical
            memory_peak_mb=320.0,   # 60% more memory - critical
            fusion_ratio=0.65       # 19% worse fusion - major
        )
        
        print("  Testing degraded performance...")
        monitor.monitor_compilation("continuous_test_model", bad_profiler, model_info)
        print(f"  Alerts triggered: {len(alerts_received)}")
        
        # Display alert details
        if alerts_received:
            print("\n3. Alert Details:")
            for i, alert in enumerate(alerts_received):
                print(f"  Alert {i+1}:")
                print(f"    Model: {alert['model_name']}")
                print(f"    Regressions: {len(alert['regressions'])}")
                
                for regression in alert['regressions'][:3]:  # Show first 3
                    print(f"      - {regression['metric_name']}: {regression['change_percent']:.1f}% change")
        
        print("\n4. Stopping Monitoring:")
        monitor.stop_monitoring()
        print("✓ Monitoring system stopped")


def demonstrate_optimization_pipeline():
    """Demonstrate advanced optimization pipeline."""
    print("\n" + "=" * 60)
    print("Advanced Optimization Pipeline Demonstration")
    print("=" * 60)
    
    # This would normally work with real DAG, but we'll simulate for demo
    print("\n1. Optimization Pipeline Overview:")
    print("  - Buffer reuse optimization")
    print("  - Memory layout optimization") 
    print("  - Advanced fusion heuristics")
    print("  - Performance estimation")
    
    # Create optimization pipeline
    optimizer = OptimizationPipeline()
    
    print("\n2. Simulated Optimization Results:")
    
    # Simulate optimization results
    optimization_results = {
        'buffer_reuse_mapping': {
            'temp_1': 'temp_0',
            'temp_3': 'temp_2',
            'temp_5': 'temp_4'
        },
        'memory_layout_hints': {
            'large_matrix': 'blocked_layout',
            'vector_data': 'vectorized_layout'
        },
        'fusion_clusters': [
            {'type': 'elementwise', 'nodes': 3, 'benefit': 8.5},
            {'type': 'reduction', 'nodes': 2, 'benefit': 12.0}
        ],
        'optimization_stats': {
            'total_nodes': 15,
            'fused_nodes': 12,
            'fusion_ratio': 0.8,
            'total_buffers': 20,
            'reused_buffers': 6,
            'buffer_reuse_ratio': 0.3,
            'memory_layout_optimizations': 8,
            'fusion_clusters': 2
        }
    }
    
    print(f"  Buffer Reuse:")
    print(f"    Reused buffers: {len(optimization_results['buffer_reuse_mapping'])}")
    print(f"    Reuse ratio: {optimization_results['optimization_stats']['buffer_reuse_ratio']:.1%}")
    
    print(f"  Memory Layout:")
    print(f"    Optimized buffers: {len(optimization_results['memory_layout_hints'])}")
    print(f"    Layout strategies: {list(optimization_results['memory_layout_hints'].values())}")
    
    print(f"  Fusion Optimization:")
    print(f"    Fusion clusters: {len(optimization_results['fusion_clusters'])}")
    print(f"    Fusion ratio: {optimization_results['optimization_stats']['fusion_ratio']:.1%}")
    
    for i, cluster in enumerate(optimization_results['fusion_clusters']):
        print(f"    Cluster {i+1}: {cluster['type']} ({cluster['nodes']} nodes, {cluster['benefit']:.1f}x benefit)")
    
    print(f"\n3. Overall Optimization Impact:")
    stats = optimization_results['optimization_stats']
    print(f"  Total operations: {stats['total_nodes']}")
    print(f"  Fused operations: {stats['fused_nodes']} ({stats['fusion_ratio']:.1%})")
    print(f"  Buffer reuse: {stats['reused_buffers']}/{stats['total_buffers']} ({stats['buffer_reuse_ratio']:.1%})")
    print(f"  Memory optimizations: {stats['memory_layout_optimizations']}")
    
    # Estimate performance improvement
    estimated_speedup = 1.0 + (stats['fusion_ratio'] * 0.5) + (stats['buffer_reuse_ratio'] * 0.2)
    print(f"  Estimated speedup: {estimated_speedup:.1f}x")


def main():
    """Main demonstration function."""
    print("Conductor Performance Monitoring and Optimization Demo")
    print("=" * 60)
    
    # Check backend availability
    if not conductor.is_backend_registered():
        print("⚠️  Conductor backend not registered - using simulation mode")
    else:
        print("✓ Conductor backend is available")
    
    try:
        # Run demonstrations
        profiler = demonstrate_performance_profiling()
        demonstrate_performance_benchmarking()
        demonstrate_regression_detection()
        demonstrate_continuous_monitoring()
        demonstrate_optimization_pipeline()
        
        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETE")
        print("=" * 70)
        
        print("""
Performance Monitoring Summary:

1. Performance Profiling:
   - Comprehensive metrics collection during compilation and execution
   - Memory usage tracking and optimization validation
   - Phase-by-phase timing analysis
   - Automatic performance requirement validation

2. Benchmarking Framework:
   - Standardized performance measurement across model sizes
   - Statistical analysis with multiple runs and warmup
   - Performance requirement validation
   - Comprehensive reporting

3. Regression Detection:
   - Baseline creation and management
   - Automatic regression detection with severity classification
   - Detailed regression reports with recommendations
   - Historical trend analysis

4. Continuous Monitoring:
   - Real-time performance monitoring during production
   - Automatic alerting for performance regressions
   - Configurable thresholds and alert callbacks
   - Integration with existing monitoring systems

5. Advanced Optimization:
   - Buffer reuse optimization for memory efficiency
   - Memory layout optimization for cache performance
   - Advanced fusion heuristics for complex patterns
   - Comprehensive optimization impact analysis

Key Benefits:
- Validates performance requirements automatically
- Detects regressions before they impact users
- Provides actionable optimization recommendations
- Enables data-driven performance improvements
- Supports both development and production workflows

Next Steps:
- Integrate profiling into your development workflow
- Set up continuous monitoring for production models
- Use regression detection for CI/CD pipelines
- Apply optimization recommendations to improve performance
        """)
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
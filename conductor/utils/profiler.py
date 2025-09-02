"""
Performance monitoring and profiling utilities.

This module provides comprehensive performance monitoring capabilities
for the Conductor compilation and execution pipeline.
"""

import time
import psutil
import os
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    compilation_time: float = 0.0
    execution_time: float = 0.0
    memory_peak_mb: float = 0.0
    memory_allocated_mb: float = 0.0
    fusion_ratio: float = 0.0
    cache_hit_ratio: float = 0.0
    kernel_launches: int = 0
    buffer_reuse_ratio: float = 0.0
    dsl_generation_time: float = 0.0
    graph_analysis_time: float = 0.0
    fusion_analysis_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'compilation_time': self.compilation_time,
            'execution_time': self.execution_time,
            'memory_peak_mb': self.memory_peak_mb,
            'memory_allocated_mb': self.memory_allocated_mb,
            'fusion_ratio': self.fusion_ratio,
            'cache_hit_ratio': self.cache_hit_ratio,
            'kernel_launches': self.kernel_launches,
            'buffer_reuse_ratio': self.buffer_reuse_ratio,
            'dsl_generation_time': self.dsl_generation_time,
            'graph_analysis_time': self.graph_analysis_time,
            'fusion_analysis_time': self.fusion_analysis_time,
        }


class ConductorProfiler:
    """
    Comprehensive profiler for Conductor operations.
    
    This class provides detailed performance monitoring and profiling
    capabilities for compilation and execution workflows.
    """
    
    def __init__(self, enable_memory_tracking: bool = True):
        """
        Initialize profiler.
        
        Args:
            enable_memory_tracking: Whether to track memory usage
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.metrics = PerformanceMetrics()
        self._start_time = 0.0
        self._memory_tracker = None
        self._active = False
        
        if enable_memory_tracking:
            self._process = psutil.Process(os.getpid())
            self._initial_memory = self._process.memory_info().rss / 1024 / 1024
    
    def __enter__(self):
        """Start profiling context."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling context."""
        self.stop()
    
    def start(self):
        """Start profiling."""
        if self._active:
            logger.warning("Profiler already active")
            return
        
        self._active = True
        self._start_time = time.perf_counter()
        
        if self.enable_memory_tracking:
            self._start_memory_tracking()
        
        logger.debug("Performance profiling started")
    
    def stop(self):
        """Stop profiling and finalize metrics."""
        if not self._active:
            logger.warning("Profiler not active")
            return
        
        self._active = False
        
        if self.enable_memory_tracking:
            self._stop_memory_tracking()
        
        logger.debug("Performance profiling stopped")
    
    def _start_memory_tracking(self):
        """Start memory usage tracking."""
        if not self.enable_memory_tracking:
            return
        
        self._memory_tracker = MemoryTracker()
        self._memory_tracker.start()
    
    def _stop_memory_tracking(self):
        """Stop memory usage tracking and update metrics."""
        if not self.enable_memory_tracking or not self._memory_tracker:
            return
        
        memory_stats = self._memory_tracker.stop()
        self.metrics.memory_peak_mb = memory_stats['peak_memory_mb']
        self.metrics.memory_allocated_mb = memory_stats['allocated_mb']
    
    def record_compilation_time(self, duration: float):
        """Record compilation time."""
        self.metrics.compilation_time = duration
        logger.debug(f"Compilation time: {duration:.3f}s")
    
    def record_execution_time(self, duration: float):
        """Record execution time."""
        self.metrics.execution_time = duration
        logger.debug(f"Execution time: {duration:.3f}s")
    
    def record_fusion_metrics(self, total_nodes: int, fused_nodes: int):
        """Record fusion effectiveness metrics."""
        if total_nodes > 0:
            self.metrics.fusion_ratio = fused_nodes / total_nodes
        logger.debug(f"Fusion ratio: {self.metrics.fusion_ratio:.2%}")
    
    def record_cache_metrics(self, hits: int, total: int):
        """Record cache performance metrics."""
        if total > 0:
            self.metrics.cache_hit_ratio = hits / total
        logger.debug(f"Cache hit ratio: {self.metrics.cache_hit_ratio:.2%}")
    
    def record_buffer_reuse(self, reused_buffers: int, total_buffers: int):
        """Record buffer reuse metrics."""
        if total_buffers > 0:
            self.metrics.buffer_reuse_ratio = reused_buffers / total_buffers
        logger.debug(f"Buffer reuse ratio: {self.metrics.buffer_reuse_ratio:.2%}")
    
    def record_kernel_launches(self, count: int):
        """Record number of kernel launches."""
        self.metrics.kernel_launches = count
        logger.debug(f"Kernel launches: {count}")
    
    def record_phase_time(self, phase: str, duration: float):
        """Record time for specific compilation phases."""
        if phase == 'dsl_generation':
            self.metrics.dsl_generation_time = duration
        elif phase == 'graph_analysis':
            self.metrics.graph_analysis_time = duration
        elif phase == 'fusion_analysis':
            self.metrics.fusion_analysis_time = duration
        
        logger.debug(f"{phase} time: {duration:.3f}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.metrics.to_dict()
    
    def get_summary(self) -> str:
        """Get human-readable performance summary."""
        stats = self.get_stats()
        
        summary_lines = [
            "Performance Summary:",
            f"  Compilation: {stats['compilation_time']:.3f}s",
            f"  Execution: {stats['execution_time']:.3f}s",
            f"  Memory Peak: {stats['memory_peak_mb']:.1f}MB",
            f"  Fusion Ratio: {stats['fusion_ratio']:.1%}",
            f"  Cache Hit Ratio: {stats['cache_hit_ratio']:.1%}",
            f"  Buffer Reuse: {stats['buffer_reuse_ratio']:.1%}",
            f"  Kernel Launches: {stats['kernel_launches']}",
        ]
        
        return "\n".join(summary_lines)


class MemoryTracker:
    """
    Memory usage tracker for performance monitoring.
    
    Tracks memory usage patterns during compilation and execution
    to identify memory optimization opportunities.
    """
    
    def __init__(self, sampling_interval: float = 0.1):
        """
        Initialize memory tracker.
        
        Args:
            sampling_interval: Memory sampling interval in seconds
        """
        self.sampling_interval = sampling_interval
        self._process = psutil.Process(os.getpid())
        self._initial_memory = 0.0
        self._peak_memory = 0.0
        self._samples = []
        self._tracking = False
        self._tracker_thread = None
    
    def start(self):
        """Start memory tracking."""
        if self._tracking:
            return
        
        self._tracking = True
        self._initial_memory = self._process.memory_info().rss / 1024 / 1024
        self._peak_memory = self._initial_memory
        self._samples = []
        
        # Start background tracking thread
        self._tracker_thread = threading.Thread(target=self._track_memory)
        self._tracker_thread.daemon = True
        self._tracker_thread.start()
    
    def stop(self) -> Dict[str, float]:
        """
        Stop memory tracking and return statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self._tracking:
            return {'peak_memory_mb': 0.0, 'allocated_mb': 0.0}
        
        self._tracking = False
        
        if self._tracker_thread:
            self._tracker_thread.join(timeout=1.0)
        
        current_memory = self._process.memory_info().rss / 1024 / 1024
        allocated_mb = current_memory - self._initial_memory
        
        return {
            'peak_memory_mb': self._peak_memory,
            'allocated_mb': max(0, allocated_mb),
            'samples': len(self._samples)
        }
    
    def _track_memory(self):
        """Background memory tracking loop."""
        while self._tracking:
            try:
                current_memory = self._process.memory_info().rss / 1024 / 1024
                self._peak_memory = max(self._peak_memory, current_memory)
                self._samples.append(current_memory)
                
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.warning(f"Memory tracking error: {e}")
                break


class PerformanceBenchmark:
    """
    Benchmarking framework for performance validation.
    
    Provides standardized benchmarking capabilities to validate
    performance requirements and detect regressions.
    """
    
    def __init__(self, warmup_runs: int = 3, benchmark_runs: int = 10):
        """
        Initialize benchmark framework.
        
        Args:
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations
        """
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = []
    
    def benchmark_function(
        self, 
        func: Callable, 
        *args, 
        name: str = "benchmark",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Benchmark a function with multiple runs.
        
        Args:
            func: Function to benchmark
            *args: Function arguments
            name: Benchmark name
            **kwargs: Function keyword arguments
            
        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Starting benchmark: {name}")
        
        # Warmup runs
        for i in range(self.warmup_runs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Warmup run {i} failed: {e}")
        
        # Benchmark runs
        times = []
        memory_usage = []
        
        for i in range(self.benchmark_runs):
            # Memory tracking
            memory_tracker = MemoryTracker()
            memory_tracker.start()
            
            # Time measurement
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                logger.error(f"Benchmark run {i} failed: {e}")
                success = False
                result = None
            
            end_time = time.perf_counter()
            memory_stats = memory_tracker.stop()
            
            if success:
                times.append(end_time - start_time)
                memory_usage.append(memory_stats['peak_memory_mb'])
        
        # Calculate statistics
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        else:
            avg_time = min_time = max_time = std_time = 0.0
        
        if memory_usage:
            avg_memory = sum(memory_usage) / len(memory_usage)
            peak_memory = max(memory_usage)
        else:
            avg_memory = peak_memory = 0.0
        
        results = {
            'name': name,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_time': std_time,
            'avg_memory_mb': avg_memory,
            'peak_memory_mb': peak_memory,
            'successful_runs': len(times),
            'total_runs': self.benchmark_runs,
            'throughput': 1.0 / avg_time if avg_time > 0 else 0.0
        }
        
        self.results.append(results)
        logger.info(f"Benchmark {name} completed: {avg_time:.3f}s avg")
        
        return results
    
    def validate_performance_requirements(self, results: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate performance against requirements.
        
        Args:
            results: Benchmark results
            
        Returns:
            Dictionary of requirement validation results
        """
        validation = {}
        
        # JIT compilation overhead < 10% of execution time
        if 'compilation_time' in results and 'execution_time' in results:
            compilation_time = results['compilation_time']
            execution_time = results['execution_time']
            
            if execution_time > 0:
                overhead_ratio = compilation_time / execution_time
                validation['jit_overhead_requirement'] = overhead_ratio < 0.1
            else:
                validation['jit_overhead_requirement'] = True
        
        # Memory usage should be reasonable
        if 'peak_memory_mb' in results:
            # Arbitrary threshold - should be configurable
            validation['memory_requirement'] = results['peak_memory_mb'] < 1024
        
        # Fusion effectiveness > 80% for fusible operations
        if 'fusion_ratio' in results:
            validation['fusion_effectiveness'] = results['fusion_ratio'] > 0.8
        
        return validation
    
    def get_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        if not self.results:
            return "No benchmark results available"
        
        report_lines = [
            "Performance Benchmark Report",
            "=" * 40,
            ""
        ]
        
        for result in self.results:
            report_lines.extend([
                f"Benchmark: {result['name']}",
                f"  Average Time: {result['avg_time']:.3f}s",
                f"  Min/Max Time: {result['min_time']:.3f}s / {result['max_time']:.3f}s",
                f"  Standard Deviation: {result['std_time']:.3f}s",
                f"  Throughput: {result['throughput']:.1f} ops/sec",
                f"  Memory Usage: {result['avg_memory_mb']:.1f}MB avg, {result['peak_memory_mb']:.1f}MB peak",
                f"  Success Rate: {result['successful_runs']}/{result['total_runs']}",
                ""
            ])
        
        return "\n".join(report_lines)


@contextmanager
def profile_operation(operation_name: str, profiler: Optional[ConductorProfiler] = None):
    """
    Context manager for profiling individual operations.
    
    Args:
        operation_name: Name of the operation being profiled
        profiler: Optional profiler instance to use
    """
    if profiler is None:
        profiler = ConductorProfiler()
    
    start_time = time.perf_counter()
    
    try:
        yield profiler
    finally:
        duration = time.perf_counter() - start_time
        profiler.record_phase_time(operation_name, duration)


def get_last_compilation_stats() -> Dict[str, Any]:
    """
    Get statistics from the last compilation.
    
    This is a placeholder for accessing compilation statistics
    from the global profiler state.
    
    Returns:
        Dictionary with compilation statistics
    """
    # In a real implementation, this would access a global profiler
    # or compilation context to get the actual statistics
    return {
        'compilation_time': 0.0,
        'fusion_ratio': 0.0,
        'cache_hit_ratio': 0.0,
        'memory_usage_mb': 0.0
    }
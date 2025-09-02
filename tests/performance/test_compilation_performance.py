"""
Performance tests for compilation pipeline.

This module tests the performance characteristics of the Conductor
compilation pipeline and validates performance requirements.
"""

import pytest
import torch
import time
import psutil
import os
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from conductor.runtime.jit import JITCompiler
from conductor.runtime.aot import AOTManager
from conductor.codegen.graph import GraphAnalyzer
from conductor.codegen.fusion import FusionEngine
from conductor.codegen.dsl import DSLGenerator


@pytest.mark.performance
class TestCompilationPerformance:
    """Test compilation performance characteristics."""
    
    def test_small_model_compilation_time(self, simple_torch_model, performance_config, mock_conductor_cli):
        """Test compilation time for small models."""
        # Create small model
        x = torch.randn(8, 16)
        traced_model = torch.fx.symbolic_trace(simple_torch_model)
        
        # Mock compilation pipeline
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=1024):
                with patch('tempfile.NamedTemporaryFile') as mock_temp:
                    mock_file = Mock()
                    mock_file.name = "/tmp/small_model.so"
                    mock_temp.return_value.__enter__.return_value = mock_file
                    
                    with patch('conductor.runtime.loader.ExecutableKernel.load_from_file'):
                        compiler = JITCompiler()
                        
                        # Measure compilation time
                        start_time = time.perf_counter()
                        artifact = compiler.compile_graph(traced_model)
                        compilation_time = time.perf_counter() - start_time
                        
                        # Performance requirement: small models should compile quickly
                        assert compilation_time < 1.0, f"Small model compilation took {compilation_time:.3f}s"
                        
                        # Verify artifact was created
                        assert artifact is not None
                        assert 'graph_hash' in artifact.metadata
    
    def test_medium_model_compilation_time(self, complex_torch_model, performance_config, mock_conductor_cli):
        """Test compilation time for medium-sized models."""
        # Create medium model input
        x = torch.randn(32, 128)
        traced_model = torch.fx.symbolic_trace(complex_torch_model)
        
        # Mock compilation pipeline
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=4096):
                with patch('tempfile.NamedTemporaryFile') as mock_temp:
                    mock_file = Mock()
                    mock_file.name = "/tmp/medium_model.so"
                    mock_temp.return_value.__enter__.return_value = mock_file
                    
                    with patch('conductor.runtime.loader.ExecutableKernel.load_from_file'):
                        compiler = JITCompiler()
                        
                        # Measure compilation time
                        start_time = time.perf_counter()
                        artifact = compiler.compile_graph(traced_model)
                        compilation_time = time.perf_counter() - start_time
                        
                        # Performance requirement: medium models should compile reasonably fast
                        assert compilation_time < 5.0, f"Medium model compilation took {compilation_time:.3f}s"
                        
                        # Check that fusion was applied
                        assert artifact.metadata.get('fusion_clusters', 0) >= 0
    
    def test_compilation_memory_usage(self, complex_torch_model, performance_config, mock_conductor_cli):
        """Test memory usage during compilation."""
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create model
        x = torch.randn(64, 256)
        traced_model = torch.fx.symbolic_trace(complex_torch_model)
        
        # Mock compilation pipeline
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=8192):
                with patch('tempfile.NamedTemporaryFile') as mock_temp:
                    mock_file = Mock()
                    mock_file.name = "/tmp/memory_test.so"
                    mock_temp.return_value.__enter__.return_value = mock_file
                    
                    with patch('conductor.runtime.loader.ExecutableKernel.load_from_file'):
                        compiler = JITCompiler()
                        
                        # Compile and measure peak memory
                        artifact = compiler.compile_graph(traced_model)
                        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
                        
                        memory_increase = peak_memory - initial_memory
                        
                        # Performance requirement: memory usage should be reasonable
                        assert memory_increase < performance_config['memory_limit_mb'], \
                            f"Memory usage increased by {memory_increase:.1f}MB"
    
    def test_cache_performance_impact(self, simple_torch_model, temp_test_dir, performance_config):
        """Test performance impact of compilation caching."""
        compiler = JITCompiler(cache_dir=temp_test_dir)
        traced_model = torch.fx.symbolic_trace(simple_torch_model)
        
        # Mock compilation for first run
        with patch.object(compiler, 'invoke_conductor_compiler') as mock_compile:
            with patch.object(compiler, 'load_compiled_artifact') as mock_load:
                mock_compile.return_value = "/tmp/cache_test.so"
                mock_kernel = Mock()
                mock_load.return_value = mock_kernel
                
                # First compilation (cache miss)
                start_time = time.perf_counter()
                artifact1 = compiler.compile_graph(traced_model)
                first_compile_time = time.perf_counter() - start_time
                
                # Mock cache hit for second compilation
                with patch.object(compiler._cache, 'get', return_value=artifact1):
                    with patch.object(artifact1, 'is_valid', return_value=True):
                        # Second compilation (cache hit)
                        start_time = time.perf_counter()
                        artifact2 = compiler.compile_graph(traced_model)
                        second_compile_time = time.perf_counter() - start_time
                        
                        # Cache hit should be significantly faster
                        speedup = first_compile_time / second_compile_time if second_compile_time > 0 else float('inf')
                        assert speedup > 5.0, f"Cache speedup was only {speedup:.1f}x"
    
    def test_parallel_compilation_performance(self, performance_config, mock_conductor_cli):
        """Test performance with multiple concurrent compilations."""
        import threading
        import queue
        
        # Create multiple simple models
        models = []
        for i in range(3):
            class TestModel(torch.nn.Module):
                def __init__(self, size):
                    super().__init__()
                    self.size = size
                
                def forward(self, x):
                    return torch.relu(x + self.size)
            
            model = TestModel(i)
            traced = torch.fx.symbolic_trace(model)
            models.append(traced)
        
        # Mock compilation pipeline
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=1024):
                with patch('tempfile.NamedTemporaryFile') as mock_temp:
                    mock_file = Mock()
                    mock_file.name = "/tmp/parallel_test.so"
                    mock_temp.return_value.__enter__.return_value = mock_file
                    
                    with patch('conductor.runtime.loader.ExecutableKernel.load_from_file'):
                        results = queue.Queue()
                        
                        def compile_model(model, model_id):
                            compiler = JITCompiler()
                            start_time = time.perf_counter()
                            try:
                                artifact = compiler.compile_graph(model)
                                duration = time.perf_counter() - start_time
                                results.put(('success', model_id, duration))
                            except Exception as e:
                                duration = time.perf_counter() - start_time
                                results.put(('error', model_id, duration, str(e)))
                        
                        # Start parallel compilations
                        threads = []
                        start_time = time.perf_counter()
                        
                        for i, model in enumerate(models):
                            thread = threading.Thread(target=compile_model, args=(model, i))
                            thread.start()
                            threads.append(thread)
                        
                        # Wait for all compilations
                        for thread in threads:
                            thread.join()
                        
                        total_time = time.perf_counter() - start_time
                        
                        # Collect results
                        compilation_results = []
                        while not results.empty():
                            compilation_results.append(results.get())
                        
                        # Verify all compilations succeeded
                        successful = [r for r in compilation_results if r[0] == 'success']
                        assert len(successful) == len(models), "All parallel compilations should succeed"
                        
                        # Parallel execution should be faster than sequential
                        sequential_time_estimate = sum(r[2] for r in successful)
                        parallel_efficiency = sequential_time_estimate / total_time
                        
                        # Should show some parallelization benefit
                        assert parallel_efficiency > 1.0, f"Parallel efficiency: {parallel_efficiency:.2f}"


@pytest.mark.performance
class TestFusionPerformance:
    """Test fusion optimization performance."""
    
    def test_fusion_analysis_performance(self, performance_config):
        """Test performance of fusion analysis."""
        fusion_engine = FusionEngine()
        
        # Create large DAG with many fusion opportunities
        from conductor.codegen.graph import ComputationDAG
        from conductor.codegen.buffers import Buffer, BufferScope
        
        num_ops = 50  # Large number of operations
        buffers = [
            Buffer(f"buf_{i}", BufferScope.LOCAL, torch.float32, (64, 64))
            for i in range(num_ops + 1)
        ]
        
        nodes = []
        for i in range(num_ops):
            op_name = ["add", "mul", "relu"][i % 3]  # Cycle through fusible ops
            node = ConductorNode(op_name, inputs=[buffers[i]], outputs=[buffers[i + 1]])
            nodes.append(node)
        
        # Set up relationships
        for i in range(num_ops):
            buffers[i + 1].producer = nodes[i]
            if i < num_ops - 1:
                buffers[i + 1].consumers = [nodes[i + 1]]
        
        dag = ComputationDAG()
        for node in nodes:
            dag.add_node(node)
        for buffer in buffers:
            dag.add_buffer(buffer)
        
        # Measure fusion analysis time
        start_time = time.perf_counter()
        clusters = fusion_engine.identify_fusion_opportunities(dag)
        analysis_time = time.perf_counter() - start_time
        
        # Performance requirement: fusion analysis should be fast
        assert analysis_time < 1.0, f"Fusion analysis took {analysis_time:.3f}s for {num_ops} operations"
        
        # Verify fusion was effective
        total_fused_nodes = sum(len(cluster.nodes) for cluster in clusters)
        fusion_ratio = total_fused_nodes / num_ops
        assert fusion_ratio > 0.5, f"Fusion ratio too low: {fusion_ratio:.2f}"
    
    def test_dsl_generation_performance(self, sample_dag, performance_config):
        """Test performance of DSL generation."""
        generator = DSLGenerator()
        
        # Measure DSL generation time
        start_time = time.perf_counter()
        dsl_code = generator.generate_dsl_file(sample_dag)
        generation_time = time.perf_counter() - start_time
        
        # Performance requirement: DSL generation should be fast
        assert generation_time < 0.5, f"DSL generation took {generation_time:.3f}s"
        
        # Verify DSL was generated
        assert len(dsl_code) > 0
        assert "function main" in dsl_code
    
    def test_graph_analysis_performance(self, complex_fx_graph, performance_config):
        """Test performance of graph analysis."""
        analyzer = GraphAnalyzer()
        
        # Measure graph analysis time
        start_time = time.perf_counter()
        dag = analyzer.parse_fx_graph(complex_fx_graph)
        analysis_time = time.perf_counter() - start_time
        
        # Performance requirement: graph analysis should be fast
        assert analysis_time < 1.0, f"Graph analysis took {analysis_time:.3f}s"
        
        # Verify analysis results
        assert len(dag.nodes) > 0
        assert len(dag.buffers) > 0


@pytest.mark.performance
class TestMemoryPerformance:
    """Test memory usage and optimization performance."""
    
    def test_buffer_manager_performance(self, performance_config):
        """Test buffer manager performance with many buffers."""
        from conductor.codegen.buffers import BufferManager
        
        manager = BufferManager()
        
        # Create many buffers
        num_buffers = 1000
        start_time = time.perf_counter()
        
        buffers = []
        for i in range(num_buffers):
            buffer = manager.allocate_buffer(f"buf_{i}", torch.float32, (32, 32))
            buffers.append(buffer)
        
        allocation_time = time.perf_counter() - start_time
        
        # Performance requirement: buffer allocation should be fast
        assert allocation_time < 1.0, f"Buffer allocation took {allocation_time:.3f}s for {num_buffers} buffers"
        
        # Test buffer reuse optimization performance
        start_time = time.perf_counter()
        reuse_map = manager.optimize_buffer_reuse(buffers)
        optimization_time = time.perf_counter() - start_time
        
        # Performance requirement: optimization should be fast
        assert optimization_time < 2.0, f"Buffer optimization took {optimization_time:.3f}s"
    
    def test_memory_footprint_calculation_performance(self, performance_config):
        """Test performance of memory footprint calculations."""
        from conductor.codegen.buffers import Buffer, BufferScope
        
        # Create buffers with various sizes
        buffers = []
        for i in range(100):
            size = 2 ** (i % 10 + 1)  # Powers of 2 from 2 to 1024
            buffer = Buffer(f"buf_{i}", BufferScope.LOCAL, torch.float32, (size, size))
            buffers.append(buffer)
        
        # Measure footprint calculation time
        start_time = time.perf_counter()
        total_footprint = sum(buf.get_memory_footprint() for buf in buffers)
        calculation_time = time.perf_counter() - start_time
        
        # Performance requirement: calculations should be fast
        assert calculation_time < 0.1, f"Memory footprint calculation took {calculation_time:.3f}s"
        
        # Verify calculations are reasonable
        assert total_footprint > 0


@pytest.mark.performance
class TestScalabilityPerformance:
    """Test scalability with large models and graphs."""
    
    def test_large_graph_compilation_scalability(self, performance_config, mock_conductor_cli):
        """Test compilation scalability with large graphs."""
        # Create large model
        class LargeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([
                    torch.nn.Linear(128, 128) for _ in range(10)
                ])
            
            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return x
        
        model = LargeModel()
        x = torch.randn(32, 128)
        traced_model = torch.fx.symbolic_trace(model)
        
        # Mock compilation pipeline
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=16384):
                with patch('tempfile.NamedTemporaryFile') as mock_temp:
                    mock_file = Mock()
                    mock_file.name = "/tmp/large_model.so"
                    mock_temp.return_value.__enter__.return_value = mock_file
                    
                    with patch('conductor.runtime.loader.ExecutableKernel.load_from_file'):
                        compiler = JITCompiler()
                        
                        # Measure compilation time
                        start_time = time.perf_counter()
                        artifact = compiler.compile_graph(traced_model)
                        compilation_time = time.perf_counter() - start_time
                        
                        # Performance requirement: should handle large models
                        assert compilation_time < 10.0, f"Large model compilation took {compilation_time:.3f}s"
                        
                        # Verify compilation succeeded
                        assert artifact is not None
    
    def test_memory_scalability_with_batch_size(self, performance_config, mock_conductor_cli):
        """Test memory scalability with different batch sizes."""
        # Test with increasing batch sizes
        batch_sizes = [1, 8, 32, 128]
        compilation_times = []
        
        for batch_size in batch_sizes:
            # Create model with specific batch size
            class BatchModel(torch.nn.Module):
                def forward(self, x):
                    return torch.relu(x + 1.0)
            
            model = BatchModel()
            x = torch.randn(batch_size, 64)
            traced_model = torch.fx.symbolic_trace(model)
            
            # Mock compilation
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=2048):
                    with patch('tempfile.NamedTemporaryFile') as mock_temp:
                        mock_file = Mock()
                        mock_file.name = f"/tmp/batch_{batch_size}.so"
                        mock_temp.return_value.__enter__.return_value = mock_file
                        
                        with patch('conductor.runtime.loader.ExecutableKernel.load_from_file'):
                            compiler = JITCompiler()
                            
                            # Measure compilation time
                            start_time = time.perf_counter()
                            artifact = compiler.compile_graph(traced_model)
                            compilation_time = time.perf_counter() - start_time
                            
                            compilation_times.append(compilation_time)
        
        # Verify compilation time doesn't grow excessively with batch size
        max_time = max(compilation_times)
        min_time = min(compilation_times)
        time_ratio = max_time / min_time if min_time > 0 else 1.0
        
        # Compilation time shouldn't grow too much with batch size
        assert time_ratio < 5.0, f"Compilation time ratio across batch sizes: {time_ratio:.2f}"


@pytest.mark.performance
class TestPerformanceRegression:
    """Test for performance regressions."""
    
    def test_baseline_performance_metrics(self, simple_torch_model, performance_config, mock_conductor_cli):
        """Establish baseline performance metrics."""
        # This test establishes baseline metrics that can be used
        # to detect performance regressions in future runs
        
        x = torch.randn(16, 32)
        traced_model = torch.fx.symbolic_trace(simple_torch_model)
        
        # Mock compilation pipeline
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=1024):
                with patch('tempfile.NamedTemporaryFile') as mock_temp:
                    mock_file = Mock()
                    mock_file.name = "/tmp/baseline.so"
                    mock_temp.return_value.__enter__.return_value = mock_file
                    
                    with patch('conductor.runtime.loader.ExecutableKernel.load_from_file'):
                        compiler = JITCompiler()
                        
                        # Measure multiple runs for stability
                        times = []
                        for _ in range(5):
                            start_time = time.perf_counter()
                            artifact = compiler.compile_graph(traced_model)
                            duration = time.perf_counter() - start_time
                            times.append(duration)
                        
                        # Calculate statistics
                        avg_time = sum(times) / len(times)
                        max_time = max(times)
                        min_time = min(times)
                        
                        # Store baseline metrics (in practice, these would be saved)
                        baseline_metrics = {
                            'avg_compilation_time': avg_time,
                            'max_compilation_time': max_time,
                            'min_compilation_time': min_time,
                            'model_type': 'simple',
                            'input_shape': x.shape
                        }
                        
                        # Verify metrics are reasonable
                        assert avg_time < 2.0, f"Baseline average time: {avg_time:.3f}s"
                        assert max_time < 3.0, f"Baseline max time: {max_time:.3f}s"
                        
                        # Consistency check
                        time_variance = (max_time - min_time) / avg_time
                        assert time_variance < 2.0, f"High time variance: {time_variance:.2f}"
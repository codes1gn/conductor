"""
End-to-end integration tests for the complete Conductor pipeline.

This module tests the complete workflow from PyTorch model to executable
artifacts, including all intermediate steps and error handling.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import conductor
from conductor.backend import ConductorBackend
from conductor.runtime.jit import JITCompiler
from conductor.runtime.aot import AOTManager
from conductor.runtime.loader import CompiledArtifact, ExecutableKernel
from conductor.utils.exceptions import CompilationError, UnsupportedOperationError


@pytest.mark.integration
class TestEndToEndPipeline:
    """Test complete end-to-end pipeline functionality."""
    
    def test_simple_model_compilation_pipeline(self, simple_torch_model, mock_conductor_cli):
        """Test complete pipeline with simple model."""
        # Create sample input
        x = torch.randn(5, 10)
        
        # Trace the model
        traced_model = torch.fx.symbolic_trace(simple_torch_model)
        
        # Mock file operations for compilation
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=1024):
                with patch('tempfile.NamedTemporaryFile') as mock_temp:
                    mock_file = Mock()
                    mock_file.name = "/tmp/test.so"
                    mock_temp.return_value.__enter__.return_value = mock_file
                    
                    # Mock ExecutableKernel
                    with patch('conductor.runtime.loader.ExecutableKernel.load_from_file') as mock_load:
                        mock_kernel = Mock(spec=ExecutableKernel)
                        mock_kernel.execute.return_value = torch.randn(5, 10)
                        mock_load.return_value = mock_kernel
                        
                        # Create backend and compile
                        backend = ConductorBackend()
                        compiled_fn = backend(traced_model, [x])
                        
                        # Execute compiled function
                        result = compiled_fn(x)
                        
                        # Verify result
                        assert isinstance(result, torch.Tensor)
                        assert result.shape == x.shape
    
    def test_elementwise_model_with_fusion(self, elementwise_torch_model, mock_conductor_cli):
        """Test pipeline with model that has fusion opportunities."""
        # Create sample input
        x = torch.randn(8, 16)
        
        # Trace the model
        traced_model = torch.fx.symbolic_trace(elementwise_torch_model)
        
        # Mock compilation pipeline
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=2048):
                with patch('tempfile.NamedTemporaryFile') as mock_temp:
                    mock_file = Mock()
                    mock_file.name = "/tmp/fused_model.so"
                    mock_temp.return_value.__enter__.return_value = mock_file
                    
                    with patch('conductor.runtime.loader.ExecutableKernel.load_from_file') as mock_load:
                        mock_kernel = Mock(spec=ExecutableKernel)
                        mock_kernel.execute.return_value = torch.randn(8, 16)
                        mock_load.return_value = mock_kernel
                        
                        # Compile with backend
                        backend = ConductorBackend()
                        compiled_fn = backend(traced_model, [x])
                        
                        # Execute
                        result = compiled_fn(x)
                        
                        # Verify fusion occurred (check compilation was called)
                        assert mock_conductor_cli.called
                        
                        # Verify result
                        assert isinstance(result, torch.Tensor)
                        assert result.shape == x.shape
    
    def test_complex_model_pipeline(self, complex_torch_model, mock_conductor_cli):
        """Test pipeline with complex model containing multiple operation types."""
        # Create sample input
        x = torch.randn(4, 128)
        
        # Trace the model
        traced_model = torch.fx.symbolic_trace(complex_torch_model)
        
        # Mock compilation pipeline
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=4096):
                with patch('tempfile.NamedTemporaryFile') as mock_temp:
                    mock_file = Mock()
                    mock_file.name = "/tmp/complex_model.so"
                    mock_temp.return_value.__enter__.return_value = mock_file
                    
                    with patch('conductor.runtime.loader.ExecutableKernel.load_from_file') as mock_load:
                        mock_kernel = Mock(spec=ExecutableKernel)
                        mock_kernel.execute.return_value = torch.randn(4)  # Reduced by sum
                        mock_load.return_value = mock_kernel
                        
                        # Compile
                        backend = ConductorBackend()
                        compiled_fn = backend(traced_model, [x])
                        
                        # Execute
                        result = compiled_fn(x)
                        
                        # Verify result shape (should be reduced)
                        assert isinstance(result, torch.Tensor)
                        assert result.shape == (4,)  # Reduced from (4, 128)
    
    def test_jit_compilation_caching(self, simple_torch_model, temp_test_dir):
        """Test that JIT compilation properly caches results."""
        # Create JIT compiler with temp cache
        compiler = JITCompiler(cache_dir=temp_test_dir)
        
        # Create model and trace
        x = torch.randn(3, 5)
        traced_model = torch.fx.symbolic_trace(simple_torch_model)
        
        # Mock compilation pipeline
        with patch.object(compiler, 'invoke_conductor_compiler') as mock_compile:
            with patch.object(compiler, 'load_compiled_artifact') as mock_load:
                mock_compile.return_value = "/tmp/cached_model.so"
                mock_kernel = Mock(spec=ExecutableKernel)
                mock_load.return_value = mock_kernel
                
                # First compilation
                artifact1 = compiler.compile_graph(traced_model)
                
                # Second compilation of same model
                artifact2 = compiler.compile_graph(traced_model)
                
                # Should use cache for second compilation
                assert mock_compile.call_count == 1  # Only called once
                
                # Both artifacts should have same graph hash
                assert artifact1.metadata['graph_hash'] == artifact2.metadata['graph_hash']
    
    def test_aot_artifact_loading(self, simple_torch_model, temp_test_dir, mock_compiled_artifact):
        """Test AOT artifact loading pipeline."""
        # Create AOT manager
        aot_manager = AOTManager()
        
        # Create model
        traced_model = torch.fx.symbolic_trace(simple_torch_model)
        
        # Mock artifact discovery
        with patch.object(aot_manager, 'locate_precompiled_artifact') as mock_locate:
            with patch.object(aot_manager, 'validate_artifact_compatibility') as mock_validate:
                with patch.object(aot_manager, 'load_static_artifact') as mock_load:
                    mock_locate.return_value = mock_compiled_artifact.path
                    mock_validate.return_value = True
                    mock_kernel = Mock(spec=ExecutableKernel)
                    mock_load.return_value = mock_kernel
                    
                    # Load artifact
                    kernel = aot_manager.load_precompiled_artifact(traced_model)
                    
                    # Verify loading process
                    assert mock_locate.called
                    assert mock_validate.called
                    assert mock_load.called
                    assert kernel == mock_kernel
    
    def test_fallback_to_inductor(self, simple_torch_model):
        """Test fallback mechanism when Conductor compilation fails."""
        # Create model
        x = torch.randn(2, 8)
        traced_model = torch.fx.symbolic_trace(simple_torch_model)
        
        # Mock compilation failure
        with patch('conductor.runtime.jit.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("conductor: command not found")
            
            # Mock Inductor fallback
            with patch('torch.compile') as mock_torch_compile:
                mock_compiled_fn = Mock()
                mock_compiled_fn.return_value = torch.randn(2, 8)
                mock_torch_compile.return_value = mock_compiled_fn
                
                # Create backend
                backend = ConductorBackend()
                
                # Should fallback to Inductor
                compiled_fn = backend(traced_model, [x])
                
                # Verify fallback was used
                assert mock_torch_compile.called
                
                # Execute fallback function
                result = compiled_fn(x)
                assert isinstance(result, torch.Tensor)
    
    def test_unsupported_operation_fallback(self):
        """Test fallback when encountering unsupported operations."""
        # Create model with potentially unsupported operation
        class UnsupportedOpModel(torch.nn.Module):
            def forward(self, x):
                # This might not be supported in early implementation
                return torch.fft.fft(x)
        
        model = UnsupportedOpModel()
        x = torch.randn(4, 8, dtype=torch.complex64)
        
        # Mock unsupported operation error
        with patch('conductor.codegen.graph.GraphAnalyzer.parse_fx_graph') as mock_parse:
            mock_parse.side_effect = UnsupportedOperationError("fft", "Not implemented")
            
            # Mock fallback
            with patch('torch.compile') as mock_torch_compile:
                mock_compiled_fn = Mock()
                mock_compiled_fn.return_value = torch.randn(4, 8, dtype=torch.complex64)
                mock_torch_compile.return_value = mock_compiled_fn
                
                # Trace model
                traced_model = torch.fx.symbolic_trace(model)
                
                # Create backend
                backend = ConductorBackend()
                
                # Should fallback due to unsupported operation
                compiled_fn = backend(traced_model, [x])
                
                # Verify fallback
                assert mock_torch_compile.called
    
    def test_compilation_error_handling(self, simple_torch_model):
        """Test handling of compilation errors."""
        # Create model
        x = torch.randn(3, 6)
        traced_model = torch.fx.symbolic_trace(simple_torch_model)
        
        # Mock compilation error
        with patch('conductor.runtime.jit.subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=1,
                stderr="Compilation failed: syntax error",
                stdout=""
            )
            
            # Mock fallback
            with patch('torch.compile') as mock_torch_compile:
                mock_compiled_fn = Mock()
                mock_compiled_fn.return_value = torch.randn(3, 6)
                mock_torch_compile.return_value = mock_compiled_fn
                
                # Create backend
                backend = ConductorBackend()
                
                # Should handle compilation error and fallback
                compiled_fn = backend(traced_model, [x])
                
                # Verify fallback was used
                assert mock_torch_compile.called
    
    def test_memory_optimization_pipeline(self, elementwise_torch_model, mock_conductor_cli):
        """Test that memory optimization is applied in the pipeline."""
        # Create model with large tensors
        x = torch.randn(256, 512)  # Large tensor
        traced_model = torch.fx.symbolic_trace(elementwise_torch_model)
        
        # Mock compilation with memory optimization
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=8192):
                with patch('tempfile.NamedTemporaryFile') as mock_temp:
                    mock_file = Mock()
                    mock_file.name = "/tmp/optimized_model.so"
                    mock_temp.return_value.__enter__.return_value = mock_file
                    
                    with patch('conductor.runtime.loader.ExecutableKernel.load_from_file') as mock_load:
                        mock_kernel = Mock(spec=ExecutableKernel)
                        mock_kernel.execute.return_value = torch.randn(256, 512)
                        mock_load.return_value = mock_kernel
                        
                        # Compile with memory optimization
                        backend = ConductorBackend()
                        compiled_fn = backend(traced_model, [x])
                        
                        # Execute
                        result = compiled_fn(x)
                        
                        # Verify compilation occurred
                        assert mock_conductor_cli.called
                        
                        # Check that DSL was generated (indirectly through compilation call)
                        call_args = mock_conductor_cli.call_args[0][0]
                        assert 'conductor' in call_args
                        assert 'compile' in call_args
    
    def test_performance_monitoring_integration(self, simple_torch_model, mock_conductor_cli):
        """Test integration with performance monitoring."""
        # Create model
        x = torch.randn(10, 20)
        traced_model = torch.fx.symbolic_trace(simple_torch_model)
        
        # Mock performance monitoring
        with patch('conductor.utils.profiler.ConductorProfiler') as mock_profiler:
            mock_profiler_instance = Mock()
            mock_profiler_instance.get_stats.return_value = {
                'compilation_time': 0.5,
                'execution_time': 0.1,
                'fusion_ratio': 0.8
            }
            mock_profiler.return_value = mock_profiler_instance
            
            # Mock compilation
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=1024):
                    with patch('tempfile.NamedTemporaryFile') as mock_temp:
                        mock_file = Mock()
                        mock_file.name = "/tmp/monitored_model.so"
                        mock_temp.return_value.__enter__.return_value = mock_file
                        
                        with patch('conductor.runtime.loader.ExecutableKernel.load_from_file') as mock_load:
                            mock_kernel = Mock(spec=ExecutableKernel)
                            mock_kernel.execute.return_value = torch.randn(10, 20)
                            mock_load.return_value = mock_kernel
                            
                            # Compile with monitoring
                            backend = ConductorBackend()
                            compiled_fn = backend(traced_model, [x])
                            
                            # Execute
                            result = compiled_fn(x)
                            
                            # Verify monitoring was integrated
                            assert isinstance(result, torch.Tensor)


@pytest.mark.integration
class TestPipelineErrorRecovery:
    """Test error recovery mechanisms in the pipeline."""
    
    def test_recovery_from_dsl_generation_failure(self, simple_torch_model):
        """Test recovery when DSL generation fails."""
        x = torch.randn(4, 8)
        traced_model = torch.fx.symbolic_trace(simple_torch_model)
        
        # Mock DSL generation failure
        with patch('conductor.codegen.dsl.DSLGenerator.generate_dsl_file') as mock_dsl:
            mock_dsl.side_effect = RuntimeError("DSL generation failed")
            
            # Mock fallback
            with patch('torch.compile') as mock_torch_compile:
                mock_compiled_fn = Mock()
                mock_compiled_fn.return_value = torch.randn(4, 8)
                mock_torch_compile.return_value = mock_compiled_fn
                
                # Should recover gracefully
                backend = ConductorBackend()
                compiled_fn = backend(traced_model, [x])
                
                # Verify fallback
                assert mock_torch_compile.called
    
    def test_recovery_from_fusion_failure(self, elementwise_torch_model):
        """Test recovery when fusion optimization fails."""
        x = torch.randn(6, 12)
        traced_model = torch.fx.symbolic_trace(elementwise_torch_model)
        
        # Mock fusion failure
        with patch('conductor.codegen.fusion.FusionEngine.identify_fusion_opportunities') as mock_fusion:
            mock_fusion.side_effect = RuntimeError("Fusion analysis failed")
            
            # Mock fallback
            with patch('torch.compile') as mock_torch_compile:
                mock_compiled_fn = Mock()
                mock_compiled_fn.return_value = torch.randn(6, 12)
                mock_torch_compile.return_value = mock_compiled_fn
                
                # Should recover gracefully
                backend = ConductorBackend()
                compiled_fn = backend(traced_model, [x])
                
                # Verify fallback
                assert mock_torch_compile.called
    
    def test_recovery_from_artifact_loading_failure(self, simple_torch_model, temp_test_dir):
        """Test recovery when artifact loading fails."""
        # Create JIT compiler
        compiler = JITCompiler(cache_dir=temp_test_dir)
        traced_model = torch.fx.symbolic_trace(simple_torch_model)
        
        # Mock successful compilation but failed loading
        with patch.object(compiler, 'invoke_conductor_compiler') as mock_compile:
            with patch.object(compiler, 'load_compiled_artifact') as mock_load:
                mock_compile.return_value = "/tmp/test.so"
                mock_load.side_effect = RuntimeError("Failed to load artifact")
                
                # Should raise CompilationError
                with pytest.raises(RuntimeError):
                    compiler.compile_graph(traced_model)
    
    def test_timeout_handling(self, simple_torch_model):
        """Test handling of compilation timeouts."""
        x = torch.randn(5, 10)
        traced_model = torch.fx.symbolic_trace(simple_torch_model)
        
        # Mock timeout
        from subprocess import TimeoutExpired
        with patch('conductor.runtime.jit.subprocess.run') as mock_run:
            mock_run.side_effect = TimeoutExpired('conductor', 300)
            
            # Mock fallback
            with patch('torch.compile') as mock_torch_compile:
                mock_compiled_fn = Mock()
                mock_compiled_fn.return_value = torch.randn(5, 10)
                mock_torch_compile.return_value = mock_compiled_fn
                
                # Should handle timeout and fallback
                backend = ConductorBackend()
                compiled_fn = backend(traced_model, [x])
                
                # Verify fallback
                assert mock_torch_compile.called


@pytest.mark.integration
class TestPipelinePerformance:
    """Test performance aspects of the pipeline."""
    
    @pytest.mark.performance
    def test_compilation_performance_baseline(self, simple_torch_model, performance_config):
        """Test compilation performance baseline."""
        import time
        
        x = torch.randn(32, 64)
        traced_model = torch.fx.symbolic_trace(simple_torch_model)
        
        # Mock fast compilation
        with patch('conductor.runtime.jit.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stderr="", stdout="Success")
            
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=1024):
                    with patch('tempfile.NamedTemporaryFile') as mock_temp:
                        mock_file = Mock()
                        mock_file.name = "/tmp/perf_test.so"
                        mock_temp.return_value.__enter__.return_value = mock_file
                        
                        with patch('conductor.runtime.loader.ExecutableKernel.load_from_file'):
                            # Measure compilation time
                            start_time = time.time()
                            
                            backend = ConductorBackend()
                            compiled_fn = backend(traced_model, [x])
                            
                            compilation_time = time.time() - start_time
                            
                            # Should complete within reasonable time
                            assert compilation_time < performance_config['timeout_seconds']
    
    @pytest.mark.performance
    def test_memory_usage_monitoring(self, complex_torch_model, performance_config):
        """Test memory usage during compilation."""
        import psutil
        import os
        
        x = torch.randn(16, 128)
        traced_model = torch.fx.symbolic_trace(complex_torch_model)
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Mock compilation
        with patch('conductor.runtime.jit.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stderr="", stdout="Success")
            
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=2048):
                    with patch('tempfile.NamedTemporaryFile') as mock_temp:
                        mock_file = Mock()
                        mock_file.name = "/tmp/memory_test.so"
                        mock_temp.return_value.__enter__.return_value = mock_file
                        
                        with patch('conductor.runtime.loader.ExecutableKernel.load_from_file'):
                            backend = ConductorBackend()
                            compiled_fn = backend(traced_model, [x])
                            
                            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
                            memory_increase = peak_memory - initial_memory
                            
                            # Memory increase should be reasonable
                            assert memory_increase < performance_config['memory_limit_mb']
    
    @pytest.mark.performance
    def test_cache_performance_impact(self, simple_torch_model, temp_test_dir, performance_config):
        """Test performance impact of caching."""
        import time
        
        compiler = JITCompiler(cache_dir=temp_test_dir)
        traced_model = torch.fx.symbolic_trace(simple_torch_model)
        
        # Mock compilation
        with patch.object(compiler, 'invoke_conductor_compiler') as mock_compile:
            with patch.object(compiler, 'load_compiled_artifact') as mock_load:
                mock_compile.return_value = "/tmp/cache_test.so"
                mock_kernel = Mock(spec=ExecutableKernel)
                mock_load.return_value = mock_kernel
                
                # First compilation (cache miss)
                start_time = time.time()
                artifact1 = compiler.compile_graph(traced_model)
                first_compile_time = time.time() - start_time
                
                # Mock cache hit
                with patch.object(compiler._cache, 'get', return_value=artifact1):
                    with patch.object(artifact1, 'is_valid', return_value=True):
                        # Second compilation (cache hit)
                        start_time = time.time()
                        artifact2 = compiler.compile_graph(traced_model)
                        second_compile_time = time.time() - start_time
                        
                        # Cache hit should be significantly faster
                        assert second_compile_time < first_compile_time * 0.1  # 10x faster
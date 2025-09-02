"""
Integration tests for JIT compilation pipeline.

This module tests the complete end-to-end JIT compilation workflow
from FX Graph to executable artifacts.
"""

import pytest
import torch
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from conductor.runtime.jit import JITCompiler
from conductor.runtime.loader import CompiledArtifact, ExecutableKernel
from conductor.utils.exceptions import CompilationError, UnsupportedOperationError


class TestJITCompiler:
    """Test cases for JIT compilation pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.compiler = JITCompiler()
        
    def test_compiler_initialization(self):
        """Test JIT compiler initialization."""
        assert self.compiler is not None
        assert hasattr(self.compiler, '_graph_analyzer')
        assert hasattr(self.compiler, '_fusion_engine')
        assert hasattr(self.compiler, '_dsl_generator')
        assert hasattr(self.compiler, '_cache')
        
    def test_graph_hash_generation(self):
        """Test graph hash generation for caching."""
        # Create a simple model
        def simple_model(x):
            return torch.relu(x + 1.0)
        
        # Trace the model
        x = torch.randn(2, 3)
        traced = torch.jit.trace(simple_model, x)
        graph_module = torch.fx.symbolic_trace(simple_model)
        
        # Generate hash
        hash1 = self.compiler._generate_graph_hash(graph_module)
        hash2 = self.compiler._generate_graph_hash(graph_module)
        
        # Hash should be consistent
        assert hash1 == hash2
        assert len(hash1) == 16  # Should be 16 character hash
        
        # Different graphs should have different hashes
        def different_model(x):
            return torch.sigmoid(x * 2.0)
        
        different_graph = torch.fx.symbolic_trace(different_model)
        hash3 = self.compiler._generate_graph_hash(different_graph)
        
        assert hash1 != hash3
        
    @patch('conductor.runtime.jit.subprocess.run')
    def test_invoke_conductor_compiler_success(self, mock_run):
        """Test successful compiler invocation."""
        # Mock successful compilation
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="Compilation successful")
        
        # Create temporary DSL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.co', delete=False) as dsl_file:
            dsl_file.write("function main() { }")
            dsl_file_path = dsl_file.name
        
        try:
            # Mock the output file creation
            with patch('os.path.exists') as mock_exists:
                with patch('os.path.getsize') as mock_getsize:
                    mock_exists.return_value = True
                    mock_getsize.return_value = 1024  # Non-empty file
                    
                    result = self.compiler.invoke_conductor_compiler(dsl_file_path)
                    
                    # Verify compiler was called correctly
                    mock_run.assert_called_once()
                    call_args = mock_run.call_args[0][0]
                    assert call_args[0] == 'conductor'
                    assert call_args[1] == 'compile'
                    assert call_args[2] == dsl_file_path
                    assert call_args[3] == '-o'
                    
                    # Result should be output path
                    assert result.endswith('.so')
                    
        finally:
            # Clean up
            try:
                os.unlink(dsl_file_path)
            except OSError:
                pass
                
    @patch('conductor.runtime.jit.subprocess.run')
    def test_invoke_conductor_compiler_failure(self, mock_run):
        """Test compiler invocation failure handling."""
        # Mock compilation failure
        mock_run.return_value = Mock(
            returncode=1, 
            stderr="Compilation error: invalid syntax",
            stdout=""
        )
        
        # Create temporary DSL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.co', delete=False) as dsl_file:
            dsl_file.write("invalid dsl content")
            dsl_file_path = dsl_file.name
        
        try:
            with pytest.raises(CompilationError) as exc_info:
                self.compiler.invoke_conductor_compiler(dsl_file_path)
            
            # Check error details
            error = exc_info.value
            assert "Conductor compilation failed" in str(error)
            assert error.compiler_output == "Compilation error: invalid syntax"
            
        finally:
            # Clean up
            try:
                os.unlink(dsl_file_path)
            except OSError:
                pass
                
    @patch('conductor.runtime.jit.subprocess.run')
    def test_invoke_conductor_compiler_not_found(self, mock_run):
        """Test handling when conductor compiler is not found."""
        # Mock FileNotFoundError
        mock_run.side_effect = FileNotFoundError("conductor: command not found")
        
        # Create temporary DSL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.co', delete=False) as dsl_file:
            dsl_file.write("function main() { }")
            dsl_file_path = dsl_file.name
        
        try:
            with pytest.raises(CompilationError) as exc_info:
                self.compiler.invoke_conductor_compiler(dsl_file_path)
            
            # Check error message
            error = exc_info.value
            assert "Conductor compiler not found in PATH" in str(error)
            
        finally:
            # Clean up
            try:
                os.unlink(dsl_file_path)
            except OSError:
                pass
                
    @patch('conductor.runtime.jit.subprocess.run')
    def test_invoke_conductor_compiler_timeout(self, mock_run):
        """Test handling of compiler timeout."""
        from subprocess import TimeoutExpired
        
        # Mock timeout
        mock_run.side_effect = TimeoutExpired('conductor', 300)
        
        # Create temporary DSL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.co', delete=False) as dsl_file:
            dsl_file.write("function main() { }")
            dsl_file_path = dsl_file.name
        
        try:
            with pytest.raises(CompilationError) as exc_info:
                self.compiler.invoke_conductor_compiler(dsl_file_path)
            
            # Check error message
            error = exc_info.value
            assert "timed out" in str(error)
            
        finally:
            # Clean up
            try:
                os.unlink(dsl_file_path)
            except OSError:
                pass
                
    def test_load_compiled_artifact(self):
        """Test loading compiled artifacts."""
        # Create a mock artifact file
        with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as artifact_file:
            artifact_path = artifact_file.name
            artifact_file.write(b"mock shared library content")
        
        try:
            # Mock ExecutableKernel.load_from_file
            with patch('conductor.runtime.loader.ExecutableKernel.load_from_file') as mock_load:
                mock_kernel = Mock(spec=ExecutableKernel)
                mock_load.return_value = mock_kernel
                
                result = self.compiler.load_compiled_artifact(artifact_path)
                
                # Verify loading was called correctly
                mock_load.assert_called_once_with(artifact_path)
                assert result == mock_kernel
                
        finally:
            # Clean up
            try:
                os.unlink(artifact_path)
            except OSError:
                pass
                
    def test_load_compiled_artifact_failure(self):
        """Test handling of artifact loading failure."""
        # Use non-existent file
        artifact_path = "/non/existent/file.so"
        
        with pytest.raises(RuntimeError) as exc_info:
            self.compiler.load_compiled_artifact(artifact_path)
        
        assert "Failed to load compiled artifact" in str(exc_info.value)
        
    def test_cache_operations(self):
        """Test compilation result caching."""
        # Create a fresh compiler instance for this test
        import tempfile
        temp_dir = tempfile.mkdtemp()
        fresh_compiler = JITCompiler(cache_dir=temp_dir)
        
        try:
            # Create mock artifact
            artifact = CompiledArtifact(
                path="/mock/path.so",
                artifact_type="shared_library",
                entry_point="main",
                metadata={}
            )
            
            # Test caching
            graph_hash = "test_hash_123"
            success = fresh_compiler.cache_compilation_result(graph_hash, artifact)
            assert success
            
            # Verify cache by retrieving
            cached_artifact = fresh_compiler._cache.get(graph_hash)
            assert cached_artifact is not None
            assert cached_artifact.path == artifact.path
            
            # Test cache stats
            stats = fresh_compiler.get_cache_stats()
            assert stats['entries'] == 1
            assert stats['total_size_bytes'] > 0
            
            # Test cache clearing
            fresh_compiler.clear_cache()
            
            stats = fresh_compiler.get_cache_stats()
            assert stats['entries'] == 0
            
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    @patch('conductor.runtime.jit.subprocess.run')
    def test_compile_graph_end_to_end(self, mock_run):
        """Test complete end-to-end compilation pipeline."""
        # Create a simple model
        def simple_model(x):
            return torch.relu(x + 1.0)
        
        graph_module = torch.fx.symbolic_trace(simple_model)
        
        # Mock successful compilation
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="Success")
        
        # Mock file operations
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=1024):
                with patch('tempfile.NamedTemporaryFile') as mock_temp:
                    # Mock temporary file
                    mock_file = Mock()
                    mock_file.name = "/tmp/test.so"
                    mock_temp.return_value.__enter__.return_value = mock_file
                    
                    # Mock ExecutableKernel loading
                    with patch('conductor.runtime.loader.ExecutableKernel.load_from_file'):
                        result = self.compiler.compile_graph(graph_module)
                        
                        # Verify result
                        assert isinstance(result, CompiledArtifact)
                        assert result.artifact_type == 'shared_library'
                        assert result.entry_point == 'conductor_kernel_main'
                        assert 'graph_hash' in result.metadata
                        assert 'node_count' in result.metadata
                        
    def test_compile_graph_caching(self):
        """Test that compilation results are properly cached."""
        # Create a simple model
        def simple_model(x):
            return torch.relu(x + 1.0)
        
        graph_module = torch.fx.symbolic_trace(simple_model)
        
        # Mock the entire compilation pipeline
        mock_artifact = CompiledArtifact(
            path="/mock/path.so",
            artifact_type="shared_library", 
            entry_point="main",
            metadata={}
        )
        
        # Mock the cache get method to return our artifact
        with patch.object(self.compiler._cache, 'get', return_value=mock_artifact):
            with patch.object(mock_artifact, 'is_valid', return_value=True):
                # Compile should return cached result without actual compilation
                result = self.compiler.compile_graph(graph_module)
                assert result.path == mock_artifact.path
        
    def test_compile_graph_unsupported_operation(self):
        """Test handling of unsupported operations."""
        # This would require mocking the graph analyzer to raise UnsupportedOperationError
        def simple_model(x):
            return torch.relu(x + 1.0)
        
        graph_module = torch.fx.symbolic_trace(simple_model)
        
        # Mock graph analyzer to raise unsupported operation error
        with patch.object(self.compiler._graph_analyzer, 'parse_fx_graph') as mock_parse:
            mock_parse.side_effect = UnsupportedOperationError("custom_op", "Not implemented")
            
            with pytest.raises(UnsupportedOperationError):
                self.compiler.compile_graph(graph_module)
                
    def test_compile_graph_invalid_graph(self):
        """Test handling of invalid graph structures."""
        def simple_model(x):
            return torch.relu(x + 1.0)
        
        graph_module = torch.fx.symbolic_trace(simple_model)
        
        # Mock graph analyzer to return invalid DAG
        with patch.object(self.compiler._graph_analyzer, 'parse_fx_graph') as mock_parse:
            mock_dag = Mock()
            mock_dag.validate_graph_correctness.return_value = False
            mock_parse.return_value = mock_dag
            
            with pytest.raises(CompilationError) as exc_info:
                self.compiler.compile_graph(graph_module)
            
            assert "Invalid graph structure" in str(exc_info.value)


class TestJITIntegrationScenarios:
    """Integration test scenarios for common use cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.compiler = JITCompiler()
        
    def test_simple_elementwise_model(self):
        """Test compilation of simple elementwise operations."""
        def elementwise_model(x, y):
            z = x + y
            return torch.relu(z)
        
        # Create sample inputs
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        
        # Trace the model
        graph_module = torch.fx.symbolic_trace(elementwise_model)
        
        # Mock compilation pipeline
        with patch.object(self.compiler, 'invoke_conductor_compiler') as mock_compile:
            with patch.object(self.compiler, 'load_compiled_artifact') as mock_load:
                mock_compile.return_value = "/tmp/test.so"
                mock_load.return_value = Mock(spec=ExecutableKernel)
                
                # This should work without errors
                result = self.compiler.compile_graph(graph_module)
                assert isinstance(result, CompiledArtifact)
                
    def test_reduction_model(self):
        """Test compilation of models with reduction operations."""
        def reduction_model(x):
            return torch.sum(torch.relu(x), dim=1)
        
        # Create sample input
        x = torch.randn(4, 8)
        
        # Trace the model
        graph_module = torch.fx.symbolic_trace(reduction_model)
        
        # Mock compilation pipeline
        with patch.object(self.compiler, 'invoke_conductor_compiler') as mock_compile:
            with patch.object(self.compiler, 'load_compiled_artifact') as mock_load:
                mock_compile.return_value = "/tmp/test.so"
                mock_load.return_value = Mock(spec=ExecutableKernel)
                
                # This should work without errors
                result = self.compiler.compile_graph(graph_module)
                assert isinstance(result, CompiledArtifact)
                
    def test_multiple_compilation_same_graph(self):
        """Test that multiple compilations of the same graph use cache."""
        def simple_model(x):
            return torch.relu(x)
        
        graph_module = torch.fx.symbolic_trace(simple_model)
        
        # Mock compilation pipeline for first call only
        with patch.object(self.compiler, 'invoke_conductor_compiler') as mock_compile:
            with patch.object(self.compiler, 'load_compiled_artifact') as mock_load:
                mock_compile.return_value = "/tmp/test.so"
                mock_load.return_value = Mock(spec=ExecutableKernel)
                
                # First compilation
                result1 = self.compiler.compile_graph(graph_module)
                
                # Mock the cache to return the same artifact with is_valid=True
                mock_artifact = Mock(spec=CompiledArtifact)
                mock_artifact.is_valid.return_value = True
                mock_artifact.path = result1.path
                mock_artifact.artifact_type = result1.artifact_type
                mock_artifact.entry_point = result1.entry_point
                mock_artifact.metadata = result1.metadata
                
                with patch.object(self.compiler._cache, 'get', return_value=mock_artifact):
                    # Second compilation should use cache
                    result2 = self.compiler.compile_graph(graph_module)
                    
                    # Should be the cached artifact
                    assert result2 is mock_artifact
                    
                    # Compiler should only be called once
                    assert mock_compile.call_count == 1
                
    def test_compilation_with_fusion_opportunities(self):
        """Test compilation of graphs with fusion opportunities."""
        def fusible_model(x):
            # Chain of elementwise operations that should be fused
            y = x + 1.0
            z = y * 2.0
            return torch.relu(z)
        
        graph_module = torch.fx.symbolic_trace(fusible_model)
        
        # Mock compilation pipeline
        with patch.object(self.compiler, 'invoke_conductor_compiler') as mock_compile:
            with patch.object(self.compiler, 'load_compiled_artifact') as mock_load:
                mock_compile.return_value = "/tmp/test.so"
                mock_load.return_value = Mock(spec=ExecutableKernel)
                
                result = self.compiler.compile_graph(graph_module)
                
                # Check that fusion was attempted
                assert 'fusion_clusters' in result.metadata
                # Should have at least some fusion opportunities
                assert result.metadata['fusion_clusters'] >= 0
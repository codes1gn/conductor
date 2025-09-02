"""
Unit tests for AOT fallback and error recovery functionality.

Tests the AOTManager fallback mechanisms including JIT fallback
and Inductor backend fallback.
"""

import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import pytest
import torch
import torch.fx

from conductor.runtime.aot import AOTManager, AOTCompatibilityError, AOTArtifactNotFoundError
from conductor.runtime.loader import ExecutableKernel


class TestAOTFallback:
    """Test cases for AOT fallback functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.aot_manager = AOTManager([self.temp_dir])
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_mock_graph_module(self) -> torch.fx.GraphModule:
        """Create a mock FX Graph module for testing."""
        def simple_model(x):
            return torch.relu(x + 1.0)
            
        model = simple_model
        traced = torch.fx.symbolic_trace(model)
        return traced
        
    @patch('conductor.runtime.aot.AOTManager._fallback_to_jit')
    def test_load_with_fallback_artifact_not_found(self, mock_jit_fallback):
        """Test fallback when artifact is not found."""
        mock_kernel = Mock(spec=ExecutableKernel)
        mock_jit_fallback.return_value = mock_kernel
        
        graph_module = self.create_mock_graph_module()
        
        result = self.aot_manager.load_with_fallback(graph_module)
        
        assert result == mock_kernel
        mock_jit_fallback.assert_called_once_with(graph_module, "artifact_not_found")
        
    @patch('conductor.runtime.aot.AOTManager._fallback_to_jit')
    def test_load_with_fallback_compatibility_failed(self, mock_jit_fallback):
        """Test fallback when compatibility validation fails."""
        mock_kernel = Mock(spec=ExecutableKernel)
        mock_jit_fallback.return_value = mock_kernel
        
        # Create an artifact that will be found but fail compatibility
        graph_module = self.create_mock_graph_module()
        signature = self.aot_manager._generate_graph_signature(graph_module)
        
        artifact_path = os.path.join(self.temp_dir, f"{signature}.so")
        with open(artifact_path, 'wb') as f:
            f.write(b'dummy_content')
            
        # Mock compatibility validation to fail
        with patch.object(self.aot_manager, 'validate_artifact_compatibility', return_value=False):
            result = self.aot_manager.load_with_fallback(graph_module)
            
        assert result == mock_kernel
        mock_jit_fallback.assert_called_once_with(graph_module, "compatibility_check_failed")
        
    @patch('conductor.runtime.aot.ExecutableKernel.load_from_file')
    def test_load_with_fallback_success(self, mock_load):
        """Test successful AOT loading without fallback."""
        mock_kernel = Mock(spec=ExecutableKernel)
        mock_load.return_value = mock_kernel
        
        graph_module = self.create_mock_graph_module()
        signature = self.aot_manager._generate_graph_signature(graph_module)
        
        # Create a compatible artifact
        artifact_path = os.path.join(self.temp_dir, f"{signature}.so")
        with open(artifact_path, 'wb') as f:
            f.write(b'dummy_content')
            
        # Mock compatibility validation to succeed
        with patch.object(self.aot_manager, 'validate_artifact_compatibility', return_value=True):
            result = self.aot_manager.load_with_fallback(graph_module)
            
        assert result == mock_kernel
        mock_load.assert_called_once_with(artifact_path)
        
    def test_load_with_fallback_disabled_artifact_not_found(self):
        """Test error when fallback is disabled and artifact not found."""
        graph_module = self.create_mock_graph_module()
        
        with pytest.raises(AOTArtifactNotFoundError):
            self.aot_manager.load_with_fallback(graph_module, fallback_to_jit=False)
            
    def test_load_with_fallback_disabled_compatibility_failed(self):
        """Test error when fallback is disabled and compatibility fails."""
        graph_module = self.create_mock_graph_module()
        signature = self.aot_manager._generate_graph_signature(graph_module)
        
        # Create an artifact
        artifact_path = os.path.join(self.temp_dir, f"{signature}.so")
        with open(artifact_path, 'wb') as f:
            f.write(b'dummy_content')
            
        # Mock compatibility validation to fail
        with patch.object(self.aot_manager, 'validate_artifact_compatibility', return_value=False):
            with pytest.raises(AOTCompatibilityError):
                self.aot_manager.load_with_fallback(graph_module, fallback_to_jit=False)
                
    @patch('conductor.runtime.jit.JITCompiler')
    def test_fallback_to_jit_success(self, mock_jit_compiler_class):
        """Test successful JIT fallback."""
        # Mock JIT compiler
        mock_compiler = Mock()
        mock_artifact = Mock()
        mock_artifact.path = "/path/to/jit/artifact.so"
        mock_compiler.compile_graph.return_value = mock_artifact
        mock_jit_compiler_class.return_value = mock_compiler
        
        # Mock ExecutableKernel loading
        mock_kernel = Mock(spec=ExecutableKernel)
        with patch('conductor.runtime.aot.ExecutableKernel.load_from_file', return_value=mock_kernel):
            graph_module = self.create_mock_graph_module()
            
            result = self.aot_manager._fallback_to_jit(graph_module, "test_reason")
            
            assert result == mock_kernel
            mock_compiler.compile_graph.assert_called_once_with(graph_module)
            
    @patch('conductor.runtime.jit.JITCompiler')
    @patch('conductor.runtime.aot.AOTManager._fallback_to_inductor')
    def test_fallback_to_jit_failure_then_inductor(self, mock_inductor_fallback, mock_jit_compiler_class):
        """Test JIT fallback failure leading to Inductor fallback."""
        # Mock JIT compiler to fail
        mock_compiler = Mock()
        mock_compiler.compile_graph.side_effect = RuntimeError("JIT compilation failed")
        mock_jit_compiler_class.return_value = mock_compiler
        
        # Mock Inductor fallback to succeed
        mock_kernel = Mock(spec=ExecutableKernel)
        mock_inductor_fallback.return_value = mock_kernel
        
        graph_module = self.create_mock_graph_module()
        
        result = self.aot_manager._fallback_to_jit(graph_module, "test_reason")
        
        assert result == mock_kernel
        mock_inductor_fallback.assert_called_once_with(graph_module)
        
    @patch('conductor.runtime.jit.JITCompiler')
    @patch('conductor.runtime.aot.AOTManager._fallback_to_inductor')
    def test_fallback_to_jit_all_methods_fail(self, mock_inductor_fallback, mock_jit_compiler_class):
        """Test when all fallback methods fail."""
        # Mock JIT compiler to fail
        mock_compiler = Mock()
        mock_compiler.compile_graph.side_effect = RuntimeError("JIT compilation failed")
        mock_jit_compiler_class.return_value = mock_compiler
        
        # Mock Inductor fallback to fail
        mock_inductor_fallback.side_effect = RuntimeError("Inductor compilation failed")
        
        graph_module = self.create_mock_graph_module()
        
        with pytest.raises(RuntimeError, match="All compilation methods failed"):
            self.aot_manager._fallback_to_jit(graph_module, "test_reason")
            
    @patch('torch.compile')
    def test_fallback_to_inductor_success(self, mock_torch_compile):
        """Test successful Inductor fallback."""
        # Mock torch.compile to return a compiled function
        mock_compiled_fn = Mock()
        mock_compiled_fn.return_value = torch.tensor([1.0, 2.0, 3.0])
        mock_torch_compile.return_value = mock_compiled_fn
        
        graph_module = self.create_mock_graph_module()
        
        result = self.aot_manager._fallback_to_inductor(graph_module)
        
        # Test the wrapper functionality
        assert hasattr(result, 'execute')
        assert hasattr(result, 'unload')
        assert hasattr(result, 'get_metadata')
        
        # Test execution
        test_input = torch.tensor([1.0])
        output = result.execute([test_input])
        
        assert len(output) == 1
        mock_compiled_fn.assert_called_once_with(test_input)
        mock_torch_compile.assert_called_once_with(graph_module, backend='inductor')
        
    @patch('torch.compile')
    def test_fallback_to_inductor_failure(self, mock_torch_compile):
        """Test Inductor fallback failure."""
        mock_torch_compile.side_effect = RuntimeError("Inductor compilation failed")
        
        graph_module = self.create_mock_graph_module()
        
        with pytest.raises(RuntimeError, match="Inductor fallback compilation failed"):
            self.aot_manager._fallback_to_inductor(graph_module)
            
    def test_diagnose_aot_failure_no_artifact(self):
        """Test AOT failure diagnosis when no artifact is found."""
        graph_module = self.create_mock_graph_module()
        
        diagnosis = self.aot_manager.diagnose_aot_failure(graph_module)
        
        assert 'graph_signature' in diagnosis
        assert 'search_paths' in diagnosis
        assert 'search_results' in diagnosis
        assert diagnosis['artifact_found'] is None
        assert len(diagnosis['recommendations']) > 0
        
    def test_diagnose_aot_failure_with_artifact(self):
        """Test AOT failure diagnosis when artifact exists but is incompatible."""
        graph_module = self.create_mock_graph_module()
        signature = self.aot_manager._generate_graph_signature(graph_module)
        
        # Create an incompatible artifact
        artifact_path = os.path.join(self.temp_dir, f"{signature}.so")
        with open(artifact_path, 'wb') as f:
            f.write(b'dummy_content')
            
        # Mock compatibility validation to fail
        with patch.object(self.aot_manager, 'validate_artifact_compatibility', return_value=False):
            diagnosis = self.aot_manager.diagnose_aot_failure(graph_module)
            
        assert diagnosis['artifact_found'] == artifact_path
        assert 'artifact_metadata' in diagnosis
        assert len(diagnosis['compatibility_issues']) > 0
        
    def test_diagnose_aot_failure_nonexistent_search_paths(self):
        """Test diagnosis with nonexistent search paths."""
        manager = AOTManager(['/nonexistent/path1', '/nonexistent/path2'])
        graph_module = self.create_mock_graph_module()
        
        diagnosis = manager.diagnose_aot_failure(graph_module)
        
        # Check that all search paths are marked as non-existent
        for path_info in diagnosis['search_results'].values():
            assert path_info['exists'] is False
            
        assert any('No artifact search paths exist' in rec for rec in diagnosis['recommendations'])
        
    def test_diagnose_aot_failure_with_metadata(self):
        """Test diagnosis with artifact metadata."""
        graph_module = self.create_mock_graph_module()
        signature = self.aot_manager._generate_graph_signature(graph_module)
        
        # Create artifact and metadata
        artifact_path = os.path.join(self.temp_dir, f"{signature}.so")
        with open(artifact_path, 'wb') as f:
            f.write(b'dummy_content')
            
        metadata_path = os.path.join(self.temp_dir, f"{signature}.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump({
                'graph_signature': 'different_signature',
                'version': '1.0'
            }, f)
            
        # Mock compatibility validation to fail
        with patch.object(self.aot_manager, 'validate_artifact_compatibility', return_value=False):
            diagnosis = self.aot_manager.diagnose_aot_failure(graph_module)
            
        assert any('Graph signature mismatch' in issue for issue in diagnosis['compatibility_issues'])


if __name__ == '__main__':
    pytest.main([__file__])
"""
Unit tests for AOT Manager artifact discovery and loading.

Tests the AOTManager class functionality including artifact discovery,
compatibility validation, and loading mechanisms.
"""

import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import pytest
import torch
import torch.fx

from conductor.runtime.aot import AOTManager, AOTCompatibilityError, AOTArtifactNotFoundError
from conductor.runtime.loader import ExecutableKernel, CompiledArtifact


class TestAOTManager:
    """Test cases for AOTManager class."""
    
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
        example_input = torch.randn(1, 4)
        
        # Create FX Graph
        traced = torch.fx.symbolic_trace(model)
        return traced
        
    def create_test_artifact(self, signature: str, artifact_type: str = '.so') -> str:
        """Create a test artifact file."""
        artifact_path = os.path.join(self.temp_dir, f"{signature}{artifact_type}")
        
        # Create a dummy file
        with open(artifact_path, 'wb') as f:
            f.write(b'dummy_artifact_content')
            
        return artifact_path
        
    def create_test_metadata(self, signature: str, metadata: dict) -> str:
        """Create a test metadata file."""
        metadata_path = os.path.join(self.temp_dir, f"{signature}.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        return metadata_path
        
    def test_init_default_search_paths(self):
        """Test AOTManager initialization with default search paths."""
        manager = AOTManager()
        
        expected_paths = [
            './artifacts',
            '~/.conductor/artifacts',
            '/usr/local/lib/conductor'
        ]
        
        assert manager.search_paths == expected_paths
        assert manager._artifact_cache == {}
        
    def test_init_custom_search_paths(self):
        """Test AOTManager initialization with custom search paths."""
        custom_paths = ['/custom/path1', '/custom/path2']
        manager = AOTManager(custom_paths)
        
        assert manager.search_paths == custom_paths
        
    def test_locate_precompiled_artifact_found_so(self):
        """Test locating a .so artifact."""
        signature = "test_signature_123"
        artifact_path = self.create_test_artifact(signature, '.so')
        
        result = self.aot_manager.locate_precompiled_artifact(signature)
        
        assert result == artifact_path
        
    def test_locate_precompiled_artifact_found_o(self):
        """Test locating a .o artifact."""
        signature = "test_signature_456"
        artifact_path = self.create_test_artifact(signature, '.o')
        
        result = self.aot_manager.locate_precompiled_artifact(signature)
        
        assert result == artifact_path
        
    def test_locate_precompiled_artifact_priority_so_over_o(self):
        """Test that .so files have priority over .o files."""
        signature = "test_signature_789"
        so_path = self.create_test_artifact(signature, '.so')
        o_path = self.create_test_artifact(signature, '.o')
        
        result = self.aot_manager.locate_precompiled_artifact(signature)
        
        # Should return .so file, not .o file
        assert result == so_path
        
    def test_locate_precompiled_artifact_with_metadata(self):
        """Test locating artifact via metadata file."""
        signature = "test_signature_meta"
        artifact_path = self.create_test_artifact("actual_artifact", '.so')
        
        metadata = {
            'artifact_path': 'actual_artifact.so',
            'graph_signature': signature
        }
        self.create_test_metadata(signature, metadata)
        
        result = self.aot_manager.locate_precompiled_artifact(signature)
        
        assert result == artifact_path
        
    def test_locate_precompiled_artifact_not_found(self):
        """Test artifact not found scenario."""
        result = self.aot_manager.locate_precompiled_artifact("nonexistent_signature")
        
        assert result is None
        
    def test_locate_precompiled_artifact_nonexistent_search_path(self):
        """Test handling of nonexistent search paths."""
        manager = AOTManager(['/nonexistent/path', self.temp_dir])
        signature = "test_signature"
        artifact_path = self.create_test_artifact(signature, '.so')
        
        result = manager.locate_precompiled_artifact(signature)
        
        assert result == artifact_path
        
    def test_generate_graph_signature_deterministic(self):
        """Test that graph signature generation is deterministic."""
        graph_module = self.create_mock_graph_module()
        
        signature1 = self.aot_manager._generate_graph_signature(graph_module)
        signature2 = self.aot_manager._generate_graph_signature(graph_module)
        
        assert signature1 == signature2
        assert len(signature1) == 16  # SHA256 truncated to 16 chars
        
    def test_generate_graph_signature_different_graphs(self):
        """Test that different graphs produce different signatures."""
        def model1(x):
            return torch.relu(x + 1.0)
            
        def model2(x):
            return torch.sigmoid(x * 2.0)
            
        graph1 = torch.fx.symbolic_trace(model1)
        graph2 = torch.fx.symbolic_trace(model2)
        
        sig1 = self.aot_manager._generate_graph_signature(graph1)
        sig2 = self.aot_manager._generate_graph_signature(graph2)
        
        assert sig1 != sig2
        
    def test_validate_artifact_compatibility_file_not_exists(self):
        """Test compatibility validation with nonexistent file."""
        graph_module = self.create_mock_graph_module()
        
        result = self.aot_manager.validate_artifact_compatibility(
            "/nonexistent/file.so", graph_module
        )
        
        assert result is False
        
    def test_validate_artifact_compatibility_unsupported_format(self):
        """Test compatibility validation with unsupported file format."""
        graph_module = self.create_mock_graph_module()
        
        # Create a file with unsupported extension
        unsupported_path = os.path.join(self.temp_dir, "test.txt")
        with open(unsupported_path, 'w') as f:
            f.write("test")
            
        result = self.aot_manager.validate_artifact_compatibility(
            unsupported_path, graph_module
        )
        
        assert result is False
        
    def test_validate_artifact_compatibility_with_metadata_match(self):
        """Test compatibility validation with matching metadata."""
        graph_module = self.create_mock_graph_module()
        signature = self.aot_manager._generate_graph_signature(graph_module)
        
        artifact_path = self.create_test_artifact(signature, '.so')
        
        metadata = {
            'graph_signature': signature,
            'inputs': [{'name': 'x', 'type': 'tensor'}],
            'outputs': [{'name': 'output', 'type': 'tensor'}],
            'operations': ['add', 'relu']
        }
        self.create_test_metadata(signature, metadata)
        
        result = self.aot_manager.validate_artifact_compatibility(
            artifact_path, graph_module
        )
        
        assert result is True
        
    def test_validate_artifact_compatibility_signature_mismatch(self):
        """Test compatibility validation with signature mismatch."""
        graph_module = self.create_mock_graph_module()
        
        artifact_path = self.create_test_artifact("wrong_signature", '.so')
        
        metadata = {
            'graph_signature': 'different_signature',
            'inputs': [{'name': 'x', 'type': 'tensor'}],
            'outputs': [{'name': 'output', 'type': 'tensor'}]
        }
        self.create_test_metadata("wrong_signature", metadata)
        
        result = self.aot_manager.validate_artifact_compatibility(
            artifact_path, graph_module
        )
        
        assert result is False
        
    def test_validate_artifact_compatibility_without_metadata(self):
        """Test compatibility validation without metadata (filename-based)."""
        graph_module = self.create_mock_graph_module()
        signature = self.aot_manager._generate_graph_signature(graph_module)
        
        artifact_path = self.create_test_artifact(signature, '.so')
        
        result = self.aot_manager.validate_artifact_compatibility(
            artifact_path, graph_module
        )
        
        assert result is True
        
    @patch('conductor.runtime.aot.ExecutableKernel.load_from_file')
    def test_load_static_artifact_so_file(self, mock_load):
        """Test loading a .so artifact."""
        mock_kernel = Mock(spec=ExecutableKernel)
        mock_load.return_value = mock_kernel
        
        artifact_path = self.create_test_artifact("test_sig", '.so')
        
        result = self.aot_manager.load_static_artifact(artifact_path)
        
        assert result == mock_kernel
        mock_load.assert_called_once_with(artifact_path)
        
    def test_load_static_artifact_file_not_found(self):
        """Test loading nonexistent artifact."""
        with pytest.raises(AOTArtifactNotFoundError):
            self.aot_manager.load_static_artifact("/nonexistent/file.so")
            
    def test_load_static_artifact_file_not_readable(self):
        """Test loading non-readable artifact."""
        artifact_path = self.create_test_artifact("test_sig", '.so')
        
        # Make file non-readable
        os.chmod(artifact_path, 0o000)
        
        try:
            with pytest.raises(AOTCompatibilityError):
                self.aot_manager.load_static_artifact(artifact_path)
        finally:
            # Restore permissions for cleanup
            os.chmod(artifact_path, 0o644)
            
    @patch('conductor.runtime.aot.AOTManager._link_object_file')
    @patch('conductor.runtime.aot.ExecutableKernel.load_from_file')
    def test_load_static_artifact_o_file(self, mock_load, mock_link):
        """Test loading a .o artifact (requires linking)."""
        mock_kernel = Mock(spec=ExecutableKernel)
        mock_load.return_value = mock_kernel
        
        linked_so_path = os.path.join(self.temp_dir, "linked.so")
        mock_link.return_value = linked_so_path
        
        artifact_path = self.create_test_artifact("test_sig", '.o')
        
        result = self.aot_manager.load_static_artifact(artifact_path)
        
        assert result == mock_kernel
        mock_link.assert_called_once_with(artifact_path)
        mock_load.assert_called_once_with(linked_so_path)
        
    def test_load_static_artifact_unsupported_format(self):
        """Test loading artifact with unsupported format."""
        unsupported_path = os.path.join(self.temp_dir, "test.txt")
        with open(unsupported_path, 'w') as f:
            f.write("test")
            
        with pytest.raises(AOTCompatibilityError):
            self.aot_manager.load_static_artifact(unsupported_path)
            
    def test_load_static_artifact_caching(self):
        """Test that artifacts are cached after loading."""
        with patch('conductor.runtime.aot.ExecutableKernel.load_from_file') as mock_load:
            mock_kernel = Mock(spec=ExecutableKernel)
            mock_load.return_value = mock_kernel
            
            artifact_path = self.create_test_artifact("test_sig", '.so')
            
            # First load
            result1 = self.aot_manager.load_static_artifact(artifact_path)
            
            # Check cache
            assert len(self.aot_manager._artifact_cache) == 1
            
            # Second load should use cache
            result2 = self.aot_manager.load_static_artifact(artifact_path)
            
            assert result1 == mock_kernel
            assert result2 == mock_kernel
            
    @patch('subprocess.run')
    def test_link_object_file_success(self, mock_run):
        """Test successful object file linking."""
        mock_run.return_value = Mock(returncode=0, stderr="")
        
        object_path = self.create_test_artifact("test", '.o')
        
        with patch('tempfile.mkdtemp') as mock_mkdtemp:
            temp_dir = os.path.join(self.temp_dir, 'temp_link')
            os.makedirs(temp_dir, exist_ok=True)
            mock_mkdtemp.return_value = temp_dir
            
            # Create the expected output file
            expected_so = os.path.join(temp_dir, "test.o.so")
            with open(expected_so, 'wb') as f:
                f.write(b'linked_library')
                
            result = self.aot_manager._link_object_file(object_path)
            
            assert result == expected_so
            mock_run.assert_called_once()
            
    @patch('subprocess.run')
    def test_link_object_file_failure(self, mock_run):
        """Test object file linking failure."""
        mock_run.return_value = Mock(returncode=1, stderr="linking error")
        
        object_path = self.create_test_artifact("test", '.o')
        
        with pytest.raises(RuntimeError, match="Linking failed"):
            self.aot_manager._link_object_file(object_path)
            
    def test_get_artifact_metadata_basic(self):
        """Test getting basic artifact metadata."""
        artifact_path = self.create_test_artifact("test_sig", '.so')
        
        metadata = self.aot_manager.get_artifact_metadata(artifact_path)
        
        assert metadata['path'] == artifact_path
        assert metadata['exists'] is True
        assert metadata['size'] > 0
        assert metadata['type'] == 'shared_library'
        assert metadata['readable'] is True
        assert 'mtime' in metadata
        
    def test_get_artifact_metadata_with_json(self):
        """Test getting artifact metadata with JSON metadata file."""
        artifact_path = self.create_test_artifact("test_sig", '.so')
        
        json_metadata = {
            'graph_signature': 'test_sig',
            'version': '1.0',
            'compiler_version': '2.1.0'
        }
        self.create_test_metadata("test_sig", json_metadata)
        
        metadata = self.aot_manager.get_artifact_metadata(artifact_path)
        
        assert metadata['graph_signature'] == 'test_sig'
        assert metadata['version'] == '1.0'
        assert metadata['compiler_version'] == '2.1.0'
        
    def test_get_artifact_metadata_nonexistent(self):
        """Test getting metadata for nonexistent artifact."""
        metadata = self.aot_manager.get_artifact_metadata("/nonexistent/file.so")
        
        assert metadata['exists'] is False
        assert metadata['size'] == 0
        assert metadata['readable'] is False
        
    def test_clear_cache(self):
        """Test clearing the artifact cache."""
        # Add something to cache
        self.aot_manager._artifact_cache['test'] = Mock()
        
        assert len(self.aot_manager._artifact_cache) == 1
        
        self.aot_manager.clear_cache()
        
        assert len(self.aot_manager._artifact_cache) == 0
        
    def test_get_cache_info(self):
        """Test getting cache information."""
        # Add items to cache
        self.aot_manager._artifact_cache['key1'] = Mock()
        self.aot_manager._artifact_cache['key2'] = Mock()
        
        info = self.aot_manager.get_cache_info()
        
        assert info['cached_artifacts'] == 2
        assert set(info['cache_keys']) == {'key1', 'key2'}
        
    def test_validate_io_compatibility_no_metadata(self):
        """Test I/O validation when metadata has no I/O info."""
        metadata = {'graph_signature': 'test'}
        graph_module = self.create_mock_graph_module()
        
        result = self.aot_manager._validate_io_compatibility(metadata, graph_module)
        
        assert result is True
        
    def test_validate_io_compatibility_input_count_mismatch(self):
        """Test I/O validation with input count mismatch."""
        metadata = {
            'inputs': [{'name': 'x'}, {'name': 'y'}],  # 2 inputs
            'outputs': [{'name': 'out'}]
        }
        graph_module = self.create_mock_graph_module()  # Has 1 input
        
        result = self.aot_manager._validate_io_compatibility(metadata, graph_module)
        
        assert result is False
        
    def test_validate_operation_compatibility_no_metadata(self):
        """Test operation validation when metadata has no operation info."""
        metadata = {'graph_signature': 'test'}
        graph_module = self.create_mock_graph_module()
        
        result = self.aot_manager._validate_operation_compatibility(metadata, graph_module)
        
        assert result is True
        
    def test_validate_operation_compatibility_missing_operation(self):
        """Test operation validation with missing operation."""
        metadata = {
            'operations': ['add', 'relu', 'missing_op']
        }
        graph_module = self.create_mock_graph_module()
        
        result = self.aot_manager._validate_operation_compatibility(metadata, graph_module)
        
        assert result is False


if __name__ == '__main__':
    pytest.main([__file__])
"""
Integration tests for AOT compilation pipeline.

Tests the complete AOT workflow including artifact discovery,
loading, and fallback mechanisms.
"""

import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch
import pytest
import torch
import torch.fx

from conductor.runtime.aot import AOTManager
from conductor.runtime.loader import ExecutableKernel


class TestAOTIntegration:
    """Integration tests for AOT pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.aot_manager = AOTManager([self.temp_dir])
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_test_model(self):
        """Create a test model for integration testing."""
        def test_model(x, y):
            z = x + y
            return torch.relu(z)
            
        return test_model
        
    def create_test_artifact_with_metadata(self, signature: str, compatible: bool = True):
        """Create a test artifact with metadata for integration testing."""
        # Create artifact file
        artifact_path = os.path.join(self.temp_dir, f"{signature}.so")
        with open(artifact_path, 'wb') as f:
            f.write(b'mock_shared_library_content')
            
        # Create metadata file
        metadata = {
            'graph_signature': signature if compatible else 'different_signature',
            'version': '1.0.0',
            'compiler_version': '2.1.0',
            'inputs': [
                {'name': 'x', 'type': 'tensor', 'shape': [-1, 4]},
                {'name': 'y', 'type': 'tensor', 'shape': [-1, 4]}
            ],
            'outputs': [
                {'name': 'output', 'type': 'tensor', 'shape': [-1, 4]}
            ],
            'operations': ['add', 'relu'],
            'optimization_level': 'O2',
            'target_device': 'gcu'
        }
        
        metadata_path = os.path.join(self.temp_dir, f"{signature}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return artifact_path, metadata_path
        
    @patch('conductor.runtime.aot.ExecutableKernel.load_from_file')
    def test_end_to_end_aot_success(self, mock_load):
        """Test successful end-to-end AOT loading."""
        # Create test model and graph
        model = self.create_test_model()
        example_inputs = (torch.randn(2, 4), torch.randn(2, 4))
        traced = torch.fx.symbolic_trace(model)
        
        # Generate signature and create compatible artifact
        signature = self.aot_manager._generate_graph_signature(traced)
        artifact_path, metadata_path = self.create_test_artifact_with_metadata(signature, compatible=True)
        
        # Mock successful kernel loading
        mock_kernel = Mock(spec=ExecutableKernel)
        mock_load.return_value = mock_kernel
        
        # Test the complete workflow
        result = self.aot_manager.load_with_fallback(traced)
        
        assert result == mock_kernel
        mock_load.assert_called_once_with(artifact_path)
        
    @patch('conductor.runtime.aot.AOTManager._fallback_to_jit')
    def test_end_to_end_aot_fallback_incompatible(self, mock_jit_fallback):
        """Test end-to-end AOT with fallback due to incompatible artifact."""
        # Create test model and graph
        model = self.create_test_model()
        traced = torch.fx.symbolic_trace(model)
        
        # Generate signature and create incompatible artifact
        signature = self.aot_manager._generate_graph_signature(traced)
        artifact_path, metadata_path = self.create_test_artifact_with_metadata(signature, compatible=False)
        
        # Mock JIT fallback
        mock_kernel = Mock(spec=ExecutableKernel)
        mock_jit_fallback.return_value = mock_kernel
        
        # Test the complete workflow
        result = self.aot_manager.load_with_fallback(traced)
        
        assert result == mock_kernel
        mock_jit_fallback.assert_called_once()
        
    @patch('conductor.runtime.aot.AOTManager._fallback_to_jit')
    def test_end_to_end_aot_fallback_no_artifact(self, mock_jit_fallback):
        """Test end-to-end AOT with fallback when no artifact exists."""
        # Create test model and graph
        model = self.create_test_model()
        traced = torch.fx.symbolic_trace(model)
        
        # Mock JIT fallback
        mock_kernel = Mock(spec=ExecutableKernel)
        mock_jit_fallback.return_value = mock_kernel
        
        # Test the complete workflow (no artifact created)
        result = self.aot_manager.load_with_fallback(traced)
        
        assert result == mock_kernel
        mock_jit_fallback.assert_called_once_with(traced, "artifact_not_found")
        
    def test_artifact_discovery_multiple_search_paths(self):
        """Test artifact discovery across multiple search paths."""
        # Create additional search paths
        temp_dir2 = tempfile.mkdtemp()
        temp_dir3 = tempfile.mkdtemp()
        
        try:
            manager = AOTManager([self.temp_dir, temp_dir2, temp_dir3])
            
            # Create artifact in the second search path
            signature = "test_multi_path_signature"
            artifact_path = os.path.join(temp_dir2, f"{signature}.so")
            with open(artifact_path, 'wb') as f:
                f.write(b'test_artifact')
                
            # Test discovery
            found_path = manager.locate_precompiled_artifact(signature)
            
            assert found_path == artifact_path
            
        finally:
            shutil.rmtree(temp_dir2, ignore_errors=True)
            shutil.rmtree(temp_dir3, ignore_errors=True)
            
    def test_artifact_priority_so_over_o(self):
        """Test that .so files have priority over .o files."""
        signature = "test_priority_signature"
        
        # Create both .so and .o files
        so_path = os.path.join(self.temp_dir, f"{signature}.so")
        o_path = os.path.join(self.temp_dir, f"{signature}.o")
        
        with open(o_path, 'wb') as f:
            f.write(b'object_file_content')
            
        with open(so_path, 'wb') as f:
            f.write(b'shared_library_content')
            
        # Test that .so is preferred
        found_path = self.aot_manager.locate_precompiled_artifact(signature)
        
        assert found_path == so_path
        
    def test_comprehensive_diagnosis(self):
        """Test comprehensive AOT failure diagnosis."""
        # Create test model
        model = self.create_test_model()
        traced = torch.fx.symbolic_trace(model)
        
        # Create some artifacts in the search path
        with open(os.path.join(self.temp_dir, "other_artifact.so"), 'wb') as f:
            f.write(b'other_content')
            
        with open(os.path.join(self.temp_dir, "metadata.json"), 'w') as f:
            json.dump({'test': 'metadata'}, f)
            
        # Run diagnosis
        diagnosis = self.aot_manager.diagnose_aot_failure(traced)
        
        # Verify diagnosis structure
        assert 'graph_signature' in diagnosis
        assert 'search_paths' in diagnosis
        assert 'search_results' in diagnosis
        assert 'compatibility_issues' in diagnosis
        assert 'recommendations' in diagnosis
        
        # Check search results
        search_result = diagnosis['search_results'][self.temp_dir]
        assert search_result['exists'] is True
        assert search_result['readable'] is True
        assert 'other_artifact.so' in search_result['artifacts_found']
        assert 'metadata.json' in search_result['artifacts_found']
        
        # Should recommend creating artifact since none found for this signature
        assert any('No artifact found' in rec for rec in diagnosis['recommendations'])
        
    @patch('conductor.runtime.aot.ExecutableKernel.load_from_file')
    def test_artifact_caching_behavior(self, mock_load):
        """Test that artifacts are properly cached."""
        # Create test model and artifact
        model = self.create_test_model()
        traced = torch.fx.symbolic_trace(model)
        signature = self.aot_manager._generate_graph_signature(traced)
        artifact_path, _ = self.create_test_artifact_with_metadata(signature)
        
        # Mock kernel loading
        mock_kernel = Mock(spec=ExecutableKernel)
        mock_load.return_value = mock_kernel
        
        # Load artifact twice
        result1 = self.aot_manager.load_with_fallback(traced)
        result2 = self.aot_manager.load_with_fallback(traced)
        
        # Both should succeed
        assert result1 == mock_kernel
        assert result2 == mock_kernel
        
        # Check that cache was used (artifact info should be cached)
        cache_info = self.aot_manager.get_cache_info()
        assert cache_info['cached_artifacts'] > 0
        
    def test_metadata_loading_variations(self):
        """Test loading metadata from different file locations."""
        signature = "test_metadata_variations"
        artifact_path = os.path.join(self.temp_dir, f"{signature}.so")
        
        # Create artifact
        with open(artifact_path, 'wb') as f:
            f.write(b'test_content')
            
        # Test 1: metadata with same base name
        metadata1_path = os.path.join(self.temp_dir, f"{signature}.json")
        metadata1 = {'type': 'base_name', 'version': '1.0'}
        with open(metadata1_path, 'w') as f:
            json.dump(metadata1, f)
            
        loaded_metadata = self.aot_manager._load_artifact_metadata(artifact_path)
        assert loaded_metadata['type'] == 'base_name'
        
        # Clean up for next test
        os.remove(metadata1_path)
        
        # Test 2: metadata with .meta.json suffix
        metadata2_path = os.path.join(self.temp_dir, f"{signature}.meta.json")
        metadata2 = {'type': 'meta_suffix', 'version': '2.0'}
        with open(metadata2_path, 'w') as f:
            json.dump(metadata2, f)
            
        loaded_metadata = self.aot_manager._load_artifact_metadata(artifact_path)
        assert loaded_metadata['type'] == 'meta_suffix'
        
    def test_error_recovery_scenarios(self):
        """Test various error recovery scenarios."""
        model = self.create_test_model()
        traced = torch.fx.symbolic_trace(model)
        
        # Test 1: Corrupted metadata file
        signature = self.aot_manager._generate_graph_signature(traced)
        artifact_path = os.path.join(self.temp_dir, f"{signature}.so")
        metadata_path = os.path.join(self.temp_dir, f"{signature}.json")
        
        with open(artifact_path, 'wb') as f:
            f.write(b'test_content')
            
        with open(metadata_path, 'w') as f:
            f.write("invalid json content")
            
        # Should still work with basic validation
        with patch.object(self.aot_manager, 'validate_artifact_compatibility', return_value=True):
            with patch('conductor.runtime.aot.ExecutableKernel.load_from_file') as mock_load:
                mock_kernel = Mock(spec=ExecutableKernel)
                mock_load.return_value = mock_kernel
                
                result = self.aot_manager.load_with_fallback(traced)
                assert result == mock_kernel
                
        # Test 2: Permission denied on artifact
        os.chmod(artifact_path, 0o000)
        
        try:
            with patch('conductor.runtime.aot.AOTManager._fallback_to_jit') as mock_fallback:
                mock_kernel = Mock(spec=ExecutableKernel)
                mock_fallback.return_value = mock_kernel
                
                result = self.aot_manager.load_with_fallback(traced)
                assert result == mock_kernel
                mock_fallback.assert_called_once()
                
        finally:
            # Restore permissions for cleanup
            os.chmod(artifact_path, 0o644)


if __name__ == '__main__':
    pytest.main([__file__])
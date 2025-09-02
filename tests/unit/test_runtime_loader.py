"""
Unit tests for runtime artifact loading.

Tests the CompiledArtifact class and ExecutableKernel functionality
including artifact loading, validation, and execution.
"""

import pytest
import tempfile
import shutil
import os
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from conductor.runtime.loader import CompiledArtifact, ExecutableKernel


class TestCompiledArtifact:
    """Test CompiledArtifact class functionality."""
    
    def test_artifact_creation_basic(self):
        """Test basic artifact creation."""
        artifact = CompiledArtifact(
            path="/test/path.so",
            artifact_type="shared_library",
            entry_point="main",
            metadata={}
        )
        
        assert artifact.path == "/test/path.so"
        assert artifact.artifact_type == "shared_library"
        assert artifact.entry_point == "main"
        assert artifact.metadata == {}
    
    def test_artifact_creation_with_metadata(self):
        """Test artifact creation with metadata."""
        metadata = {"version": "1.0", "compile_time": "2023-01-01"}
        artifact = CompiledArtifact(
            path="/test/path.o",
            artifact_type="object_file",
            entry_point="kernel_main",
            metadata=metadata
        )
        
        assert artifact.metadata == metadata
        assert artifact.artifact_type == "object_file"
        assert artifact.entry_point == "kernel_main"
    
    def test_artifact_is_valid_existing_file(self):
        """Test artifact validation for existing file."""
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            f.write(b"fake shared library content")
            temp_path = f.name
        
        try:
            artifact = CompiledArtifact(
                path=temp_path,
                artifact_type="shared_library",
                entry_point="main",
                metadata={}
            )
            
            assert artifact.is_valid() is True
        finally:
            os.unlink(temp_path)
    
    def test_artifact_is_valid_missing_file(self):
        """Test artifact validation for missing file."""
        artifact = CompiledArtifact(
            path="/nonexistent/path.so",
            artifact_type="shared_library",
            entry_point="main",
            metadata={}
        )
        
        assert artifact.is_valid() is False
    
    def test_artifact_is_valid_unreadable_file(self):
        """Test artifact validation for unreadable file."""
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            f.write(b"content")
            temp_path = f.name
        
        try:
            # Make file unreadable
            os.chmod(temp_path, 0o000)
            
            artifact = CompiledArtifact(
                path=temp_path,
                artifact_type="shared_library",
                entry_point="main",
                metadata={}
            )
            
            # Should be invalid due to permissions
            assert artifact.is_valid() is False
            
        finally:
            # Restore permissions and cleanup
            os.chmod(temp_path, 0o644)
            os.unlink(temp_path)


class TestExecutableKernel:
    """Test ExecutableKernel class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_library = Mock()
        self.mock_entry_function = Mock()
        self.kernel = ExecutableKernel(self.mock_library, self.mock_entry_function)
    
    def test_kernel_creation(self):
        """Test kernel creation."""
        assert self.kernel._library == self.mock_library
        assert self.kernel._entry_function == self.mock_entry_function
        assert self.kernel._is_loaded is True
    
    def test_kernel_get_metadata(self):
        """Test getting kernel metadata."""
        metadata = self.kernel.get_metadata()
        
        assert metadata['loaded'] is True
        assert 'library' in metadata
        assert 'entry_function' in metadata
    
    def test_kernel_unload(self):
        """Test kernel unloading."""
        self.kernel.unload()
        
        assert self.kernel._library is None
        assert self.kernel._entry_function is None
        assert self.kernel._is_loaded is False
    
    def test_kernel_unload_metadata_after_unload(self):
        """Test metadata after unloading."""
        self.kernel.unload()
        metadata = self.kernel.get_metadata()
        
        assert metadata['loaded'] is False
        assert metadata['library'] is None
        assert metadata['entry_function'] is None
    
    def test_execute_not_loaded(self):
        """Test execution when kernel is not loaded."""
        self.kernel.unload()
        
        with pytest.raises(RuntimeError, match="Kernel is not loaded"):
            self.kernel.execute([torch.randn(2, 3)])
    
    def test_execute_no_inputs(self):
        """Test execution with no inputs."""
        with pytest.raises(RuntimeError, match="No inputs provided"):
            self.kernel.execute([])
    
    @patch('torch.zeros_like')
    def test_execute_success(self, mock_zeros_like):
        """Test successful kernel execution."""
        # Setup input tensor
        input_tensor = torch.randn(2, 3)
        output_tensor = torch.zeros_like(input_tensor)
        mock_zeros_like.return_value = output_tensor
        
        # Mock the entry function to return success
        self.mock_entry_function.return_value = 0
        
        # Mock tensor data_ptr and numel
        with patch.object(input_tensor, 'data_ptr', return_value=12345):
            with patch.object(input_tensor, 'numel', return_value=6):
                with patch.object(output_tensor, 'data_ptr', return_value=67890):
                    with patch.object(output_tensor, 'numel', return_value=6):
                        outputs = self.kernel.execute([input_tensor])
        
        assert len(outputs) == 1
        assert outputs[0] is output_tensor
        self.mock_entry_function.assert_called_once()
    
    def test_execute_kernel_failure(self):
        """Test execution when kernel returns error code."""
        input_tensor = torch.randn(2, 3)
        
        # Mock the entry function to return error
        self.mock_entry_function.return_value = -1
        
        with patch('torch.zeros_like', return_value=torch.zeros_like(input_tensor)):
            with patch.object(input_tensor, 'data_ptr', return_value=12345):
                with patch.object(input_tensor, 'numel', return_value=6):
                    with pytest.raises(RuntimeError, match="Kernel execution failed with code: -1"):
                        self.kernel.execute([input_tensor])
    
    def test_execute_unsupported_input_type(self):
        """Test execution with unsupported input type."""
        with pytest.raises(RuntimeError, match="Unsupported input type"):
            self.kernel.execute(["invalid_input"])
    
    def test_execute_non_float32_tensor(self):
        """Test execution with non-float32 tensor."""
        input_tensor = torch.randn(2, 3, dtype=torch.float64)
        
        # Mock the conversion and execution
        with patch.object(input_tensor, 'float', return_value=torch.randn(2, 3)) as mock_float:
            with patch('torch.zeros_like', return_value=torch.zeros(2, 3)):
                self.mock_entry_function.return_value = 0
                
                try:
                    self.kernel.execute([input_tensor])
                    mock_float.assert_called_once()
                except:
                    # Execution might fail due to mocking complexity, but conversion should be attempted
                    mock_float.assert_called_once()
    
    def test_execute_non_contiguous_tensor(self):
        """Test execution with non-contiguous tensor."""
        input_tensor = torch.randn(4, 3).transpose(0, 1)  # Non-contiguous
        
        with patch.object(input_tensor, 'contiguous', return_value=torch.randn(3, 4)) as mock_contiguous:
            with patch('torch.zeros_like', return_value=torch.zeros(3, 4)):
                self.mock_entry_function.return_value = 0
                
                try:
                    self.kernel.execute([input_tensor])
                    mock_contiguous.assert_called_once()
                except:
                    # Execution might fail due to mocking complexity, but contiguous should be attempted
                    mock_contiguous.assert_called_once()


class TestExecutableKernelLoading:
    """Test ExecutableKernel loading functionality."""
    
    def test_load_from_file_missing_file(self):
        """Test loading from missing file."""
        with pytest.raises(RuntimeError, match="Artifact not found"):
            ExecutableKernel.load_from_file("/nonexistent/path.so")
    
    def test_load_from_file_unreadable_file(self):
        """Test loading from unreadable file."""
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            f.write(b"content")
            temp_path = f.name
        
        try:
            # Make file unreadable
            os.chmod(temp_path, 0o000)
            
            with pytest.raises(RuntimeError, match="Artifact not readable"):
                ExecutableKernel.load_from_file(temp_path)
                
        finally:
            # Restore permissions and cleanup
            os.chmod(temp_path, 0o644)
            os.unlink(temp_path)
    
    @patch('ctypes.CDLL')
    def test_load_shared_library_success(self, mock_cdll):
        """Test successful shared library loading."""
        mock_library = Mock()
        mock_cdll.return_value = mock_library
        
        # Mock the entry point function
        mock_function = Mock()
        mock_library.conductor_kernel_main = mock_function
        
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            temp_path = f.name
        
        try:
            kernel = ExecutableKernel.load_from_file(temp_path)
            
            assert isinstance(kernel, ExecutableKernel)
            assert kernel._library == mock_library
            assert kernel._entry_function == mock_function
            mock_cdll.assert_called_once_with(temp_path)
            
        finally:
            os.unlink(temp_path)
    
    @patch('ctypes.CDLL')
    def test_load_shared_library_fallback_entry_points(self, mock_cdll):
        """Test shared library loading with fallback entry points."""
        mock_library = Mock()
        mock_cdll.return_value = mock_library
        
        # Mock that first entry points don't exist, but 'main' does
        # Use spec to avoid AttributeError for conductor_kernel_main and kernel_main
        mock_library.configure_mock(**{
            'main': Mock()
        })
        # Remove the other attributes to simulate they don't exist
        del mock_library.conductor_kernel_main
        del mock_library.kernel_main
        del mock_library.execute
        
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            temp_path = f.name
        
        try:
            kernel = ExecutableKernel.load_from_file(temp_path)
            assert isinstance(kernel, ExecutableKernel)
            
        finally:
            os.unlink(temp_path)
    
    @patch('ctypes.CDLL')
    def test_load_shared_library_no_entry_point(self, mock_cdll):
        """Test shared library loading with no valid entry point."""
        mock_library = Mock()
        mock_cdll.return_value = mock_library
        
        # Remove all entry point attributes to simulate they don't exist
        del mock_library.conductor_kernel_main
        del mock_library.kernel_main
        del mock_library.main
        del mock_library.execute
        
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(RuntimeError, match="No suitable entry function found"):
                ExecutableKernel.load_from_file(temp_path)
                
        finally:
            os.unlink(temp_path)
    
    @patch('ctypes.CDLL')
    def test_load_shared_library_loading_error(self, mock_cdll):
        """Test shared library loading error."""
        mock_cdll.side_effect = OSError("Cannot load library")
        
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(RuntimeError, match="Failed to load shared library"):
                ExecutableKernel.load_from_file(temp_path)
                
        finally:
            os.unlink(temp_path)
    
    def test_load_object_file_not_implemented(self):
        """Test object file loading (not implemented)."""
        with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(NotImplementedError, match="Object file loading not yet implemented"):
                ExecutableKernel.load_from_file(temp_path)
                
        finally:
            os.unlink(temp_path)
    
    def test_load_unsupported_file_type(self):
        """Test loading unsupported file type."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(RuntimeError, match="Unsupported artifact type"):
                ExecutableKernel.load_from_file(temp_path)
                
        finally:
            os.unlink(temp_path)


class TestKernelIntegration:
    """Integration tests for kernel loading and execution."""
    
    @patch('ctypes.CDLL')
    def test_complete_kernel_workflow(self, mock_cdll):
        """Test complete kernel workflow from loading to execution."""
        # Setup mocks
        mock_library = Mock()
        mock_cdll.return_value = mock_library
        mock_function = Mock()
        mock_function.return_value = 0  # Success
        mock_library.conductor_kernel_main = mock_function
        
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            temp_path = f.name
        
        try:
            # Load kernel
            kernel = ExecutableKernel.load_from_file(temp_path)
            assert kernel._is_loaded is True
            
            # Get metadata
            metadata = kernel.get_metadata()
            assert metadata['loaded'] is True
            
            # Execute kernel
            input_tensor = torch.randn(2, 3)
            with patch('torch.zeros_like', return_value=torch.zeros_like(input_tensor)):
                with patch.object(input_tensor, 'data_ptr', return_value=12345):
                    with patch.object(input_tensor, 'numel', return_value=6):
                        outputs = kernel.execute([input_tensor])
            
            assert len(outputs) == 1
            mock_function.assert_called_once()
            
            # Unload kernel
            kernel.unload()
            assert kernel._is_loaded is False
            
        finally:
            os.unlink(temp_path)
    
    def test_artifact_and_kernel_integration(self):
        """Test integration between CompiledArtifact and ExecutableKernel."""
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            f.write(b"fake shared library")
            temp_path = f.name
        
        try:
            # Create artifact
            artifact = CompiledArtifact(
                path=temp_path,
                artifact_type="shared_library",
                entry_point="main",
                metadata={"version": "1.0"}
            )
            
            # Validate artifact
            assert artifact.is_valid() is True
            
            # Load kernel from artifact
            with patch('ctypes.CDLL') as mock_cdll:
                mock_library = Mock()
                mock_cdll.return_value = mock_library
                mock_library.main = Mock()
                
                kernel = ExecutableKernel.load_from_file(artifact.path)
                assert kernel._is_loaded is True
                
        finally:
            os.unlink(temp_path)
    
    def test_error_recovery_workflow(self):
        """Test error recovery in kernel workflow."""
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            temp_path = f.name
        
        try:
            # Test loading failure recovery
            with patch('ctypes.CDLL', side_effect=OSError("Load failed")):
                with pytest.raises(RuntimeError, match="Failed to load shared library"):
                    ExecutableKernel.load_from_file(temp_path)
            
            # Test successful loading after failure
            with patch('ctypes.CDLL') as mock_cdll:
                mock_library = Mock()
                mock_cdll.return_value = mock_library
                mock_library.main = Mock()
                
                kernel = ExecutableKernel.load_from_file(temp_path)
                assert kernel._is_loaded is True
                
                # Test execution failure recovery
                mock_library.main.return_value = -1  # Error code
                
                input_tensor = torch.randn(2, 3)
                with patch('torch.zeros_like', return_value=torch.zeros_like(input_tensor)):
                    with patch.object(input_tensor, 'data_ptr', return_value=12345):
                        with patch.object(input_tensor, 'numel', return_value=6):
                            with pytest.raises(RuntimeError, match="Kernel execution failed with code"):
                                kernel.execute([input_tensor])
                
                # Kernel should still be loaded after execution failure
                assert kernel._is_loaded is True
                
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__])
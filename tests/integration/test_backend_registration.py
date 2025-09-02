"""
Integration tests for PyTorch backend registration and torch.compile compatibility.

These tests verify that the Conductor backend integrates properly with
PyTorch's compilation system and provides the expected API.
"""

import pytest
import torch
import warnings
from unittest.mock import patch, MagicMock

from conductor.backend import (
    register_backend, 
    is_backend_registered, 
    get_backend_info,
    list_supported_operations,
    get_backend,
    configure_backend,
    ConductorBackend
)
from conductor.utils.exceptions import ConductorError


class TestBackendRegistration:
    """Test backend registration functionality."""
    
    def test_backend_registration_success(self):
        """Test successful backend registration."""
        # Mock PyTorch's backend registration
        with patch('torch._dynamo.register_backend') as mock_register:
            with patch('torch._dynamo.list_backends') as mock_list:
                mock_list.return_value = {'gcu': MagicMock()}
                
                # Register backend
                register_backend()
                
                # Verify registration was called
                mock_register.assert_called_once()
                args, kwargs = mock_register.call_args
                assert kwargs['name'] == 'gcu'
                assert isinstance(kwargs['compiler_fn'], ConductorBackend)
    
    def test_backend_registration_pytorch_version_check(self):
        """Test PyTorch version compatibility check."""
        with patch('torch.__version__', '1.13.0'):
            with pytest.warns(UserWarning, match="PyTorch 2.0\\+ required"):
                register_backend()
    
    def test_backend_registration_no_dynamo_support(self):
        """Test handling when PyTorch doesn't support backend registration."""
        with patch('torch._dynamo', spec=[]):  # Remove register_backend attribute
            with pytest.warns(UserWarning, match="does not support backend registration"):
                register_backend()
    
    def test_is_backend_registered_true(self):
        """Test backend registration check when registered."""
        with patch('torch._dynamo.list_backends') as mock_list:
            mock_list.return_value = {'gcu': MagicMock(), 'inductor': MagicMock()}
            
            assert is_backend_registered() is True
    
    def test_is_backend_registered_false(self):
        """Test backend registration check when not registered."""
        with patch('torch._dynamo.list_backends') as mock_list:
            mock_list.return_value = {'inductor': MagicMock()}
            
            assert is_backend_registered() is False
    
    def test_is_backend_registered_exception_handling(self):
        """Test backend registration check with exceptions."""
        with patch('torch._dynamo.list_backends', side_effect=AttributeError()):
            assert is_backend_registered() is False
    
    def test_get_backend_info(self):
        """Test backend information retrieval."""
        with patch('conductor.backend.is_backend_registered', return_value=True):
            with patch('torch._dynamo.list_backends') as mock_list:
                mock_backend = MagicMock()
                mock_list.return_value = {'gcu': mock_backend}
                
                info = get_backend_info()
                
                assert info['name'] == 'gcu'
                assert info['registered'] is True
                assert 'pytorch_version' in info
                assert 'python_version' in info
                assert 'backend_function' in info
    
    def test_list_supported_operations(self):
        """Test listing supported operations."""
        operations = list_supported_operations()
        
        # Check that common operations are supported
        expected_ops = ['add', 'mul', 'relu', 'sum', 'matmul', 'reshape']
        for op in expected_ops:
            assert op in operations
        
        # Verify it returns a list
        assert isinstance(operations, list)
        assert len(operations) > 0


class TestConductorBackend:
    """Test ConductorBackend class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = ConductorBackend()
    
    def test_backend_initialization(self):
        """Test backend initialization."""
        assert self.backend.name == "gcu"
        assert self.backend._registered is False
        assert self.backend.enable_fusion is True
        assert self.backend.enable_fallback is True
        assert self.backend.compilation_mode == "jit"
    
    def test_set_compilation_mode_valid(self):
        """Test setting valid compilation modes."""
        self.backend.set_compilation_mode("aot")
        assert self.backend.compilation_mode == "aot"
        
        self.backend.set_compilation_mode("jit")
        assert self.backend.compilation_mode == "jit"
    
    def test_set_compilation_mode_invalid(self):
        """Test setting invalid compilation mode."""
        with pytest.raises(ValueError, match="Mode must be 'jit' or 'aot'"):
            self.backend.set_compilation_mode("invalid")
    
    def test_enable_fusion_optimization(self):
        """Test enabling/disabling fusion optimization."""
        self.backend.enable_fusion_optimization(False)
        assert self.backend.enable_fusion is False
        
        self.backend.enable_fusion_optimization(True)
        assert self.backend.enable_fusion is True
    
    def test_enable_fallback_mechanism(self):
        """Test enabling/disabling fallback mechanism."""
        self.backend.enable_fallback_mechanism(False)
        assert self.backend.enable_fallback is False
        
        self.backend.enable_fallback_mechanism(True)
        assert self.backend.enable_fallback is True


class TestTorchCompileIntegration:
    """Test torch.compile integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = ConductorBackend()
    
    def create_simple_model(self):
        """Create a simple model for testing."""
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x + 1.0)
        
        return SimpleModel()
    
    def create_fx_graph(self, model, example_input):
        """Create FX Graph from model."""
        # Use torch.fx.symbolic_trace to create FX Graph
        traced = torch.fx.symbolic_trace(model)
        return traced
    
    @patch('conductor.backend.ConductorBackend._compile_jit_mode')
    def test_backend_call_jit_mode(self, mock_jit_compile):
        """Test backend compilation in JIT mode."""
        # Setup mock
        mock_compiled_fn = MagicMock()
        mock_jit_compile.return_value = mock_compiled_fn
        
        # Create test model and graph
        model = self.create_simple_model()
        example_input = torch.randn(2, 3)
        fx_graph = self.create_fx_graph(model, example_input)
        
        # Set backend to JIT mode
        self.backend.set_compilation_mode("jit")
        
        # Call backend
        result = self.backend(fx_graph, [example_input])
        
        # Verify JIT compilation was called
        mock_jit_compile.assert_called_once()
        assert result == mock_compiled_fn
    
    @patch('conductor.backend.ConductorBackend._compile_aot_mode')
    @patch('conductor.backend.ConductorBackend._compile_jit_mode')
    def test_backend_call_aot_mode_success(self, mock_jit_compile, mock_aot_compile):
        """Test backend compilation in AOT mode with successful artifact loading."""
        # Setup mocks
        mock_compiled_fn = MagicMock()
        mock_aot_compile.return_value = mock_compiled_fn
        
        # Create test model and graph
        model = self.create_simple_model()
        example_input = torch.randn(2, 3)
        fx_graph = self.create_fx_graph(model, example_input)
        
        # Set backend to AOT mode
        self.backend.set_compilation_mode("aot")
        
        # Call backend
        result = self.backend(fx_graph, [example_input])
        
        # Verify AOT compilation was called, JIT was not
        mock_aot_compile.assert_called_once()
        mock_jit_compile.assert_not_called()
        assert result == mock_compiled_fn
    
    @patch('conductor.backend.ConductorBackend._compile_aot_mode')
    @patch('conductor.backend.ConductorBackend._compile_jit_mode')
    def test_backend_call_aot_fallback_to_jit(self, mock_jit_compile, mock_aot_compile):
        """Test AOT mode falling back to JIT when artifacts not found."""
        # Setup mocks
        mock_aot_compile.return_value = None  # AOT fails
        mock_jit_fn = MagicMock()
        mock_jit_compile.return_value = mock_jit_fn
        
        # Create test model and graph
        model = self.create_simple_model()
        example_input = torch.randn(2, 3)
        fx_graph = self.create_fx_graph(model, example_input)
        
        # Set backend to AOT mode
        self.backend.set_compilation_mode("aot")
        
        # Call backend
        result = self.backend(fx_graph, [example_input])
        
        # Verify both AOT and JIT were called
        mock_aot_compile.assert_called_once()
        mock_jit_compile.assert_called_once()
        assert result == mock_jit_fn
    
    @patch('conductor.backend.ConductorBackend._execute_fallback')
    def test_backend_fallback_on_unsupported_operation(self, mock_fallback):
        """Test fallback to Inductor on unsupported operations."""
        # Setup mock
        mock_fallback_fn = MagicMock()
        mock_fallback.return_value = mock_fallback_fn
        
        # Create test model and graph
        model = self.create_simple_model()
        example_input = torch.randn(2, 3)
        fx_graph = self.create_fx_graph(model, example_input)
        
        # Mock graph analyzer to raise UnsupportedOperationError
        with patch.object(self.backend.graph_analyzer, 'parse_fx_graph') as mock_parse:
            from conductor.utils.exceptions import UnsupportedOperationError
            mock_parse.side_effect = UnsupportedOperationError("test_op", "not implemented")
            
            # Call backend
            result = self.backend(fx_graph, [example_input])
            
            # Verify fallback was called
            mock_fallback.assert_called_once()
            args = mock_fallback.call_args[0]
            assert args[0] == fx_graph
            assert args[1] == [example_input]
            assert "test_op" in args[2]  # reason should contain operation name
            assert result == mock_fallback_fn
    
    def test_backend_fallback_disabled_raises_exception(self):
        """Test that disabling fallback raises exceptions on errors."""
        # Disable fallback
        self.backend.enable_fallback_mechanism(False)
        
        # Create test model and graph
        model = self.create_simple_model()
        example_input = torch.randn(2, 3)
        fx_graph = self.create_fx_graph(model, example_input)
        
        # Mock graph analyzer to raise exception
        with patch.object(self.backend.graph_analyzer, 'parse_fx_graph') as mock_parse:
            mock_parse.side_effect = Exception("Test error")
            
            # Call backend and expect exception
            with pytest.raises(ConductorError, match="Backend compilation failed"):
                self.backend(fx_graph, [example_input])
    
    @patch('torch.compile')
    def test_fallback_to_inductor_success(self, mock_torch_compile):
        """Test successful fallback to Inductor backend."""
        # Setup mock
        mock_compiled_fn = MagicMock()
        mock_torch_compile.return_value = mock_compiled_fn
        
        # Create test inputs
        model = self.create_simple_model()
        example_input = torch.randn(2, 3)
        fx_graph = self.create_fx_graph(model, example_input)
        
        # Call fallback
        result = self.backend._fallback_to_inductor(fx_graph, [example_input])
        
        # Verify torch.compile was called with inductor backend
        mock_torch_compile.assert_called_once_with(fx_graph, backend='inductor')
        assert result == mock_compiled_fn
    
    @patch('torch.compile', side_effect=Exception("Inductor failed"))
    def test_fallback_to_inductor_not_available(self, mock_torch_compile):
        """Test fallback when Inductor is not available."""
        # Create test inputs
        model = self.create_simple_model()
        example_input = torch.randn(2, 3)
        fx_graph = self.create_fx_graph(model, example_input)
        
        # Call fallback
        result = self.backend._fallback_to_inductor(fx_graph, [example_input])
        
        # Should return original forward function as final fallback
        assert result == fx_graph.forward
    
    def test_generate_graph_signature(self):
        """Test graph signature generation."""
        # Create test model and graph
        model = self.create_simple_model()
        example_input = torch.randn(2, 3)
        fx_graph = self.create_fx_graph(model, example_input)
        
        # Generate signature
        signature1 = self.backend._generate_graph_signature(fx_graph)
        signature2 = self.backend._generate_graph_signature(fx_graph)
        
        # Signatures should be consistent
        assert signature1 == signature2
        assert isinstance(signature1, str)
        assert len(signature1) == 64  # SHA256 hex digest length


class TestBackendConfiguration:
    """Test backend configuration functionality."""
    
    def test_configure_backend_success(self):
        """Test successful backend configuration."""
        mock_backend = MagicMock(spec=ConductorBackend)
        
        with patch('conductor.backend.get_backend', return_value=mock_backend):
            configure_backend(
                mode='aot',
                enable_fusion=False,
                enable_fallback=True
            )
            
            mock_backend.set_compilation_mode.assert_called_once_with('aot')
            mock_backend.enable_fusion_optimization.assert_called_once_with(False)
            mock_backend.enable_fallback_mechanism.assert_called_once_with(True)
    
    def test_configure_backend_not_registered(self):
        """Test configuration when backend is not registered."""
        with patch('conductor.backend.get_backend', return_value=None):
            with pytest.raises(RuntimeError, match="Backend not registered"):
                configure_backend(mode='jit')
    
    def test_get_backend_success(self):
        """Test getting backend instance when registered."""
        mock_backend = MagicMock(spec=ConductorBackend)
        
        with patch('conductor.backend.is_backend_registered', return_value=True):
            with patch('torch._dynamo.list_backends') as mock_list:
                mock_list.return_value = {'gcu': mock_backend}
                
                backend = get_backend()
                assert backend == mock_backend
    
    def test_get_backend_not_registered(self):
        """Test getting backend when not registered."""
        with patch('conductor.backend.is_backend_registered', return_value=False):
            with patch('conductor.backend._global_backend', None):
                backend = get_backend()
                assert backend is None


if __name__ == '__main__':
    pytest.main([__file__])
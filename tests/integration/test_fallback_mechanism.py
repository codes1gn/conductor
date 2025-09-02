"""
Integration tests for fallback mechanism to Inductor backend.

These tests verify that the Conductor backend gracefully falls back
to the Inductor backend when operations are unsupported or compilation fails.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from conductor.backend import ConductorBackend
from conductor.utils.exceptions import (
    UnsupportedOperationError,
    CompilationError,
    DeviceError,
    FallbackHandler,
    get_fallback_handler
)


class TestFallbackHandler:
    """Test FallbackHandler class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = FallbackHandler()
        # Reset statistics for clean test state
        self.handler.reset_stats()
    
    def test_should_fallback_unsupported_operation(self):
        """Test fallback decision for unsupported operations."""
        error = UnsupportedOperationError("custom_op", "not implemented")
        assert self.handler.should_fallback(error) is True
    
    def test_should_fallback_compilation_error(self):
        """Test fallback decision for compilation errors."""
        error = CompilationError("DSL compilation failed", "test_dsl", "error output")
        assert self.handler.should_fallback(error) is True
    
    def test_should_fallback_device_error(self):
        """Test fallback decision for device errors."""
        error = DeviceError("GCU device not available", device_id=0)
        assert self.handler.should_fallback(error) is True
    
    def test_should_not_fallback_other_errors(self):
        """Test fallback decision for other error types."""
        error = ValueError("Invalid argument")
        assert self.handler.should_fallback(error) is False
        
        error = RuntimeError("Generic runtime error")
        assert self.handler.should_fallback(error) is False
    
    @patch('torch.compile')
    def test_execute_fallback_success(self, mock_torch_compile):
        """Test successful fallback execution."""
        # Setup mock
        mock_compiled_fn = MagicMock()
        mock_torch_compile.return_value = mock_compiled_fn
        
        # Create test graph
        graph_module = MagicMock(spec=torch.fx.GraphModule)
        
        # Execute fallback
        result = self.handler.execute_fallback(graph_module, "test reason")
        
        # Verify torch.compile was called with inductor backend
        mock_torch_compile.assert_called_once_with(graph_module, backend='inductor')
        assert result == mock_compiled_fn
        
        # Check statistics
        stats = self.handler.get_fallback_stats()
        assert stats['total_fallbacks'] == 1
        assert stats['fallback_reasons']['test reason'] == 1
    
    @patch('torch.compile', side_effect=AttributeError("torch.compile not available"))
    def test_execute_fallback_no_torch_compile(self, mock_torch_compile):
        """Test fallback when torch.compile is not available."""
        # Create test graph
        graph_module = MagicMock(spec=torch.fx.GraphModule)
        graph_module.forward = MagicMock()
        
        # Execute fallback
        result = self.handler.execute_fallback(graph_module, "no torch.compile")
        
        # Should return original forward function
        assert result == graph_module.forward
        
        # Check statistics were updated
        stats = self.handler.get_fallback_stats()
        assert stats['total_fallbacks'] == 1
    
    @patch('torch.compile', side_effect=Exception("Compilation failed"))
    def test_execute_fallback_compilation_fails(self, mock_torch_compile):
        """Test fallback when Inductor compilation fails."""
        # Create test graph
        graph_module = MagicMock(spec=torch.fx.GraphModule)
        
        # Execute fallback should raise RuntimeError
        with pytest.raises(RuntimeError, match="Fallback compilation failed"):
            self.handler.execute_fallback(graph_module, "inductor fails")
    
    def test_fallback_statistics(self):
        """Test fallback statistics tracking."""
        # Initially no fallbacks
        stats = self.handler.get_fallback_stats()
        assert stats['total_fallbacks'] == 0
        assert stats['fallback_reasons'] == {}
        assert stats['most_common_reason'] is None
        
        # Mock torch.compile for successful fallbacks
        with patch('torch.compile', return_value=MagicMock()):
            graph_module = MagicMock(spec=torch.fx.GraphModule)
            
            # Execute multiple fallbacks
            self.handler.execute_fallback(graph_module, "reason1")
            self.handler.execute_fallback(graph_module, "reason2")
            self.handler.execute_fallback(graph_module, "reason1")
            
            # Check updated statistics
            stats = self.handler.get_fallback_stats()
            assert stats['total_fallbacks'] == 3
            assert stats['fallback_reasons']['reason1'] == 2
            assert stats['fallback_reasons']['reason2'] == 1
            assert stats['most_common_reason'] == 'reason1'
    
    def test_reset_statistics(self):
        """Test resetting fallback statistics."""
        # Mock torch.compile for successful fallback
        with patch('torch.compile', return_value=MagicMock()):
            graph_module = MagicMock(spec=torch.fx.GraphModule)
            self.handler.execute_fallback(graph_module, "test")
            
            # Verify statistics exist
            stats = self.handler.get_fallback_stats()
            assert stats['total_fallbacks'] == 1
            
            # Reset and verify
            self.handler.reset_stats()
            stats = self.handler.get_fallback_stats()
            assert stats['total_fallbacks'] == 0
            assert stats['fallback_reasons'] == {}


class TestBackendFallbackIntegration:
    """Test fallback integration in ConductorBackend."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = ConductorBackend()
        # Reset fallback statistics for clean test state
        self.backend.reset_fallback_stats()
    
    def create_simple_model(self):
        """Create a simple model for testing."""
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x + 1.0)
        
        return SimpleModel()
    
    def create_fx_graph(self, model, example_input):
        """Create FX Graph from model."""
        traced = torch.fx.symbolic_trace(model)
        return traced
    
    @patch('conductor.backend.ConductorBackend._execute_fallback')
    def test_fallback_on_unsupported_operation(self, mock_fallback):
        """Test fallback when unsupported operation is encountered."""
        # Setup mock
        mock_compiled_fn = MagicMock()
        mock_fallback.return_value = mock_compiled_fn
        
        # Create test inputs
        model = self.create_simple_model()
        example_input = torch.randn(2, 3)
        fx_graph = self.create_fx_graph(model, example_input)
        
        # Mock graph analyzer to raise UnsupportedOperationError
        with patch.object(self.backend.graph_analyzer, 'parse_fx_graph') as mock_parse:
            mock_parse.side_effect = UnsupportedOperationError("custom_op", "not implemented")
            
            # Call backend
            result = self.backend(fx_graph, [example_input])
            
            # Verify fallback was called
            mock_fallback.assert_called_once()
            args = mock_fallback.call_args[0]
            assert args[0] == fx_graph
            assert args[1] == [example_input]
            assert "custom_op" in args[2]  # reason should contain operation name
            
            assert result == mock_compiled_fn
    
    @patch('conductor.backend.ConductorBackend._execute_fallback')
    def test_fallback_on_compilation_error(self, mock_fallback):
        """Test fallback when DSL compilation fails."""
        # Setup mock
        mock_compiled_fn = MagicMock()
        mock_fallback.return_value = mock_compiled_fn
        
        # Create test inputs
        model = self.create_simple_model()
        example_input = torch.randn(2, 3)
        fx_graph = self.create_fx_graph(model, example_input)
        
        # Mock DSL generator to raise CompilationError
        with patch.object(self.backend.dsl_generator, 'generate_dsl_file') as mock_dsl:
            mock_dsl.side_effect = CompilationError("DSL generation failed", "test_dsl", "error")
            
            # Call backend
            result = self.backend(fx_graph, [example_input])
            
            # Verify fallback was called
            mock_fallback.assert_called_once()
            assert result == mock_compiled_fn
    
    @patch('conductor.backend.ConductorBackend._execute_fallback')
    def test_fallback_on_device_error(self, mock_fallback):
        """Test fallback when device error occurs."""
        # Setup mock
        mock_compiled_fn = MagicMock()
        mock_fallback.return_value = mock_compiled_fn
        
        # Create test inputs
        model = self.create_simple_model()
        example_input = torch.randn(2, 3)
        fx_graph = self.create_fx_graph(model, example_input)
        
        # Mock JIT compiler to raise DeviceError
        with patch.object(self.backend, '_compile_jit_mode') as mock_jit:
            mock_jit.side_effect = DeviceError("GCU device not available")
            
            # Call backend
            result = self.backend(fx_graph, [example_input])
            
            # Verify fallback was called
            mock_fallback.assert_called_once()
            assert result == mock_compiled_fn
    
    def test_fallback_disabled_raises_exception(self):
        """Test that disabling fallback raises exceptions."""
        # Disable fallback
        self.backend.enable_fallback_mechanism(False)
        
        # Create test inputs
        model = self.create_simple_model()
        example_input = torch.randn(2, 3)
        fx_graph = self.create_fx_graph(model, example_input)
        
        # Mock to raise UnsupportedOperationError
        with patch.object(self.backend.graph_analyzer, 'parse_fx_graph') as mock_parse:
            mock_parse.side_effect = UnsupportedOperationError("custom_op", "not implemented")
            
            # Should raise the original exception
            with pytest.raises(UnsupportedOperationError):
                self.backend(fx_graph, [example_input])
    
    @patch('torch.compile')
    def test_execute_fallback_success(self, mock_torch_compile):
        """Test successful fallback execution."""
        # Setup mock
        mock_compiled_fn = MagicMock()
        mock_torch_compile.return_value = mock_compiled_fn
        
        # Create test inputs
        model = self.create_simple_model()
        example_input = torch.randn(2, 3)
        fx_graph = self.create_fx_graph(model, example_input)
        
        # Execute fallback
        result = self.backend._execute_fallback(fx_graph, [example_input], "test reason")
        
        # Verify torch.compile was called
        mock_torch_compile.assert_called_once_with(fx_graph, backend='inductor')
        assert result == mock_compiled_fn
    
    @patch('torch.compile', side_effect=Exception("Inductor failed"))
    def test_execute_fallback_final_fallback(self, mock_torch_compile):
        """Test final fallback to eager execution."""
        # Create test inputs
        model = self.create_simple_model()
        example_input = torch.randn(2, 3)
        fx_graph = self.create_fx_graph(model, example_input)
        
        # Execute fallback
        result = self.backend._execute_fallback(fx_graph, [example_input], "inductor fails")
        
        # Should return original forward function as final fallback
        assert result == fx_graph.forward
    
    def test_get_fallback_stats(self):
        """Test getting fallback statistics from backend."""
        # Initially no fallbacks
        stats = self.backend.get_fallback_stats()
        assert stats['total_fallbacks'] == 0
        
        # Mock successful fallback
        with patch('torch.compile', return_value=MagicMock()):
            model = self.create_simple_model()
            example_input = torch.randn(2, 3)
            fx_graph = self.create_fx_graph(model, example_input)
            
            # Execute fallback
            self.backend._execute_fallback(fx_graph, [example_input], "test")
            
            # Check statistics
            stats = self.backend.get_fallback_stats()
            assert stats['total_fallbacks'] == 1
    
    def test_reset_fallback_stats(self):
        """Test resetting fallback statistics."""
        # Execute a fallback first
        with patch('torch.compile', return_value=MagicMock()):
            model = self.create_simple_model()
            example_input = torch.randn(2, 3)
            fx_graph = self.create_fx_graph(model, example_input)
            
            self.backend._execute_fallback(fx_graph, [example_input], "test")
            
            # Verify statistics exist
            stats = self.backend.get_fallback_stats()
            assert stats['total_fallbacks'] == 1
            
            # Reset and verify
            self.backend.reset_fallback_stats()
            stats = self.backend.get_fallback_stats()
            assert stats['total_fallbacks'] == 0


class TestFallbackScenarios:
    """Test various fallback scenarios and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = ConductorBackend()
        # Reset fallback statistics for clean test state
        self.backend.reset_fallback_stats()
    
    def create_complex_model(self):
        """Create a more complex model with potential unsupported operations."""
        class ComplexModel(torch.nn.Module):
            def forward(self, x):
                # Mix of supported and potentially unsupported operations
                x = torch.relu(x)
                x = torch.add(x, 1.0)
                x = torch.sum(x, dim=-1)
                return x
        
        return ComplexModel()
    
    @patch('conductor.codegen.graph.GraphAnalyzer.parse_fx_graph')
    @patch('torch.compile')
    def test_graph_parsing_failure_fallback(self, mock_torch_compile, mock_parse):
        """Test fallback when FX Graph parsing fails."""
        # Setup mocks
        mock_parse.side_effect = Exception("Graph parsing failed")
        mock_compiled_fn = MagicMock()
        mock_torch_compile.return_value = mock_compiled_fn
        
        # Create test inputs
        model = self.create_complex_model()
        example_input = torch.randn(4, 8)
        fx_graph = torch.fx.symbolic_trace(model)
        
        # Call backend
        result = self.backend(fx_graph, [example_input])
        
        # Should fallback to Inductor
        mock_torch_compile.assert_called_once_with(fx_graph, backend='inductor')
        assert result == mock_compiled_fn
    
    @patch('conductor.codegen.fusion.FusionEngine.identify_fusion_opportunities')
    @patch('torch.compile')
    def test_fusion_failure_fallback(self, mock_torch_compile, mock_fusion):
        """Test fallback when fusion optimization fails."""
        # Setup mocks
        mock_fusion.side_effect = Exception("Fusion failed")
        mock_compiled_fn = MagicMock()
        mock_torch_compile.return_value = mock_compiled_fn
        
        # Create test inputs
        model = self.create_complex_model()
        example_input = torch.randn(4, 8)
        fx_graph = torch.fx.symbolic_trace(model)
        
        # Call backend
        result = self.backend(fx_graph, [example_input])
        
        # Should fallback to Inductor
        mock_torch_compile.assert_called_once_with(fx_graph, backend='inductor')
        assert result == mock_compiled_fn
    
    @patch('conductor.codegen.dsl.DSLGenerator.generate_dsl_file')
    @patch('torch.compile')
    def test_dsl_generation_failure_fallback(self, mock_torch_compile, mock_dsl):
        """Test fallback when DSL generation fails."""
        # Setup mocks
        mock_dsl.side_effect = Exception("DSL generation failed")
        mock_compiled_fn = MagicMock()
        mock_torch_compile.return_value = mock_compiled_fn
        
        # Create test inputs
        model = self.create_complex_model()
        example_input = torch.randn(4, 8)
        fx_graph = torch.fx.symbolic_trace(model)
        
        # Call backend
        result = self.backend(fx_graph, [example_input])
        
        # Should fallback to Inductor
        mock_torch_compile.assert_called_once_with(fx_graph, backend='inductor')
        assert result == mock_compiled_fn
    
    def test_multiple_fallback_attempts(self):
        """Test multiple fallback attempts with statistics tracking."""
        with patch('torch.compile', return_value=MagicMock()) as mock_torch_compile:
            model = self.create_complex_model()
            example_input = torch.randn(4, 8)
            fx_graph = torch.fx.symbolic_trace(model)
            
            # Mock different failure scenarios
            scenarios = [
                UnsupportedOperationError("op1", "not supported"),
                CompilationError("DSL failed", "dsl", "error"),
                DeviceError("Device unavailable"),
            ]
            
            for i, error in enumerate(scenarios):
                with patch.object(self.backend.graph_analyzer, 'parse_fx_graph', side_effect=error):
                    self.backend(fx_graph, [example_input])
            
            # Check statistics
            stats = self.backend.get_fallback_stats()
            assert stats['total_fallbacks'] == 3
            
            # Verify torch.compile was called for each fallback
            assert mock_torch_compile.call_count == 3


class TestGlobalFallbackHandler:
    """Test global fallback handler functionality."""
    
    def test_get_fallback_handler_singleton(self):
        """Test that get_fallback_handler returns the same instance."""
        handler1 = get_fallback_handler()
        handler2 = get_fallback_handler()
        
        assert handler1 is handler2
        assert isinstance(handler1, FallbackHandler)
    
    def test_fallback_handler_shared_state(self):
        """Test that fallback statistics are shared across instances."""
        # Get handler and execute fallback
        handler = get_fallback_handler()
        
        with patch('torch.compile', return_value=MagicMock()):
            graph_module = MagicMock(spec=torch.fx.GraphModule)
            handler.execute_fallback(graph_module, "shared test")
        
        # Create backend and check if it sees the same statistics
        backend = ConductorBackend()
        stats = backend.get_fallback_stats()
        
        assert stats['total_fallbacks'] >= 1
        assert 'shared test' in stats['fallback_reasons']


if __name__ == '__main__':
    pytest.main([__file__])
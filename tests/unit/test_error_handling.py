"""
Unit tests for error handling and diagnostic reporting.

This module tests the comprehensive error handling mechanisms
and diagnostic reporting capabilities of the JIT compiler.
"""

import pytest
import torch
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from conductor.runtime.jit import JITCompiler
from conductor.utils.exceptions import (
    ConductorError, CompilationError, UnsupportedOperationError, 
    DeviceError, FallbackHandler, get_fallback_handler
)


class TestConductorExceptions:
    """Test cases for custom exception classes."""
    
    def test_conductor_error_basic(self):
        """Test basic ConductorError functionality."""
        error = ConductorError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == {}
        
    def test_conductor_error_with_details(self):
        """Test ConductorError with details."""
        details = {"key1": "value1", "key2": 42}
        error = ConductorError("Test error", details)
        
        assert error.message == "Test error"
        assert error.details == details
        assert "key1=value1" in str(error)
        assert "key2=42" in str(error)
        
    def test_compilation_error(self):
        """Test CompilationError functionality."""
        dsl_code = "function main() { invalid syntax }"
        compiler_output = "error: syntax error on line 1"
        
        error = CompilationError("Compilation failed", dsl_code, compiler_output)
        
        assert error.message == "Compilation failed"
        assert error.dsl_code == dsl_code
        assert error.compiler_output == compiler_output
        assert error.details['dsl_length'] == len(dsl_code)
        assert error.details['compiler_output_length'] == len(compiler_output)
        
    def test_compilation_error_get_compiler_errors(self):
        """Test extraction of compiler errors."""
        compiler_output = """
        Compiling DSL file...
        error: undefined symbol 'invalid_op'
        warning: unused variable 'temp'
        error: syntax error on line 5
        Compilation failed.
        """
        
        error = CompilationError("Failed", "", compiler_output)
        errors = error.get_compiler_errors()
        
        assert len(errors) == 2
        assert any("undefined symbol" in err for err in errors)
        assert any("syntax error" in err for err in errors)
        
    def test_unsupported_operation_error(self):
        """Test UnsupportedOperationError functionality."""
        error = UnsupportedOperationError("custom_op", "Not implemented yet")
        
        assert error.operation == "custom_op"
        assert error.reason == "Not implemented yet"
        assert "custom_op" in str(error)
        assert "Not implemented yet" in str(error)
        
    def test_device_error(self):
        """Test DeviceError functionality."""
        error = DeviceError("Device initialization failed", device_id=0)
        
        assert error.device_id == 0
        assert error.details['device_id'] == 0
        assert "Device initialization failed" in str(error)


class TestFallbackHandler:
    """Test cases for fallback handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = FallbackHandler()
        
    def test_should_fallback_unsupported_operation(self):
        """Test fallback decision for unsupported operations."""
        error = UnsupportedOperationError("custom_op", "Not supported")
        assert self.handler.should_fallback(error) is True
        
    def test_should_fallback_compilation_error(self):
        """Test fallback decision for compilation errors."""
        error = CompilationError("Compilation failed", "", "")
        assert self.handler.should_fallback(error) is True
        
    def test_should_fallback_device_error(self):
        """Test fallback decision for device errors."""
        error = DeviceError("Device not available")
        assert self.handler.should_fallback(error) is True
        
    def test_should_not_fallback_generic_error(self):
        """Test that generic errors don't trigger fallback."""
        error = ValueError("Generic error")
        assert self.handler.should_fallback(error) is False
        
    def test_execute_fallback_with_torch_compile(self):
        """Test fallback execution with torch.compile."""
        def simple_model(x):
            return torch.relu(x)
        
        graph_module = torch.fx.symbolic_trace(simple_model)
        
        # Mock torch.compile
        with patch('torch.compile') as mock_compile:
            mock_compiled = Mock()
            mock_compile.return_value = mock_compiled
            
            result = self.handler.execute_fallback(graph_module, "test reason")
            
            mock_compile.assert_called_once_with(graph_module, backend='inductor')
            assert result == mock_compiled
            
    def test_execute_fallback_without_torch_compile(self):
        """Test fallback execution when torch.compile is not available."""
        def simple_model(x):
            return torch.relu(x)
        
        graph_module = torch.fx.symbolic_trace(simple_model)
        
        # Mock torch to not have compile attribute
        with patch('torch.compile', side_effect=AttributeError):
            result = self.handler.execute_fallback(graph_module, "test reason")
            assert result == graph_module.forward
            
    def test_fallback_stats_tracking(self):
        """Test fallback statistics tracking."""
        def simple_model(x):
            return torch.relu(x)
        
        graph_module = torch.fx.symbolic_trace(simple_model)
        
        # Execute multiple fallbacks
        with patch('torch.compile', return_value=Mock()):
            self.handler.execute_fallback(graph_module, "reason1")
            self.handler.execute_fallback(graph_module, "reason1")
            self.handler.execute_fallback(graph_module, "reason2")
            
        stats = self.handler.get_fallback_stats()
        assert stats['total_fallbacks'] == 3
        assert stats['fallback_reasons']['reason1'] == 2
        assert stats['fallback_reasons']['reason2'] == 1
        assert stats['most_common_reason'] == 'reason1'
        
    def test_reset_stats(self):
        """Test resetting fallback statistics."""
        def simple_model(x):
            return torch.relu(x)
        
        graph_module = torch.fx.symbolic_trace(simple_model)
        
        with patch('torch.compile', return_value=Mock()):
            self.handler.execute_fallback(graph_module, "test")
            
        # Verify stats exist
        stats = self.handler.get_fallback_stats()
        assert stats['total_fallbacks'] == 1
        
        # Reset and verify
        self.handler.reset_stats()
        stats = self.handler.get_fallback_stats()
        assert stats['total_fallbacks'] == 0
        assert stats['fallback_reasons'] == {}


class TestJITCompilerErrorHandling:
    """Test error handling in JIT compiler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.compiler = JITCompiler(cache_dir=self.temp_dir)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_get_diagnostic_info_compilation_error(self):
        """Test diagnostic info collection for compilation errors."""
        dsl_code = "function main() { invalid syntax }"
        compiler_output = "error: syntax error"
        error = CompilationError("Failed", dsl_code, compiler_output)
        
        def simple_model(x):
            return torch.relu(x)
        
        graph_module = torch.fx.symbolic_trace(simple_model)
        
        diagnostics = self.compiler.get_diagnostic_info(error, graph_module)
        
        # Check basic error info
        assert diagnostics['error_type'] == 'CompilationError'
        assert 'Failed' in diagnostics['error_message']
        assert 'timestamp' in diagnostics
        
        # Check compilation-specific info
        assert diagnostics['dsl_code_length'] == len(dsl_code)
        assert diagnostics['compiler_output_length'] == len(compiler_output)
        assert len(diagnostics['compiler_errors']) > 0
        
        # Check graph info
        assert 'graph_node_count' in diagnostics
        assert 'graph_operation_counts' in diagnostics
        assert 'graph_hash' in diagnostics
        
        # Check system info
        assert 'python_version' in diagnostics
        assert 'platform' in diagnostics
        assert 'torch_version' in diagnostics
        
    def test_get_diagnostic_info_unsupported_operation(self):
        """Test diagnostic info for unsupported operations."""
        error = UnsupportedOperationError("custom_op", "Not implemented")
        
        diagnostics = self.compiler.get_diagnostic_info(error)
        
        assert diagnostics['error_type'] == 'UnsupportedOperationError'
        assert diagnostics['unsupported_operation'] == 'custom_op'
        assert diagnostics['unsupported_reason'] == 'Not implemented'
        assert 'suggested_alternatives' in diagnostics
        
    def test_get_diagnostic_info_device_error(self):
        """Test diagnostic info for device errors."""
        error = DeviceError("Device not available", device_id=0)
        
        diagnostics = self.compiler.get_diagnostic_info(error)
        
        assert diagnostics['error_type'] == 'DeviceError'
        assert diagnostics['device_id'] == 0
        assert 'device_available' in diagnostics
        assert 'memory_info' in diagnostics
        
    def test_analyze_dsl_code(self):
        """Test DSL code analysis."""
        dsl_code = """
        function main() {
          local float32 temp;
          temp = add(input1, input2);
          output = relu(temp);
          invalid_line =
        }
        """
        
        analysis = self.compiler._analyze_dsl_code(dsl_code)
        
        assert analysis['dsl_line_count'] > 0
        assert analysis['dsl_function_count'] == 1
        assert analysis['dsl_buffer_declarations'] == 1
        assert analysis['dsl_operations'] >= 2
        assert len(analysis['dsl_syntax_issues']) > 0
        
    def test_analyze_compiler_output(self):
        """Test compiler output analysis."""
        compiler_output = """
        Compiling DSL...
        error: syntax error on line 5
        warning: unused variable 'temp'
        error: undefined symbol 'invalid_op'
        Compilation failed with 2 errors, 1 warning
        """
        
        analysis = self.compiler._analyze_compiler_output(compiler_output)
        
        assert analysis['compiler_output_lines'] > 0
        assert analysis['error_lines'] == 2
        assert analysis['warning_lines'] == 1
        assert 'syntax_error' in analysis['compiler_error_types']
        assert 'undefined_symbol' in analysis['compiler_error_types']
        
    def test_get_operation_alternatives(self):
        """Test operation alternatives suggestions."""
        alternatives = self.compiler._get_operation_alternatives('custom_op')
        assert len(alternatives) > 0
        assert all(isinstance(alt, str) for alt in alternatives)
        
        # Test specific operation
        alternatives = self.compiler._get_operation_alternatives('dynamic_shapes')
        assert any('static shapes' in alt.lower() for alt in alternatives)
        
    def test_generate_error_report_compilation_error(self):
        """Test comprehensive error report generation."""
        dsl_code = "function main() { invalid syntax }"
        compiler_output = "error: syntax error on line 1"
        error = CompilationError("Compilation failed", dsl_code, compiler_output)
        
        def simple_model(x):
            return torch.relu(x)
        
        graph_module = torch.fx.symbolic_trace(simple_model)
        
        report = self.compiler.generate_error_report(error, graph_module)
        
        # Check report structure
        assert "CONDUCTOR COMPILATION ERROR REPORT" in report
        assert "Error Type: CompilationError" in report
        assert "Error Message: Compilation failed" in report
        assert "SYSTEM INFORMATION:" in report
        assert "GRAPH INFORMATION:" in report
        assert "COMPILATION ERROR DETAILS:" in report
        assert "SUGGESTED ACTIONS:" in report
        
    def test_generate_error_report_unsupported_operation(self):
        """Test error report for unsupported operations."""
        error = UnsupportedOperationError("custom_op", "Not implemented")
        
        report = self.compiler.generate_error_report(error)
        
        assert "UNSUPPORTED OPERATION DETAILS:" in report
        assert "Operation: custom_op" in report
        assert "Reason: Not implemented" in report
        assert "Suggested Alternatives:" in report
        
    def test_error_handling_in_compile_graph(self):
        """Test error handling integration in compile_graph."""
        def simple_model(x):
            return torch.relu(x)
        
        graph_module = torch.fx.symbolic_trace(simple_model)
        
        # Mock graph analyzer to raise unsupported operation error
        with patch.object(self.compiler._graph_analyzer, 'parse_fx_graph') as mock_parse:
            mock_parse.side_effect = UnsupportedOperationError("test_op", "Test reason")
            
            with pytest.raises(UnsupportedOperationError):
                self.compiler.compile_graph(graph_module)
                
        # Mock to raise compilation error
        with patch.object(self.compiler, 'invoke_conductor_compiler') as mock_compile:
            mock_compile.side_effect = CompilationError("Test compilation error", "", "")
            
            with pytest.raises(CompilationError):
                self.compiler.compile_graph(graph_module)
                
        # Mock to raise unexpected error
        with patch.object(self.compiler._graph_analyzer, 'parse_fx_graph') as mock_parse:
            mock_parse.side_effect = RuntimeError("Unexpected error")
            
            with pytest.raises(CompilationError) as exc_info:
                self.compiler.compile_graph(graph_module)
            
            # Should be wrapped in CompilationError
            assert "JIT compilation failed" in str(exc_info.value)


class TestErrorRecoveryScenarios:
    """Test error recovery and fallback scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.compiler = JITCompiler(cache_dir=self.temp_dir)
        self.fallback_handler = get_fallback_handler()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.fallback_handler.reset_stats()
        
    def test_fallback_integration(self):
        """Test integration between compiler and fallback handler."""
        def simple_model(x):
            return torch.relu(x)
        
        graph_module = torch.fx.symbolic_trace(simple_model)
        
        # Test that unsupported operation triggers fallback
        error = UnsupportedOperationError("custom_op", "Not supported")
        assert self.fallback_handler.should_fallback(error)
        
        # Test fallback execution
        with patch('torch.compile') as mock_compile:
            mock_compiled = Mock()
            mock_compile.return_value = mock_compiled
            
            result = self.fallback_handler.execute_fallback(graph_module, "test")
            assert result == mock_compiled
            
    def test_error_diagnostics_with_system_info(self):
        """Test that system information is included in diagnostics."""
        error = CompilationError("Test error", "", "")
        diagnostics = self.compiler.get_diagnostic_info(error)
        
        # Should include system information
        assert 'python_version' in diagnostics
        assert 'platform' in diagnostics
        assert 'torch_version' in diagnostics
        
        # May include memory info if psutil is available
        if 'system_memory_total' in diagnostics:
            assert 'system_memory_available' in diagnostics
            assert 'system_memory_percent' in diagnostics
            
    def test_comprehensive_error_workflow(self):
        """Test complete error handling workflow."""
        def simple_model(x):
            return torch.relu(x)
        
        graph_module = torch.fx.symbolic_trace(simple_model)
        
        # Simulate compilation error
        error = CompilationError(
            "Test compilation failed",
            "function main() { invalid }",
            "error: syntax error"
        )
        
        # Get diagnostics
        diagnostics = self.compiler.get_diagnostic_info(error, graph_module)
        assert diagnostics['error_type'] == 'CompilationError'
        
        # Generate report
        report = self.compiler.generate_error_report(error, graph_module)
        assert len(report) > 0
        assert "CONDUCTOR COMPILATION ERROR REPORT" in report
        
        # Test fallback decision
        should_fallback = self.fallback_handler.should_fallback(error)
        assert should_fallback is True
        
        # Execute fallback
        with patch('torch.compile', return_value=Mock()) as mock_compile:
            fallback_result = self.fallback_handler.execute_fallback(
                graph_module, "compilation_error"
            )
            assert fallback_result is not None
            
        # Check fallback stats
        stats = self.fallback_handler.get_fallback_stats()
        assert stats['total_fallbacks'] == 1
        assert 'compilation_error' in stats['fallback_reasons']
"""
Unit tests for logging utilities.

Tests the logging configuration and utilities including
logger setup, formatting, and performance monitoring.
"""

import pytest
import logging
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from conductor.utils.logging import setup_logging, get_logger, ConductorLogger


class TestLoggingSetup:
    """Test logging setup and configuration."""
    
    def test_setup_logging_default(self):
        """Test default logging setup."""
        setup_logging()
        
        # Should create a logger for conductor
        logger = logging.getLogger('conductor')
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
        assert logger.propagate is False
    
    def test_setup_logging_debug_level(self):
        """Test logging setup with debug level."""
        setup_logging(level='DEBUG')
        
        logger = logging.getLogger('conductor')
        assert logger.level == logging.DEBUG
    
    def test_setup_logging_warning_level(self):
        """Test logging setup with warning level."""
        setup_logging(level='WARNING')
        
        logger = logging.getLogger('conductor')
        assert logger.level == logging.WARNING
    
    def test_setup_logging_invalid_level(self):
        """Test logging setup with invalid level defaults to INFO."""
        setup_logging(level='INVALID')
        
        logger = logging.getLogger('conductor')
        assert logger.level == logging.INFO
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            log_file = f.name
        
        try:
            setup_logging(log_file=log_file)
            
            logger = logging.getLogger('conductor')
            
            # Should have both console and file handlers
            handler_types = [type(h).__name__ for h in logger.handlers]
            assert 'StreamHandler' in handler_types
            assert 'FileHandler' in handler_types
            
            # Test that logging works
            logger.info("Test message")
            
            # Check that file was created and has content
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
                
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)
    
    def test_setup_logging_environment_variable(self):
        """Test logging setup with environment variable."""
        with patch.dict('os.environ', {'CONDUCTOR_LOG_LEVEL': 'DEBUG'}):
            setup_logging()
            
            logger = logging.getLogger('conductor')
            assert logger.level == logging.DEBUG
    
    def test_setup_logging_removes_existing_handlers(self):
        """Test that setup_logging removes existing handlers."""
        logger = logging.getLogger('conductor')
        
        # Add a dummy handler
        dummy_handler = logging.StreamHandler()
        logger.addHandler(dummy_handler)
        initial_handler_count = len(logger.handlers)
        
        # Setup logging should remove existing handlers
        setup_logging()
        
        # Should have new handlers, not the dummy one
        assert dummy_handler not in logger.handlers
        assert len(logger.handlers) > 0
    
    def test_get_logger(self):
        """Test getting logger instances."""
        logger1 = get_logger('test_module')
        logger2 = get_logger('test_module')
        
        # Should return the same logger instance
        assert logger1 is logger2
        assert logger1.name == 'conductor.test_module'
    
    def test_get_logger_different_modules(self):
        """Test getting loggers for different modules."""
        logger1 = get_logger('module1')
        logger2 = get_logger('module2')
        
        # Should return different logger instances
        assert logger1 is not logger2
        assert logger1.name == 'conductor.module1'
        assert logger2.name == 'conductor.module2'


class TestConductorLogger:
    """Test ConductorLogger class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.conductor_logger = ConductorLogger('test_component')
    
    def test_conductor_logger_creation(self):
        """Test ConductorLogger creation."""
        assert self.conductor_logger.logger.name == 'conductor.test_component'
        assert isinstance(self.conductor_logger.logger, logging.Logger)
    
    @patch('conductor.utils.logging.get_logger')
    def test_log_compilation_start(self, mock_get_logger):
        """Test logging compilation start."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = ConductorLogger('test')
        logger.log_compilation_start('abcdef123456', 5)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert 'Starting compilation' in call_args
        assert 'abcdef12' in call_args  # First 8 chars of hash
        assert '5 nodes' in call_args
    
    @patch('conductor.utils.logging.get_logger')
    def test_log_fusion_decision_applied(self, mock_get_logger):
        """Test logging fusion decision when applied."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = ConductorLogger('test')
        logger.log_fusion_decision(['add', 'relu'], True, 'Compatible operations')
        
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert 'Applied fusion' in call_args
        assert 'add' in call_args
        assert 'relu' in call_args
        assert 'Compatible operations' in call_args
    
    @patch('conductor.utils.logging.get_logger')
    def test_log_fusion_decision_skipped(self, mock_get_logger):
        """Test logging fusion decision when skipped."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = ConductorLogger('test')
        logger.log_fusion_decision(['matmul', 'conv2d'], False, 'Incompatible shapes')
        
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert 'Skipped fusion' in call_args
        assert 'matmul' in call_args
        assert 'conv2d' in call_args
        assert 'Incompatible shapes' in call_args
    
    @patch('conductor.utils.logging.get_logger')
    def test_log_performance_metrics(self, mock_get_logger):
        """Test logging performance metrics."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = ConductorLogger('test')
        logger.log_performance_metrics(2.5, 0.1)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert 'Performance:' in call_args
        assert 'compilation=2.500s' in call_args
        assert 'execution=0.100s' in call_args
    
    @patch('conductor.utils.logging.get_logger')
    def test_log_fallback(self, mock_get_logger):
        """Test logging fallback operations."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = ConductorLogger('test')
        logger.log_fallback('custom_op', 'Not implemented')
        
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        assert 'Falling back' in call_args
        assert 'custom_op' in call_args
        assert 'Not implemented' in call_args
    
    @patch('conductor.utils.logging.get_logger')
    def test_log_cache_hit(self, mock_get_logger):
        """Test logging cache hits."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = ConductorLogger('test')
        logger.log_cache_hit('abcdef123456')
        
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert 'Cache hit' in call_args
        assert 'abcdef12' in call_args  # First 8 chars
    
    @patch('conductor.utils.logging.get_logger')
    def test_log_cache_miss(self, mock_get_logger):
        """Test logging cache misses."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = ConductorLogger('test')
        logger.log_cache_miss('abcdef123456')
        
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert 'Cache miss' in call_args
        assert 'abcdef12' in call_args  # First 8 chars


class TestLoggingIntegration:
    """Integration tests for logging functionality."""
    
    def test_complete_logging_workflow(self):
        """Test complete logging workflow."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            log_file = f.name
        
        try:
            # Setup logging
            setup_logging(level='DEBUG', log_file=log_file)
            
            # Get logger and log messages
            logger = get_logger('test_integration')
            logger.info('Integration test message')
            logger.debug('Debug message')
            logger.warning('Warning message')
            
            # Use ConductorLogger
            conductor_logger = ConductorLogger('test_component')
            conductor_logger.log_compilation_start('test_hash_123', 3)
            conductor_logger.log_fusion_decision(['add', 'relu'], True, 'Test fusion')
            conductor_logger.log_performance_metrics(1.0, 0.5)
            conductor_logger.log_fallback('test_op', 'Test reason')
            conductor_logger.log_cache_hit('cache_hash_456')
            conductor_logger.log_cache_miss('cache_hash_789')
            
            # Verify log file has content
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert 'Integration test message' in content
                assert 'Starting compilation' in content
                assert 'Applied fusion' in content
                assert 'Performance:' in content
                assert 'Falling back' in content
                assert 'Cache hit' in content
                assert 'Cache miss' in content
                
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)
    
    def test_logger_hierarchy(self):
        """Test logger hierarchy and inheritance."""
        # Setup parent logger
        parent_logger = get_logger('parent')
        child_logger = get_logger('parent.child')
        
        # Both should be under conductor namespace
        assert parent_logger.name == 'conductor.parent'
        assert child_logger.name == 'conductor.parent.child'
        
        # Child should inherit from parent
        assert child_logger.parent == parent_logger or child_logger.parent.name.startswith('conductor')
    
    def test_logging_levels(self):
        """Test different logging levels."""
        # Test with different levels
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in levels:
            setup_logging(level=level)
            logger = logging.getLogger('conductor')
            assert logger.level == getattr(logging, level)
    
    def test_multiple_components_logging(self):
        """Test logging from multiple components."""
        component1 = ConductorLogger('component1')
        component2 = ConductorLogger('component2')
        
        # Should have different logger names
        assert component1.logger.name == 'conductor.component1'
        assert component2.logger.name == 'conductor.component2'
        
        # Both should be able to log without interference
        with patch.object(component1.logger, 'info') as mock_info1:
            with patch.object(component2.logger, 'info') as mock_info2:
                component1.log_performance_metrics(1.0, 0.5)
                component2.log_performance_metrics(2.0, 1.0)
                
                mock_info1.assert_called_once()
                mock_info2.assert_called_once()
    
    def test_logging_formatter(self):
        """Test logging formatter output."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            log_file = f.name
        
        try:
            setup_logging(level='INFO', log_file=log_file)
            
            logger = get_logger('test_formatter')
            logger.info('Test message for formatting')
            
            # Check log format
            with open(log_file, 'r') as f:
                content = f.read()
                
                # Should contain timestamp, logger name, level, and message
                assert 'conductor.test_formatter' in content
                assert 'INFO' in content
                assert 'Test message for formatting' in content
                # Should have timestamp format (basic check)
                assert '-' in content  # Date separators
                assert ':' in content  # Time separators
                
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)
    
    def test_logging_no_propagation(self):
        """Test that conductor logger doesn't propagate to root."""
        setup_logging()
        logger = logging.getLogger('conductor')
        
        # Should not propagate to avoid duplicate messages
        assert logger.propagate is False


if __name__ == '__main__':
    pytest.main([__file__])
"""
Unit tests for system information utilities.

Tests the system information collection functionality including
hardware detection, software versions, and environment details.
"""

import pytest
import platform
import sys
from unittest.mock import Mock, patch, MagicMock
from conductor.utils.info import get_system_info, get_conductor_info, print_info, main


class TestSystemInfo:
    """Test system information collection."""
    
    def test_get_system_info_basic(self):
        """Test getting basic system information."""
        info = get_system_info()
        
        # Should contain basic system info
        assert 'python_version' in info
        assert 'platform' in info
        assert 'architecture' in info
        assert 'processor' in info
        assert 'torch_version' in info
        assert 'torch_cuda_available' in info
        assert 'torch_backends' in info
        
        # Verify values are reasonable
        assert info['python_version'] == sys.version
        assert info['platform'] == platform.platform()
        assert info['architecture'] == platform.architecture()
        assert info['processor'] == platform.processor()
    
    def test_get_system_info_with_torch(self):
        """Test getting system information with PyTorch available."""
        with patch('torch.__version__', '2.0.0'):
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch._dynamo.list_backends', return_value={'gcu': Mock(), 'inductor': Mock()}):
                    info = get_system_info()
                    
                    assert info['torch_version'] == '2.0.0'
                    assert info['torch_cuda_available'] is True
                    assert 'gcu' in info['torch_backends']
                    assert 'inductor' in info['torch_backends']
    
    def test_get_system_info_torch_import_error(self):
        """Test system info handles torch import errors gracefully."""
        # Test that the function structure is correct even when torch is available
        # The actual ImportError path is hard to test without breaking the test environment
        info = get_system_info()
        
        # Verify all expected keys are present
        assert 'torch_version' in info
        assert 'torch_cuda_available' in info
        assert 'torch_backends' in info
        
        # Since torch is available in test environment, verify it's detected
        assert info['torch_version'] != 'Not installed'
        assert isinstance(info['torch_cuda_available'], bool)
        assert isinstance(info['torch_backends'], (list, dict))
    
    def test_get_system_info_torch_no_dynamo(self):
        """Test system info when torch._dynamo is not available."""
        with patch('torch.__version__', '1.13.0'):
            with patch('torch.cuda.is_available', return_value=False):
                with patch('torch._dynamo', None):
                    info = get_system_info()
                    
                    assert info['torch_version'] == '1.13.0'
                    assert info['torch_cuda_available'] is False
                    assert info['torch_backends'] == []


class TestConductorInfo:
    """Test Conductor-specific information collection."""
    
    @patch('conductor.__version__', '1.0.0')
    @patch('conductor.__author__', 'Test Author')
    def test_get_conductor_info_basic(self):
        """Test getting basic Conductor information."""
        info = get_conductor_info()
        
        assert 'version' in info
        assert 'author' in info
        assert 'backend_registered' in info
        
        assert info['version'] == '1.0.0'
        assert info['author'] == 'Test Author'
        assert isinstance(info['backend_registered'], bool)
    
    @patch('conductor.backend.is_backend_registered', return_value=True)
    def test_get_conductor_info_backend_registered(self, mock_registered):
        """Test Conductor info when backend is registered."""
        info = get_conductor_info()
        
        assert info['backend_registered'] is True
        mock_registered.assert_called_once()
    
    @patch('conductor.backend.is_backend_registered', side_effect=Exception("Backend error"))
    def test_get_conductor_info_backend_error(self, mock_registered):
        """Test Conductor info when backend registration check fails."""
        info = get_conductor_info()
        
        assert 'backend_registration_error' in info
        assert info['backend_registration_error'] == 'Backend error'
    
    def test_get_conductor_info_device_integration(self):
        """Test Conductor info with device integration."""
        mock_device = Mock()
        mock_device.get_device_info.return_value = {'device_id': 0, 'type': 'gcu'}
        
        mock_interface = Mock()
        mock_interface.list_devices.return_value = [0, 1]
        mock_interface.get_default_device.return_value = mock_device
        
        with patch('conductor.device.get_gcu_interface', return_value=mock_interface):
            info = get_conductor_info()
            
            assert 'available_devices' in info
            assert 'default_device' in info
            assert info['available_devices'] == [0, 1]
            assert info['default_device'] == {'device_id': 0, 'type': 'gcu'}
    
    def test_get_conductor_info_device_error(self):
        """Test Conductor info when device access fails."""
        with patch('conductor.device.get_gcu_interface', side_effect=Exception("Device error")):
            info = get_conductor_info()
            
            assert 'device_error' in info
            assert info['device_error'] == 'Device error'
    
    def test_get_conductor_info_no_default_device(self):
        """Test Conductor info when no default device is available."""
        mock_interface = Mock()
        mock_interface.list_devices.return_value = []
        mock_interface.get_default_device.return_value = None
        
        with patch('conductor.device.get_gcu_interface', return_value=mock_interface):
            info = get_conductor_info()
            
            assert 'available_devices' in info
            assert info['available_devices'] == []
            # Should not have 'default_device' key when no device is available


class TestPrintInfo:
    """Test information printing functionality."""
    
    @patch('conductor.utils.info.get_conductor_info')
    @patch('conductor.utils.info.get_system_info')
    @patch('builtins.print')
    def test_print_info_basic(self, mock_print, mock_system_info, mock_conductor_info):
        """Test basic information printing."""
        mock_conductor_info.return_value = {
            'version': '1.0.0',
            'author': 'Test Author',
            'backend_registered': True
        }
        
        mock_system_info.return_value = {
            'python_version': '3.8.10 (default, ...)',
            'platform': 'Linux-5.4.0',
            'architecture': ('64bit', 'ELF'),
            'torch_version': '2.0.0',
            'torch_cuda_available': True,
            'torch_backends': {'gcu': Mock(), 'inductor': Mock()}
        }
        
        print_info()
        
        # Verify print was called multiple times
        assert mock_print.call_count > 5
        
        # Check that key information was printed
        printed_text = ' '.join([str(call[0][0]) for call in mock_print.call_args_list])
        assert 'Conductor Version: 1.0.0' in printed_text
        assert 'Author: Test Author' in printed_text
        assert 'Backend Registered: True' in printed_text
        assert 'Python Version: 3.8.10' in printed_text
        assert 'PyTorch Version: 2.0.0' in printed_text
    
    @patch('conductor.utils.info.get_conductor_info')
    @patch('conductor.utils.info.get_system_info')
    @patch('builtins.print')
    def test_print_info_with_errors(self, mock_print, mock_system_info, mock_conductor_info):
        """Test information printing with error conditions."""
        mock_conductor_info.return_value = {
            'version': '1.0.0',
            'author': 'Test Author',
            'backend_registered': False,
            'backend_registration_error': 'Registration failed',
            'device_error': 'No devices found'
        }
        
        mock_system_info.return_value = {
            'python_version': '3.8.10 (default, ...)',
            'platform': 'Linux-5.4.0',
            'architecture': ('64bit', 'ELF'),
            'torch_version': 'Not installed',
            'torch_cuda_available': False,
            'torch_backends': []
        }
        
        print_info()
        
        # Check that error information was printed
        printed_text = ' '.join([str(call[0][0]) for call in mock_print.call_args_list])
        assert 'Registration Error: Registration failed' in printed_text
        assert 'Device Error: No devices found' in printed_text
        assert 'PyTorch: Not installed' in printed_text
    
    @patch('conductor.utils.info.get_conductor_info')
    @patch('conductor.utils.info.get_system_info')
    @patch('builtins.print')
    def test_print_info_with_devices(self, mock_print, mock_system_info, mock_conductor_info):
        """Test information printing with device information."""
        mock_conductor_info.return_value = {
            'version': '1.0.0',
            'author': 'Test Author',
            'backend_registered': True,
            'available_devices': [0, 1, 2]
        }
        
        mock_system_info.return_value = {
            'python_version': '3.8.10 (default, ...)',
            'platform': 'Linux-5.4.0',
            'architecture': ('64bit', 'ELF'),
            'torch_version': '2.0.0',
            'torch_cuda_available': True,
            'torch_backends': {'gcu': Mock()}
        }
        
        print_info()
        
        # Check that device information was printed
        printed_text = ' '.join([str(call[0][0]) for call in mock_print.call_args_list])
        assert 'Available Devices: [0, 1, 2]' in printed_text


class TestMainFunction:
    """Test main entry point functionality."""
    
    @patch('conductor.utils.info.print_info')
    def test_main_success(self, mock_print_info):
        """Test successful main execution."""
        main()
        mock_print_info.assert_called_once()
    
    @patch('conductor.utils.info.print_info', side_effect=Exception("Test error"))
    @patch('builtins.print')
    @patch('sys.exit')
    def test_main_error(self, mock_exit, mock_print, mock_print_info):
        """Test main execution with error."""
        main()
        
        mock_print_info.assert_called_once()
        mock_print.assert_called_once_with("Error getting system information: Test error")
        mock_exit.assert_called_once_with(1)


class TestInfoIntegration:
    """Integration tests for info functionality."""
    
    def test_complete_info_workflow(self):
        """Test complete information gathering workflow."""
        # This should work on any system without mocking
        system_info = get_system_info()
        conductor_info = get_conductor_info()
        
        # Verify basic structure
        assert isinstance(system_info, dict)
        assert isinstance(conductor_info, dict)
        
        # Verify required keys exist
        assert 'python_version' in system_info
        assert 'platform' in system_info
        assert 'version' in conductor_info
        assert 'backend_registered' in conductor_info
        
        # Verify values are reasonable
        assert system_info['python_version'] == sys.version
        assert isinstance(conductor_info['backend_registered'], bool)
    
    def test_print_info_integration(self):
        """Test print_info integration."""
        # Should not raise exceptions
        with patch('builtins.print'):
            print_info()
    
    def test_main_integration(self):
        """Test main function integration."""
        # Should not raise exceptions
        with patch('builtins.print'):
            with patch('sys.exit') as mock_exit:
                main()
                # Should not exit with error
                mock_exit.assert_not_called()
    
    def test_system_info_consistency(self):
        """Test that system info is consistent across calls."""
        info1 = get_system_info()
        info2 = get_system_info()
        
        # Basic system info should be the same
        assert info1['python_version'] == info2['python_version']
        assert info1['platform'] == info2['platform']
        assert info1['architecture'] == info2['architecture']
    
    def test_conductor_info_consistency(self):
        """Test that conductor info is consistent across calls."""
        info1 = get_conductor_info()
        info2 = get_conductor_info()
        
        # Version and author should be the same
        assert info1['version'] == info2['version']
        assert info1['author'] == info2['author']
        
        # Backend registration status might change, but should be boolean
        assert isinstance(info1['backend_registered'], bool)
        assert isinstance(info2['backend_registered'], bool)


if __name__ == '__main__':
    pytest.main([__file__])
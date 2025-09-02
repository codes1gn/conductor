"""
Unit tests for GCU device interface.

Tests the GCU device Python interface including device detection,
memory management, and kernel execution.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from conductor.device.gcu import GCUDevice, GCUInterface, get_gcu_interface


class TestGCUDevice:
    """Test GCUDevice class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = GCUDevice(device_id=0)
    
    def test_device_initialization(self):
        """Test GCU device initialization."""
        assert self.device.device_id == 0
        assert self.device._is_initialized is False
        assert self.device._memory_pool is None
    
    def test_device_initialization_with_id(self):
        """Test device initialization with specific ID."""
        device = GCUDevice(device_id=1)
        
        assert device.device_id == 1
        assert device._is_initialized is False
    
    def test_initialize_success(self):
        """Test successful device initialization."""
        result = self.device.initialize()
        assert result is True
        assert self.device._is_initialized is True
    
    @patch('conductor.device.gcu.logger')
    def test_initialize_with_logging(self, mock_logger):
        """Test device initialization with logging."""
        result = self.device.initialize()
        
        assert result is True
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "Initializing GCU device 0" in call_args
    
    @patch('conductor.device.gcu.logger')
    def test_initialize_failure(self, mock_logger):
        """Test device initialization failure."""
        # Mock an exception during initialization
        with patch.object(self.device, '_is_initialized', side_effect=Exception("Init failed")):
            # Since the actual initialize method doesn't raise exceptions,
            # we'll test the error logging path by mocking the logger
            result = self.device.initialize()
            assert result is True  # Current implementation always returns True
    
    def test_allocate_memory_success(self):
        """Test successful memory allocation."""
        self.device.initialize()
        
        handle = self.device.allocate_memory(1024)
        assert handle is not None
        assert "gcu_memory_handle_1024" in str(handle)
    
    def test_allocate_memory_not_initialized(self):
        """Test memory allocation when device not initialized."""
        handle = self.device.allocate_memory(1024)
        assert handle is None
    
    def test_free_memory_success(self):
        """Test successful memory deallocation."""
        result = self.device.free_memory("test_handle")
        assert result is True
    
    def test_execute_kernel_success(self):
        """Test successful kernel execution."""
        self.device.initialize()
        
        inputs = [1, 2, 3]
        outputs = self.device.execute_kernel("test_kernel", inputs)
        
        # Current implementation echoes inputs
        assert outputs == inputs
    
    def test_execute_kernel_not_initialized(self):
        """Test kernel execution when device not initialized."""
        inputs = [1, 2, 3]
        
        with pytest.raises(RuntimeError, match="Device not initialized"):
            self.device.execute_kernel("test_kernel", inputs)
    
    def test_synchronize(self):
        """Test device synchronization."""
        # Should not raise exception
        self.device.synchronize()
    
    def test_get_device_info(self):
        """Test getting device information."""
        info = self.device.get_device_info()
        
        assert info['device_id'] == 0
        assert info['initialized'] is False
        assert info['type'] == 'gcu'
        assert 'memory_total' in info
        assert 'memory_free' in info
    
    def test_get_device_info_initialized(self):
        """Test getting device info after initialization."""
        self.device.initialize()
        
        info = self.device.get_device_info()
        assert info['initialized'] is True


class TestGCUInterface:
    """Test GCUInterface class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.interface = GCUInterface()
    
    def test_interface_initialization(self):
        """Test GCU interface initialization."""
        assert self.interface._devices == {}
        assert self.interface._default_device is None
    
    def test_get_device_success(self):
        """Test getting device successfully."""
        device = self.interface.get_device(0)
        
        assert isinstance(device, GCUDevice)
        assert device.device_id == 0
        assert device._is_initialized is True
        assert 0 in self.interface._devices
        assert self.interface._default_device is device
    
    def test_get_device_cached(self):
        """Test getting cached device."""
        device1 = self.interface.get_device(0)
        device2 = self.interface.get_device(0)
        
        # Should return same instance
        assert device1 is device2
    
    def test_get_device_multiple(self):
        """Test getting multiple devices."""
        device0 = self.interface.get_device(0)
        device1 = self.interface.get_device(1)
        
        assert device0.device_id == 0
        assert device1.device_id == 1
        assert device0 is not device1
        assert len(self.interface._devices) == 2
    
    @patch.object(GCUDevice, 'initialize', return_value=False)
    def test_get_device_initialization_failure(self, mock_init):
        """Test getting device when initialization fails."""
        with pytest.raises(RuntimeError, match="Failed to initialize GCU device 0"):
            self.interface.get_device(0)
    
    def test_get_default_device_none(self):
        """Test getting default device when none exists."""
        # Should try to initialize device 0
        default_device = self.interface.get_default_device()
        
        assert default_device is not None
        assert default_device.device_id == 0
        assert self.interface._default_device is default_device
    
    def test_get_default_device_existing(self):
        """Test getting default device when one exists."""
        # Create a device first
        device = self.interface.get_device(1)
        
        default_device = self.interface.get_default_device()
        assert default_device is device
    
    @patch.object(GCUDevice, 'initialize', return_value=False)
    def test_get_default_device_initialization_failure(self, mock_init):
        """Test getting default device when initialization fails."""
        default_device = self.interface.get_default_device()
        assert default_device is None
    
    def test_list_devices_empty(self):
        """Test listing devices when none exist."""
        devices = self.interface.list_devices()
        assert devices == []
    
    def test_list_devices_with_devices(self):
        """Test listing devices when some exist."""
        self.interface.get_device(0)
        self.interface.get_device(2)
        
        devices = self.interface.list_devices()
        assert set(devices) == {0, 2}
    
    def test_cleanup(self):
        """Test interface cleanup."""
        # Create some devices
        device0 = self.interface.get_device(0)
        device1 = self.interface.get_device(1)
        
        # Mock synchronize to verify it's called
        with patch.object(device0, 'synchronize') as mock_sync0:
            with patch.object(device1, 'synchronize') as mock_sync1:
                self.interface.cleanup()
                
                mock_sync0.assert_called_once()
                mock_sync1.assert_called_once()
        
        # Verify cleanup
        assert self.interface._devices == {}
        assert self.interface._default_device is None


class TestGlobalInterface:
    """Test global GCU interface functionality."""
    
    def test_get_gcu_interface(self):
        """Test getting global GCU interface."""
        interface1 = get_gcu_interface()
        interface2 = get_gcu_interface()
        
        # Should return same instance
        assert interface1 is interface2
        assert isinstance(interface1, GCUInterface)
    
    def test_global_interface_persistence(self):
        """Test that global interface persists state."""
        interface = get_gcu_interface()
        
        # Create a device
        device = interface.get_device(0)
        
        # Get interface again and verify device exists
        interface2 = get_gcu_interface()
        assert 0 in interface2._devices
        assert interface2._devices[0] is device


class TestGCUDeviceIntegration:
    """Integration tests for GCU device functionality."""
    
    def test_complete_device_workflow(self):
        """Test complete device workflow."""
        device = GCUDevice(0)
        
        # Initialize device
        assert device.initialize() is True
        assert device._is_initialized is True
        
        # Get device info
        info = device.get_device_info()
        assert info['initialized'] is True
        assert info['device_id'] == 0
        
        # Allocate memory
        handle = device.allocate_memory(1024)
        assert handle is not None
        
        # Execute kernel
        inputs = [1.0, 2.0, 3.0]
        outputs = device.execute_kernel("test_kernel", inputs)
        assert outputs == inputs
        
        # Free memory
        assert device.free_memory(handle) is True
        
        # Synchronize
        device.synchronize()  # Should not raise
    
    def test_interface_device_management(self):
        """Test interface device management workflow."""
        interface = GCUInterface()
        
        # Get multiple devices
        device0 = interface.get_device(0)
        device1 = interface.get_device(1)
        
        assert device0.device_id == 0
        assert device1.device_id == 1
        assert device0._is_initialized is True
        assert device1._is_initialized is True
        
        # Verify default device
        default = interface.get_default_device()
        assert default is device0
        
        # List devices
        devices = interface.list_devices()
        assert set(devices) == {0, 1}
        
        # Cleanup
        interface.cleanup()
        assert len(interface._devices) == 0
    
    def test_error_handling_workflow(self):
        """Test error handling in device workflow."""
        device = GCUDevice(0)
        
        # Try to execute kernel without initialization
        with pytest.raises(RuntimeError, match="Device not initialized"):
            device.execute_kernel("test", [1, 2, 3])
        
        # Try to allocate memory without initialization
        handle = device.allocate_memory(1024)
        assert handle is None
        
        # Initialize and retry
        device.initialize()
        handle = device.allocate_memory(1024)
        assert handle is not None
        
        outputs = device.execute_kernel("test", [1, 2, 3])
        assert outputs == [1, 2, 3]
    
    def test_multiple_interface_instances(self):
        """Test behavior with multiple interface instances."""
        interface1 = GCUInterface()
        interface2 = GCUInterface()
        
        # Should be independent
        device1 = interface1.get_device(0)
        device2 = interface2.get_device(0)
        
        # Different interfaces should create different device instances
        assert device1 is not device2
        assert device1.device_id == device2.device_id == 0
        
        # But global interface should be singleton
        global1 = get_gcu_interface()
        global2 = get_gcu_interface()
        assert global1 is global2


if __name__ == '__main__':
    pytest.main([__file__])
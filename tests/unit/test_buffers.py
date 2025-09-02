"""
Unit tests for buffer management and scoping.

Tests the Buffer class, BufferScope enum, and BufferManager functionality
including scope promotion, memory footprint calculation, and buffer reuse.
"""

import pytest
import torch
from conductor.codegen.buffers import Buffer, BufferScope, BufferManager


class TestBufferScope:
    """Test BufferScope enum functionality."""
    
    def test_scope_values(self):
        """Test that scope enum has correct values."""
        assert BufferScope.LOCAL.value == "local"
        assert BufferScope.SHARED.value == "shared"
        assert BufferScope.GLOBAL.value == "global"


class TestBuffer:
    """Test Buffer class functionality."""
    
    def test_buffer_creation(self):
        """Test basic buffer creation with required fields."""
        buffer = Buffer(
            name="test_buffer",
            scope=BufferScope.LOCAL,
            dtype=torch.float32
        )
        
        assert buffer.name == "test_buffer"
        assert buffer.scope == BufferScope.LOCAL
        assert buffer.dtype == torch.float32
        assert buffer.shape is None
        assert buffer.producer is None
        assert buffer.consumers == []
        assert buffer.is_temporary is False
    
    def test_buffer_creation_with_shape(self):
        """Test buffer creation with shape information."""
        buffer = Buffer(
            name="shaped_buffer",
            scope=BufferScope.SHARED,
            dtype=torch.int32,
            shape=(10, 20, 30)
        )
        
        assert buffer.shape == (10, 20, 30)
        assert buffer.dtype == torch.int32
        assert buffer.scope == BufferScope.SHARED
    
    def test_buffer_creation_temporary(self):
        """Test creation of temporary buffer."""
        buffer = Buffer(
            name="temp_buffer",
            scope=BufferScope.LOCAL,
            dtype=torch.float16,
            is_temporary=True
        )
        
        assert buffer.is_temporary is True
    
    def test_scope_promotion_local_to_shared(self):
        """Test promoting buffer from LOCAL to SHARED scope."""
        buffer = Buffer(
            name="test_buffer",
            scope=BufferScope.LOCAL,
            dtype=torch.float32
        )
        
        buffer.promote_scope(BufferScope.SHARED)
        assert buffer.scope == BufferScope.SHARED
    
    def test_scope_promotion_local_to_global(self):
        """Test promoting buffer from LOCAL to GLOBAL scope."""
        buffer = Buffer(
            name="test_buffer",
            scope=BufferScope.LOCAL,
            dtype=torch.float32
        )
        
        buffer.promote_scope(BufferScope.GLOBAL)
        assert buffer.scope == BufferScope.GLOBAL
    
    def test_scope_promotion_shared_to_global(self):
        """Test promoting buffer from SHARED to GLOBAL scope."""
        buffer = Buffer(
            name="test_buffer",
            scope=BufferScope.SHARED,
            dtype=torch.float32
        )
        
        buffer.promote_scope(BufferScope.GLOBAL)
        assert buffer.scope == BufferScope.GLOBAL
    
    def test_scope_promotion_no_downgrade(self):
        """Test that scope promotion doesn't allow downgrading."""
        buffer = Buffer(
            name="test_buffer",
            scope=BufferScope.GLOBAL,
            dtype=torch.float32
        )
        
        # Try to "promote" to lower scope - should not change
        buffer.promote_scope(BufferScope.SHARED)
        assert buffer.scope == BufferScope.GLOBAL
        
        buffer.promote_scope(BufferScope.LOCAL)
        assert buffer.scope == BufferScope.GLOBAL
    
    def test_scope_promotion_same_scope(self):
        """Test that promoting to same scope is safe."""
        buffer = Buffer(
            name="test_buffer",
            scope=BufferScope.SHARED,
            dtype=torch.float32
        )
        
        buffer.promote_scope(BufferScope.SHARED)
        assert buffer.scope == BufferScope.SHARED
    
    def test_memory_footprint_unknown_shape(self):
        """Test memory footprint calculation with unknown shape."""
        buffer = Buffer(
            name="test_buffer",
            scope=BufferScope.LOCAL,
            dtype=torch.float32,
            shape=None
        )
        
        assert buffer.get_memory_footprint() == -1
    
    def test_memory_footprint_float32(self):
        """Test memory footprint calculation for float32 tensor."""
        buffer = Buffer(
            name="test_buffer",
            scope=BufferScope.LOCAL,
            dtype=torch.float32,
            shape=(10, 20)
        )
        
        # 10 * 20 * 4 bytes = 800 bytes
        assert buffer.get_memory_footprint() == 800
    
    def test_memory_footprint_float16(self):
        """Test memory footprint calculation for float16 tensor."""
        buffer = Buffer(
            name="test_buffer",
            scope=BufferScope.LOCAL,
            dtype=torch.float16,
            shape=(5, 10, 2)
        )
        
        # 5 * 10 * 2 * 2 bytes = 200 bytes
        assert buffer.get_memory_footprint() == 200
    
    def test_memory_footprint_int32(self):
        """Test memory footprint calculation for int32 tensor."""
        buffer = Buffer(
            name="test_buffer",
            scope=BufferScope.LOCAL,
            dtype=torch.int32,
            shape=(8, 8)
        )
        
        # 8 * 8 * 4 bytes = 256 bytes
        assert buffer.get_memory_footprint() == 256
    
    def test_memory_footprint_int64(self):
        """Test memory footprint calculation for int64 tensor."""
        buffer = Buffer(
            name="test_buffer",
            scope=BufferScope.LOCAL,
            dtype=torch.int64,
            shape=(4, 4, 4)
        )
        
        # 4 * 4 * 4 * 8 bytes = 512 bytes
        assert buffer.get_memory_footprint() == 512
    
    def test_memory_footprint_bool(self):
        """Test memory footprint calculation for bool tensor."""
        buffer = Buffer(
            name="test_buffer",
            scope=BufferScope.LOCAL,
            dtype=torch.bool,
            shape=(100,)
        )
        
        # 100 * 1 byte = 100 bytes
        assert buffer.get_memory_footprint() == 100
    
    def test_memory_footprint_unknown_dtype(self):
        """Test memory footprint calculation for unknown dtype."""
        buffer = Buffer(
            name="test_buffer",
            scope=BufferScope.LOCAL,
            dtype=torch.complex64,  # Not in the dtype_sizes dict
            shape=(10, 10)
        )
        
        # Should default to 4 bytes per element
        # 10 * 10 * 4 bytes = 400 bytes
        assert buffer.get_memory_footprint() == 400
    
    def test_memory_footprint_scalar(self):
        """Test memory footprint calculation for scalar (empty shape)."""
        buffer = Buffer(
            name="test_buffer",
            scope=BufferScope.LOCAL,
            dtype=torch.float32,
            shape=()
        )
        
        # Scalar: 1 element * 4 bytes = 4 bytes
        assert buffer.get_memory_footprint() == 4


class TestBufferManager:
    """Test BufferManager functionality."""
    
    def test_buffer_manager_creation(self):
        """Test BufferManager initialization."""
        manager = BufferManager()
        assert len(manager.list_buffers()) == 0
    
    def test_allocate_buffer_basic(self):
        """Test basic buffer allocation."""
        manager = BufferManager()
        buffer = manager.allocate_buffer("test", torch.float32)
        
        assert buffer.name == "test"
        assert buffer.dtype == torch.float32
        assert buffer.scope == BufferScope.LOCAL
        assert len(manager.list_buffers()) == 1
    
    def test_allocate_buffer_with_shape(self):
        """Test buffer allocation with shape."""
        manager = BufferManager()
        buffer = manager.allocate_buffer("test", torch.int32, (5, 10))
        
        assert buffer.shape == (5, 10)
        assert buffer.dtype == torch.int32
    
    def test_allocate_buffer_duplicate_names(self):
        """Test allocation with duplicate names generates unique names."""
        manager = BufferManager()
        buffer1 = manager.allocate_buffer("test", torch.float32)
        buffer2 = manager.allocate_buffer("test", torch.int32)
        
        assert buffer1.name == "test"
        assert buffer2.name == "test_1"
        assert len(manager.list_buffers()) == 2
    
    def test_allocate_buffer_multiple_duplicates(self):
        """Test multiple duplicate names increment counter."""
        manager = BufferManager()
        buffer1 = manager.allocate_buffer("test", torch.float32)
        buffer2 = manager.allocate_buffer("test", torch.int32)
        buffer3 = manager.allocate_buffer("test", torch.float16)
        
        assert buffer1.name == "test"
        assert buffer2.name == "test_1"
        assert buffer3.name == "test_2"
    
    def test_get_buffer_existing(self):
        """Test retrieving existing buffer by name."""
        manager = BufferManager()
        original = manager.allocate_buffer("test", torch.float32)
        retrieved = manager.get_buffer("test")
        
        assert retrieved is original
        assert retrieved.name == "test"
    
    def test_get_buffer_nonexistent(self):
        """Test retrieving non-existent buffer returns None."""
        manager = BufferManager()
        result = manager.get_buffer("nonexistent")
        assert result is None
    
    def test_promote_buffer_scope(self):
        """Test buffer scope promotion through manager."""
        manager = BufferManager()
        buffer = manager.allocate_buffer("test", torch.float32)
        
        assert buffer.scope == BufferScope.LOCAL
        manager.promote_buffer_scope(buffer, BufferScope.GLOBAL)
        assert buffer.scope == BufferScope.GLOBAL
    
    def test_optimize_buffer_reuse_empty_list(self):
        """Test buffer reuse optimization with empty list."""
        manager = BufferManager()
        reuse_map = manager.optimize_buffer_reuse([])
        assert reuse_map == {}
    
    def test_optimize_buffer_reuse_single_buffer(self):
        """Test buffer reuse optimization with single buffer."""
        manager = BufferManager()
        buffer = manager.allocate_buffer("test", torch.float32, (10, 10))
        reuse_map = manager.optimize_buffer_reuse([buffer])
        assert reuse_map == {}
    
    def test_optimize_buffer_reuse_compatible_temporary_buffers(self):
        """Test buffer reuse optimization with compatible temporary buffers."""
        manager = BufferManager()
        buffer1 = Buffer("temp1", BufferScope.LOCAL, torch.float32, (10, 10), is_temporary=True)
        buffer2 = Buffer("temp2", BufferScope.LOCAL, torch.float32, (10, 10), is_temporary=True)
        
        reuse_map = manager.optimize_buffer_reuse([buffer1, buffer2])
        assert reuse_map == {"temp2": "temp1"}
    
    def test_optimize_buffer_reuse_incompatible_shapes(self):
        """Test buffer reuse optimization with incompatible shapes."""
        manager = BufferManager()
        buffer1 = Buffer("temp1", BufferScope.LOCAL, torch.float32, (10, 10), is_temporary=True)
        buffer2 = Buffer("temp2", BufferScope.LOCAL, torch.float32, (5, 5), is_temporary=True)
        
        reuse_map = manager.optimize_buffer_reuse([buffer1, buffer2])
        assert reuse_map == {}
    
    def test_optimize_buffer_reuse_incompatible_dtypes(self):
        """Test buffer reuse optimization with incompatible dtypes."""
        manager = BufferManager()
        buffer1 = Buffer("temp1", BufferScope.LOCAL, torch.float32, (10, 10), is_temporary=True)
        buffer2 = Buffer("temp2", BufferScope.LOCAL, torch.int32, (10, 10), is_temporary=True)
        
        reuse_map = manager.optimize_buffer_reuse([buffer1, buffer2])
        assert reuse_map == {}
    
    def test_optimize_buffer_reuse_non_temporary_buffers(self):
        """Test buffer reuse optimization with non-temporary buffers."""
        manager = BufferManager()
        buffer1 = Buffer("buf1", BufferScope.LOCAL, torch.float32, (10, 10), is_temporary=False)
        buffer2 = Buffer("buf2", BufferScope.LOCAL, torch.float32, (10, 10), is_temporary=False)
        
        reuse_map = manager.optimize_buffer_reuse([buffer1, buffer2])
        assert reuse_map == {}
    
    def test_list_buffers(self):
        """Test listing all managed buffers."""
        manager = BufferManager()
        buffer1 = manager.allocate_buffer("test1", torch.float32)
        buffer2 = manager.allocate_buffer("test2", torch.int32)
        
        buffers = manager.list_buffers()
        assert len(buffers) == 2
        assert buffer1 in buffers
        assert buffer2 in buffers


class TestBufferIntegration:
    """Integration tests for buffer functionality."""
    
    def test_buffer_lifecycle(self):
        """Test complete buffer lifecycle through manager."""
        manager = BufferManager()
        
        # Allocate buffer
        buffer = manager.allocate_buffer("lifecycle_test", torch.float32, (100, 100))
        assert buffer.scope == BufferScope.LOCAL
        
        # Promote scope
        manager.promote_buffer_scope(buffer, BufferScope.SHARED)
        assert buffer.scope == BufferScope.SHARED
        
        # Further promote scope
        manager.promote_buffer_scope(buffer, BufferScope.GLOBAL)
        assert buffer.scope == BufferScope.GLOBAL
        
        # Check memory footprint
        assert buffer.get_memory_footprint() == 100 * 100 * 4  # 40KB
        
        # Retrieve buffer
        retrieved = manager.get_buffer("lifecycle_test")
        assert retrieved is buffer
    
    def test_multiple_buffer_management(self):
        """Test managing multiple buffers with different properties."""
        manager = BufferManager()
        
        # Create buffers with different properties
        float_buffer = manager.allocate_buffer("float_buf", torch.float32, (50, 50))
        int_buffer = manager.allocate_buffer("int_buf", torch.int64, (25, 25))
        temp_buffer = Buffer("temp", BufferScope.LOCAL, torch.float16, (10, 10), is_temporary=True)
        
        # Promote scopes differently
        manager.promote_buffer_scope(float_buffer, BufferScope.SHARED)
        manager.promote_buffer_scope(int_buffer, BufferScope.GLOBAL)
        
        # Verify states
        assert float_buffer.scope == BufferScope.SHARED
        assert int_buffer.scope == BufferScope.GLOBAL
        assert temp_buffer.scope == BufferScope.LOCAL
        
        # Check memory footprints
        assert float_buffer.get_memory_footprint() == 50 * 50 * 4  # 10KB
        assert int_buffer.get_memory_footprint() == 25 * 25 * 8   # 5KB
        assert temp_buffer.get_memory_footprint() == 10 * 10 * 2  # 200B
        
        # Test reuse optimization
        temp_buffer2 = Buffer("temp2", BufferScope.LOCAL, torch.float16, (10, 10), is_temporary=True)
        reuse_map = manager.optimize_buffer_reuse([temp_buffer, temp_buffer2])
        assert reuse_map == {"temp2": "temp"}
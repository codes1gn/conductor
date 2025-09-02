"""
Unit tests for compilation result caching system.

This module tests the caching mechanisms used to store and retrieve
compiled artifacts for performance optimization.
"""

import pytest
import tempfile
import shutil
import time
import os
from pathlib import Path
from unittest.mock import Mock, patch
from conductor.utils.caching import CompilationCache, generate_cache_key
from conductor.runtime.loader import CompiledArtifact


class TestCompilationCache:
    """Test cases for compilation cache."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CompilationCache(cache_dir=self.temp_dir, max_size_mb=10)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_cache_initialization(self):
        """Test cache initialization."""
        assert self.cache.cache_dir.exists()
        assert self.cache.max_size_bytes == 10 * 1024 * 1024
        assert isinstance(self.cache._metadata, dict)
        
    def test_cache_put_and_get(self):
        """Test basic cache put and get operations."""
        # Create test artifact
        artifact = CompiledArtifact(
            path="/test/path.so",
            artifact_type="shared_library",
            entry_point="main",
            metadata={"test": "data"}
        )
        
        key = "test_key_123"
        
        # Put artifact in cache
        success = self.cache.put(key, artifact)
        assert success
        
        # Get artifact from cache
        retrieved = self.cache.get(key)
        assert retrieved is not None
        assert retrieved.path == artifact.path
        assert retrieved.artifact_type == artifact.artifact_type
        assert retrieved.entry_point == artifact.entry_point
        assert retrieved.metadata == artifact.metadata
        
    def test_cache_miss(self):
        """Test cache miss behavior."""
        result = self.cache.get("nonexistent_key")
        assert result is None
        
    def test_cache_overwrite(self):
        """Test overwriting existing cache entries."""
        key = "test_key"
        
        # Put first artifact
        artifact1 = CompiledArtifact(
            path="/test/path1.so",
            artifact_type="shared_library",
            entry_point="main1",
            metadata={"version": 1}
        )
        self.cache.put(key, artifact1)
        
        # Put second artifact with same key
        artifact2 = CompiledArtifact(
            path="/test/path2.so", 
            artifact_type="shared_library",
            entry_point="main2",
            metadata={"version": 2}
        )
        self.cache.put(key, artifact2)
        
        # Should get the second artifact
        retrieved = self.cache.get(key)
        assert retrieved.path == artifact2.path
        assert retrieved.metadata["version"] == 2
        
    def test_cache_invalidation(self):
        """Test cache entry invalidation."""
        artifact = CompiledArtifact(
            path="/test/path.so",
            artifact_type="shared_library", 
            entry_point="main",
            metadata={}
        )
        
        key = "test_key"
        self.cache.put(key, artifact)
        
        # Verify it's cached
        assert self.cache.get(key) is not None
        
        # Invalidate
        success = self.cache.invalidate(key)
        assert success
        
        # Should be gone
        assert self.cache.get(key) is None
        
        # Invalidating again should return False
        success = self.cache.invalidate(key)
        assert not success
        
    def test_cache_clear(self):
        """Test clearing entire cache."""
        # Add multiple entries
        for i in range(3):
            artifact = CompiledArtifact(
                path=f"/test/path{i}.so",
                artifact_type="shared_library",
                entry_point="main",
                metadata={"index": i}
            )
            self.cache.put(f"key_{i}", artifact)
        
        # Verify entries exist
        stats = self.cache.get_stats()
        assert stats['entries'] == 3
        
        # Clear cache
        self.cache.clear()
        
        # Verify cache is empty
        stats = self.cache.get_stats()
        assert stats['entries'] == 0
        
        # Verify entries are gone
        for i in range(3):
            assert self.cache.get(f"key_{i}") is None
            
    def test_cache_stats(self):
        """Test cache statistics."""
        # Initially empty
        stats = self.cache.get_stats()
        assert stats['entries'] == 0
        assert stats['total_size_bytes'] == 0
        
        # Add an entry
        artifact = CompiledArtifact(
            path="/test/path.so",
            artifact_type="shared_library",
            entry_point="main", 
            metadata={}
        )
        self.cache.put("test_key", artifact)
        
        # Check stats
        stats = self.cache.get_stats()
        assert stats['entries'] == 1
        assert stats['total_size_bytes'] > 0
        assert stats['total_size_mb'] > 0
        assert 'cache_dir' in stats
        assert 'max_size_mb' in stats
        
    def test_cache_size_limit_enforcement(self):
        """Test that cache enforces size limits."""
        # Create cache with very small limit
        small_cache = CompilationCache(
            cache_dir=tempfile.mkdtemp(),
            max_size_mb=1  # 1MB limit
        )
        
        try:
            # Create large artifacts that exceed the limit
            large_metadata = {"data": "x" * 1024 * 512}  # ~512KB each
            
            artifacts = []
            for i in range(5):  # 5 * 512KB = ~2.5MB total
                artifact = CompiledArtifact(
                    path=f"/test/path{i}.so",
                    artifact_type="shared_library",
                    entry_point="main",
                    metadata=large_metadata
                )
                artifacts.append(artifact)
                small_cache.put(f"key_{i}", artifact)
                
                # Add small delay to ensure different access times
                time.sleep(0.01)
            
            # Cache should have evicted some entries
            stats = small_cache.get_stats()
            assert stats['total_size_mb'] <= 1.1  # Allow small margin for overhead
            
            # Earlier entries should be evicted (LRU)
            assert small_cache.get("key_0") is None
            assert small_cache.get("key_4") is not None  # Most recent should remain
            
        finally:
            shutil.rmtree(small_cache.cache_dir, ignore_errors=True)
            
    def test_cache_persistence(self):
        """Test that cache persists across instances."""
        artifact = CompiledArtifact(
            path="/test/path.so",
            artifact_type="shared_library",
            entry_point="main",
            metadata={"persistent": True}
        )
        
        key = "persistent_key"
        
        # Put in first cache instance
        self.cache.put(key, artifact)
        
        # Create new cache instance with same directory
        cache2 = CompilationCache(cache_dir=self.temp_dir)
        
        # Should be able to retrieve from new instance
        retrieved = cache2.get(key)
        assert retrieved is not None
        assert retrieved.metadata["persistent"] is True
        
    def test_cache_missing_file_handling(self):
        """Test handling when cached file is missing."""
        artifact = CompiledArtifact(
            path="/test/path.so",
            artifact_type="shared_library",
            entry_point="main",
            metadata={}
        )
        
        key = "test_key"
        self.cache.put(key, artifact)
        
        # Manually delete the cached file (not the metadata file)
        cache_files = list(self.cache.cache_dir.glob("*.pkl"))
        assert len(cache_files) == 2  # artifact file + metadata file
        
        # Find and delete the artifact file (not metadata)
        artifact_file = None
        for f in cache_files:
            if f.name != 'cache_metadata.pkl':
                artifact_file = f
                break
        
        assert artifact_file is not None
        artifact_file.unlink()
        
        # Getting should return None and clean up metadata
        result = self.cache.get(key)
        assert result is None
        
        # Metadata should be cleaned up
        assert key not in self.cache._metadata
        
    def test_cache_corrupted_file_handling(self):
        """Test handling of corrupted cache files."""
        artifact = CompiledArtifact(
            path="/test/path.so",
            artifact_type="shared_library",
            entry_point="main",
            metadata={}
        )
        
        key = "test_key"
        self.cache.put(key, artifact)
        
        # Corrupt the cached file (not the metadata file)
        cache_files = list(self.cache.cache_dir.glob("*.pkl"))
        assert len(cache_files) == 2  # artifact file + metadata file
        
        # Find and corrupt the artifact file (not metadata)
        artifact_file = None
        for f in cache_files:
            if f.name != 'cache_metadata.pkl':
                artifact_file = f
                break
        
        assert artifact_file is not None
        with open(artifact_file, 'w') as f:
            f.write("corrupted data")
        
        # Getting should return None and clean up
        result = self.cache.get(key)
        assert result is None
        
        # Entry should be removed
        assert key not in self.cache._metadata
        
    def test_access_time_updates(self):
        """Test that access times are updated on cache hits."""
        artifact = CompiledArtifact(
            path="/test/path.so",
            artifact_type="shared_library",
            entry_point="main",
            metadata={}
        )
        
        key = "test_key"
        self.cache.put(key, artifact)
        
        # Get initial access time
        initial_time = self.cache._metadata[key]['last_accessed']
        
        # Wait a bit and access again
        time.sleep(0.1)
        self.cache.get(key)
        
        # Access time should be updated
        updated_time = self.cache._metadata[key]['last_accessed']
        assert updated_time > initial_time


class TestCacheKeyGeneration:
    """Test cases for cache key generation."""
    
    def test_generate_cache_key_consistency(self):
        """Test that cache key generation is consistent."""
        args = ["arg1", "arg2", 123, {"key": "value"}]
        
        key1 = generate_cache_key(*args)
        key2 = generate_cache_key(*args)
        
        assert key1 == key2
        assert len(key1) == 16  # Should be 16 character hash
        
    def test_generate_cache_key_uniqueness(self):
        """Test that different inputs generate different keys."""
        key1 = generate_cache_key("input1", "input2")
        key2 = generate_cache_key("input1", "input3")
        key3 = generate_cache_key("input2", "input1")  # Different order
        
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3
        
    def test_generate_cache_key_types(self):
        """Test cache key generation with different types."""
        # Should handle various types without errors
        key1 = generate_cache_key("string", 123, 45.67, True, None)
        key2 = generate_cache_key(["list", "items"], {"dict": "value"})
        
        assert isinstance(key1, str)
        assert isinstance(key2, str)
        assert len(key1) == 16
        assert len(key2) == 16


class TestJITCompilerCaching:
    """Test JIT compiler caching integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from conductor.runtime.jit import JITCompiler
        self.temp_dir = tempfile.mkdtemp()
        self.compiler = JITCompiler(cache_dir=self.temp_dir, max_cache_size_mb=10)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_jit_compiler_cache_integration(self):
        """Test JIT compiler cache integration."""
        # Test cache stats
        stats = self.compiler.get_cache_stats()
        assert 'entries' in stats
        assert 'total_size_mb' in stats
        
        # Test cache clearing
        self.compiler.clear_cache()
        stats = self.compiler.get_cache_stats()
        assert stats['entries'] == 0
        
    def test_jit_compiler_cache_operations(self):
        """Test JIT compiler cache operations."""
        artifact = CompiledArtifact(
            path="/test/path.so",
            artifact_type="shared_library",
            entry_point="main",
            metadata={}
        )
        
        graph_hash = "test_hash_123"
        
        # Test caching
        success = self.compiler.cache_compilation_result(graph_hash, artifact)
        assert success
        
        # Test invalidation
        success = self.compiler.invalidate_cache_entry(graph_hash)
        assert success
        
        # Test validation
        validation = self.compiler.validate_cache_integrity()
        assert 'total_entries' in validation
        assert 'valid_entries' in validation
        
    def test_jit_compiler_cache_validation(self):
        """Test cache validation functionality."""
        # Add some entries
        for i in range(3):
            artifact = CompiledArtifact(
                path=f"/test/path{i}.so",
                artifact_type="shared_library",
                entry_point="main",
                metadata={"index": i}
            )
            self.compiler.cache_compilation_result(f"hash_{i}", artifact)
        
        # Validate cache
        validation = self.compiler.validate_cache_integrity()
        assert validation['total_entries'] == 3
        assert validation['valid_entries'] == 3
        assert validation['invalid_entries'] == 0
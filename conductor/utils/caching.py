"""
Compilation result caching.

This module provides caching mechanisms for compiled artifacts
to improve performance by avoiding redundant compilations.
"""

import os
import pickle
import hashlib
import tempfile
import shutil
from typing import Any, Optional, Dict
from pathlib import Path
import time
from .logging import get_logger

logger = get_logger(__name__)


class CompilationCache:
    """
    Manages caching of compiled artifacts.
    
    This class provides persistent caching of compilation results
    with automatic cleanup and cache validation.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: int = 1024):
        """
        Initialize compilation cache.
        
        Args:
            cache_dir: Directory for cache storage (default: system temp)
            max_size_mb: Maximum cache size in megabytes
        """
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), 'conductor_cache')
            
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._metadata_file = self.cache_dir / 'cache_metadata.pkl'
        self._metadata = self._load_metadata()
        
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached artifact by key.
        
        Args:
            key: Cache key (typically graph hash)
            
        Returns:
            Cached artifact if found and valid, None otherwise
        """
        if key not in self._metadata:
            logger.debug(f"Cache miss: key '{key}' not found")
            return None
            
        entry = self._metadata[key]
        artifact_path = self.cache_dir / entry['filename']
        
        # Check if file still exists
        if not artifact_path.exists():
            logger.warning(f"Cache entry exists but file missing: {artifact_path}")
            del self._metadata[key]
            self._save_metadata()
            return None
            
        # Update access time
        entry['last_accessed'] = time.time()
        self._save_metadata()
        
        try:
            with open(artifact_path, 'rb') as f:
                artifact = pickle.load(f)
            logger.debug(f"Cache hit: loaded artifact for key '{key}'")
            return artifact
        except Exception as e:
            logger.error(f"Failed to load cached artifact: {e}")
            self._remove_entry(key)
            return None
            
    def put(self, key: str, artifact: Any) -> bool:
        """
        Store artifact in cache.
        
        Args:
            key: Cache key (typically graph hash)
            artifact: Artifact to cache
            
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            # Generate filename
            filename = f"{key}.pkl"
            artifact_path = self.cache_dir / filename
            
            # Serialize artifact
            with open(artifact_path, 'wb') as f:
                pickle.dump(artifact, f)
                
            # Update metadata
            file_size = artifact_path.stat().st_size
            self._metadata[key] = {
                'filename': filename,
                'size': file_size,
                'created': time.time(),
                'last_accessed': time.time()
            }
            
            # Enforce cache size limit
            self._enforce_size_limit()
            self._save_metadata()
            
            logger.debug(f"Cached artifact for key '{key}' ({file_size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache artifact: {e}")
            return False
            
    def invalidate(self, key: str) -> bool:
        """
        Remove specific entry from cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if entry was removed, False if not found
        """
        return self._remove_entry(key)
        
    def clear(self) -> None:
        """Remove all entries from cache."""
        try:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._metadata.clear()
            self._save_metadata()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        total_size = sum(entry['size'] for entry in self._metadata.values())
        return {
            'entries': len(self._metadata),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'max_size_mb': self.max_size_bytes / (1024 * 1024)
        }
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        try:
            if self._metadata_file.exists():
                with open(self._metadata_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
        return {}
        
    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self._metadata_file, 'wb') as f:
                pickle.dump(self._metadata, f)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
            
    def _remove_entry(self, key: str) -> bool:
        """Remove cache entry and associated file."""
        if key not in self._metadata:
            return False
            
        try:
            entry = self._metadata[key]
            artifact_path = self.cache_dir / entry['filename']
            
            if artifact_path.exists():
                artifact_path.unlink()
                
            del self._metadata[key]
            self._save_metadata()
            
            logger.debug(f"Removed cache entry for key '{key}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove cache entry: {e}")
            return False
            
    def _enforce_size_limit(self) -> None:
        """Remove oldest entries if cache exceeds size limit."""
        total_size = sum(entry['size'] for entry in self._metadata.values())
        
        if total_size <= self.max_size_bytes:
            return
            
        # Sort by last accessed time (oldest first)
        sorted_entries = sorted(
            self._metadata.items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        # Remove oldest entries until under limit
        for key, entry in sorted_entries:
            if total_size <= self.max_size_bytes:
                break
                
            if self._remove_entry(key):
                total_size -= entry['size']
                logger.debug(f"Evicted cache entry '{key}' to enforce size limit")


def generate_cache_key(*args) -> str:
    """
    Generate cache key from arguments.
    
    Args:
        *args: Arguments to hash for cache key
        
    Returns:
        Hexadecimal hash string suitable for use as cache key
    """
    hasher = hashlib.sha256()
    for arg in args:
        hasher.update(str(arg).encode('utf-8'))
    return hasher.hexdigest()[:16]  # Use first 16 characters
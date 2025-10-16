import os
import logging
from typing import Dict, Any, Optional
from threading import Lock
import pandas as pd

logger = logging.getLogger(__name__)


class ResponseCache:
    """Caches LLM responses to avoid duplicate API calls."""

    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize the response cache.

        Args:
            cache_file: Optional file path to save/load cache
        """
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.lock = Lock()  # Thread-safe lock for cache access
        self.cache_file = cache_file

        # Load cache from file if it exists
        if cache_file and os.path.exists(cache_file):
            try:
                loaded_cache = pd.read_pickle(cache_file)
                with self.lock:
                    self.cache = loaded_cache
                logger.info(f"Loaded {len(self.cache)} items from cache file")
            except Exception as e:
                logger.error(f"Failed to load cache from file: {e}")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get value from cache and update stats.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self.lock:
            result = self.cache.get(key)
            if result is not None:
                self.hits += 1
                logger.debug(f"Cache hit for key: {key}")
            else:
                self.misses += 1
                logger.debug(f"Cache miss for key: {key}")
            return result

    def set(self, key: str, value: Dict[str, Any]):
        """
        Set value in cache and save to file if configured.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            self.cache[key] = value

        # Save cache to file if configured
        if self.cache_file:
            try:
                pd.to_pickle(self.cache, self.cache_file)
                logger.debug(f"Cache saved to {self.cache_file} ({len(self.cache)} items)")
            except Exception as e:
                logger.error(f"Failed to save cache to file: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            return {
                'total_requests': total,
                'cache_hits': self.hits,
                'cache_misses': self.misses,
                'hit_rate': hit_rate
            }

"""Cache module for file analysis results.

Caches analysis results to avoid re-analyzing unchanged files.
Similar to trace_cache.py but for analyse operations.
"""

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class AnalyseCacheData(BaseModel):
    """Cached analysis data structure."""

    version: int = Field(default=1, description='Cache format version')
    file_path: str = Field(..., description='Absolute path to analyzed file')
    file_size: int = Field(..., description='File size in bytes')
    file_mtime: str = Field(..., description='File modification time (ISO format)')
    cached_at: str = Field(..., description='When this cache was created (ISO format)')

    # Analysis results
    analysis_result: dict = Field(..., description='Complete FileAnalysisResult as dict')


def get_analyse_cache_dir() -> Path:
    """Get the analyse cache directory path.

    Returns:
        Path to ~/.cache/rx/analyse_cache/
    """
    cache_home = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
    cache_dir = Path(cache_home) / 'rx' / 'analyse_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key(file_path: str) -> str:
    """Generate cache key for a file.

    Args:
        file_path: Absolute path to file

    Returns:
        Cache key (filename-safe hash)
    """
    # Hash the absolute path
    path_hash = hashlib.sha256(file_path.encode('utf-8')).hexdigest()[:16]

    # Include basename for readability
    basename = os.path.basename(file_path)
    # Sanitize basename to be filename-safe
    safe_basename = ''.join(c if c.isalnum() or c in '._-' else '_' for c in basename)

    return f'{safe_basename}_{path_hash}'


def get_cache_path(file_path: str) -> Path:
    """Get the cache file path for a given file.

    Args:
        file_path: Absolute path to file

    Returns:
        Path to cache file
    """
    cache_dir = get_analyse_cache_dir()
    cache_key = get_cache_key(file_path)
    return cache_dir / f'{cache_key}.json'


def is_cache_valid(file_path: str, cache_data: AnalyseCacheData) -> bool:
    """Check if cache is still valid for the file.

    Cache is valid if file size and mtime haven't changed.

    Args:
        file_path: Absolute path to file
        cache_data: Cached data

    Returns:
        True if cache is valid, False otherwise
    """
    try:
        stat = os.stat(file_path)
        current_size = stat.st_size
        current_mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()

        # Check if size and mtime match
        return cache_data.file_size == current_size and cache_data.file_mtime == current_mtime
    except (OSError, FileNotFoundError):
        return False


def load_cache(file_path: str) -> dict | None:
    """Load cached analysis result if valid.

    Args:
        file_path: Absolute path to file

    Returns:
        Cached analysis result dict, or None if cache doesn't exist or is invalid
    """
    cache_path = get_cache_path(file_path)

    if not cache_path.exists():
        logger.debug(f'No cache found for {file_path}')
        return None

    try:
        with open(cache_path) as f:
            cache_json = json.load(f)

        cache_data = AnalyseCacheData(**cache_json)

        # Validate cache
        if not is_cache_valid(file_path, cache_data):
            logger.debug(f'Cache invalid (file changed) for {file_path}')
            return None

        logger.info(f'Cache hit for {file_path}')
        return cache_data.analysis_result

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f'Failed to load cache for {file_path}: {e}')
        return None


def save_cache(file_path: str, analysis_result: dict) -> bool:
    """Save analysis result to cache.

    Args:
        file_path: Absolute path to file
        analysis_result: FileAnalysisResult as dict

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        stat = os.stat(file_path)

        cache_data = AnalyseCacheData(
            file_path=file_path,
            file_size=stat.st_size,
            file_mtime=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            cached_at=datetime.now(UTC).isoformat(),
            analysis_result=analysis_result,
        )

        cache_path = get_cache_path(file_path)

        with open(cache_path, 'w') as f:
            json.dump(cache_data.model_dump(), f, indent=2)

        logger.info(f'Saved cache for {file_path}')
        return True

    except OSError as e:
        logger.warning(f'Failed to save cache for {file_path}: {e}')
        return False


def delete_cache(file_path: str) -> bool:
    """Delete cache for a file.

    Args:
        file_path: Absolute path to file

    Returns:
        True if deleted, False if didn't exist or error
    """
    cache_path = get_cache_path(file_path)

    try:
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f'Deleted cache for {file_path}')
            return True
        return False
    except OSError as e:
        logger.warning(f'Failed to delete cache for {file_path}: {e}')
        return False


def clear_all_caches() -> int:
    """Clear all analyse caches.

    Returns:
        Number of cache files deleted
    """
    cache_dir = get_analyse_cache_dir()
    count = 0

    for cache_file in cache_dir.glob('*.json'):
        try:
            cache_file.unlink()
            count += 1
        except OSError as e:
            logger.warning(f'Failed to delete {cache_file}: {e}')

    logger.info(f'Cleared {count} analyse cache files')
    return count

"""Large file indexing module for efficient line-based access.

This module provides functionality to create and manage line-offset indexes
for large text files, enabling efficient random access by line number.

Index files are stored in ~/.cache/rx/indexes/ and contain:
- Source file metadata (path, size, modification time)
- Analysis results (line counts, statistics)
- Line-to-offset mapping at regular intervals
"""

import bisect
import hashlib
import json
import logging
import os
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rx.utils import NEWLINE_SYMBOL, get_int_env

logger = logging.getLogger(__name__)

# Constants
INDEX_VERSION = 1
DEFAULT_LARGE_FILE_MB = 100
CACHE_DIR_NAME = "rx/indexes"


def get_cache_dir() -> Path:
    """Get the cache directory path, creating it if necessary."""
    # Use XDG_CACHE_HOME if set, otherwise ~/.cache
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        base = Path(xdg_cache)
    else:
        base = Path.home() / ".cache"

    cache_dir = base / CACHE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_large_file_threshold_bytes() -> int:
    """Get the threshold in bytes for creating indexes.

    Controlled by RX_LARGE_TEXT_FILE_MB environment variable.
    Default: 100MB
    """
    threshold_mb = get_int_env("RX_LARGE_TEXT_FILE_MB")
    if threshold_mb <= 0:
        threshold_mb = DEFAULT_LARGE_FILE_MB
    return threshold_mb * 1024 * 1024


def get_index_step_bytes() -> int:
    """Get the index step size in bytes.

    Step size is threshold / 10.
    Default: 10MB (when threshold is 100MB)
    """
    return get_large_file_threshold_bytes() // 10


def get_index_path(source_path: str) -> Path:
    """Generate the index file path for a source file.

    Uses SHA256 hash of absolute path (first 16 chars) plus filename
    to create a unique but identifiable index filename.

    Args:
        source_path: Path to the source file

    Returns:
        Path to the index file in cache directory
    """
    abs_path = os.path.abspath(source_path)
    path_hash = hashlib.sha256(abs_path.encode()).hexdigest()[:16]
    filename = os.path.basename(source_path)
    index_filename = f"{path_hash}_{filename}.json"
    return get_cache_dir() / index_filename


def is_index_valid(source_path: str) -> bool:
    """Check if a valid index exists for the source file.

    An index is valid if:
    - Index file exists
    - Source file modification time matches
    - Source file size matches

    Args:
        source_path: Path to the source file

    Returns:
        True if valid index exists, False otherwise
    """
    index_path = get_index_path(source_path)
    if not index_path.exists():
        return False

    try:
        index_data = load_index(index_path)
        if index_data is None:
            return False

        source_stat = os.stat(source_path)
        source_mtime = datetime.fromtimestamp(source_stat.st_mtime).isoformat()

        return (
            index_data.get("source_modified_at") == source_mtime
            and index_data.get("source_size_bytes") == source_stat.st_size
        )
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Index validation failed for {source_path}: {e}")
        return False


def load_index(index_path: Path | str) -> dict | None:
    """Load an index file from disk.

    Args:
        index_path: Path to the index file

    Returns:
        Index data dictionary, or None if loading fails
    """
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate version
        if data.get("version") != INDEX_VERSION:
            logger.warning(f"Index version mismatch: {data.get('version')} != {INDEX_VERSION}")
            return None

        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(f"Failed to load index {index_path}: {e}")
        return None


def save_index(index_data: dict, index_path: Path | str) -> bool:
    """Save an index to disk.

    Args:
        index_data: Index data dictionary
        index_path: Path to save the index

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Ensure parent directory exists
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
        return True
    except OSError as e:
        logger.error(f"Failed to save index {index_path}: {e}")
        return False


def delete_index(source_path: str) -> bool:
    """Delete the index file for a source file.

    Args:
        source_path: Path to the source file

    Returns:
        True if deleted (or didn't exist), False on error
    """
    index_path = get_index_path(source_path)
    try:
        if index_path.exists():
            index_path.unlink()
            logger.info(f"Deleted index for {source_path}")
        return True
    except OSError as e:
        logger.error(f"Failed to delete index {index_path}: {e}")
        return False


@dataclass
class IndexBuildResult:
    """Result of building an index."""

    line_index: list[list[int]]  # [[line_number, byte_offset], ...]
    line_count: int
    empty_line_count: int
    line_length_max: int
    line_length_avg: float
    line_length_median: float
    line_length_p95: float
    line_length_p99: float
    line_length_stddev: float
    line_length_max_line_number: int
    line_length_max_byte_offset: int
    line_ending: str


def _percentile(data: list[int], p: float) -> float:
    """Calculate the p-th percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    k = (n - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < n else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _detect_line_ending(sample_bytes: bytes) -> str:
    """Detect line ending style from a sample of bytes."""
    crlf_count = sample_bytes.count(b"\r\n")
    cr_count = sample_bytes.count(b"\r") - crlf_count
    lf_count = sample_bytes.count(b"\n") - crlf_count

    endings = []
    if crlf_count > 0:
        endings.append(("CRLF", crlf_count))
    if lf_count > 0:
        endings.append(("LF", lf_count))
    if cr_count > 0:
        endings.append(("CR", cr_count))

    if len(endings) == 0:
        return "LF"  # Default
    elif len(endings) == 1:
        return endings[0][0]
    else:
        return "mixed"


def build_index(source_path: str, step_bytes: int | None = None) -> IndexBuildResult:
    """Build a line-offset index for a file.

    Creates index entries at approximately every step_bytes interval,
    with all offsets aligned to line starts.

    Args:
        source_path: Path to the source file
        step_bytes: Bytes between index entries (default: from config)

    Returns:
        IndexBuildResult with line index and analysis data
    """
    if step_bytes is None:
        step_bytes = get_index_step_bytes()

    line_index: list[list[int]] = [[1, 0]]  # First line always at offset 0

    current_offset = 0
    current_line = 0
    next_checkpoint = step_bytes

    line_lengths: list[int] = []
    empty_line_count = 0
    max_line_length = 0
    max_line_number = 0
    max_line_offset = 0

    # Sample for line ending detection (first 64KB)
    line_ending_sample = b""
    sample_collected = False

    with open(source_path, "rb") as f:
        for line in f:
            current_line += 1
            line_len_bytes = len(line)

            # Collect sample for line ending detection
            if not sample_collected:
                line_ending_sample += line
                if len(line_ending_sample) >= 65536:
                    sample_collected = True

            # Strip line ending for length calculation
            stripped = line.rstrip(b"\r\n")
            content_len = len(stripped)

            # Track line statistics (for non-empty lines)
            if stripped.strip():  # Has non-whitespace content
                line_lengths.append(content_len)
                if content_len > max_line_length:
                    max_line_length = content_len
                    max_line_number = current_line
                    max_line_offset = current_offset
            else:
                empty_line_count += 1

            current_offset += line_len_bytes

            # Check if we've passed the next checkpoint
            if current_offset >= next_checkpoint:
                # Record the start of the NEXT line
                # current_offset is now at the start of the next line
                line_index.append([current_line + 1, current_offset])
                next_checkpoint = current_offset + step_bytes

    # Calculate statistics
    if line_lengths:
        line_length_avg = statistics.mean(line_lengths)
        line_length_median = statistics.median(line_lengths)
        line_length_p95 = _percentile(line_lengths, 95)
        line_length_p99 = _percentile(line_lengths, 99)
        line_length_stddev = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0.0
    else:
        line_length_avg = 0.0
        line_length_median = 0.0
        line_length_p95 = 0.0
        line_length_p99 = 0.0
        line_length_stddev = 0.0

    line_ending = _detect_line_ending(line_ending_sample)

    return IndexBuildResult(
        line_index=line_index,
        line_count=current_line,
        empty_line_count=empty_line_count,
        line_length_max=max_line_length,
        line_length_avg=line_length_avg,
        line_length_median=line_length_median,
        line_length_p95=line_length_p95,
        line_length_p99=line_length_p99,
        line_length_stddev=line_length_stddev,
        line_length_max_line_number=max_line_number,
        line_length_max_byte_offset=max_line_offset,
        line_ending=line_ending,
    )


def create_index_file(source_path: str, force: bool = False) -> dict | None:
    """Create or update an index file for a source file.

    Args:
        source_path: Path to the source file
        force: If True, rebuild even if valid index exists

    Returns:
        Index data dictionary, or None on failure
    """
    abs_path = os.path.abspath(source_path)

    # Check if valid index exists (unless forcing)
    if not force and is_index_valid(source_path):
        logger.info(f"Valid index exists for {source_path}")
        return load_index(get_index_path(source_path))

    # Build new index
    logger.info(f"Building index for {source_path}")
    try:
        source_stat = os.stat(source_path)
        build_result = build_index(source_path)

        index_data = {
            "version": INDEX_VERSION,
            "source_path": abs_path,
            "source_modified_at": datetime.fromtimestamp(source_stat.st_mtime).isoformat(),
            "source_size_bytes": source_stat.st_size,
            "index_step_bytes": get_index_step_bytes(),
            "created_at": datetime.now().isoformat(),
            "analysis": {
                "line_count": build_result.line_count,
                "empty_line_count": build_result.empty_line_count,
                "line_length_max": build_result.line_length_max,
                "line_length_avg": build_result.line_length_avg,
                "line_length_median": build_result.line_length_median,
                "line_length_p95": build_result.line_length_p95,
                "line_length_p99": build_result.line_length_p99,
                "line_length_stddev": build_result.line_length_stddev,
                "line_length_max_line_number": build_result.line_length_max_line_number,
                "line_length_max_byte_offset": build_result.line_length_max_byte_offset,
                "line_ending": build_result.line_ending,
            },
            "line_index": build_result.line_index,
        }

        index_path = get_index_path(source_path)
        if save_index(index_data, index_path):
            logger.info(f"Index saved to {index_path}")
            return index_data
        return None

    except Exception as e:
        logger.error(f"Failed to build index for {source_path}: {e}")
        return None


def find_line_offset(line_index: list[list[int]], target_line: int) -> tuple[int, int]:
    """Find the closest indexed line before or at target_line.

    Uses binary search for efficient lookup.

    Args:
        line_index: List of [line_number, byte_offset] pairs
        target_line: The line number to find

    Returns:
        Tuple of (line_number, byte_offset) for the closest previous indexed line
    """
    if not line_index:
        return (1, 0)

    # Extract line numbers for binary search
    lines = [entry[0] for entry in line_index]

    # Find rightmost entry with line <= target_line
    idx = bisect.bisect_right(lines, target_line) - 1
    if idx < 0:
        idx = 0

    return (line_index[idx][0], line_index[idx][1])


def get_index_info(source_path: str) -> dict | None:
    """Get information about an existing index.

    Args:
        source_path: Path to the source file

    Returns:
        Dictionary with index info, or None if no index exists
    """
    index_path = get_index_path(source_path)
    if not index_path.exists():
        return None

    index_data = load_index(index_path)
    if index_data is None:
        return None

    return {
        "index_path": str(index_path),
        "source_path": index_data.get("source_path"),
        "source_size_bytes": index_data.get("source_size_bytes"),
        "source_modified_at": index_data.get("source_modified_at"),
        "created_at": index_data.get("created_at"),
        "index_step_bytes": index_data.get("index_step_bytes"),
        "index_entries": len(index_data.get("line_index", [])),
        "is_valid": is_index_valid(source_path),
        "analysis": index_data.get("analysis"),
    }

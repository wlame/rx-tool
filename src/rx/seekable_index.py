"""Seekable zstd index module.

This module manages companion index files for seekable zstd files.
Index files are stored in ~/.cache/rx/indexes/ and contain:
1. Seek table cache (avoid re-reading from file each time)
2. Line-to-frame mapping for fast line access
3. Frame-to-line-range mapping

The index enables O(1) lookup of which frame contains a given line,
enabling fast samples extraction without full decompression.
"""

import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from rx.seekable_zstd import (
    DEFAULT_FRAME_SIZE_BYTES,
    FrameInfo,
    decompress_frame,
    get_seekable_zstd_info,
    is_seekable_zstd,
    read_seek_table,
)


logger = logging.getLogger(__name__)

# Index version for compatibility checking
SEEKABLE_INDEX_VERSION = 1

# Index directory name under XDG cache
INDEX_DIR_NAME = 'rx/indexes'

# Sampling interval for line index (every N lines)
LINE_INDEX_INTERVAL = 10000


@dataclass
class FrameLineInfo:
    """Frame information with line mapping."""

    index: int
    compressed_offset: int
    compressed_size: int
    decompressed_offset: int
    decompressed_size: int
    first_line: int
    last_line: int
    line_count: int

    @classmethod
    def from_frame_info(cls, frame: FrameInfo, first_line: int, last_line: int) -> 'FrameLineInfo':
        """Create FrameLineInfo from FrameInfo with line data."""
        return cls(
            index=frame.index,
            compressed_offset=frame.compressed_offset,
            compressed_size=frame.compressed_size,
            decompressed_offset=frame.decompressed_offset,
            decompressed_size=frame.decompressed_size,
            first_line=first_line,
            last_line=last_line,
            line_count=last_line - first_line + 1,
        )


@dataclass
class SeekableIndex:
    """Complete index for a seekable zstd file."""

    version: int
    source_zst_path: str
    source_zst_modified_at: str
    source_zst_size_bytes: int
    decompressed_size_bytes: int
    total_lines: int
    frame_count: int
    frame_size_target: int
    frames: list[FrameLineInfo] = field(default_factory=list)
    # Sampled line index: list of (line_number, decompressed_offset, frame_index)
    line_index: list[tuple[int, int, int]] = field(default_factory=list)
    created_at: str = ''

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'version': self.version,
            'source_zst_path': self.source_zst_path,
            'source_zst_modified_at': self.source_zst_modified_at,
            'source_zst_size_bytes': self.source_zst_size_bytes,
            'decompressed_size_bytes': self.decompressed_size_bytes,
            'total_lines': self.total_lines,
            'frame_count': self.frame_count,
            'frame_size_target': self.frame_size_target,
            'frames': [asdict(f) for f in self.frames],
            'line_index': self.line_index,
            'created_at': self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SeekableIndex':
        """Create SeekableIndex from dictionary."""
        frames = [FrameLineInfo(**f) for f in data.get('frames', [])]
        return cls(
            version=data.get('version', SEEKABLE_INDEX_VERSION),
            source_zst_path=data.get('source_zst_path', ''),
            source_zst_modified_at=data.get('source_zst_modified_at', ''),
            source_zst_size_bytes=data.get('source_zst_size_bytes', 0),
            decompressed_size_bytes=data.get('decompressed_size_bytes', 0),
            total_lines=data.get('total_lines', 0),
            frame_count=data.get('frame_count', 0),
            frame_size_target=data.get('frame_size_target', DEFAULT_FRAME_SIZE_BYTES),
            frames=frames,
            line_index=data.get('line_index', []),
            created_at=data.get('created_at', ''),
        )


def get_index_dir() -> Path:
    """Get the index directory path, creating it if necessary."""
    xdg_cache = os.environ.get('XDG_CACHE_HOME')
    if xdg_cache:
        base = Path(xdg_cache)
    else:
        base = Path.home() / '.cache'

    index_dir = base / INDEX_DIR_NAME
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir


def get_index_path(zst_path: str | Path) -> Path:
    """Get the index file path for a seekable zstd file.

    Index files are stored in ~/.cache/rx/indexes/ with names based on
    hash of the absolute path plus the filename for readability.

    Args:
        zst_path: Path to the seekable zstd file

    Returns:
        Path to the index file
    """
    zst_path = Path(zst_path)
    abs_path = str(zst_path.resolve())
    path_hash = hashlib.sha256(abs_path.encode()).hexdigest()[:16]
    filename = zst_path.name
    index_filename = f'{path_hash}_{filename}.idx.json'
    return get_index_dir() / index_filename


def is_index_valid(zst_path: str | Path) -> bool:
    """Check if a valid index exists for the given zstd file.

    An index is valid if:
    - Index file exists
    - Zst file modification time matches
    - Zst file size matches
    - Index version matches

    Args:
        zst_path: Path to the seekable zstd file

    Returns:
        True if valid index exists, False otherwise
    """
    zst_path = Path(zst_path)
    index_path = get_index_path(zst_path)

    if not index_path.exists():
        return False

    if not zst_path.exists():
        return False

    try:
        index = load_index(index_path)
        if index is None:
            return False

        # Check version
        if index.version != SEEKABLE_INDEX_VERSION:
            logger.debug(f'Index version mismatch: {index.version} != {SEEKABLE_INDEX_VERSION}')
            return False

        # Check file metadata
        zst_stat = zst_path.stat()
        zst_mtime = datetime.fromtimestamp(zst_stat.st_mtime).isoformat()

        if index.source_zst_modified_at != zst_mtime:
            logger.debug(f'Index invalid: mtime mismatch for {zst_path}')
            return False

        if index.source_zst_size_bytes != zst_stat.st_size:
            logger.debug(f'Index invalid: size mismatch for {zst_path}')
            return False

        return True

    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.debug(f'Index validation failed for {zst_path}: {e}')
        return False


def load_index(index_path: Path | str) -> SeekableIndex | None:
    """Load an index file from disk.

    Args:
        index_path: Path to the index file

    Returns:
        SeekableIndex object, or None if loading fails
    """
    try:
        with open(index_path, encoding='utf-8') as f:
            data = json.load(f)
        return SeekableIndex.from_dict(data)
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(f'Failed to load index {index_path}: {e}')
        return None


def save_index(index: SeekableIndex, index_path: Path | str) -> bool:
    """Save an index to disk.

    Args:
        index: SeekableIndex object
        index_path: Path to save the index

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index.to_dict(), f, indent=2)

        logger.info(f'Seekable index saved to {index_path}')
        return True

    except OSError as e:
        logger.error(f'Failed to save index {index_path}: {e}')
        return False


def get_index(zst_path: str | Path) -> SeekableIndex | None:
    """Get index for a seekable zstd file, loading from cache if valid.

    Args:
        zst_path: Path to the seekable zstd file

    Returns:
        SeekableIndex if available and valid, None otherwise
    """
    zst_path = Path(zst_path)

    if not is_index_valid(zst_path):
        return None

    return load_index(get_index_path(zst_path))


def build_index(
    zst_path: str | Path,
    progress_callback: Callable | None = None,
) -> SeekableIndex:
    """Build a comprehensive index for a seekable zstd file.

    This decompresses each frame to count lines and build the line-to-frame
    mapping. The index is saved to the cache directory.

    Args:
        zst_path: Path to the seekable zstd file
        progress_callback: Optional callback(frame_index, total_frames)

    Returns:
        SeekableIndex with complete line mapping

    Raises:
        ValueError: If file is not a valid seekable zstd
    """
    zst_path = Path(zst_path)

    if not is_seekable_zstd(zst_path):
        raise ValueError(f'Not a seekable zstd file: {zst_path}')

    logger.info(f'Building index for {zst_path}...')

    # Get basic info
    zst_info = get_seekable_zstd_info(zst_path)
    frames = read_seek_table(zst_path)

    zst_stat = zst_path.stat()
    zst_mtime = datetime.fromtimestamp(zst_stat.st_mtime).isoformat()

    # Build frame line info by decompressing each frame
    frame_line_infos = []
    line_index = []
    current_line = 1
    total_lines = 0

    for i, frame in enumerate(frames):
        if progress_callback:
            progress_callback(i, len(frames))

        # Decompress frame to count lines
        frame_data = decompress_frame(zst_path, frame.index, frames)
        frame_text = frame_data.decode('utf-8', errors='replace')

        # Count lines in this frame
        lines_in_frame = frame_text.count('\n')
        # Only add 1 for partial line if this is the last frame
        # (intermediate frames may be split mid-line, which would cause double-counting)
        is_last_frame = i == len(frames) - 1
        if is_last_frame and frame_text and not frame_text.endswith('\n'):
            # Partial line at end of file counts as a line
            lines_in_frame += 1

        first_line = current_line
        last_line = current_line + lines_in_frame - 1 if lines_in_frame > 0 else current_line

        frame_line_infos.append(FrameLineInfo.from_frame_info(frame, first_line, last_line))

        # Add sampled line index entries for this frame
        # Store entry for first line of frame
        line_index.append((first_line, frame.decompressed_offset, frame.index))

        # Add intermediate entries at LINE_INDEX_INTERVAL
        if lines_in_frame > LINE_INDEX_INTERVAL:
            # Track byte offset within frame for sampled lines
            byte_offset = 0
            line_num = first_line

            for line in frame_text.split('\n'):
                if line_num > first_line and (line_num - first_line) % LINE_INDEX_INTERVAL == 0:
                    decompressed_offset = frame.decompressed_offset + byte_offset
                    line_index.append((line_num, decompressed_offset, frame.index))

                byte_offset += len(line.encode('utf-8')) + 1  # +1 for newline
                line_num += 1

        current_line = last_line + 1
        total_lines += lines_in_frame

    if progress_callback:
        progress_callback(len(frames), len(frames))

    # Create index
    index = SeekableIndex(
        version=SEEKABLE_INDEX_VERSION,
        source_zst_path=str(zst_path.resolve()),
        source_zst_modified_at=zst_mtime,
        source_zst_size_bytes=zst_stat.st_size,
        decompressed_size_bytes=zst_info.decompressed_size,
        total_lines=total_lines,
        frame_count=len(frames),
        frame_size_target=zst_info.frame_size_target,
        frames=frame_line_infos,
        line_index=line_index,
        created_at=datetime.now().isoformat(),
    )

    # Save index
    index_path = get_index_path(zst_path)
    save_index(index, index_path)

    logger.info(f'Index built: {total_lines} lines in {len(frames)} frames')
    return index


def get_or_build_index(
    zst_path: str | Path,
    progress_callback: Callable | None = None,
) -> SeekableIndex:
    """Get existing index or build a new one.

    Args:
        zst_path: Path to the seekable zstd file
        progress_callback: Optional callback for build progress

    Returns:
        SeekableIndex (from cache or newly built)
    """
    zst_path = Path(zst_path)

    # Try to get existing valid index
    index = get_index(zst_path)
    if index is not None:
        logger.debug(f'Using cached index for {zst_path}')
        return index

    # Build new index
    return build_index(zst_path, progress_callback)


def find_frame_for_line(index: SeekableIndex, line_number: int) -> int:
    """Find which frame contains the given line number.

    Args:
        index: SeekableIndex object
        line_number: 1-based line number

    Returns:
        Frame index (0-based)

    Raises:
        ValueError: If line_number is out of range
    """
    if line_number < 1 or line_number > index.total_lines:
        raise ValueError(f'Line number {line_number} out of range (1-{index.total_lines})')

    # Binary search through frames
    for frame in index.frames:
        if frame.first_line <= line_number <= frame.last_line:
            return frame.index

    # Shouldn't happen if index is consistent
    raise ValueError(f'Line {line_number} not found in any frame')


def find_frames_for_lines(index: SeekableIndex, line_numbers: list[int]) -> dict[int, list[int]]:
    """Find frames for multiple line numbers.

    Args:
        index: SeekableIndex object
        line_numbers: List of 1-based line numbers

    Returns:
        Dictionary mapping frame_index to list of line numbers in that frame
    """
    from collections import defaultdict

    frames_to_lines = defaultdict(list)

    for line_num in line_numbers:
        if 1 <= line_num <= index.total_lines:
            frame_idx = find_frame_for_line(index, line_num)
            frames_to_lines[frame_idx].append(line_num)

    return dict(frames_to_lines)


def find_frames_for_byte_range(index: SeekableIndex, start_offset: int, end_offset: int) -> list[int]:
    """Find frames that cover a decompressed byte range.

    Args:
        index: SeekableIndex object
        start_offset: Start offset in decompressed stream
        end_offset: End offset in decompressed stream

    Returns:
        List of frame indices that overlap with the range
    """
    result = []

    for frame in index.frames:
        frame_start = frame.decompressed_offset
        frame_end = frame.decompressed_offset + frame.decompressed_size

        # Check for overlap
        if frame_start < end_offset and frame_end > start_offset:
            result.append(frame.index)

    return result


def get_frame_info(index: SeekableIndex, frame_index: int) -> FrameLineInfo:
    """Get frame info by index.

    Args:
        index: SeekableIndex object
        frame_index: 0-based frame index

    Returns:
        FrameLineInfo for the specified frame

    Raises:
        ValueError: If frame_index is out of range
    """
    if frame_index < 0 or frame_index >= index.frame_count:
        raise ValueError(f'Frame index {frame_index} out of range (0-{index.frame_count - 1})')

    return index.frames[frame_index]


def delete_index(zst_path: str | Path) -> bool:
    """Delete the index file for a seekable zstd file.

    Args:
        zst_path: Path to the seekable zstd file

    Returns:
        True if deleted (or didn't exist), False on error
    """
    index_path = get_index_path(zst_path)
    try:
        if index_path.exists():
            index_path.unlink()
            logger.info(f'Deleted index for {zst_path}')
        return True
    except OSError as e:
        logger.error(f'Failed to delete index {index_path}: {e}')
        return False


def get_index_info(zst_path: str | Path) -> dict | None:
    """Get information about an existing index.

    Args:
        zst_path: Path to the seekable zstd file

    Returns:
        Dictionary with index info, or None if no index exists
    """
    index_path = get_index_path(zst_path)

    if not index_path.exists():
        return None

    index = load_index(index_path)
    if index is None:
        return None

    return {
        'index_path': str(index_path),
        'source_zst_path': index.source_zst_path,
        'source_zst_size_bytes': index.source_zst_size_bytes,
        'source_zst_modified_at': index.source_zst_modified_at,
        'decompressed_size_bytes': index.decompressed_size_bytes,
        'total_lines': index.total_lines,
        'frame_count': index.frame_count,
        'frame_size_target': index.frame_size_target,
        'created_at': index.created_at,
        'is_valid': is_index_valid(zst_path),
        'line_index_entries': len(index.line_index),
    }

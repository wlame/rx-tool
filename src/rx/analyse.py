"""Experimental features!! File analysis module"""

import logging
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import Any, Callable

from rx.index import (
    create_index_file,
    get_index_path,
    get_large_file_threshold_bytes,
    is_index_valid,
    load_index,
)
from rx.parse import is_text_file
from rx.regex import calculate_regex_complexity
from rx.utils import NEWLINE_SYMBOL

logger = logging.getLogger(__name__)


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


@dataclass
class FileAnalysisResult:
    """Result of analyzing a single file."""

    file_id: str
    filepath: str
    size_bytes: int
    size_human: str
    is_text: bool

    # Metadata
    created_at: str | None = None
    modified_at: str | None = None
    permissions: str | None = None
    owner: str | None = None

    # Text file metrics (only if is_text=True)
    line_count: int | None = None
    empty_line_count: int | None = None
    line_length_max: int | None = None
    line_length_avg: float | None = None
    line_length_median: float | None = None
    line_length_p95: float | None = None
    line_length_p99: float | None = None
    line_length_stddev: float | None = None

    # Longest line info
    line_length_max_line_number: int | None = None
    line_length_max_byte_offset: int | None = None

    # Line ending info
    line_ending: str | None = None  # "LF", "CRLF", "CR", or "mixed"

    # Additional metrics can be added by plugins
    custom_metrics: dict[str, Any] = field(default_factory=dict)


class FileAnalyzer:
    """
    Pluggable file analysis system.

    Supports adding custom analysis functions that can:
    - Analyze file metadata
    - Process file content line by line
    - Compute custom metrics

    For large files (>= RX_LARGE_TEXT_FILE_MB), analysis results are cached
    in index files for faster subsequent access.
    """

    def __init__(self, use_index_cache: bool = True):
        """Initialize the analyzer.

        Args:
            use_index_cache: If True, use cached analysis from index files
                           when available. Default: True
        """
        self.file_hooks: list[Callable] = []
        self.line_hooks: list[Callable] = []
        self.post_hooks: list[Callable] = []
        self.use_index_cache = use_index_cache

    def register_file_hook(self, hook: Callable):
        """
        Register a hook that processes file metadata.

        Hook signature: hook(filepath: str, result: FileAnalysisResult) -> None
        """
        self.file_hooks.append(hook)

    def register_line_hook(self, hook: Callable):
        """
        Register a hook that processes each line.

        Hook signature: hook(line: str, line_num: int, result: FileAnalysisResult) -> None
        """
        self.line_hooks.append(hook)

    def register_post_hook(self, hook: Callable):
        """
        Register a hook that runs after file processing.

        Hook signature: hook(result: FileAnalysisResult) -> None
        """
        self.post_hooks.append(hook)

    def _try_load_from_cache(self, filepath: str, file_id: str) -> FileAnalysisResult | None:
        """Try to load analysis results from cached index.

        Args:
            filepath: Path to the file
            file_id: File ID for the result

        Returns:
            FileAnalysisResult if valid cache exists, None otherwise
        """
        if not self.use_index_cache:
            return None

        if not is_index_valid(filepath):
            return None

        index_data = load_index(get_index_path(filepath))
        if index_data is None:
            return None

        analysis = index_data.get("analysis")
        if analysis is None:
            return None

        logger.debug(f"Using cached analysis for {filepath}")

        try:
            stat_info = os.stat(filepath)

            result = FileAnalysisResult(
                file_id=file_id,
                filepath=filepath,
                size_bytes=index_data.get("source_size_bytes", stat_info.st_size),
                size_human=human_readable_size(index_data.get("source_size_bytes", stat_info.st_size)),
                is_text=True,  # Index only exists for text files
            )

            # File metadata from stat
            result.created_at = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
            result.modified_at = index_data.get("source_modified_at")
            result.permissions = oct(stat_info.st_mode)[-3:]

            try:
                import pwd

                result.owner = pwd.getpwuid(stat_info.st_uid).pw_name
            except (ImportError, KeyError):
                result.owner = str(stat_info.st_uid)

            # Analysis data from cache
            result.line_count = analysis.get("line_count")
            result.empty_line_count = analysis.get("empty_line_count")
            result.line_length_max = analysis.get("line_length_max")
            result.line_length_avg = analysis.get("line_length_avg")
            result.line_length_median = analysis.get("line_length_median")
            result.line_length_p95 = analysis.get("line_length_p95")
            result.line_length_p99 = analysis.get("line_length_p99")
            result.line_length_stddev = analysis.get("line_length_stddev")
            result.line_length_max_line_number = analysis.get("line_length_max_line_number")
            result.line_length_max_byte_offset = analysis.get("line_length_max_byte_offset")
            result.line_ending = analysis.get("line_ending")

            return result

        except Exception as e:
            logger.debug(f"Failed to load from cache for {filepath}: {e}")
            return None

    def analyze_file(self, filepath: str, file_id: str) -> FileAnalysisResult:
        """Analyze a single file with all registered hooks.

        For large text files with valid cached indexes, analysis data is
        loaded from cache for better performance.
        """
        # Try to use cached analysis first
        cached_result = self._try_load_from_cache(filepath, file_id)
        if cached_result is not None:
            # Still run hooks on cached result
            for hook in self.file_hooks:
                try:
                    hook(filepath, cached_result)
                except Exception as e:
                    logger.warning(f"File hook failed for {filepath}: {e}")
            for hook in self.post_hooks:
                try:
                    hook(cached_result)
                except Exception as e:
                    logger.warning(f"Post hook failed for {filepath}: {e}")
            return cached_result

        try:
            stat_info = os.stat(filepath)
            size_bytes = stat_info.st_size

            # Initialize result
            result = FileAnalysisResult(
                file_id=file_id,
                filepath=filepath,
                size_bytes=size_bytes,
                size_human=human_readable_size(size_bytes),
                is_text=is_text_file(filepath),
            )

            # File metadata
            result.created_at = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
            result.modified_at = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
            result.permissions = oct(stat_info.st_mode)[-3:]

            try:
                import pwd

                result.owner = pwd.getpwuid(stat_info.st_uid).pw_name
            except (ImportError, KeyError):
                result.owner = str(stat_info.st_uid)

            # Run file-level hooks
            for hook in self.file_hooks:
                try:
                    hook(filepath, result)
                except Exception as e:
                    logger.warning(f"File hook failed for {filepath}: {e}")

            # Analyze text files
            if result.is_text:
                self._analyze_text_file(filepath, result)

                # Create index for large files
                if size_bytes >= get_large_file_threshold_bytes():
                    self._create_index_for_result(filepath, result)

            # Run post-processing hooks
            for hook in self.post_hooks:
                try:
                    hook(result)
                except Exception as e:
                    logger.warning(f"Post hook failed for {filepath}: {e}")

            return result

        except Exception as e:
            logger.error(f"Failed to analyze {filepath}: {e}")
            # Return minimal result for failed files
            return FileAnalysisResult(
                file_id=file_id,
                filepath=filepath,
                size_bytes=0,
                size_human="0 B",
                is_text=False,
            )

    def _create_index_for_result(self, filepath: str, result: FileAnalysisResult):
        """Create an index file from analysis result.

        This is called after analyzing large files to cache the results.
        """
        try:
            logger.info(f"Creating index for large file: {filepath}")
            # Use create_index_file which will build a full index with line offsets
            create_index_file(filepath, force=True)
        except Exception as e:
            logger.warning(f"Failed to create index for {filepath}: {e}")

    def _analyze_text_file(self, filepath: str, result: FileAnalysisResult):
        """Analyze text file content."""
        try:
            # Read raw bytes first to detect line endings
            with open(filepath, 'rb') as f:
                raw_content = f.read()

            # Detect line endings
            result.line_ending = self._detect_line_ending(raw_content)

            # Now read as text for line analysis
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # Basic line metrics
            result.line_count = len(lines)

            # Track byte offsets for each line
            byte_offset = 0
            line_data = []  # [(line_num, stripped_line, byte_offset), ...]

            for line_num, line in enumerate(lines, 1):
                stripped = line.rstrip(NEWLINE_SYMBOL + '\r')
                if line.strip():  # non-empty line
                    line_data.append((line_num, stripped, byte_offset))
                byte_offset += len(line.encode('utf-8'))

            result.empty_line_count = len(lines) - len(line_data)

            if line_data:
                line_lengths = [len(stripped) for _, stripped, _ in line_data]
                result.line_length_max = max(line_lengths)
                result.line_length_avg = statistics.mean(line_lengths)
                result.line_length_median = statistics.median(line_lengths)

                # Calculate percentiles
                result.line_length_p95 = self._percentile(line_lengths, 95)
                result.line_length_p99 = self._percentile(line_lengths, 99)

                # Find longest line info
                max_idx = line_lengths.index(result.line_length_max)
                result.line_length_max_line_number = line_data[max_idx][0]
                result.line_length_max_byte_offset = line_data[max_idx][2]

                if len(line_lengths) > 1:
                    result.line_length_stddev = statistics.stdev(line_lengths)
                else:
                    result.line_length_stddev = 0.0
            else:
                result.line_length_max = 0
                result.line_length_avg = 0.0
                result.line_length_median = 0.0
                result.line_length_p95 = 0.0
                result.line_length_p99 = 0.0
                result.line_length_stddev = 0.0

            # Run line-level hooks
            for line_num, line in enumerate(lines, 1):
                for hook in self.line_hooks:
                    try:
                        hook(line, line_num, result)
                    except Exception as e:
                        logger.warning(f"Line hook {hook.__name__} failed at {filepath}:{line_num}: {e}")

        except Exception as e:
            logger.error(f"Failed to analyze text content of {filepath}: {e}")

    @staticmethod
    def _percentile(data: list[int | float], p: float) -> float:
        """Calculate the p-th percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        n = len(sorted_data)
        k = (n - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < n else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    @staticmethod
    def _detect_line_ending(content: bytes) -> str:
        """Detect the line ending style used in the file."""
        crlf_count = content.count(b'\r\n')
        # Count standalone CR (not followed by LF)
        cr_count = content.count(b'\r') - crlf_count
        # Count standalone LF (not preceded by CR)
        lf_count = content.count(b'\n') - crlf_count

        endings = []
        if crlf_count > 0:
            endings.append(('CRLF', crlf_count))
        if lf_count > 0:
            endings.append(('LF', lf_count))
        if cr_count > 0:
            endings.append(('CR', cr_count))

        if len(endings) == 0:
            return 'LF'  # Default for single-line files
        elif len(endings) == 1:
            return endings[0][0]
        else:
            return 'mixed'


def analyse_path(paths: list[str], max_workers: int = 10) -> dict[str, Any]:
    """
    Analyze files at given paths.

    For directories, only text files are analyzed (binary files are skipped).

    Args:
        paths: List of file or directory paths
        max_workers: Maximum number of parallel workers

    Returns:
        Dictionary with analysis results in ID-based format
    """
    start_time = time()

    # Collect all files to analyze
    files_to_analyze = []
    skipped_binary_files = []
    for path in paths:
        if os.path.isfile(path):
            # Single file - always analyze (even if binary)
            files_to_analyze.append(path)
        elif os.path.isdir(path):
            # Scan directory for text files only
            for root, dirs, files in os.walk(path):
                for file in files:
                    filepath = os.path.join(root, file)
                    if is_text_file(filepath):
                        files_to_analyze.append(filepath)
                    else:
                        skipped_binary_files.append(filepath)
        else:
            logger.warning(f"Path not found: {path}")

    # Create file IDs
    file_ids = {f"f{i + 1}": filepath for i, filepath in enumerate(files_to_analyze)}

    # Analyze files in parallel
    analyzer = FileAnalyzer()
    results = []
    scanned_files = []
    skipped_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(analyzer.analyze_file, filepath, file_id): (file_id, filepath)
            for file_id, filepath in file_ids.items()
        }

        for future in as_completed(future_to_file):
            file_id, filepath = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                scanned_files.append(filepath)
            except Exception as e:
                logger.error(f"Analysis failed for {filepath}: {e}")
                skipped_files.append(filepath)

    elapsed_time = time() - start_time

    # Build response in ID-based format
    return {
        'path': ', '.join(paths) if len(paths) > 1 else paths[0],
        'time': elapsed_time,
        'files': file_ids,
        'results': [
            {
                'file': r.file_id,
                'size_bytes': r.size_bytes,
                'size_human': r.size_human,
                'is_text': r.is_text,
                'created_at': r.created_at,
                'modified_at': r.modified_at,
                'permissions': r.permissions,
                'owner': r.owner,
                'line_count': r.line_count,
                'empty_line_count': r.empty_line_count,
                'line_length_max': r.line_length_max,
                'line_length_avg': r.line_length_avg,
                'line_length_median': r.line_length_median,
                'line_length_p95': r.line_length_p95,
                'line_length_p99': r.line_length_p99,
                'line_length_stddev': r.line_length_stddev,
                'line_length_max_line_number': r.line_length_max_line_number,
                'line_length_max_byte_offset': r.line_length_max_byte_offset,
                'line_ending': r.line_ending,
                'custom_metrics': r.custom_metrics,
            }
            for r in results
        ],
        'scanned_files': scanned_files,
        'skipped_files': skipped_files + skipped_binary_files,
    }


# Re-export calculate_regex_complexity for backward compatibility
__all__ = ['analyse_path', 'calculate_regex_complexity', 'FileAnalyzer', 'FileAnalysisResult']

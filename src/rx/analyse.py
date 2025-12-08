"""Experimental features!! File analysis module"""

import logging
import os
import random
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import Any

from rx import index, seekable_index, seekable_zstd
from rx.compression import decompress_to_file, detect_compression, is_compressed
from rx.file_utils import is_text_file


logger = logging.getLogger(__name__)


def get_sample_size_lines() -> int:
    """Get the sample size for line length statistics from environment variable.

    Returns:
        Sample size in number of lines. Default is 1,000,000.
        Files with fewer non-empty lines than this will have exact statistics.
    """
    try:
        return int(os.environ.get('RX_SAMPLE_SIZE_LINES', '1000000'))
    except (ValueError, TypeError):
        logger.warning('Invalid RX_SAMPLE_SIZE_LINES value, using default 1000000')
        return 1000000


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f'{size_bytes:.2f} {unit}'
        size_bytes /= 1024
    return f'{size_bytes:.2f} PB'


@dataclass
class FileAnalysisState:
    """Internal state for analyzing a single file.

    This is used during the analysis process and contains internal identifiers.
    For API responses, use the Pydantic FileAnalysisResult model from rx.models.
    """

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

    # Compression information
    is_compressed: bool = False
    compression_format: str | None = None
    is_seekable_zstd: bool = False
    compressed_size: int | None = None
    decompressed_size: int | None = None
    compression_ratio: float | None = None

    # Index information
    has_index: bool = False
    index_path: str | None = None
    index_valid: bool = False
    index_checkpoint_count: int | None = None

    # Additional metrics can be added by plugins
    custom_metrics: dict[str, Any] = field(default_factory=dict)


class FileAnalyzer:
    """
    Pluggable file analysis system.

    Supports custom analysis via overridable hook methods:
    - file_hook(): Processes file metadata after basic info is gathered
    - line_hook(): Processes each line during iteration
    - post_hook(): Runs after file processing is complete

    For large files (>= RX_LARGE_TEXT_FILE_MB), analysis results are cached
    in index files for faster subsequent access.

    To add custom analysis, subclass FileAnalyzer and override the hook methods.
    """

    def __init__(self, use_index_cache: bool = True):
        """Initialize the analyzer.

        Args:
            use_index_cache: If True, use cached analysis from index files
                           when available. Default: True
        """
        self.use_index_cache = use_index_cache

    def file_hook(self, filepath: str, result: FileAnalysisState) -> None:
        """
        Hook that processes file metadata.

        Override this method to add custom file-level analysis.

        Args:
            filepath: Path to the file being analyzed
            result: FileAnalysisState object to update with custom metrics
        """
        pass

    def line_hook(self, line: str, line_num: int, result: FileAnalysisState) -> None:
        """
        Hook that processes each line.

        Override this method to add custom line-level analysis.

        Args:
            line: The current line content
            line_num: The line number (1-indexed)
            result: FileAnalysisState object to update with custom metrics
        """
        pass

    def post_hook(self, result: FileAnalysisState) -> None:
        """
        Hook that runs after file processing.

        Override this method to add custom post-processing analysis.

        Args:
            result: FileAnalysisState object to update with custom metrics
        """
        pass

    def _dict_to_state(self, data: dict, file_id: str, filepath: str) -> FileAnalysisState:
        """Convert cached dict to FileAnalysisState."""
        return FileAnalysisState(
            file_id=file_id,
            filepath=filepath,
            size_bytes=data.get('size_bytes', 0),
            size_human=data.get('size_human', '0 B'),
            is_text=data.get('is_text', False),
            created_at=data.get('created_at'),
            modified_at=data.get('modified_at'),
            permissions=data.get('permissions'),
            owner=data.get('owner'),
            line_count=data.get('line_count'),
            empty_line_count=data.get('empty_line_count'),
            line_length_max=data.get('line_length_max'),
            line_length_avg=data.get('line_length_avg'),
            line_length_median=data.get('line_length_median'),
            line_length_p95=data.get('line_length_p95'),
            line_length_p99=data.get('line_length_p99'),
            line_length_stddev=data.get('line_length_stddev'),
            line_length_max_line_number=data.get('line_length_max_line_number'),
            line_length_max_byte_offset=data.get('line_length_max_byte_offset'),
            line_ending=data.get('line_ending'),
            custom_metrics=data.get('custom_metrics', {}),
            # Compression fields
            is_compressed=data.get('is_compressed', False),
            compression_format=data.get('compression_format'),
            is_seekable_zstd=data.get('is_seekable_zstd', False),
            compressed_size=data.get('compressed_size'),
            decompressed_size=data.get('decompressed_size'),
            compression_ratio=data.get('compression_ratio'),
            # Index fields
            has_index=data.get('has_index', False),
            index_path=data.get('index_path'),
            index_valid=data.get('index_valid', False),
            index_checkpoint_count=data.get('index_checkpoint_count'),
        )

    def _state_to_dict(self, result: FileAnalysisState) -> dict:
        """Convert FileAnalysisState to dict for caching."""
        return {
            'file': result.file_id,
            'size_bytes': result.size_bytes,
            'size_human': result.size_human,
            'is_text': result.is_text,
            'created_at': result.created_at,
            'modified_at': result.modified_at,
            'permissions': result.permissions,
            'owner': result.owner,
            'line_count': result.line_count,
            'empty_line_count': result.empty_line_count,
            'line_length_max': result.line_length_max,
            'line_length_avg': result.line_length_avg,
            'line_length_median': result.line_length_median,
            'line_length_p95': result.line_length_p95,
            'line_length_p99': result.line_length_p99,
            'line_length_stddev': result.line_length_stddev,
            'line_length_max_line_number': result.line_length_max_line_number,
            'line_length_max_byte_offset': result.line_length_max_byte_offset,
            'line_ending': result.line_ending,
            'custom_metrics': result.custom_metrics,
            # Compression fields
            'is_compressed': result.is_compressed,
            'compression_format': result.compression_format,
            'is_seekable_zstd': result.is_seekable_zstd,
            'compressed_size': result.compressed_size,
            'decompressed_size': result.decompressed_size,
            'compression_ratio': result.compression_ratio,
            # Index fields
            'has_index': result.has_index,
            'index_path': result.index_path,
            'index_valid': result.index_valid,
            'index_checkpoint_count': result.index_checkpoint_count,
        }

    def _add_index_info(self, filepath: str, result: FileAnalysisState):
        """Add index information to analysis result."""
        try:
            if seekable_zstd.seekable_zstd.is_seekable_zstd(filepath):
                # Check for seekable zstd index
                index_path = seekable_index.index.get_index_path(filepath)
                if os.path.exists(index_path):
                    result.has_index = True
                    result.index_path = str(index_path)
                    result.index_valid = seekable_index.is_index_valid(filepath)

                    if result.index_valid:
                        try:
                            index_data = seekable_index.load_seekable_index(filepath)
                            result.index_checkpoint_count = len(index_data.get('frames', []))
                        except Exception as e:
                            logger.warning(f'Failed to load seekable index: {e}')
            else:
                # Check for regular file index
                index_path = index.index.get_index_path(filepath)
                if os.path.exists(str(index_path)):
                    result.has_index = True
                    result.index_path = str(index_path)
                    result.index_valid = index.is_index_valid(filepath)

                    if result.index_valid:
                        try:
                            index_data = index.load_index(index_path)
                            result.index_checkpoint_count = len(index_data.get('line_index', []))
                        except Exception as e:
                            logger.warning(f'Failed to load index: {e}')
        except Exception as e:
            logger.warning(f'Failed to add index info: {e}')

    def _analyze_compressed_file(self, filepath: str, result: FileAnalysisState):
        """Analyze compressed file by decompressing to /tmp and analyzing."""
        temp_file = None
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tf:
                temp_file = tf.name

            logger.info(f'Decompressing {filepath} to {temp_file}')

            # Decompress to temp file
            try:
                decompress_to_file(filepath, temp_file)
            except OSError as e:
                if 'No space left' in str(e) or 'Disk quota exceeded' in str(e):
                    logger.warning(f'No space left on device, skipping decompression of {filepath}')
                    return
                raise

            # Get decompressed size
            stat = os.stat(temp_file)
            result.decompressed_size = stat.st_size

            # Calculate compression ratio
            if result.compressed_size and result.decompressed_size:
                result.compression_ratio = result.decompressed_size / result.compressed_size

            # Check if decompressed file is text
            if is_text_file(temp_file):
                result.is_text = True
                # Analyze the decompressed content
                self._analyze_text_file(temp_file, result)
            else:
                logger.info(f'Decompressed file is not text: {filepath}')

        except Exception as e:
            logger.error(f'Failed to analyze compressed file {filepath}: {e}')
        finally:
            # IMPORTANT: Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.debug(f'Cleaned up temp file: {temp_file}')
                except OSError as e:
                    logger.warning(f'Failed to remove temp file {temp_file}: {e}')

    def _try_load_from_cache(self, filepath: str, file_id: str) -> FileAnalysisState | None:
        """Try to load analysis results from cached index.

        Args:
            filepath: Path to the file
            file_id: File ID for the result

        Returns:
            FileAnalysisState if valid cache exists, None otherwise
        """
        if not self.use_index_cache:
            return None

        if not index.is_index_valid(filepath):
            return None

        index_data = index.load_index(index.get_index_path(filepath))
        if index_data is None:
            return None

        analysis = index_data.get('analysis')
        if analysis is None:
            return None

        logger.debug(f'Using cached analysis for {filepath}')

        try:
            stat_info = os.stat(filepath)

            result = FileAnalysisState(
                file_id=file_id,
                filepath=filepath,
                size_bytes=index_data.get('source_size_bytes', stat_info.st_size),
                size_human=human_readable_size(index_data.get('source_size_bytes', stat_info.st_size)),
                is_text=True,  # Index only exists for text files
            )

            # File metadata from stat
            result.created_at = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
            result.modified_at = index_data.get('source_modified_at')
            result.permissions = oct(stat_info.st_mode)[-3:]

            try:
                import pwd

                result.owner = pwd.getpwuid(stat_info.st_uid).pw_name
            except (ImportError, KeyError):
                result.owner = str(stat_info.st_uid)

            # Analysis data from cache
            result.line_count = analysis.get('line_count')
            result.empty_line_count = analysis.get('empty_line_count')
            result.line_length_max = analysis.get('line_length_max')
            result.line_length_avg = analysis.get('line_length_avg')
            result.line_length_median = analysis.get('line_length_median')
            result.line_length_p95 = analysis.get('line_length_p95')
            result.line_length_p99 = analysis.get('line_length_p99')
            result.line_length_stddev = analysis.get('line_length_stddev')
            result.line_length_max_line_number = analysis.get('line_length_max_line_number')
            result.line_length_max_byte_offset = analysis.get('line_length_max_byte_offset')
            result.line_ending = analysis.get('line_ending')

            return result

        except Exception as e:
            logger.debug(f'Failed to load from cache for {filepath}: {e}')
            return None

    def analyze_file(self, filepath: str, file_id: str) -> FileAnalysisState:
        """Analyze a single file with all registered hooks.

        For large text files with valid cached indexes, analysis data is
        loaded from cache for better performance.
        """
        # STEP 1: Try analyse_cache first (new cache system)
        from rx.analyse_cache import load_cache, save_cache

        cached = load_cache(filepath)
        if cached:
            logger.info(f'Loaded from analyse_cache: {filepath}')
            # Convert dict back to FileAnalysisState
            result = self._dict_to_state(cached, file_id, filepath)
            # Still run hooks
            try:
                self.file_hook(filepath, result)
            except Exception as e:
                logger.warning(f'File hook failed: {e}')
            try:
                self.post_hook(result)
            except Exception as e:
                logger.warning(f'Post hook failed: {e}')
            return result

        # STEP 2: Try old index cache (keep existing logic)
        cached_result = self._try_load_from_cache(filepath, file_id)
        if cached_result is not None:
            # Still run hooks on cached result
            try:
                self.file_hook(filepath, cached_result)
            except Exception as e:
                logger.warning(f'File hook failed for {filepath}: {e}')
            try:
                self.post_hook(cached_result)
            except Exception as e:
                logger.warning(f'Post hook failed for {filepath}: {e}')
            return cached_result

        # STEP 3: Fresh analysis
        try:
            stat_info = os.stat(filepath)
            size_bytes = stat_info.st_size

            # Initialize result
            result = FileAnalysisState(
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

            # STEP 4: Detect compression (NEW)
            if is_compressed(filepath):
                result.is_compressed = True
                comp_format = detect_compression(filepath)
                result.compression_format = comp_format.value if comp_format else None
                result.compressed_size = size_bytes

                if seekable_zstd.is_seekable_zstd(filepath):
                    result.is_seekable_zstd = True
                    # Get decompressed size from seekable index if available
                    try:
                        index_data = seekable_index.load_index(filepath)
                        if index_data:
                            result.decompressed_size = index_data.get('decompressed_size_bytes')
                            if result.decompressed_size and result.compressed_size:
                                result.compression_ratio = result.decompressed_size / result.compressed_size
                    except Exception as e:
                        logger.debug(f'Could not get seekable zstd info: {e}')

            # STEP 5: Run file-level hooks
            try:
                self.file_hook(filepath, result)
            except Exception as e:
                logger.warning(f'File hook failed for {filepath}: {e}')

            # STEP 6: Analyze content
            if result.is_text:
                self._analyze_text_file(filepath, result)

                # Create index for large files
                if size_bytes >= index.get_large_file_threshold_bytes():
                    self._create_index_for_result(filepath, result)
            elif result.is_compressed:
                # NEW: Handle compressed files
                self._analyze_compressed_file(filepath, result)

            # STEP 7: Add index information (NEW)
            self._add_index_info(filepath, result)

            # STEP 8: Run post-processing hooks
            try:
                self.post_hook(result)
            except Exception as e:
                logger.warning(f'Post hook failed for {filepath}: {e}')

            # STEP 9: Save to analyse_cache (NEW)
            try:
                result_dict = self._state_to_dict(result)
                save_cache(filepath, result_dict)
            except Exception as e:
                logger.warning(f'Failed to save cache: {e}')

            return result

        except Exception as e:
            logger.error(f'Failed to analyze {filepath}: {e}')
            # Return minimal result for failed files
            return FileAnalysisState(
                file_id=file_id,
                filepath=filepath,
                size_bytes=0,
                size_human='0 B',
                is_text=False,
            )

    def _create_index_for_result(self, filepath: str, result: FileAnalysisState):
        """Create an index file from analysis result.

        This is called after analyzing large files to cache the results.
        """
        try:
            logger.info(f'Creating index for large file: {filepath}')
            # Use create_index_file which will build a full index with line offsets
            index.create_index_file(filepath, force=True)
        except Exception as e:
            logger.warning(f'Failed to create index for {filepath}: {e}')

    def _analyze_text_file(self, filepath: str, result: FileAnalysisState):
        """Analyze text file content using streaming to avoid loading entire file in memory."""
        try:
            # Sample first chunk for line ending detection (up to 10MB)
            SAMPLE_SIZE = 10 * 1024 * 1024
            with open(filepath, 'rb') as f:
                sample = f.read(SAMPLE_SIZE)

            # Detect line endings from sample
            result.line_ending = self._detect_line_ending(sample)

            # Now process file line by line (streaming)
            # Use reservoir sampling for percentiles to avoid storing all line lengths
            # IMPORTANT: Use binary mode to count lines consistently with wc -l
            # Text mode treats \r, \n, and \r\n all as line separators, which can
            # double-count lines in files with CRLF that also have bare \r characters
            empty_line_count = 0
            byte_offset = 0
            max_line_length = 0
            max_line_number = 0
            max_line_offset = 0
            last_line_num = 0

            # Streaming statistics
            total_length = 0
            non_empty_count = 0
            sum_of_squares = 0.0

            # Reservoir sampling for percentiles
            # Files with fewer non-empty lines will have exact statistics
            sample_size = get_sample_size_lines()
            line_length_sample = []

            # Use binary mode and split on \n to match wc -l behavior
            with open(filepath, 'rb') as f:
                for line_num, line_bytes in enumerate(f, 1):
                    last_line_num = line_num

                    # Decode line
                    try:
                        line = line_bytes.decode('utf-8', errors='ignore')
                    except Exception:
                        line = ''

                    # Calculate byte offset
                    line_byte_length = len(line_bytes)

                    # Strip line ending for length calculation
                    # Remove \n and \r from the end
                    stripped = line.rstrip('\n\r')

                    if stripped.strip():  # non-empty line
                        line_len = len(stripped)
                        non_empty_count += 1
                        total_length += line_len
                        sum_of_squares += line_len * line_len

                        # Reservoir sampling: keep a random sample of line lengths
                        if len(line_length_sample) < sample_size:
                            line_length_sample.append(line_len)
                        else:
                            # Randomly replace an element with decreasing probability
                            j = random.randint(0, non_empty_count - 1)
                            if j < sample_size:
                                line_length_sample[j] = line_len

                        # Track longest line
                        if line_len > max_line_length:
                            max_line_length = line_len
                            max_line_number = line_num
                            max_line_offset = byte_offset
                    else:
                        empty_line_count += 1

                    # Run line-level hooks on the fly
                    try:
                        self.line_hook(line, line_num, result)
                    except Exception as e:
                        logger.warning(f'Line hook failed at {filepath}:{line_num}: {e}')

                    byte_offset += line_byte_length

            # Set basic line metrics
            result.line_count = last_line_num
            result.empty_line_count = empty_line_count

            # Calculate statistics from streaming data and sample
            if non_empty_count > 0:
                result.line_length_max = max_line_length
                result.line_length_avg = total_length / non_empty_count
                result.line_length_max_line_number = max_line_number
                result.line_length_max_byte_offset = max_line_offset

                # Calculate stddev from sum of squares
                mean = result.line_length_avg
                variance = (sum_of_squares / non_empty_count) - (mean * mean)
                result.line_length_stddev = variance**0.5 if variance > 0 else 0.0

                # Use sample for percentiles
                if line_length_sample:
                    result.line_length_median = statistics.median(line_length_sample)
                    result.line_length_p95 = self._percentile(line_length_sample, 95)
                    result.line_length_p99 = self._percentile(line_length_sample, 99)
                else:
                    result.line_length_median = 0.0
                    result.line_length_p95 = 0.0
                    result.line_length_p99 = 0.0
            else:
                result.line_length_max = 0
                result.line_length_avg = 0.0
                result.line_length_median = 0.0
                result.line_length_p95 = 0.0
                result.line_length_p99 = 0.0
                result.line_length_stddev = 0.0

        except Exception as e:
            logger.error(f'Failed to analyze text content of {filepath}: {e}')

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
            logger.warning(f'Path not found: {path}')

    # Create file IDs
    file_ids = {f'f{i + 1}': filepath for i, filepath in enumerate(files_to_analyze)}

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
                logger.error(f'Analysis failed for {filepath}: {e}')
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
                # Compression fields
                'is_compressed': r.is_compressed,
                'compression_format': r.compression_format,
                'is_seekable_zstd': r.is_seekable_zstd,
                'compressed_size': r.compressed_size,
                'decompressed_size': r.decompressed_size,
                'compression_ratio': r.compression_ratio,
                # Index fields
                'has_index': r.has_index,
                'index_path': r.index_path,
                'index_valid': r.index_valid,
                'index_checkpoint_count': r.index_checkpoint_count,
            }
            for r in results
        ],
        'scanned_files': scanned_files,
        'skipped_files': skipped_files + skipped_binary_files,
    }


__all__ = ['FileAnalysisState', 'FileAnalyzer', 'analyse_path']

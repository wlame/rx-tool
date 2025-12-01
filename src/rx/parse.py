"""Parse functionality using ripgrep"""

import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Import noop prometheus stub by default (CLI mode)
# Real prometheus is only imported in web.py for server mode
from rx.cli import prometheus as prom
from rx.utils import NEWLINE_SYMBOL, NEWLINE_SYMBOL_BYTES

logger = logging.getLogger(__name__)


LINE_SIZE_ASSUMPTION_KB = int(os.getenv('RX_MAX_LINE_SIZE_KB', '8'))  # for context retrieving we assume max avg length
MAX_SUBPROCESSES = int(os.getenv('RX_MAX_SUBPROCESSES', '20'))
MIN_CHUNK_SIZE = int(os.getenv('RX_MIN_CHUNK_SIZE_MB', str(20))) * 1024 * 1024
MAX_FILES = int(os.getenv('RX_MAX_FILES', '1000'))  #


@dataclass
class FileTask:
    """Represents a unit of work for parallel processing"""

    task_id: int
    filepath: str
    offset: int  # Starting byte offset (aligned to newline)
    count: int  # Number of bytes to read

    @property
    def end_offset(self) -> int:
        return self.offset + self.count


def is_text_file(filepath: str, sample_size: int = 8192) -> bool:
    """
    Check if a file is a text file by reading a sample and looking for null bytes.
    Binary files typically contain null bytes, while text files don't.
    """
    try:
        with open(filepath, 'rb') as f:
            sample = f.read(sample_size)
            if b'\x00' in sample:
                return False
            return True
    except Exception:
        return False


def scan_directory_for_text_files(dirpath: str, max_files: int = MAX_FILES) -> tuple[list[str], list[str]]:
    text_files = []
    skipped_files = []

    logger.info(f"[SCAN] Scanning directory: {dirpath}")

    try:
        entries = os.listdir(dirpath)
        logger.debug(f"[SCAN] Found {len(entries)} entries in directory")

        for entry in entries:
            if len(text_files) + len(skipped_files) >= max_files:
                logger.warning(f"[SCAN] Reached max_files limit ({max_files}), stopping scan")
                break

            filepath = os.path.join(dirpath, entry)

            # Skip directories, symlinks, etc - only process regular files
            if not os.path.isfile(filepath):
                logger.debug(f"[SCAN] Skipping non-file: {entry}")
                continue

            # Check if text file
            if is_text_file(filepath):
                text_files.append(filepath)
                logger.debug(f"[SCAN] Added text file: {entry}")
            else:
                skipped_files.append(filepath)
                logger.debug(f"[SCAN] Skipped binary file: {entry}")

        logger.info(f"[SCAN] Completed: {len(text_files)} text files, {len(skipped_files)} skipped")
        return text_files, skipped_files

    except Exception as e:
        logger.error(f"[SCAN] Error scanning directory: {e}")
        raise RuntimeError(f"Directory scan failed: {e}")


def validate_file(filepath: str) -> None:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if not os.path.isfile(filepath):
        raise ValueError(f"Path is not a file: {filepath}")

    if not is_text_file(filepath):
        raise ValueError("File appears to be binary, not text")


def find_next_newline(filename: str, offset: int) -> int:
    """
    Find the position of the next newline after the given offset.
    This ensures we split on line boundaries.

    Args:
        filename: Path to the file
        offset: Starting byte offset

    Returns:
        Byte position of next newline, or offset if at end of file
    """
    with open(filename, 'rb') as f:
        f.seek(offset)
        # Read up to 256KB to find next newline
        chunk = f.read(256 * 1024)
        if not chunk:
            return offset

        newline_pos = chunk.find(NEWLINE_SYMBOL_BYTES)
        if newline_pos == -1:
            # No newline found in this chunk, return end of chunk
            return offset + len(chunk)

        return offset + newline_pos + len(NEWLINE_SYMBOL_BYTES)  # Position after the newline


def get_file_offsets(filename: str, file_size_bytes: int) -> list[int]:
    """
    Calculate byte offsets for splitting a file into equal length parallel processing chunks.
    Offsets are aligned to line boundaries to ensure patterns are not split.

    Rules:
    - Each chunk should be at least MIN_CHUNK_SIZE bytes
    - Maximum number of chunks is MAX_SUBPROCESSES
    - Chunks are aligned to line boundaries (newlines)
    - Returns list of starting byte offsets (always starts with 0)

    Args:
        filename: Path to the file

    Returns:
        List of byte offsets where each chunk starts, aligned to line boundaries

    Examples:
        60MB file -> [0] (whole file goes to one worker)
        140MB file -> [0, ~70MB] (2 chunks, aligned to newlines)
        180MB file -> [0, ~60MB, ~120MB] (3 chunks, aligned to newlines)
    """
    logger.debug(
        f"[OFFSETS] Calculating offsets for file_size_bytes={file_size_bytes} bytes ({file_size_bytes / (1024 * 1024):.2f} MB)"
    )

    # Calculate maximum possible number of chunks given MIN_CHUNK_SIZE
    max_chunks_by_size = file_size_bytes // MIN_CHUNK_SIZE
    logger.debug(f"[OFFSETS] max_chunks_by_size={max_chunks_by_size} (file_size_bytes // MIN_CHUNK_SIZE)")

    # Actual number of chunks is limited by both MIN_CHUNK_SIZE and MAX_SUBPROCESSES
    num_chunks = min(max_chunks_by_size, MAX_SUBPROCESSES)
    logger.debug(f"[OFFSETS] num_chunks after min with MAX_SUBPROCESSES={num_chunks}")

    # Ensure at least 1 chunk
    num_chunks = max(1, num_chunks)
    logger.debug(f"[OFFSETS] Final num_chunks={num_chunks}")

    # Calculate chunk size
    chunk_size = file_size_bytes // num_chunks
    logger.debug(f"[OFFSETS] chunk_size={chunk_size} bytes ({chunk_size / (1024 * 1024):.2f} MB)")

    # Generate initial offsets
    raw_offsets = [i * chunk_size for i in range(num_chunks)]
    logger.debug(f"[OFFSETS] Generated raw offsets: {raw_offsets}")

    # Adjust offsets to line boundaries (except first offset which is always 0)
    aligned_offsets = [0]
    for i in range(1, num_chunks):
        aligned_offset = find_next_newline(filename, raw_offsets[i])
        aligned_offsets.append(aligned_offset)
        logger.debug(f"[OFFSETS] Aligned offset[{i}]: {raw_offsets[i]} -> {aligned_offset} (line boundary)")

    logger.debug(f"[OFFSETS] Final line-aligned offsets: {aligned_offsets}")
    return aligned_offsets


def create_file_tasks(filename: str) -> list[FileTask]:
    file_size = os.path.getsize(filename)
    offsets = get_file_offsets(filename, file_size)

    tasks = []
    for i, offset in enumerate(offsets):
        if i == len(offsets) - 1:
            count = file_size - offset
        else:
            count = offsets[i + 1] - offset
        tasks.append(FileTask(task_id=i, filepath=filename, offset=offset, count=count))

    logger.debug(f"[TASKS] Created {len(tasks)} tasks for {filename}")
    return tasks


def _process_task_worker(
    task: FileTask, pattern_ids: dict[str, str], rg_extra_args: list | None = None
) -> tuple[FileTask, list[tuple[int, list[str]]], float]:
    """
    Worker function to process a single FileTask with multiple patterns.
    Runs dd | rg pipeline and returns byte offsets with matched pattern IDs.

    Args:
        task: FileTask to process
        pattern_ids: Dictionary mapping pattern_id -> pattern string
        rg_extra_args: Optional list of extra arguments to pass to ripgrep

    Returns:
        Tuple of (task, list_of_(offset, pattern_ids), execution_time)
    """
    if rg_extra_args is None:
        rg_extra_args = []

    start_time = time.time()
    thread_id = threading.current_thread().name

    # Track active workers
    prom.active_workers.inc()

    logger.debug(
        f"[WORKER {thread_id}] Starting task {task.task_id}: "
        f"file={task.filepath}, offset={task.offset}, count={task.count}"
    )

    try:
        # Calculate dd block parameters
        # We use 1MB blocks for dd to balance performance and memory
        bs = 1024 * 1024  # 1MB block size

        # Split task.offset into complete blocks and remainder bytes
        # Example: task.offset=75,000,000, bs=1,048,576
        #   skip_blocks = 71 (skip 71 complete MB blocks)
        #   skip_remainder = 552,448 bytes (the "extra" bytes into block 72)
        skip_blocks = task.offset // bs
        skip_remainder = task.offset % bs

        # dd actually starts reading at the last complete block boundary
        # This is BEFORE our desired task.offset by skip_remainder bytes
        # Example: actual_dd_offset = 71 * 1,048,576 = 74,448,896
        #   (which is 552,448 bytes before task.offset of 75,000,000)
        actual_dd_offset = skip_blocks * bs

        # Calculate how many blocks dd needs to read to ensure we get all task.count bytes
        # We add skip_remainder because dd starts before task.offset
        # We add (bs - 1) for ceiling division to ensure we read enough
        # Example: if task.count=20MB and skip_remainder=552,448:
        #   count_blocks = ceil((20MB + 552,448) / 1MB) = 21 blocks
        count_blocks = (task.count + skip_remainder + bs - 1) // bs

        logger.debug(f"[WORKER {thread_id}] Task {task.task_id}: dd bs={bs} skip={skip_blocks} count={count_blocks}")

        # Run dd | rg with --byte-offset and multiple -e patterns
        dd_proc = subprocess.Popen(
            ['dd', f'if={task.filepath}', f'bs={bs}', f'skip={skip_blocks}', f'count={count_blocks}', 'status=none'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Build ripgrep command with multiple -e patterns
        rg_cmd = ['rg', '--byte-offset', '--no-heading', '--only-matching', '--color=never']

        # Add all patterns with -e flag
        for pattern in pattern_ids.values():
            rg_cmd.extend(['-e', pattern])

        rg_cmd.extend(rg_extra_args)
        rg_cmd.append('-')  # Read from stdin

        # print(' '.join(rg_cmd))
        rg_proc = subprocess.Popen(
            rg_cmd,
            stdin=dd_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if dd_proc.stdout is not None:
            dd_proc.stdout.close()

        # Collect output and identify matching patterns
        # Store as list of (offset, [pattern_ids])
        offset_matches = []

        for line in rg_proc.stdout or []:
            line_str = line.decode('utf-8').strip()
            if not line_str:
                continue
            parts = line_str.split(':', 1)
            if not parts[0].isdigit():
                # ripgrep returns byte offset relative to its stdin (dd's output)
                # dd's output starts at actual_dd_offset in the file
                continue
            rg_byte_offset = int(parts[0])
            # print(line_str)
            # Convert to absolute position in the original file
            absolute_byte_offset = actual_dd_offset + rg_byte_offset

            # Critical: Only include matches that fall within THIS task's designated range
            # task.offset is where this task should start (inclusive)
            # task.offset + task.count is where this task should end (exclusive)
            # We read extra bytes (skip_remainder) at the start via dd, so we must filter them out
            # We also read extra bytes at the end (up to bs-1), so we must filter those too
            # This prevents duplicate matches across adjacent tasks
            if task.offset <= absolute_byte_offset < task.offset + task.count:
                # Since ripgrep already matched with all flags (including -i, -w, etc.),
                # we trust its output. For multi-pattern search, we can't reliably determine
                # which specific pattern matched when flags like -i are used, so we record
                # all patterns. This is acceptable since the user is searching for all these
                # patterns anyway.
                matching_pattern_ids = list(pattern_ids.keys())

                offset_matches.append((absolute_byte_offset, matching_pattern_ids))

        rg_proc.wait()
        dd_proc.wait()

        elapsed = time.time() - start_time

        logger.debug(
            f"[WORKER {thread_id}] Task {task.task_id} completed: found {len(offset_matches)} matches in {elapsed:.3f}s"
        )

        # Track task completion
        prom.worker_tasks_completed.inc()
        prom.active_workers.dec()

        return (task, offset_matches, elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[WORKER {thread_id}] Task {task.task_id} failed after {elapsed:.3f}s: {e}")

        # Track task failure
        prom.worker_tasks_failed.inc()
        prom.active_workers.dec()

        return (task, [], elapsed)


def parse_paths(
    paths: list[str], regexps: list[str], max_results: int | None = None, rg_extra_args: list | None = None
) -> dict:
    """
    Parse files or directories for multiple regex patterns using streaming worker pool.
    Returns ID-based structure for efficient response format.

    Args:
        paths: List of file or directory paths to search
        regexps: List of regular expression patterns to search for
        max_results: Optional maximum number of results to find before stopping (applies to all patterns combined)
        rg_extra_args: Optional list of extra arguments to pass to ripgrep

    Returns:
        Dictionary with ID-based structure:
        {
            'patterns': {'p1': 'error', 'p2': 'warning'},
            'files': {'f1': '/path/file.log', 'f2': '/other.log'},
            'matches': [
                {'pattern': 'p1', 'file': 'f1', 'offset': 100},
                {'pattern': 'p2', 'file': 'f1', 'offset': 200}
            ],
            'scanned_files': [...],
            'skipped_files': [...]
        }

    Raises:
        RuntimeError: If parsing fails
        FileNotFoundError: If path doesn't exist
    """
    if rg_extra_args is None:
        rg_extra_args = []

    pattern_ids = {f"p{i + 1}": pattern for i, pattern in enumerate(regexps)}
    logger.info(f"[PARSE] Processing {len(pattern_ids)} pattern(s) across {len(paths)} path(s)")

    # Collect all files to parse from all provided paths
    all_files_to_parse = []
    all_skipped_files = []
    all_scanned_dirs = []

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

        if os.path.isdir(path):
            logger.info(f"[PARSE] Path '{path}' is directory, scanning for text files...")
            text_files, skipped_files = scan_directory_for_text_files(path)

            if text_files:
                all_files_to_parse.extend(text_files)
                all_scanned_dirs.append(path)
            all_skipped_files.extend(skipped_files)
            logger.info(f"[PARSE] Found {len(text_files)} text file(s) in '{path}'")
        else:
            # Single file - validate and add to list
            try:
                validate_file(path)
                all_files_to_parse.append(path)
                logger.info(f"[PARSE] Added file '{path}'")
            except ValueError as e:
                logger.warning(f"[PARSE] Skipping invalid file '{path}': {e}")
                all_skipped_files.append(path)

    if not all_files_to_parse:
        logger.warning(f"[PARSE] No valid files found across all paths")
        return {
            'patterns': pattern_ids,
            'files': {},
            'matches': [],
            'scanned_files': [],
            'skipped_files': all_skipped_files,
        }

    # Generate file IDs for all files
    file_ids = {f"f{i + 1}": filepath for i, filepath in enumerate(all_files_to_parse)}

    # Parse all files using unified multi-file approach
    logger.info(f"[PARSE] Parsing {len(all_files_to_parse)} file(s) with {len(pattern_ids)} pattern(s)")
    matches = _parse_multiple_files_multipattern(all_files_to_parse, pattern_ids, file_ids, max_results, rg_extra_args)

    return {
        'patterns': pattern_ids,
        'files': file_ids,
        'matches': matches,
        'scanned_files': all_files_to_parse if all_scanned_dirs else [],
        'skipped_files': all_skipped_files,
    }


def _parse_multiple_files_multipattern(
    filepaths: list[str],
    pattern_ids: dict[str, str],
    file_ids: dict[str, str],
    max_results: int | None = None,
    rg_extra_args: list | None = None,
) -> list[dict]:
    """
    Parse multiple files with multiple patterns and return matches in ID-based format.

    Args:
        filepaths: List of file paths
        pattern_ids: Dictionary mapping pattern_id -> pattern
        file_ids: Dictionary mapping file_id -> filepath
        max_results: Optional maximum number of results
        rg_extra_args: Optional list of extra arguments to pass to ripgrep

    Returns:
        List of match dicts: [{'pattern': 'p1', 'file': 'f1', 'offset': 100}, ...]
    """
    if rg_extra_args is None:
        rg_extra_args = []

    # Validate regex patterns
    for pattern in pattern_ids.values():
        test_proc = subprocess.run(['rg', '--', pattern], input=b'', capture_output=True, timeout=1)
        if test_proc.returncode == 2:
            error_msg = test_proc.stderr.decode('utf-8').strip()
            raise RuntimeError(f"Invalid regex pattern: {error_msg}")

    logger.info(f"[PARSE_MULTI_MULTI] Parsing {len(filepaths)} files with {len(pattern_ids)} patterns")

    # Create reverse mapping: filepath -> file_id
    filepath_to_id = {v: k for k, v in file_ids.items()}

    # Create tasks from all files
    all_tasks = []
    for filepath in filepaths:
        try:
            file_tasks = create_file_tasks(filepath)
            all_tasks.extend(file_tasks)
        except Exception as e:
            logger.warning(f"[PARSE_MULTI_MULTI] Skipping {filepath}: {e}")

    logger.info(f"[PARSE_MULTI_MULTI] Created {len(all_tasks)} tasks from {len(filepaths)} files")

    # Track parallel tasks created
    prom.parallel_tasks_created.observe(len(all_tasks))

    # Process all tasks
    matches = []
    total_time = 0.0

    with ThreadPoolExecutor(max_workers=MAX_SUBPROCESSES, thread_name_prefix="Worker") as executor:
        future_to_task = {
            executor.submit(_process_task_worker, task, pattern_ids, rg_extra_args): task for task in all_tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]

            try:
                task_result, offset_pattern_list, elapsed = future.result()
                total_time += elapsed

                # Get file_id for this task's filepath
                file_id = filepath_to_id.get(task.filepath, 'f?')

                # Convert to match format
                for offset, pattern_id_list in offset_pattern_list:
                    for pattern_id in pattern_id_list:
                        matches.append({'pattern': pattern_id, 'file': file_id, 'offset': offset})

                logger.debug(
                    f"[PARSE_MULTI_MULTI] Task {task.task_id} ({task.filepath}) contributed {len(offset_pattern_list)} offsets"
                )

                # Check max_results
                if max_results and len(matches) >= max_results:
                    logger.info(f"[PARSE_MULTI_MULTI] Reached max_results={max_results}, cancelling remaining")
                    for f in future_to_task:
                        f.cancel()
                    break

            except Exception as e:
                logger.error(f"[PARSE_MULTI_MULTI] Task failed: {e}")

    # Sort by file, offset, then pattern
    matches.sort(key=lambda m: (m['file'], m['offset'], m['pattern']))

    if max_results and len(matches) > max_results:
        matches = matches[:max_results]

    logger.info(f"[PARSE_MULTI_MULTI] Completed: {len(matches)} matches, total worker time: {total_time:.3f}s")
    return matches


def parse_multiple_files(
    filepaths: list[str], regex: str, max_results: int | None = None, rg_extra_args: list = None
) -> list[dict]:
    """
    Parse multiple files for regex pattern using streaming worker pool.
    Creates tasks from all files and processes them in parallel.

    Args:
        filepaths: List of file paths to parse
        regex: Regular expression pattern to search for
        max_results: Optional maximum number of total results
        rg_extra_args: Optional list of extra arguments to pass to ripgrep

    Returns:
        List of dicts with 'filepath' and 'offset'
    """
    if rg_extra_args is None:
        rg_extra_args = []
    # Validate regex
    test_proc = subprocess.run(['rg', '--', regex], input=b'', capture_output=True, timeout=1)
    if test_proc.returncode == 2:
        error_msg = test_proc.stderr.decode('utf-8').strip()
        raise RuntimeError(f"Invalid regex pattern: {error_msg}")

    logger.info(f"[PARSE_MULTI] Parsing {len(filepaths)} files")

    # Create pattern_ids dict for single pattern
    pattern_ids = {'p1': regex}

    # Create tasks from all files
    all_tasks = []
    for filepath in filepaths:
        try:
            file_tasks = create_file_tasks(filepath)
            all_tasks.extend(file_tasks)
        except Exception as e:
            logger.warning(f"[PARSE_MULTI] Skipping {filepath}: {e}")

    logger.info(f"[PARSE_MULTI] Created {len(all_tasks)} tasks from {len(filepaths)} files")

    # Process all tasks in streaming pool
    matches = []
    total_time = 0.0

    with ThreadPoolExecutor(max_workers=MAX_SUBPROCESSES, thread_name_prefix="Worker") as executor:
        future_to_task = {
            executor.submit(_process_task_worker, task, pattern_ids, rg_extra_args): task for task in all_tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]

            try:
                task_result, offset_pattern_list, elapsed = future.result()
                total_time += elapsed

                # Add matches with filepath (flatten pattern list since we only have one pattern)
                for offset, _ in offset_pattern_list:
                    matches.append({'filepath': task.filepath, 'offset': offset})

                logger.debug(
                    f"[PARSE_MULTI] Task {task.task_id} ({task.filepath}) contributed {len(offset_pattern_list)} matches"
                )

                # Check max_results
                if max_results and len(matches) >= max_results:
                    logger.info(f"[PARSE_MULTI] Reached max_results={max_results}, cancelling remaining")
                    for f in future_to_task:
                        f.cancel()
                    break

            except Exception as e:
                logger.error(f"[PARSE_MULTI] Task failed: {e}")

    # Sort by filepath, then offset
    matches.sort(key=lambda m: (m['filepath'], m['offset']))

    if max_results and len(matches) > max_results:
        matches = matches[:max_results]

    logger.info(f"[PARSE_MULTI] Completed: {len(matches)} matches, total worker time: {total_time:.3f}s")
    return matches


# TODO: make it more efficient without invoking dd, just opening file in 'rb' and seek + read it many times.
def get_context(
    filename: str, offsets: list[int], before_context: int = 3, after_context: int = 3
) -> dict[int, list[str]]:
    """
    Get lines of context around specified byte offsets in a file.

    Args:
        filename: Path to the file
        offsets: List of byte offsets to get context for
        before_context: Number of lines before each offset (default: 3)
        after_context: Number of lines after each offset (default: 3)

    Returns:
        Dictionary mapping each offset to list of context lines

    Raises:
        ValueError: If file doesn't exist or parameters are invalid
    """
    if not os.path.exists(filename):
        raise ValueError(f"File not found: {filename}")

    if before_context < 0 or after_context < 0:
        raise ValueError("Context values must be non-negative")

    file_size = os.path.getsize(filename)
    result: dict[int, list[str]] = {}

    max_line_length = LINE_SIZE_ASSUMPTION_KB * 1024

    for offset in offsets:
        if offset < 0 or offset >= file_size:
            result[offset] = []
            continue

        # Calculate how much to read before and after the offset
        # We need enough bytes to capture the requested lines
        bytes_before = max_line_length * (before_context + 1)  # +1 for the line containing offset
        bytes_after = max_line_length * (after_context + 1)

        # Calculate read range
        start_pos = max(0, offset - bytes_before)
        end_pos = min(file_size, offset + bytes_after)
        read_size = end_pos - start_pos

        # Read the chunk using dd
        bs = 1024 * 1024  # 1MB block size
        skip_blocks = start_pos // bs
        skip_remainder = start_pos % bs
        count_blocks = (read_size + skip_remainder + bs - 1) // bs

        dd_proc = subprocess.Popen(
            ['dd', f'if={filename}', f'bs={bs}', f'skip={skip_blocks}', f'count={count_blocks}', 'status=none'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        chunk_data, _ = dd_proc.communicate()
        dd_proc.wait()

        # Adjust for remainder bytes we read extra
        chunk_data = chunk_data[skip_remainder : skip_remainder + read_size]

        # Split into lines
        lines = chunk_data.decode('utf-8', errors='replace').splitlines(keepends=True)

        # Find which line contains our offset
        current_pos = start_pos
        target_line_idx = None

        for idx, line in enumerate(lines):
            line_start = current_pos
            line_end = current_pos + len(line.encode('utf-8'))

            if line_start <= offset < line_end:
                target_line_idx = idx
                break

            current_pos = line_end

        if target_line_idx is None:
            # Offset not found in expected range
            result[offset] = []
            continue

        # Extract context lines
        start_idx = max(0, target_line_idx - before_context)
        end_idx = min(len(lines), target_line_idx + after_context + 1)

        # Strip the configured newline symbol and also \r as fallback
        context_lines = [line.rstrip(NEWLINE_SYMBOL + '\r') for line in lines[start_idx:end_idx]]
        result[offset] = context_lines

    return result


def get_context_by_lines(
    filename: str,
    line_numbers: list[int],
    before_context: int = 3,
    after_context: int = 3,
    use_index: bool = True,
) -> dict[int, list[str]]:
    """
    Get lines of context around specified line numbers in a file.

    For large files (above threshold), uses or creates an index for efficient access.
    For small files, reads the entire file.

    Args:
        filename: Path to the file
        line_numbers: List of 1-based line numbers to get context for
        before_context: Number of lines before each target line (default: 3)
        after_context: Number of lines after each target line (default: 3)
        use_index: Whether to use/create index for large files (default: True)

    Returns:
        Dictionary mapping each line number to list of context lines

    Raises:
        ValueError: If file doesn't exist or parameters are invalid
    """
    from rx.index import (
        create_index_file,
        find_line_offset,
        get_index_path,
        get_large_file_threshold_bytes,
        is_index_valid,
        load_index,
    )

    if not os.path.exists(filename):
        raise ValueError(f"File not found: {filename}")

    if before_context < 0 or after_context < 0:
        raise ValueError("Context values must be non-negative")

    file_size = os.path.getsize(filename)
    threshold = get_large_file_threshold_bytes()
    result: dict[int, list[str]] = {}

    # For small files, just read the entire file
    if file_size < threshold or not use_index:
        return _get_context_by_lines_simple(filename, line_numbers, before_context, after_context)

    # For large files, use index
    index_path = get_index_path(filename)
    if is_index_valid(filename):
        index_data = load_index(index_path)
    else:
        # Create index if it doesn't exist
        logger.info(f"Creating index for large file: {filename}")
        index_data = create_index_file(filename)

    if index_data is None:
        # Fall back to simple method if index creation failed
        logger.warning(f"Index unavailable for {filename}, falling back to simple method")
        return _get_context_by_lines_simple(filename, line_numbers, before_context, after_context)

    line_index = index_data.get("line_index", [[1, 0]])
    total_lines = index_data.get("analysis", {}).get("line_count", 0)

    max_line_length = LINE_SIZE_ASSUMPTION_KB * 1024

    for target_line in line_numbers:
        if target_line < 1:
            result[target_line] = []
            continue

        if total_lines > 0 and target_line > total_lines:
            result[target_line] = []
            continue

        # Find the closest indexed position before our target
        start_line_needed = max(1, target_line - before_context)
        indexed_line, indexed_offset = find_line_offset(line_index, start_line_needed)

        # Calculate how many lines we need to skip from the indexed position
        lines_to_skip = start_line_needed - indexed_line

        # Calculate how many lines total we need to read
        lines_to_read = before_context + 1 + after_context + lines_to_skip

        # Read enough bytes (estimate based on max line length)
        bytes_to_read = lines_to_read * max_line_length

        # Read the chunk
        try:
            with open(filename, "rb") as f:
                f.seek(indexed_offset)
                chunk = f.read(bytes_to_read)
        except IOError as e:
            logger.error(f"Failed to read file {filename}: {e}")
            result[target_line] = []
            continue

        # Split into lines
        lines = chunk.decode("utf-8", errors="replace").splitlines(keepends=True)

        # Skip to the start line and extract context
        if lines_to_skip < len(lines):
            # Calculate actual indices within read lines
            start_idx = lines_to_skip
            # Target line is at: lines_to_skip + before_context
            target_idx = lines_to_skip + before_context
            end_idx = min(len(lines), target_idx + after_context + 1)

            # Adjust start if we're at the beginning of file
            if start_line_needed == 1 and indexed_line == 1:
                start_idx = 0
                target_idx = target_line - 1
                end_idx = min(len(lines), target_idx + after_context + 1)
                start_idx = max(0, target_idx - before_context)

            context_lines = [line.rstrip(NEWLINE_SYMBOL + "\r") for line in lines[start_idx:end_idx]]
            result[target_line] = context_lines
        else:
            result[target_line] = []

    return result


def _get_context_by_lines_simple(
    filename: str,
    line_numbers: list[int],
    before_context: int,
    after_context: int,
) -> dict[int, list[str]]:
    """
    Simple implementation that reads the file line by line.
    Used for small files or when index is unavailable.
    """
    result: dict[int, list[str]] = {}

    # Read all lines
    try:
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except IOError as e:
        logger.error(f"Failed to read file {filename}: {e}")
        for line_num in line_numbers:
            result[line_num] = []
        return result

    total_lines = len(all_lines)

    for target_line in line_numbers:
        if target_line < 1 or target_line > total_lines:
            result[target_line] = []
            continue

        # Convert to 0-based index
        target_idx = target_line - 1

        start_idx = max(0, target_idx - before_context)
        end_idx = min(total_lines, target_idx + after_context + 1)

        context_lines = [line.rstrip(NEWLINE_SYMBOL + "\r") for line in all_lines[start_idx:end_idx]]
        result[target_line] = context_lines

    return result

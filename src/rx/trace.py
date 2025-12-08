"""Main tracing/search engine using ripgrep

This module provides the core tracing functionality that uses ripgrep's --json
output format for richer match data and context extraction.
"""

import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime

from rx.cli import prometheus as prom
from rx.compression import CompressionFormat, detect_compression, get_decompressor_command, is_compressed
from rx.file_utils import MAX_SUBPROCESSES, FileTask, create_file_tasks, scan_directory_for_text_files, validate_file
from rx.index import get_large_file_threshold_bytes
from rx.models import ContextLine, FileScannedPayload, MatchFoundPayload, ParseResult, Submatch
from rx.rg_json import RgContextEvent, RgMatchEvent, parse_rg_json_event
from rx.seekable_index import get_or_build_index
from rx.seekable_zstd import is_seekable_zstd, read_seek_table
from rx.trace_cache import (
    build_cache_from_matches,
    get_cached_matches,
    get_compressed_cache_info,
    get_trace_cache_path,
    reconstruct_match_data,
    reconstruct_seekable_zstd_matches,
    save_trace_cache,
    should_cache_compressed_file,
    should_cache_file,
)
from rx.utils import NEWLINE_SYMBOL


logger = logging.getLogger(__name__)


def process_compressed_file(
    filepath: str,
    pattern_ids: dict[str, str],
    rg_extra_args: list | None = None,
    context_before: int = 0,
    context_after: int = 0,
    max_results: int | None = None,
) -> tuple[list[dict], list[ContextLine], float]:
    """
    Process a compressed file by decompressing to stdout and piping to rg.

    Compressed files cannot be chunked like regular files, so they are
    processed sequentially in a single pass.

    Args:
        filepath: Path to the compressed file
        pattern_ids: Dictionary mapping pattern_id -> pattern string
        rg_extra_args: Optional list of extra arguments to pass to ripgrep
        context_before: Number of context lines before each match
        context_after: Number of context lines after each match
        max_results: Optional maximum number of results to return

    Returns:
        Tuple of (list_of_match_dicts, list_of_context_lines, execution_time)
    """
    if rg_extra_args is None:
        rg_extra_args = []

    start_time = time.time()
    thread_id = threading.current_thread().name

    logger.info(f'[COMPRESSED {thread_id}] Processing compressed file: {filepath}')

    # Detect compression format and get decompressor command
    compression_format = detect_compression(filepath)
    if compression_format == CompressionFormat.NONE:
        raise ValueError(f'File is not compressed: {filepath}')

    decompress_cmd = get_decompressor_command(compression_format, filepath)

    logger.debug(f'[COMPRESSED {thread_id}] Using decompressor: {" ".join(decompress_cmd)}')

    try:
        # Start decompression process
        decompress_proc = subprocess.Popen(
            decompress_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Build ripgrep command with --json and multiple -e patterns
        rg_cmd = ['rg', '--json', '--no-heading', '--color=never']

        # Add context flags if requested
        if context_before > 0:
            rg_cmd.extend(['-B', str(context_before)])
        if context_after > 0:
            rg_cmd.extend(['-A', str(context_after)])

        # Add all patterns with -e flag
        for pattern in pattern_ids.values():
            rg_cmd.extend(['-e', pattern])

        # Add extra args (but filter out incompatible ones)
        filtered_extra_args = [arg for arg in rg_extra_args if arg not in ['--byte-offset', '--only-matching']]
        rg_cmd.extend(filtered_extra_args)

        rg_cmd.append('-')  # Read from stdin

        logger.debug(f'[COMPRESSED {thread_id}] Running: {" ".join(rg_cmd)}')

        rg_proc = subprocess.Popen(
            rg_cmd,
            stdin=decompress_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if decompress_proc.stdout:
            decompress_proc.stdout.close()

        # Parse JSON events from ripgrep output
        matches = []
        context_lines = []
        match_count = 0

        for line in rg_proc.stdout or []:
            event = parse_rg_json_event(line)

            if isinstance(event, RgMatchEvent):
                # Check max_results limit
                if max_results and match_count >= max_results:
                    break

                match_data = event.data

                # Extract submatches
                submatches = [Submatch(text=sm.text, start=sm.start, end=sm.end) for sm in match_data.submatches]

                # For compressed files, absolute_offset is in the decompressed stream
                # We don't know absolute line number for compressed files (no index)
                matches.append(
                    {
                        'offset': match_data.absolute_offset,  # Decompressed byte offset
                        'pattern_ids': list(pattern_ids.keys()),
                        'line_number': match_data.line_number,
                        'absolute_line_number': -1,  # Unknown for compressed files
                        'line_text': match_data.lines.text.rstrip(NEWLINE_SYMBOL),
                        'submatches': submatches,
                        'is_compressed': True,
                    }
                )
                match_count += 1

                logger.debug(
                    f'[COMPRESSED {thread_id}] Match: line={match_data.line_number}, '
                    f'offset={match_data.absolute_offset}, submatches={len(submatches)}'
                )

            elif isinstance(event, RgContextEvent):
                context_data = event.data
                context_lines.append(
                    ContextLine(
                        relative_line_number=context_data.line_number,
                        absolute_line_number=-1,  # Unknown for compressed files
                        line_text=context_data.lines.text.rstrip(NEWLINE_SYMBOL),
                        absolute_offset=context_data.absolute_offset,
                    )
                )

        rg_proc.wait()
        decompress_proc.wait()

        # Check for decompression errors
        if decompress_proc.returncode != 0:
            stderr = decompress_proc.stderr.read().decode() if decompress_proc.stderr else ''
            logger.warning(f'[COMPRESSED {thread_id}] Decompression warning: {stderr}')

        elapsed = time.time() - start_time

        logger.info(
            f'[COMPRESSED {thread_id}] Completed: {len(matches)} matches, '
            f'{len(context_lines)} context lines in {elapsed:.3f}s'
        )

        return (matches, context_lines, elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f'[COMPRESSED {thread_id}] Failed after {elapsed:.3f}s: {e}')
        raise


def process_seekable_zstd_frame_batch(
    filepath: str,
    frame_indices: list[int],
    frame_infos: list,
    pattern_ids: dict[str, str],
    rg_extra_args: list | None = None,
    context_before: int = 0,
    context_after: int = 0,
) -> tuple[list[int], list[dict], list[ContextLine], float]:
    """
    Process a batch of consecutive frames from a seekable zstd file.

    This batches multiple frames to reduce subprocess overhead. Consecutive
    compressed frames are concatenated and piped through a single zstd|rg pipeline.

    Args:
        filepath: Path to the seekable zstd file
        frame_indices: List of frame indices to process
        frame_infos: List of all FrameInfo objects (indexed by frame_index)
        pattern_ids: Dictionary mapping pattern_id -> pattern string
        rg_extra_args: Optional list of extra arguments to pass to ripgrep
        context_before: Number of context lines before each match
        context_after: Number of context lines after each match

    Returns:
        Tuple of (frame_indices, list_of_match_dicts, list_of_context_lines, execution_time)
    """
    if rg_extra_args is None:
        rg_extra_args = []

    start_time = time.time()
    thread_id = threading.current_thread().name

    logger.debug(
        f'[SEEKABLE {thread_id}] Processing {len(frame_indices)} frames: {frame_indices[0]}-{frame_indices[-1]}'
    )

    # Track active workers
    prom.active_workers.inc()

    try:
        # Read all compressed frames and concatenate them
        # Consecutive zstd frames can be concatenated and decompressed as a stream
        compressed_chunks = []
        total_compressed_size = 0

        with open(filepath, 'rb') as f:
            for frame_idx in frame_indices:
                frame = frame_infos[frame_idx]
                f.seek(frame.compressed_offset)
                compressed_chunks.append(f.read(frame.compressed_size))
                total_compressed_size += frame.compressed_size

        compressed_data = b''.join(compressed_chunks)

        logger.debug(
            f'[SEEKABLE {thread_id}] Read {total_compressed_size} compressed bytes for {len(frame_indices)} frames'
        )

        # Build ripgrep command
        rg_cmd = ['rg', '--json', '--no-heading', '--color=never']

        if context_before > 0:
            rg_cmd.extend(['-B', str(context_before)])
        if context_after > 0:
            rg_cmd.extend(['-A', str(context_after)])

        for pattern in pattern_ids.values():
            rg_cmd.extend(['-e', pattern])

        filtered_extra_args = [arg for arg in rg_extra_args if arg not in ['--byte-offset', '--only-matching']]
        rg_cmd.extend(filtered_extra_args)
        rg_cmd.append('-')

        # Create native pipeline: zstd -d | rg (no Python in the middle!)
        zstd_proc = subprocess.Popen(
            ['zstd', '-d', '-c'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        rg_proc = subprocess.Popen(
            rg_cmd,
            stdin=zstd_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        zstd_proc.stdout.close()

        # Use line counts from the index (which are accurate since frames are line-aligned)
        # The index was built by decompressing each frame and counting lines precisely
        frame_line_counts = []
        for frame_idx in frame_indices:
            frame_info = frame_infos[frame_idx]
            # line_count in the index is accurate
            frame_line_counts.append(frame_info.line_count)

        # Now feed compressed data to pipeline and read results
        zstd_proc.stdin.write(compressed_data)
        zstd_proc.stdin.close()

        stdout, stderr = rg_proc.communicate()
        zstd_proc.wait()

        # Parse results and adjust offsets/line numbers
        matches = []
        context_lines = []

        for line in stdout.splitlines():
            event = parse_rg_json_event(line)

            if isinstance(event, RgMatchEvent):
                match_data = event.data

                # The offset from rg is relative to the concatenated decompressed stream
                # We need to find which frame it belongs to and adjust
                batch_offset = match_data.absolute_offset

                # Find which frame this match belongs to
                current_offset = 0
                for i, frame_idx in enumerate(frame_indices):
                    frame_info = frame_infos[frame_idx]
                    frame_size = frame_info.decompressed_size

                    if batch_offset < current_offset + frame_size:
                        # Match is in this frame
                        offset_in_frame = batch_offset - current_offset
                        absolute_offset = frame_info.decompressed_offset + offset_in_frame

                        # Adjust line number using ACTUAL line counts
                        # rg line numbers are 1-based within the batch
                        line_in_batch = match_data.line_number

                        # Count ACTUAL lines in previous frames in this batch
                        lines_before = sum(frame_line_counts[:i])

                        # Line within current frame
                        line_in_frame = line_in_batch - lines_before
                        adjusted_line_number = frame_info.first_line + line_in_frame - 1

                        # Debug logging
                        logger.debug(
                            f'[SEEKABLE {thread_id}] Match: frame_idx={frame_idx}, '
                            f'first_line={frame_info.first_line}, line_in_batch={line_in_batch}, '
                            f'lines_before={lines_before}, line_in_frame={line_in_frame}, '
                            f'adjusted={adjusted_line_number}, actual_line_count={frame_line_counts[i]}'
                        )

                        submatches = [
                            Submatch(text=sm.text, start=sm.start, end=sm.end) for sm in match_data.submatches
                        ]

                        matches.append(
                            {
                                'offset': absolute_offset,
                                'frame_index': frame_idx,
                                'pattern_ids': list(pattern_ids.keys()),
                                'line_number': adjusted_line_number,
                                'absolute_line_number': adjusted_line_number,  # We know absolute line number
                                'line_text': match_data.lines.text.rstrip(NEWLINE_SYMBOL),
                                'submatches': submatches,
                                'is_compressed': True,
                                'is_seekable_zstd': True,
                            }
                        )
                        break

                    current_offset += frame_size

            elif isinstance(event, RgContextEvent):
                # Similar logic for context lines
                context_data = event.data
                batch_offset = context_data.absolute_offset

                current_offset = 0
                for i, frame_idx in enumerate(frame_indices):
                    frame_info = frame_infos[frame_idx]
                    frame_size = frame_info.decompressed_size

                    if batch_offset < current_offset + frame_size:
                        offset_in_frame = batch_offset - current_offset
                        absolute_offset = frame_info.decompressed_offset + offset_in_frame

                        line_in_batch = context_data.line_number
                        # Count ACTUAL lines in previous frames in this batch
                        lines_before = sum(frame_line_counts[:i])
                        line_in_frame = line_in_batch - lines_before
                        adjusted_line_number = frame_info.first_line + line_in_frame - 1

                        context_lines.append(
                            ContextLine(
                                relative_line_number=adjusted_line_number,
                                absolute_line_number=adjusted_line_number,  # We know absolute line number
                                line_text=context_data.lines.text.rstrip(NEWLINE_SYMBOL),
                                absolute_offset=absolute_offset,
                            )
                        )
                        break

                    current_offset += frame_size

        elapsed = time.time() - start_time

        logger.debug(
            f'[SEEKABLE {thread_id}] Batch completed: '
            f'{len(matches)} matches, {len(context_lines)} context lines in {elapsed:.3f}s'
        )

        prom.worker_tasks_completed.inc()
        prom.active_workers.dec()

        return (frame_indices, matches, context_lines, elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f'[SEEKABLE {thread_id}] Batch failed after {elapsed:.3f}s: {e}')
        prom.worker_tasks_failed.inc()
        prom.active_workers.dec()
        raise


def process_seekable_zstd_frame(
    filepath: str,
    frame_index: int,
    first_line: int,
    decompressed_offset: int,
    frames: list,
    pattern_ids: dict[str, str],
    rg_extra_args: list | None = None,
    context_before: int = 0,
    context_after: int = 0,
) -> tuple[int, list[dict], list[ContextLine], float]:
    """
    Process a single frame from a seekable zstd file.

    Decompresses the frame and pipes the content to ripgrep for pattern matching.
    Line numbers and byte offsets are adjusted based on the frame's position in the file.

    Args:
        filepath: Path to the seekable zstd file
        frame_index: Index of the frame to process (0-based)
        first_line: First line number in this frame (1-based)
        decompressed_offset: Byte offset of this frame in the decompressed file
        frames: Pre-loaded list of FrameInfo objects (avoids redundant seek table reads)
        pattern_ids: Dictionary mapping pattern_id -> pattern string
        rg_extra_args: Optional list of extra arguments to pass to ripgrep
        context_before: Number of context lines before each match
        context_after: Number of context lines after each match

    Returns:
        Tuple of (frame_index, list_of_match_dicts, list_of_context_lines, execution_time)
    """
    if rg_extra_args is None:
        rg_extra_args = []

    start_time = time.time()
    thread_id = threading.current_thread().name

    logger.debug(f'[SEEKABLE {thread_id}] Processing frame {frame_index} from {filepath}')

    # Track active workers
    prom.active_workers.inc()

    try:
        # Get frame info
        frame = frames[frame_index]
        compressed_offset = frame.compressed_offset
        compressed_size = frame.compressed_size

        # Read compressed frame bytes (I/O bound, releases GIL)
        with open(filepath, 'rb') as f:
            f.seek(compressed_offset)
            compressed_data = f.read(compressed_size)

        # Use CLI pipeline for decompression:
        # zstd -d (native C decompression) | rg (search)
        # This avoids Python GIL for CPU-bound decompression

        # zstd decompress command
        zstd_cmd = ['zstd', '-d', '-c']

        # Build ripgrep command with --json and multiple -e patterns
        rg_cmd = ['rg', '--json', '--no-heading', '--color=never']

        # Add context flags if requested
        if context_before > 0:
            rg_cmd.extend(['-B', str(context_before)])
        if context_after > 0:
            rg_cmd.extend(['-A', str(context_after)])

        # Add all patterns with -e flag
        for pattern in pattern_ids.values():
            rg_cmd.extend(['-e', pattern])

        # Add extra args (but filter out incompatible ones)
        filtered_extra_args = [arg for arg in rg_extra_args if arg not in ['--byte-offset', '--only-matching']]
        rg_cmd.extend(filtered_extra_args)

        rg_cmd.append('-')  # Read from stdin

        logger.debug(f'[SEEKABLE {thread_id}] Pipeline: zstd -d | rg')

        # Create pipeline: zstd | rg
        zstd_proc = subprocess.Popen(
            zstd_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        rg_proc = subprocess.Popen(
            rg_cmd,
            stdin=zstd_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        zstd_proc.stdout.close()  # Allow zstd to receive SIGPIPE if rg exits

        # Feed compressed data to zstd (this happens in parallel with rg processing)
        zstd_proc.stdin.write(compressed_data)
        zstd_proc.stdin.close()

        stdout, stderr = rg_proc.communicate()

        # Wait for zstd to finish
        zstd_proc.wait()

        # Parse JSON events from ripgrep output
        matches = []
        context_lines = []

        for line in stdout.splitlines():
            event = parse_rg_json_event(line)

            if isinstance(event, RgMatchEvent):
                match_data = event.data

                # Adjust line number by adding frame's first_line offset
                # rg returns 1-based line numbers within the frame content
                adjusted_line_number = first_line + match_data.line_number - 1

                # Calculate absolute byte offset in decompressed file
                # match_data.absolute_offset is relative to the frame
                absolute_offset = decompressed_offset + match_data.absolute_offset

                # Extract submatches
                submatches = [Submatch(text=sm.text, start=sm.start, end=sm.end) for sm in match_data.submatches]

                matches.append(
                    {
                        'offset': absolute_offset,  # Absolute offset in decompressed file
                        'frame_index': frame_index,
                        'pattern_ids': list(pattern_ids.keys()),
                        'line_number': adjusted_line_number,
                        'absolute_line_number': adjusted_line_number,  # We know absolute line number
                        'line_text': match_data.lines.text.rstrip(NEWLINE_SYMBOL),
                        'submatches': submatches,
                        'is_compressed': True,
                        'is_seekable_zstd': True,
                    }
                )

                logger.debug(
                    f'[SEEKABLE {thread_id}] Match: frame={frame_index}, '
                    f'line={adjusted_line_number}, submatches={len(submatches)}'
                )

            elif isinstance(event, RgContextEvent):
                context_data = event.data
                adjusted_line_number = first_line + context_data.line_number - 1
                absolute_context_offset = decompressed_offset + context_data.absolute_offset

                context_lines.append(
                    ContextLine(
                        relative_line_number=adjusted_line_number,
                        absolute_line_number=adjusted_line_number,  # We know absolute line number
                        line_text=context_data.lines.text.rstrip(NEWLINE_SYMBOL),
                        absolute_offset=absolute_context_offset,
                    )
                )

        elapsed = time.time() - start_time

        logger.debug(
            f'[SEEKABLE {thread_id}] Frame {frame_index} completed: '
            f'{len(matches)} matches, {len(context_lines)} context lines in {elapsed:.3f}s'
        )

        # Track task completion
        prom.worker_tasks_completed.inc()
        prom.active_workers.dec()

        return (frame_index, matches, context_lines, elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f'[SEEKABLE {thread_id}] Frame {frame_index} failed after {elapsed:.3f}s: {e}')

        # Track task failure
        prom.worker_tasks_failed.inc()
        prom.active_workers.dec()

        return (frame_index, [], [], elapsed)


def process_seekable_zstd_file(
    filepath: str,
    pattern_ids: dict[str, str],
    rg_extra_args: list | None = None,
    context_before: int = 0,
    context_after: int = 0,
    max_results: int | None = None,
) -> tuple[list[dict], list[ContextLine], float]:
    """
    Process a seekable zstd file using parallel frame decompression.

    Each frame is processed independently in parallel, enabling fast search
    on large compressed files.

    Args:
        filepath: Path to the seekable zstd file
        pattern_ids: Dictionary mapping pattern_id -> pattern string
        rg_extra_args: Optional list of extra arguments to pass to ripgrep
        context_before: Number of context lines before each match
        context_after: Number of context lines after each match
        max_results: Optional maximum number of results to return

    Returns:
        Tuple of (list_of_match_dicts, list_of_context_lines, execution_time)
    """
    if rg_extra_args is None:
        rg_extra_args = []

    start_time = time.time()

    logger.info(f'[SEEKABLE] Processing seekable zstd file: {filepath}')

    # Get or build the index to get frame-to-line mapping
    index = get_or_build_index(filepath)

    logger.info(
        f'[SEEKABLE] File has {index.frame_count} frames, '
        f'{index.total_lines:,} lines, {index.decompressed_size_bytes:,} bytes decompressed'
    )

    # Create enhanced frame info list with line mapping
    # We'll use FrameLineInfo objects directly (they have all needed fields)
    # Create a list indexed by frame_index for O(1) lookup
    zstd_frames = index.frames  # These are FrameLineInfo objects

    # Check if frames are line-aligned by testing first frame
    # Line-aligned frames end with newline; non-aligned frames split lines across boundaries
    from rx.seekable_zstd import decompress_frame

    first_frame_data = decompress_frame(filepath, 0, read_seek_table(filepath))
    frames_are_line_aligned = first_frame_data.endswith(b'\n')

    # Batch frames to reduce subprocess overhead
    # Process FRAMES_PER_BATCH consecutive frames in each worker
    # NOTE: Batching only works correctly for line-aligned frames
    if frames_are_line_aligned:
        FRAMES_PER_BATCH = 100  # Batching enabled for line-aligned frames
        logger.info(f'[SEEKABLE] Frames are line-aligned, batching enabled ({FRAMES_PER_BATCH} frames/batch)')
    else:
        FRAMES_PER_BATCH = 1  # Disable batching for non-line-aligned frames
        logger.warning(
            '[SEEKABLE] Frames are NOT line-aligned, batching disabled for accuracy. '
            'Consider recreating this .zst file for better performance.'
        )

    frame_batches = []
    for i in range(0, len(index.frames), FRAMES_PER_BATCH):
        batch_indices = list(range(i, min(i + FRAMES_PER_BATCH, len(index.frames))))
        frame_batches.append(batch_indices)

    logger.info(f'[SEEKABLE] Created {len(frame_batches)} batches ({FRAMES_PER_BATCH} frames/batch)')

    # Track parallel tasks created
    prom.parallel_tasks_created.observe(len(frame_batches))

    # Process frame batches in parallel
    all_matches = []
    all_context_lines = []
    total_time = 0.0

    with ThreadPoolExecutor(max_workers=MAX_SUBPROCESSES, thread_name_prefix='SeekableWorker') as executor:
        # Submit batch tasks
        future_to_batch = {}
        for batch_indices in frame_batches:
            future = executor.submit(
                process_seekable_zstd_frame_batch,
                filepath,
                batch_indices,
                zstd_frames,
                pattern_ids,
                rg_extra_args,
                context_before,
                context_after,
            )
            future_to_batch[future] = batch_indices

        # Collect results
        for future in as_completed(future_to_batch):
            batch_indices = future_to_batch[future]

            try:
                _, matches, context_lines, elapsed = future.result()
                total_time += elapsed

                all_matches.extend(matches)
                all_context_lines.extend(context_lines)

                logger.debug(
                    f'[SEEKABLE] Batch {batch_indices[0]}-{batch_indices[-1]} contributed '
                    f'{len(matches)} matches, {len(context_lines)} context lines'
                )

                # Note: We don't break early on max_results because we need to sort all matches
                # by line number first to ensure we return the FIRST matches in the file,
                # not just the first matches found by parallel workers

            except Exception as e:
                logger.error(f'[SEEKABLE] Batch {batch_indices[0]}-{batch_indices[-1]} failed: {e}')

    # Sort matches by line number
    all_matches.sort(key=lambda m: m['line_number'])

    # Apply max_results limit
    if max_results and len(all_matches) > max_results:
        all_matches = all_matches[:max_results]

    elapsed = time.time() - start_time

    logger.info(
        f'[SEEKABLE] Completed: {len(all_matches)} matches, '
        f'{len(all_context_lines)} context lines in {elapsed:.3f}s '
        f'(worker time: {total_time:.3f}s)'
    )

    return (all_matches, all_context_lines, elapsed)


@dataclass
class HookCallbacks:
    """Callbacks for hook events during parsing.

    These callbacks are called synchronously during parsing.
    The caller is responsible for making them async/non-blocking if needed.
    """

    on_match_found: Callable[[dict], None] | None = None
    on_file_scanned: Callable[[dict], None] | None = None

    # Request metadata for hook payloads
    request_id: str = ''
    patterns: dict = field(default_factory=dict)  # pattern_id -> pattern string
    files: dict = field(default_factory=dict)  # file_id -> filepath


# Debug mode - creates debug files with full rg commands and output
DEBUG_MODE = os.getenv('RX_DEBUG', '').lower() in ('1', 'true', 'yes')


def identify_matching_patterns(
    line_text: str, submatches: list[Submatch], pattern_ids: dict[str, str], rg_extra_args: list[str]
) -> list[str]:
    """
    Identify which patterns actually matched the line by testing each pattern.

    Since ripgrep doesn't tell us which pattern matched when multiple patterns are used,
    we need to determine this in Python by checking which patterns match the submatches.

    A line may match multiple different patterns, so we return a list of pattern IDs.

    Args:
        line_text: The matched line text
        submatches: List of Submatch objects with matched text and positions
        pattern_ids: Dictionary mapping pattern_id -> pattern string

    Returns:
        List of pattern_ids that matched (may contain multiple patterns)
    """
    import re

    if not submatches:
        # No submatches, return first pattern
        return [list(pattern_ids.keys())[0]]

    # Get the matched text from submatches
    matched_texts = set(sm.text for sm in submatches)

    matching_pattern_ids = []

    # Try each pattern to see which ones match
    for pattern_id, pattern in pattern_ids.items():
        try:
            # Compile the pattern
            # regex = re.compile(pattern, re.IGNORECASE if '-i' in os.environ.get('RG_FLAGS', '') else 0)
            flags = re.NOFLAG
            if rg_extra_args and '-i' in rg_extra_args:
                flags |= re.IGNORECASE
            regex = re.compile(pattern, flags)

            # Find all matches in the line
            pattern_matches = set(m.group() for m in regex.finditer(line_text))

            # Check if any of this pattern's matches are in the submatches
            if pattern_matches & matched_texts:  # Set intersection
                matching_pattern_ids.append(pattern_id)

        except re.error:
            # Invalid regex, skip
            continue

    # Fallback: return first pattern if we couldn't identify any
    if not matching_pattern_ids:
        matching_pattern_ids = [list(pattern_ids.keys())[0]]

    return matching_pattern_ids


def process_task_worker(
    task: FileTask,
    pattern_ids: dict[str, str],
    rg_extra_args: list | None = None,
    context_before: int = 0,
    context_after: int = 0,
) -> tuple[FileTask, list[dict], list[ContextLine], float]:
    """
    Worker function to process a single FileTask with multiple patterns using ripgrep.
    Runs dd | rg --json pipeline and returns rich match data with optional context.

    Args:
        task: FileTask to process
        pattern_ids: Dictionary mapping pattern_id -> pattern string
        rg_extra_args: Optional list of extra arguments to pass to ripgrep
        context_before: Number of context lines before each match (0 = disabled)
        context_after: Number of context lines after each match (0 = disabled)

    Returns:
        Tuple of (task, list_of_match_dicts, list_of_context_lines, execution_time)

        Match dict structure:
        {
            'offset': int,           # Absolute byte offset of matched line start
            'pattern_ids': [str],    # List of pattern IDs that could have matched
            'line_number': int,      # Line number (1-indexed)
            'line_text': str,        # The matched line content
            'submatches': [Submatch] # Detailed submatch info
        }
    """
    if rg_extra_args is None:
        rg_extra_args = []

    start_time = time.time()
    thread_id = threading.current_thread().name

    # Track active workers
    prom.active_workers.inc()

    logger.debug(
        f'[WORKER {thread_id}] Starting JSON task {task.task_id}: '
        f'file={task.filepath}, offset={task.offset}, count={task.count}'
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

        logger.debug(
            f'[WORKER {thread_id}] Task {task.task_id}: '
            f'dd bs={bs} skip={skip_blocks} count={count_blocks}, '
            f'actual_dd_offset={actual_dd_offset}'
        )

        # Run dd | rg with --json mode
        dd_proc = subprocess.Popen(
            ['dd', f'if={task.filepath}', f'bs={bs}', f'skip={skip_blocks}', f'count={count_blocks}', 'status=none'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Build ripgrep command with --json and multiple -e patterns
        rg_cmd = ['rg', '--json', '--no-heading', '--color=never']

        # Add context flags if requested
        if context_before > 0:
            rg_cmd.extend(['-B', str(context_before)])
        if context_after > 0:
            rg_cmd.extend(['-A', str(context_after)])

        # Add all patterns with -e flag
        for pattern in pattern_ids.values():
            rg_cmd.extend(['-e', pattern])

        # Add extra args (but filter out incompatible ones)
        filtered_extra_args = [arg for arg in rg_extra_args if arg not in ['--byte-offset', '--only-matching']]
        rg_cmd.extend(filtered_extra_args)

        rg_cmd.append('-')  # Read from stdin

        logger.debug(f'[WORKER {thread_id}] Running: {" ".join(rg_cmd)}')

        rg_proc = subprocess.Popen(
            rg_cmd,
            stdin=dd_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if dd_proc.stdout:
            dd_proc.stdout.close()

        # Debug mode: capture all output for debugging
        debug_output = []
        debug_file = None

        if DEBUG_MODE:
            # Create debug file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # milliseconds
            debug_file = f'.debug_{timestamp}_thread{thread_id}_task{task.task_id}.txt'

            # Write header with command and metadata
            with open(debug_file, 'w') as f:
                f.write(f'Timestamp: {datetime.now().isoformat()}\n')
                f.write(f'Thread ID: {thread_id}\n')
                f.write(f'Task ID: {task.task_id}\n')
                f.write(f'File: {task.filepath}\n')
                f.write(f'Offset: {task.offset}\n')
                f.write(f'Count: {task.count}\n')
                f.write('\nDD Command:\n')
                f.write(f'  dd if={task.filepath} bs={bs} skip={skip_blocks} count={count_blocks} status=none\n')
                f.write('\nRipgrep Command:\n')
                f.write(f'  {" ".join(rg_cmd)}\n')
                f.write('\nRipgrep JSON Output:\n')
                f.write('=' * 80 + '\n')

        # Parse JSON events from ripgrep output
        # Collect matches and context lines
        matches = []
        context_lines = []

        for line in rg_proc.stdout or []:
            # Save to debug file if enabled
            if DEBUG_MODE and line:
                debug_output.append(line)

            event = parse_rg_json_event(line)

            if isinstance(event, RgMatchEvent):
                # Match event - extract rich data
                match_data = event.data

                # ripgrep's absolute_offset is relative to dd's output (which starts at actual_dd_offset)
                # Convert to true absolute offset in the original file
                absolute_offset = actual_dd_offset + match_data.absolute_offset

                # Critical: Only include matches that fall within THIS task's designated range
                # task.offset is where this task should start (inclusive)
                # task.offset + task.count is where this task should end (exclusive)
                # We read extra bytes (skip_remainder) at the start via dd, so we must filter them out
                # We also read extra bytes at the end (up to bs-1), so we must filter those too
                # This prevents duplicate matches across adjacent tasks
                if task.offset <= absolute_offset < task.offset + task.count:
                    # Extract submatches
                    submatches = [Submatch(text=sm.text, start=sm.start, end=sm.end) for sm in match_data.submatches]

                    # Since ripgrep can't tell us which specific pattern matched when using
                    # multiple patterns with flags like -i, we record all pattern IDs
                    matching_pattern_ids = list(pattern_ids.keys())

                    # For regular files, we don't track absolute line numbers during chunked processing
                    # Line numbers from rg are relative to the chunk, not the whole file
                    matches.append(
                        {
                            'offset': absolute_offset,
                            'pattern_ids': matching_pattern_ids,
                            'line_number': match_data.line_number,
                            'absolute_line_number': -1,  # Unknown for chunked processing
                            'line_text': match_data.lines.text.rstrip(NEWLINE_SYMBOL),
                            'submatches': submatches,
                        }
                    )

                    logger.debug(
                        f'[WORKER {thread_id}] Match: line={match_data.line_number}, '
                        f'offset={absolute_offset}, submatches={len(submatches)}'
                    )

            elif isinstance(event, RgContextEvent):
                # Context event - only include if within task range
                context_data = event.data
                absolute_offset = actual_dd_offset + context_data.absolute_offset

                if task.offset <= absolute_offset < task.offset + task.count:
                    context_lines.append(
                        ContextLine(
                            relative_line_number=context_data.line_number,
                            absolute_line_number=-1,  # Unknown for chunked processing
                            line_text=context_data.lines.text.rstrip(NEWLINE_SYMBOL),
                            absolute_offset=absolute_offset,
                        )
                    )

                    logger.debug(
                        f'[WORKER {thread_id}] Context: line={context_data.line_number}, offset={absolute_offset}'
                    )

        rg_proc.wait()
        dd_proc.wait()

        elapsed = time.time() - start_time

        # Write debug output to file if enabled
        if DEBUG_MODE and debug_file and debug_output:
            try:
                with open(debug_file, 'a') as f:
                    for output_line in debug_output:
                        f.write(output_line.decode('utf-8', errors='replace'))
                    f.write('\n' + '=' * 80 + '\n')
                    f.write('\nSummary:\n')
                    f.write(f'  Matches found: {len(matches)}\n')
                    f.write(f'  Context lines: {len(context_lines)}\n')
                    f.write(f'  Duration: {elapsed:.3f}s\n')
                    f.write(f'  Return code (rg): {rg_proc.returncode}\n')
                    f.write(f'  Return code (dd): {dd_proc.returncode}\n')
                logger.info(f'[WORKER {thread_id}] Debug output written to {debug_file}')
            except Exception as e:
                logger.warning(f'[WORKER {thread_id}] Failed to write debug file: {e}')

        logger.debug(
            f'[WORKER {thread_id}] Task {task.task_id} completed: '
            f'found {len(matches)} matches, {len(context_lines)} context lines in {elapsed:.3f}s'
        )

        # Track task completion
        prom.worker_tasks_completed.inc()
        prom.active_workers.dec()

        return (task, matches, context_lines, elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f'[WORKER {thread_id}] Task {task.task_id} failed after {elapsed:.3f}s: {e}')

        # Track task failure
        prom.worker_tasks_failed.inc()
        prom.active_workers.dec()

        return (task, [], [], elapsed)


def parse_multiple_files_multipattern(
    filepaths: list[str],
    pattern_ids: dict[str, str],
    file_ids: dict[str, str],
    max_results: int | None = None,
    rg_extra_args: list | None = None,
    context_before: int = 0,
    context_after: int = 0,
    hooks: HookCallbacks | None = None,
    use_cache: bool = True,
    use_index: bool = True,
) -> tuple[list[dict], dict[str, list[ContextLine]], dict[str, int]]:
    """
    Parse multiple files with multiple patterns and return rich match data.

    This function supports trace caching for large files. When a valid cache exists
    for a large file, matches are reconstructed from the cache instead of running
    the dd|rg pipeline. Cache is written for large files after successful complete scans.

    Args:
        filepaths: List of file paths
        pattern_ids: Dictionary mapping pattern_id -> pattern
        file_ids: Dictionary mapping file_id -> filepath
        max_results: Optional maximum number of results
        rg_extra_args: Optional list of extra arguments to pass to ripgrep
        context_before: Number of context lines before each match
        context_after: Number of context lines after each match
        hooks: Optional HookCallbacks for event notifications

    Returns:
        Tuple of (matches_list, context_dict, file_chunk_counts)

        matches_list: [{'pattern': 'p1', 'file': 'f1', 'offset': 100, 'line_number': 42, ...}, ...]
        context_dict: {'p1:f1:100': [ContextLine(...), ...], ...}
        file_chunk_counts: {'f1': 1, 'f2': 5, ...} - number of chunks per file
    """
    if rg_extra_args is None:
        rg_extra_args = []

    # Validate regex patterns
    for pattern in pattern_ids.values():
        test_proc = subprocess.run(['rg', '--', pattern], input=b'', capture_output=True, timeout=1)
        if test_proc.returncode == 2:
            error_msg = test_proc.stderr.decode('utf-8').strip()
            raise RuntimeError(f'Invalid regex pattern: {error_msg}')

    logger.info(f'[PARSE_JSON] Parsing {len(filepaths)} files with {len(pattern_ids)} patterns (JSON mode)')
    if context_before > 0 or context_after > 0:
        logger.info(f'[PARSE_JSON] Context requested: {context_before} before, {context_after} after')

    # Create reverse mapping: filepath -> file_id
    filepath_to_id = {v: k for k, v in file_ids.items()}

    # Get patterns list for cache operations
    patterns_list = list(pattern_ids.values())
    large_file_threshold = get_large_file_threshold_bytes()

    # Separate files into categories:
    # 1. Cached files (large files with valid cache)
    # 2. Seekable zstd files (parallel frame decompression)
    # 3. Compressed files (need sequential decompression)
    # 4. Regular files (can be chunked for parallel processing)
    files_to_process = []
    compressed_files = []  # List of compressed file paths
    seekable_zstd_files = []  # List of seekable zstd file paths to process
    seekable_zstd_cached = []  # (filepath, cache_info, file_size) for cache hits
    cached_files = []  # (filepath, cached_matches, file_size)

    for filepath in filepaths:
        try:
            file_size = os.path.getsize(filepath)
        except OSError:
            file_size = 0

        # Check if file is seekable zstd (prioritize over regular compressed)
        if is_seekable_zstd(filepath):
            # Check cache for seekable zstd files (if cache enabled)
            if use_cache:
                cache_info = get_compressed_cache_info(filepath, patterns_list, rg_extra_args)
                if cache_info is not None:
                    logger.info(
                        f'[PARSE_JSON] Seekable zstd cache hit for {filepath} '
                        f'({cache_info["match_count"]} matches in {len(cache_info["frames_with_matches"])} frames)'
                    )
                    seekable_zstd_cached.append((filepath, cache_info, file_size))
                    prom.trace_cache_hits_total.inc()
                    continue

            seekable_zstd_files.append((filepath, file_size))
            logger.info(f'[PARSE_JSON] Detected seekable zstd file: {filepath}')
            continue

        # Check if file is compressed
        if is_compressed(filepath):
            compressed_files.append((filepath, file_size))
            logger.info(f'[PARSE_JSON] Detected compressed file: {filepath}')
            continue

        # Check cache for large files (if cache enabled)
        if use_cache and file_size >= large_file_threshold:
            cached_matches = get_cached_matches(filepath, patterns_list, rg_extra_args)
            if cached_matches is not None:
                logger.info(f'[PARSE_JSON] Cache hit for {filepath} ({len(cached_matches)} matches)')
                cached_files.append((filepath, cached_matches, file_size))
                continue

        files_to_process.append(filepath)

    # Create tasks from regular (non-compressed) files that need processing
    all_tasks = []
    file_chunk_counts = {}  # file_id -> number of chunks/workers

    for filepath in files_to_process:
        try:
            file_tasks = create_file_tasks(filepath)
            all_tasks.extend(file_tasks)

            # Track how many chunks this file was split into
            file_id = filepath_to_id.get(filepath)
            if file_id:
                file_chunk_counts[file_id] = len(file_tasks)
                if len(file_tasks) > 1:
                    logger.info(f'[PARSE_JSON] {filepath} split into {len(file_tasks)} chunks for parallel processing')
        except Exception as e:
            logger.warning(f'[PARSE_JSON] Skipping {filepath}: {e}')

    # Mark cached files with chunk count of 0 (served from cache)
    for filepath, _, _ in cached_files:
        file_id = filepath_to_id.get(filepath)
        if file_id:
            file_chunk_counts[file_id] = 0  # 0 indicates cache hit

    # Mark seekable zstd files with their frame count (parallel processing)
    for filepath, _ in seekable_zstd_files:
        file_id = filepath_to_id.get(filepath)
        if file_id:
            # Get frame count for this file
            try:
                frames = read_seek_table(filepath)
                file_chunk_counts[file_id] = len(frames)
            except Exception:
                file_chunk_counts[file_id] = 1  # Fallback

    # Mark compressed files with chunk count of 1 (processed sequentially)
    for filepath, _ in compressed_files:
        file_id = filepath_to_id.get(filepath)
        if file_id:
            file_chunk_counts[file_id] = 1  # Compressed files are single-threaded

    logger.info(
        f'[PARSE_JSON] Created {len(all_tasks)} tasks from {len(files_to_process)} files '
        f'({len(cached_files)} cached, {len(seekable_zstd_cached)} seekable zstd cached, '
        f'{len(seekable_zstd_files)} seekable zstd to process, {len(compressed_files)} compressed)'
    )

    # Track parallel tasks created
    prom.parallel_tasks_created.observe(len(all_tasks))

    # Process cached files first - reconstruct matches
    matches = []
    all_context_lines = []
    total_time = 0.0

    for filepath, cached_matches, file_size in cached_files:
        file_id = filepath_to_id.get(filepath, 'f?')
        reconstruction_start = time.time()

        for cached_match in cached_matches:
            try:
                match_dict, ctx_lines = reconstruct_match_data(
                    filepath,
                    cached_match,
                    patterns_list,
                    pattern_ids,
                    file_id,
                    rg_extra_args,
                    context_before,
                    context_after,
                    use_index,
                )
                matches.append(match_dict)

                # Collect context lines with file_id association
                for ctx_line in ctx_lines:
                    all_context_lines.append((file_id, ctx_line))

                # Call on_match_found hook if configured
                if hooks and hooks.on_match_found:
                    try:
                        payload: dict = MatchFoundPayload(
                            request_id=hooks.request_id,
                            file_path=filepath,
                            pattern=patterns_list[cached_match.get('pattern_index', 0)],
                            offset=cached_match.get('offset', 0),
                            line_number=cached_match.get('line_number', 0),
                        ).model_dump()
                        hooks.on_match_found(payload)
                    except Exception as e:
                        logger.warning(f'[PARSE_JSON] on_match_found hook failed: {e}')

            except Exception as e:
                logger.warning(f'[PARSE_JSON] Failed to reconstruct match from cache: {e}')

        reconstruction_time = time.time() - reconstruction_start
        prom.trace_cache_reconstruction_seconds.observe(reconstruction_time)

        # Call on_file_scanned hook for cached file
        if hooks and hooks.on_file_scanned:
            try:
                payload = FileScannedPayload(
                    request_id=hooks.request_id,
                    file_path=filepath,
                    file_size_bytes=file_size,
                    scan_time_ms=int(reconstruction_time * 1000),
                    matches_count=len(cached_matches),
                ).model_dump()
                hooks.on_file_scanned(payload)
            except Exception as e:
                logger.warning(f'[PARSE_JSON] on_file_scanned hook failed: {e}')

        # Check max_results after cache reconstruction
        if max_results and len(matches) >= max_results:
            logger.info(f'[PARSE_JSON] Reached max_results={max_results} from cache')

    # Process seekable zstd cached files - reconstruct matches from decompressed frames
    for filepath, cache_info, file_size in seekable_zstd_cached:
        if max_results and len(matches) >= max_results:
            logger.info(f'[PARSE_JSON] Reached max_results={max_results}, skipping remaining seekable zstd cached')
            break

        file_id = filepath_to_id.get(filepath, 'f?')
        reconstruction_start = time.time()

        try:
            cached_matches = cache_info.get('matches', [])
            frames_with_matches = cache_info.get('frames_with_matches', [])

            # Reconstruct all matches from cached data (decompresses only needed frames)
            reconstructed_matches, reconstructed_context = reconstruct_seekable_zstd_matches(
                filepath,
                cached_matches,
                frames_with_matches,
                patterns_list,
                pattern_ids,
                file_id,
                rg_extra_args,
                context_before,
                context_after,
            )

            matches.extend(reconstructed_matches)

            # Collect context lines with file_id association
            for ctx_line in reconstructed_context:
                all_context_lines.append((file_id, ctx_line))

            # Call on_match_found hooks for each match
            if hooks and hooks.on_match_found:
                for match_dict in reconstructed_matches:
                    try:
                        payload: dict = MatchFoundPayload(
                            request_id=hooks.request_id,
                            file_path=filepath,
                            pattern=pattern_ids.get(match_dict.get('pattern', 'p1'), ''),
                            offset=match_dict.get('offset', 0),
                            line_number=match_dict.get('relative_line_number', 0),
                        ).model_dump()
                        hooks.on_match_found(payload)
                    except Exception as e:
                        logger.warning(f'[PARSE_JSON] on_match_found hook failed: {e}')

            reconstruction_time = time.time() - reconstruction_start

            # Call on_file_scanned hook
            if hooks and hooks.on_file_scanned:
                try:
                    payload = FileScannedPayload(
                        request_id=hooks.request_id,
                        file_path=filepath,
                        file_size_bytes=file_size,
                        scan_time_ms=int(reconstruction_time * 1000),
                        matches_count=len(reconstructed_matches),
                    ).model_dump()
                    hooks.on_file_scanned(payload)
                except Exception as e:
                    logger.warning(f'[PARSE_JSON] on_file_scanned hook failed: {e}')

            logger.info(
                f'[PARSE_JSON] Seekable zstd cache reconstruction for {filepath}: '
                f'{len(reconstructed_matches)} matches from {len(frames_with_matches)} frames in {reconstruction_time:.3f}s'
            )

        except Exception as e:
            logger.error(f'[PARSE_JSON] Failed to reconstruct seekable zstd cache for {filepath}: {e}')

    # Process compressed files (sequential decompression pipeline)
    for filepath, file_size in compressed_files:
        if max_results and len(matches) >= max_results:
            logger.info(f'[PARSE_JSON] Reached max_results={max_results}, skipping remaining compressed files')
            break

        file_id = filepath_to_id.get(filepath, 'f?')
        compression_start = time.time()

        try:
            # Calculate remaining results allowed
            remaining_results = None
            if max_results:
                remaining_results = max_results - len(matches)
                if remaining_results <= 0:
                    break

            compressed_matches, compressed_context, elapsed = process_compressed_file(
                filepath,
                pattern_ids,
                rg_extra_args,
                context_before,
                context_after,
                remaining_results,
            )
            total_time += elapsed

            # Convert compressed matches to API format
            for match_dict in compressed_matches:
                matching_pattern_ids = identify_matching_patterns(
                    match_dict['line_text'], match_dict['submatches'], pattern_ids, rg_extra_args
                )

                for matching_pattern_id in matching_pattern_ids:
                    match_entry = {
                        'pattern': matching_pattern_id,
                        'file': file_id,
                        'offset': match_dict['offset'],  # Decompressed byte offset
                        'relative_line_number': match_dict['line_number'],
                        'absolute_line_number': match_dict.get('absolute_line_number', -1),
                        'line_text': match_dict['line_text'],
                        'submatches': match_dict['submatches'],
                        'is_compressed': True,
                    }
                    matches.append(match_entry)

                    # Call on_match_found hook if configured
                    if hooks and hooks.on_match_found:
                        try:
                            payload: dict = MatchFoundPayload(
                                request_id=hooks.request_id,
                                file_path=filepath,
                                pattern=pattern_ids.get(matching_pattern_id, matching_pattern_id),
                                offset=match_dict['offset'],
                                line_number=match_dict['line_number'],
                            ).model_dump()
                            hooks.on_match_found(payload)
                        except Exception as e:
                            logger.warning(f'[PARSE_JSON] on_match_found hook failed: {e}')

            # Collect context lines with file_id association
            for ctx_line in compressed_context:
                all_context_lines.append((file_id, ctx_line))

            # Call on_file_scanned hook for compressed file
            if hooks and hooks.on_file_scanned:
                try:
                    payload = FileScannedPayload(
                        request_id=hooks.request_id,
                        file_path=filepath,
                        file_size_bytes=file_size,
                        scan_time_ms=int(elapsed * 1000),
                        matches_count=len(compressed_matches),
                    ).model_dump()
                    hooks.on_file_scanned(payload)
                except Exception as e:
                    logger.warning(f'[PARSE_JSON] on_file_scanned hook failed: {e}')

            logger.info(f'[PARSE_JSON] Compressed file {filepath}: {len(compressed_matches)} matches in {elapsed:.3f}s')

        except Exception as e:
            logger.error(f'[PARSE_JSON] Failed to process compressed file {filepath}: {e}')

    # Process seekable zstd files (parallel frame decompression)
    for filepath, file_size in seekable_zstd_files:
        if max_results and len(matches) >= max_results:
            logger.info(f'[PARSE_JSON] Reached max_results={max_results}, skipping remaining seekable zstd files')
            break

        file_id = filepath_to_id.get(filepath, 'f?')
        seekable_start = time.time()
        scan_completed = True  # Track if scan completed fully

        try:
            # Calculate remaining results allowed
            remaining_results = None
            if max_results:
                remaining_results = max_results - len(matches)
                if remaining_results <= 0:
                    break

            seekable_matches, seekable_context, elapsed = process_seekable_zstd_file(
                filepath,
                pattern_ids,
                rg_extra_args,
                context_before,
                context_after,
                remaining_results,
            )
            total_time += elapsed

            # Collect matches for caching (before converting to API format)
            matches_for_cache = []

            # Convert seekable zstd matches to API format
            for match_dict in seekable_matches:
                matching_pattern_ids = identify_matching_patterns(
                    match_dict['line_text'], match_dict['submatches'], pattern_ids, rg_extra_args
                )

                for matching_pattern_id in matching_pattern_ids:
                    match_entry = {
                        'pattern': matching_pattern_id,
                        'file': file_id,
                        'offset': match_dict['offset'],
                        'relative_line_number': match_dict['line_number'],
                        'absolute_line_number': match_dict.get('absolute_line_number', -1),
                        'line_text': match_dict['line_text'],
                        'submatches': match_dict['submatches'],
                        'frame_index': match_dict.get('frame_index'),  # Include for caching
                        'is_compressed': True,
                        'is_seekable_zstd': True,
                    }
                    matches.append(match_entry)
                    matches_for_cache.append(match_entry)

                    # Call on_match_found hook if configured
                    if hooks and hooks.on_match_found:
                        try:
                            payload: dict = MatchFoundPayload(
                                request_id=hooks.request_id,
                                file_path=filepath,
                                pattern=pattern_ids.get(matching_pattern_id, matching_pattern_id),
                                offset=match_dict['offset'],
                                line_number=match_dict['line_number'],
                            ).model_dump()
                            hooks.on_match_found(payload)
                        except Exception as e:
                            logger.warning(f'[PARSE_JSON] on_match_found hook failed: {e}')

            # Collect context lines with file_id association
            for ctx_line in seekable_context:
                all_context_lines.append((file_id, ctx_line))

            # Call on_file_scanned hook for seekable zstd file
            if hooks and hooks.on_file_scanned:
                try:
                    payload = FileScannedPayload(
                        request_id=hooks.request_id,
                        file_path=filepath,
                        file_size_bytes=file_size,
                        scan_time_ms=int(elapsed * 1000),
                        matches_count=len(seekable_matches),
                    ).model_dump()
                    hooks.on_file_scanned(payload)
                except Exception as e:
                    logger.warning(f'[PARSE_JSON] on_file_scanned hook failed: {e}')

            logger.info(
                f'[PARSE_JSON] Seekable zstd file {filepath}: {len(seekable_matches)} matches in {elapsed:.3f}s'
            )

            # Cache results for seekable zstd files (if cache enabled)
            if use_cache and should_cache_compressed_file(file_size, max_results, scan_completed):
                cache_data = build_cache_from_matches(
                    filepath,
                    patterns_list,
                    rg_extra_args,
                    matches_for_cache,
                    compression_format='zstd-seekable',
                )
                cache_path = get_trace_cache_path(filepath, patterns_list, rg_extra_args)
                save_trace_cache(cache_data, cache_path)
                logger.info(
                    f'[PARSE_JSON] Wrote seekable zstd trace cache for {filepath} ({len(matches_for_cache)} matches)'
                )

        except Exception as e:
            logger.error(f'[PARSE_JSON] Failed to process seekable zstd file {filepath}: {e}')

    # Track per-file statistics for hooks and caching
    file_stats: dict[str, dict] = {}  # file_id -> {start_time, matches_count, file_size, ...}
    for file_id, filepath in file_ids.items():
        # Skip files already served from cache
        if any(fp == filepath for fp, _, _ in cached_files):
            continue
        # Skip seekable zstd cached files (already processed above)
        if any(fp == filepath for fp, _, _ in seekable_zstd_cached):
            continue
        # Skip compressed files (already processed above)
        if any(fp == filepath for fp, _ in compressed_files):
            continue
        # Skip seekable zstd files (already processed above)
        if any(fp == filepath for fp, _ in seekable_zstd_files):
            continue
        try:
            file_size = os.path.getsize(filepath)
        except OSError:
            file_size = 0
        file_stats[file_id] = {
            'start_time': time.time(),
            'matches_count': 0,
            'file_size': file_size,
            'filepath': filepath,
            'tasks_completed': 0,
            'tasks_total': file_chunk_counts.get(file_id, 1),
            'matches_for_cache': [],  # Collect matches for caching
        }

    # Track whether max_results was hit (affects caching)
    max_results_hit = max_results and len(matches) >= max_results

    if all_tasks and not max_results_hit:
        with ThreadPoolExecutor(max_workers=MAX_SUBPROCESSES, thread_name_prefix='Worker') as executor:
            future_to_task = {
                executor.submit(
                    process_task_worker, task, pattern_ids, rg_extra_args, context_before, context_after
                ): task
                for task in all_tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]

                try:
                    task_result, match_dicts, context_lines, elapsed = future.result()
                    total_time += elapsed

                    # Get file_id for this task's filepath
                    file_id = filepath_to_id.get(task.filepath, 'f?')

                    # Convert match_dicts to API format
                    # Identify which patterns actually matched (a line may match multiple patterns)
                    for match_dict in match_dicts:
                        # Determine which patterns matched by analyzing the submatches
                        matching_pattern_ids = identify_matching_patterns(
                            match_dict['line_text'], match_dict['submatches'], pattern_ids, rg_extra_args
                        )

                        # Create one match per pattern that matched this line
                        for matching_pattern_id in matching_pattern_ids:
                            match_entry = {
                                'pattern': matching_pattern_id,
                                'file': file_id,
                                'offset': match_dict['offset'],
                                'relative_line_number': match_dict['line_number'],
                                'absolute_line_number': match_dict.get('absolute_line_number', -1),
                                'line_text': match_dict['line_text'],
                                'submatches': match_dict['submatches'],
                            }
                            matches.append(match_entry)

                            # Collect match for caching
                            if file_id in file_stats:
                                file_stats[file_id]['matches_for_cache'].append(match_entry)
                                file_stats[file_id]['matches_count'] += 1

                            # Call on_match_found hook if configured
                            if hooks and hooks.on_match_found:
                                try:
                                    payload: dict = MatchFoundPayload(
                                        request_id=hooks.request_id,
                                        file_path=file_stats[file_id]['filepath']
                                        if file_id in file_stats
                                        else task.filepath,
                                        pattern=pattern_ids.get(matching_pattern_id, matching_pattern_id),
                                        offset=match_dict['offset'],
                                        line_number=match_dict['line_number'],
                                    ).model_dump()
                                    hooks.on_match_found(payload)
                                except Exception as e:
                                    logger.warning(f'[PARSE_JSON] on_match_found hook failed: {e}')

                    # Collect context lines with file_id association
                    for ctx_line in context_lines:
                        # Add file_id as metadata for later grouping
                        ctx_line_with_file = (file_id, ctx_line)
                        all_context_lines.append(ctx_line_with_file)

                    # Track task completion for file
                    if file_id in file_stats:
                        file_stats[file_id]['tasks_completed'] += 1

                        # Check if all tasks for this file are complete
                        stats = file_stats[file_id]
                        if stats['tasks_completed'] >= stats['tasks_total']:
                            # Call on_file_scanned hook if configured
                            if hooks and hooks.on_file_scanned:
                                scan_time_ms = int((time.time() - stats['start_time']) * 1000)
                                try:
                                    payload = FileScannedPayload(
                                        request_id=hooks.request_id,
                                        file_path=stats['filepath'],
                                        file_size_bytes=stats['file_size'],
                                        scan_time_ms=scan_time_ms,
                                        matches_count=stats['matches_count'],
                                    ).model_dump()
                                    hooks.on_file_scanned(payload)
                                except Exception as e:
                                    logger.warning(f'[PARSE_JSON] on_file_scanned hook failed: {e}')

                    logger.debug(
                        f'[PARSE_JSON] Task {task.task_id} ({task.filepath}) contributed '
                        f'{len(match_dicts)} matches, {len(context_lines)} context lines'
                    )

                    # Check max_results
                    if max_results and len(matches) >= max_results:
                        logger.info(f'[PARSE_JSON] Reached max_results={max_results}, cancelling remaining')
                        max_results_hit = True
                        for f in future_to_task:
                            f.cancel()
                        break

                except Exception as e:
                    logger.error(f'[PARSE_JSON] Task failed: {e}')

    # Write cache for large files that completed successfully
    # Only cache if: no max_results limit, all tasks completed
    if use_cache and max_results is None and not max_results_hit:
        for file_id, stats in file_stats.items():
            filepath = stats['filepath']
            file_size = stats['file_size']

            # Check if this file should be cached
            if should_cache_file(file_size, max_results, stats['tasks_completed'] >= stats['tasks_total']):
                cache_data = build_cache_from_matches(
                    filepath,
                    patterns_list,
                    rg_extra_args,
                    stats['matches_for_cache'],
                )
                cache_path = get_trace_cache_path(filepath, patterns_list, rg_extra_args)
                save_trace_cache(cache_data, cache_path)
                logger.info(
                    f'[PARSE_JSON] Wrote trace cache for {filepath} ({len(stats["matches_for_cache"])} matches)'
                )

    # Sort matches by file, offset, then pattern
    matches.sort(key=lambda m: (m['file'], m['offset'], m['pattern']))

    # Apply max_results limit
    if max_results and len(matches) > max_results:
        matches = matches[:max_results]

    # Group context lines by match
    # Build composite key: "pattern:file:offset" -> [ContextLine, ...]
    context_dict = {}

    # Build a mapping of (file_id, line_number) -> match data for all matches
    # This helps us fill in matched lines that should appear in context but aren't in all_context_lines
    match_line_map = {(match['file'], match['relative_line_number']): match for match in matches}

    # For each match, build context lines
    for match in matches:
        match_line = match['relative_line_number']
        match_file = match['file']
        match_pattern = match['pattern']
        composite_key = f'{match_pattern}:{match_file}:{match["offset"]}'

        # Always create a ContextLine for the matched line itself
        matched_context_line = ContextLine(
            relative_line_number=match['relative_line_number'],
            absolute_line_number=match.get('absolute_line_number', -1),
            line_text=match['line_text'],
            absolute_offset=match['offset'],
        )

        if context_before > 0 or context_after > 0:
            # Find nearby context lines for this file near this match
            # Include ALL lines in the context range, even if they are matches
            # all_context_lines is now a list of (file_id, ContextLine) tuples
            nearby_context = [
                ctx_line
                for file_id, ctx_line in all_context_lines
                if (
                    file_id == match_file
                    and abs(ctx_line.relative_line_number - match_line) <= max(context_before, context_after)
                )
            ]

            # Also check if any matched lines fall in the context range but aren't in nearby_context
            # This can happen when a line matches multiple patterns
            context_line_numbers = {ctx.relative_line_number for ctx in nearby_context}
            for other_match in matches:
                if (
                    other_match['file'] == match_file
                    and other_match['relative_line_number'] != match_line
                    and abs(other_match['relative_line_number'] - match_line) <= max(context_before, context_after)
                    and other_match['relative_line_number'] not in context_line_numbers
                ):
                    # This matched line is in range but missing from context - add it
                    nearby_context.append(
                        ContextLine(
                            relative_line_number=other_match['relative_line_number'],
                            absolute_line_number=other_match.get('absolute_line_number', -1),
                            line_text=other_match['line_text'],
                            absolute_offset=other_match['offset'],
                        )
                    )

            # Combine context lines with the matched line and sort by line number
            all_lines = nearby_context + [matched_context_line]
            all_lines.sort(key=lambda ctx: ctx.relative_line_number)
            context_dict[composite_key] = all_lines
        else:
            # context=0: only show the matched line itself
            context_dict[composite_key] = [matched_context_line]

    logger.info(
        f'[PARSE_JSON] Completed: {len(matches)} matches, '
        f'{len(context_dict)} context groups, total worker time: {total_time:.3f}s'
    )

    return matches, context_dict, file_chunk_counts


def parse_paths(
    paths: list[str],
    regexps: list[str],
    max_results: int | None = None,
    rg_extra_args: list | None = None,
    context_before: int = 0,
    context_after: int = 0,
    hooks: HookCallbacks | None = None,
    use_cache: bool = True,
    use_index: bool = True,
) -> ParseResult:
    """
    Parse files or directories for multiple regex patterns.
    Returns ID-based structure with rich match data and optional context.

    Args:
        paths: List of file or directory paths to search
        regexps: List of regular expression patterns to search for
        max_results: Optional maximum number of results to find before stopping
        rg_extra_args: Optional list of extra arguments to pass to ripgrep
        context_before: Number of context lines before each match (0 = disabled)
        context_after: Number of context lines after each match (0 = disabled)
        hooks: Optional HookCallbacks for event notifications
        use_cache: Whether to use trace cache for reading/writing (default: True)
        use_index: Whether to use file indexes for faster processing (default: True)

    Returns:
        Dictionary with ID-based structure including rich match data and optional context.
    """
    if rg_extra_args is None:
        rg_extra_args = []

    pattern_ids = {f'p{i + 1}': pattern for i, pattern in enumerate(regexps)}

    # Update hooks with pattern info if provided
    if hooks:
        hooks.patterns = pattern_ids
    logger.info(f'[PARSE_JSON] Processing {len(pattern_ids)} pattern(s) across {len(paths)} path(s)')

    # Collect all files to parse from all provided paths
    all_files_to_parse = []
    all_skipped_files = []
    all_scanned_dirs = []

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Path not found: {path}')

        if os.path.isdir(path):
            logger.info(f"[PARSE_JSON] Path '{path}' is directory, scanning for text files...")
            text_files, skipped_files = scan_directory_for_text_files(path)

            if text_files:
                all_files_to_parse.extend(text_files)
                all_scanned_dirs.append(path)
            all_skipped_files.extend(skipped_files)
            logger.info(f"[PARSE_JSON] Found {len(text_files)} text file(s) in '{path}'")
        else:
            # Single file - validate and add to list
            try:
                validate_file(path)
                all_files_to_parse.append(path)
                logger.info(f"[PARSE_JSON] Added file '{path}'")
            except ValueError as e:
                logger.warning(f"[PARSE_JSON] Skipping invalid file '{path}': {e}")
                all_skipped_files.append(path)

    if not all_files_to_parse:
        logger.warning('[PARSE_JSON] No valid files found across all paths')
        return ParseResult(
            patterns=pattern_ids,
            files={},
            matches=[],
            scanned_files=[],
            skipped_files=all_skipped_files,
            file_chunks={},
            context_lines={},
            before_context=context_before,
            after_context=context_after,
        )

    # Generate file IDs for all files
    file_ids = {f'f{i + 1}': filepath for i, filepath in enumerate(all_files_to_parse)}

    # Update hooks with file info if provided
    if hooks:
        hooks.files = file_ids

    # Parse all files
    logger.info(f'[PARSE_JSON] Parsing {len(all_files_to_parse)} file(s) with {len(pattern_ids)} pattern(s)')
    matches, context_dict, file_chunk_counts = parse_multiple_files_multipattern(
        all_files_to_parse,
        pattern_ids,
        file_ids,
        max_results,
        rg_extra_args,
        context_before,
        context_after,
        hooks,
        use_cache,
        use_index,
    )

    return ParseResult(
        patterns=pattern_ids,
        files=file_ids,
        matches=matches,
        scanned_files=all_files_to_parse if all_scanned_dirs else [],
        skipped_files=all_skipped_files,
        file_chunks=file_chunk_counts,
        context_lines=context_dict,
        before_context=context_before,
        after_context=context_after,
    )

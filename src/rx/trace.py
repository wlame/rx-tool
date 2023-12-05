"""Main tracing/search engine using ripgrep

This module provides the core tracing functionality that uses ripgrep's --json
output format for richer match data and context extraction.
"""

import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from rx.cli import prometheus as prom
from rx.file_utils import MAX_SUBPROCESSES, FileTask, create_file_tasks, scan_directory_for_text_files, validate_file
from rx.models import ContextLine, FileScannedPayload, MatchFoundPayload, ParseResult, Submatch
from rx.rg_json import RgContextEvent, RgMatchEvent, parse_rg_json_event
from rx.utils import NEWLINE_SYMBOL

logger = logging.getLogger(__name__)


@dataclass
class HookCallbacks:
    """Callbacks for hook events during parsing.

    These callbacks are called synchronously during parsing.
    The caller is responsible for making them async/non-blocking if needed.
    """

    on_match_found: Optional[Callable[[dict], None]] = None
    on_file_scanned: Optional[Callable[[dict], None]] = None

    # Request metadata for hook payloads
    request_id: str = ""
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
        f"[WORKER {thread_id}] Starting JSON task {task.task_id}: "
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

        logger.debug(
            f"[WORKER {thread_id}] Task {task.task_id}: "
            f"dd bs={bs} skip={skip_blocks} count={count_blocks}, "
            f"actual_dd_offset={actual_dd_offset}"
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

        logger.debug(f"[WORKER {thread_id}] Running: {' '.join(rg_cmd)}")

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
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Thread ID: {thread_id}\n")
                f.write(f"Task ID: {task.task_id}\n")
                f.write(f"File: {task.filepath}\n")
                f.write(f"Offset: {task.offset}\n")
                f.write(f"Count: {task.count}\n")
                f.write(f"\nDD Command:\n")
                f.write(f"  dd if={task.filepath} bs={bs} skip={skip_blocks} count={count_blocks} status=none\n")
                f.write(f"\nRipgrep Command:\n")
                f.write(f"  {' '.join(rg_cmd)}\n")
                f.write(f"\nRipgrep JSON Output:\n")
                f.write("=" * 80 + "\n")

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

                    matches.append(
                        {
                            'offset': absolute_offset,
                            'pattern_ids': matching_pattern_ids,
                            'line_number': match_data.line_number,
                            'line_text': match_data.lines.text.rstrip(NEWLINE_SYMBOL),
                            'submatches': submatches,
                        }
                    )

                    logger.debug(
                        f"[WORKER {thread_id}] Match: line={match_data.line_number}, "
                        f"offset={absolute_offset}, submatches={len(submatches)}"
                    )

            elif isinstance(event, RgContextEvent):
                # Context event - only include if within task range
                context_data = event.data
                absolute_offset = actual_dd_offset + context_data.absolute_offset

                if task.offset <= absolute_offset < task.offset + task.count:
                    context_lines.append(
                        ContextLine(
                            relative_line_number=context_data.line_number,
                            line_text=context_data.lines.text.rstrip(NEWLINE_SYMBOL),
                            absolute_offset=absolute_offset,
                        )
                    )

                    logger.debug(
                        f"[WORKER {thread_id}] Context: line={context_data.line_number}, offset={absolute_offset}"
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
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"\nSummary:\n")
                    f.write(f"  Matches found: {len(matches)}\n")
                    f.write(f"  Context lines: {len(context_lines)}\n")
                    f.write(f"  Duration: {elapsed:.3f}s\n")
                    f.write(f"  Return code (rg): {rg_proc.returncode}\n")
                    f.write(f"  Return code (dd): {dd_proc.returncode}\n")
                logger.info(f"[WORKER {thread_id}] Debug output written to {debug_file}")
            except Exception as e:
                logger.warning(f"[WORKER {thread_id}] Failed to write debug file: {e}")

        logger.debug(
            f"[WORKER {thread_id}] Task {task.task_id} completed: "
            f"found {len(matches)} matches, {len(context_lines)} context lines in {elapsed:.3f}s"
        )

        # Track task completion
        prom.worker_tasks_completed.inc()
        prom.active_workers.dec()

        return (task, matches, context_lines, elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[WORKER {thread_id}] Task {task.task_id} failed after {elapsed:.3f}s: {e}")

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
    hooks: Optional[HookCallbacks] = None,
) -> tuple[list[dict], dict[str, list[ContextLine]], dict[str, int]]:
    """
    Parse multiple files with multiple patterns and return rich match data.

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
            raise RuntimeError(f"Invalid regex pattern: {error_msg}")

    logger.info(f"[PARSE_JSON] Parsing {len(filepaths)} files with {len(pattern_ids)} patterns (JSON mode)")
    if context_before > 0 or context_after > 0:
        logger.info(f"[PARSE_JSON] Context requested: {context_before} before, {context_after} after")

    # Create reverse mapping: filepath -> file_id
    filepath_to_id = {v: k for k, v in file_ids.items()}

    # Create tasks from all files and track chunk counts per file
    all_tasks = []
    file_chunk_counts = {}  # file_id -> number of chunks/workers

    for filepath in filepaths:
        try:
            file_tasks = create_file_tasks(filepath)
            all_tasks.extend(file_tasks)

            # Track how many chunks this file was split into
            file_id = filepath_to_id.get(filepath)
            if file_id:
                file_chunk_counts[file_id] = len(file_tasks)
                if len(file_tasks) > 1:
                    logger.info(f"[PARSE_JSON] {filepath} split into {len(file_tasks)} chunks for parallel processing")
        except Exception as e:
            logger.warning(f"[PARSE_JSON] Skipping {filepath}: {e}")

    logger.info(f"[PARSE_JSON] Created {len(all_tasks)} tasks from {len(filepaths)} files")

    # Track parallel tasks created
    prom.parallel_tasks_created.observe(len(all_tasks))

    # Process all tasks with JSON worker
    matches = []
    all_context_lines = []
    total_time = 0.0

    # Track per-file statistics for hooks
    file_stats: dict[str, dict] = {}  # file_id -> {start_time, matches_count, file_size}
    for file_id, filepath in file_ids.items():
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
        }

    with ThreadPoolExecutor(max_workers=MAX_SUBPROCESSES, thread_name_prefix="Worker") as executor:
        future_to_task = {
            executor.submit(process_task_worker, task, pattern_ids, rg_extra_args, context_before, context_after): task
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
                            'line_text': match_dict['line_text'],
                            'submatches': match_dict['submatches'],
                        }
                        matches.append(match_entry)

                        # Update file stats for hook
                        if file_id in file_stats:
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
                                logger.warning(f"[PARSE_JSON] on_match_found hook failed: {e}")

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
                                logger.warning(f"[PARSE_JSON] on_file_scanned hook failed: {e}")

                logger.debug(
                    f"[PARSE_JSON] Task {task.task_id} ({task.filepath}) contributed "
                    f"{len(match_dicts)} matches, {len(context_lines)} context lines"
                )

                # Check max_results
                if max_results and len(matches) >= max_results:
                    logger.info(f"[PARSE_JSON] Reached max_results={max_results}, cancelling remaining")
                    for f in future_to_task:
                        f.cancel()
                    break

            except Exception as e:
                logger.error(f"[PARSE_JSON] Task failed: {e}")

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
        composite_key = f"{match_pattern}:{match_file}:{match['offset']}"

        # Always create a ContextLine for the matched line itself
        matched_context_line = ContextLine(
            relative_line_number=match['relative_line_number'],
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
        f"[PARSE_JSON] Completed: {len(matches)} matches, "
        f"{len(context_dict)} context groups, total worker time: {total_time:.3f}s"
    )

    return matches, context_dict, file_chunk_counts


def parse_paths(
    paths: list[str],
    regexps: list[str],
    max_results: int | None = None,
    rg_extra_args: list | None = None,
    context_before: int = 0,
    context_after: int = 0,
    hooks: Optional[HookCallbacks] = None,
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

    Returns:
        Dictionary with ID-based structure including rich match data and optional context.
    """
    if rg_extra_args is None:
        rg_extra_args = []

    pattern_ids = {f"p{i + 1}": pattern for i, pattern in enumerate(regexps)}

    # Update hooks with pattern info if provided
    if hooks:
        hooks.patterns = pattern_ids
    logger.info(f"[PARSE_JSON] Processing {len(pattern_ids)} pattern(s) across {len(paths)} path(s)")

    # Collect all files to parse from all provided paths
    all_files_to_parse = []
    all_skipped_files = []
    all_scanned_dirs = []

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

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
        logger.warning(f"[PARSE_JSON] No valid files found across all paths")
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
    file_ids = {f"f{i + 1}": filepath for i, filepath in enumerate(all_files_to_parse)}

    # Update hooks with file info if provided
    if hooks:
        hooks.files = file_ids

    # Parse all files
    logger.info(f"[PARSE_JSON] Parsing {len(all_files_to_parse)} file(s) with {len(pattern_ids)} pattern(s)")
    matches, context_dict, file_chunk_counts = parse_multiple_files_multipattern(
        all_files_to_parse,
        pattern_ids,
        file_ids,
        max_results,
        rg_extra_args,
        context_before,
        context_after,
        hooks,
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

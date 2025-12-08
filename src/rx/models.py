"""Pydantic models for API requests and responses"""

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f'{size_bytes:.2f} {unit}'
        size_bytes /= 1024
    return f'{size_bytes:.2f} PB'


# Trace Cache Models


class TraceCacheMatch(BaseModel):
    """A single cached match entry.

    Stores the minimal information needed to reconstruct a full match:
    - pattern_index: Index into the patterns array (0-based)
    - offset: Absolute byte offset in file where matched line starts
    - line_number: 1-based line number of the match
    """

    pattern_index: int = Field(..., description='Index into patterns array (0-based)')
    offset: int = Field(..., description='Byte offset in file where matched line starts')
    line_number: int = Field(..., description='Line number (1-based)')


class TraceCacheData(BaseModel):
    """Full trace cache file structure.

    Contains all metadata needed to validate and use cached trace results.
    """

    version: int = Field(..., description='Cache format version')
    source_path: str = Field(..., description='Absolute path to source file')
    source_modified_at: str = Field(..., description='Source file modification time (ISO format)')
    source_size_bytes: int = Field(..., description='Source file size in bytes')
    patterns: list[str] = Field(..., description='Regex patterns used for this cache')
    patterns_hash: str = Field(..., description='Hash of patterns + relevant flags')
    rg_flags: list[str] = Field(default_factory=list, description='Ripgrep flags that affect matching')
    created_at: str = Field(..., description='Cache creation time (ISO format)')
    matches: list[TraceCacheMatch] = Field(default_factory=list, description='Cached match entries')


class HealthResponse(BaseModel):
    """Health check response with system introspection data"""

    status: str = Field(..., example='ok')
    ripgrep_available: bool = Field(..., example=True)
    app_version: str = Field(..., example='0.1.0', description='Application version')
    python_version: str = Field(..., example='3.13.1', description='Python interpreter version')
    os_info: dict[str, str] = Field(
        ...,
        example={'system': 'Darwin', 'release': '23.0.0', 'version': 'Darwin Kernel Version 23.0.0'},
        description='Operating system information',
    )
    system_resources: dict[str, Any] = Field(
        ...,
        example={'cpu_cores': 8, 'cpu_cores_physical': 4, 'ram_total_gb': 16.0, 'ram_available_gb': 8.5},
        description='System resources (CPU cores and RAM)',
    )
    python_packages: dict[str, str] = Field(
        default_factory=dict,
        example={'fastapi': '0.115.6', 'pydantic': '2.11.0', 'uvicorn': '0.34.0'},
        description='Key Python package versions',
    )
    constants: dict[str, Any] = Field(
        default_factory=dict,
        example={'LOG_LEVEL': 'INFO', 'MIN_CHUNK_SIZE_MB': 20, 'MAX_SUBPROCESSES': 20},
        description='Application configuration constants',
    )
    environment: dict[str, str] = Field(
        default_factory=dict, example={'RX_LOG_LEVEL': 'INFO'}, description='Application-related environment variables'
    )
    docs_url: str = Field(
        ..., example='https://github.com/wlame/rx-tool', description='Link to application documentation'
    )


class RequestInfo(BaseModel):
    """Information about a trace request for monitoring and tracking."""

    model_config = ConfigDict(
        # Allow attribute assignment after creation (needed for increment_hook_counter)
        validate_assignment=True
    )

    request_id: str = Field(..., description='Unique identifier for the trace request')
    paths: list[str] = Field(..., description='Paths that were searched')
    patterns: list[str] = Field(..., description='Regex patterns used')
    max_results: int | None = Field(None, description='Maximum number of results requested')
    started_at: datetime = Field(..., description='When the request started')
    completed_at: datetime | None = Field(None, description='When the request completed')
    total_matches: int = Field(default=0, description='Total number of matches found')
    total_files_scanned: int = Field(default=0, description='Total number of files scanned')
    total_files_skipped: int = Field(default=0, description='Total number of files skipped')
    total_time_ms: int = Field(default=0, description='Total processing time in milliseconds')
    hook_on_file_success: int = Field(default=0, description='Successful on_file hook calls')
    hook_on_file_failed: int = Field(default=0, description='Failed on_file hook calls')
    hook_on_match_success: int = Field(default=0, description='Successful on_match hook calls')
    hook_on_match_failed: int = Field(default=0, description='Failed on_match hook calls')
    hook_on_complete_success: int = Field(default=0, description='Successful on_complete hook calls')
    hook_on_complete_failed: int = Field(default=0, description='Failed on_complete hook calls')

    @computed_field
    @property
    def hooks(self) -> dict:
        """Computed field that returns hook statistics in nested structure."""
        return {
            'on_file': {
                'success': self.hook_on_file_success,
                'failed': self.hook_on_file_failed,
            },
            'on_match': {
                'success': self.hook_on_match_success,
                'failed': self.hook_on_match_failed,
            },
            'on_complete': {
                'success': self.hook_on_complete_success,
                'failed': self.hook_on_complete_failed,
            },
        }


class Submatch(BaseModel):
    """A submatch within a matched line

    Attributes:
        text: The actual matched text
        start: Byte offset from start of line where match begins
        end: Byte offset from start of line where match ends
    """

    text: str = Field(..., example='error', description='The matched text')
    start: int = Field(..., example=10, description='Start position in line (bytes)')
    end: int = Field(..., example=15, description='End position in line (bytes)')


class ParseResult(BaseModel):
    """Result from parsing files for regex patterns

    This is the internal result structure returned by parse_paths() before
    being converted to API response formats.

    Attributes:
        patterns: Mapping of pattern IDs to pattern strings (e.g., {"p1": "error", "p2": "warning"})
        files: Mapping of file IDs to file paths (e.g., {"f1": "/path/file.log"})
        matches: List of match dictionaries with pattern_id, file_id, offset, line info
        scanned_files: List of all files that were scanned (for directory scans)
        skipped_files: List of files that were skipped (binary, inaccessible, etc.)
        file_chunks: Mapping of file IDs to number of chunks processed
        context_lines: Mapping of composite keys to context line lists (e.g., "p1:f1:100" -> [ContextLine, ...])
        before_context: Number of context lines before matches
        after_context: Number of context lines after matches
    """

    patterns: dict[str, str] = Field(default_factory=dict, description='Pattern ID to pattern string mapping')
    files: dict[str, str] = Field(default_factory=dict, description='File ID to filepath mapping')
    matches: list[dict[str, Any]] = Field(default_factory=list, description='List of match dictionaries')
    scanned_files: list[str] = Field(default_factory=list, description='Files that were scanned')
    skipped_files: list[str] = Field(default_factory=list, description='Files that were skipped')
    file_chunks: dict[str, int] = Field(default_factory=dict, description='File ID to chunk count mapping')
    context_lines: dict[str, list['ContextLine']] = Field(
        default_factory=dict, description='Context lines by composite key'
    )
    before_context: int = Field(default=0, description='Number of lines before matches')
    after_context: int = Field(default=0, description='Number of lines after matches')


class ContextLine(BaseModel):
    """A context line (non-matching line shown for context)

    Attributes:
        relative_line_number: Line number relative to file start (1-indexed).
                             When file_chunks=1, this is the absolute line number.
                             When file_chunks>1, this may be relative to chunk boundary.
        absolute_line_number: Absolute line number from start of file (1-indexed), or -1 if unknown.
                             This is always set when possible (for indexed files, complete scans, etc.)
        line_text: The actual line content
        absolute_offset: Byte offset from start of file to start of this line
    """

    relative_line_number: int = Field(
        ..., example=42, description='Line number in file (1-indexed, see file_chunks to determine if absolute)'
    )
    absolute_line_number: int = Field(
        default=-1, example=42, description='Absolute line number from file start (1-indexed), or -1 if unknown'
    )
    line_text: str = Field(..., example='  previous line', description='Line content')
    absolute_offset: int = Field(..., example=1234, description='Byte offset in file')


class Match(BaseModel):
    """A single match with pattern ID, file ID, and rich metadata

    Attributes:
        pattern: Pattern ID (e.g., 'p1', 'p2')
        file: File ID (e.g., 'f1', 'f2')
        offset: Absolute byte offset in file where the matched LINE starts
        relative_line_number: Line number relative to file start (1-indexed, optional).
                             When file_chunks=1, this is the absolute line number.
                             When file_chunks>1, this may be relative to chunk boundary.
        absolute_line_number: Absolute line number from start of file (1-indexed), or -1 if unknown.
                             This is always set when possible (for indexed files, complete scans, etc.)
        line_text: The actual line content that matched (optional)
        submatches: List of submatch details with positions (optional)
    """

    pattern: str = Field(..., example='p1', description='Pattern ID (p1, p2, ...)')
    file: str = Field(..., example='f1', description='File ID (f1, f2, ...)')
    offset: int = Field(..., example=123, description='Byte offset in file where matched line starts')
    relative_line_number: int | None = Field(
        None, example=42, description='Line number in file (1-indexed, see file_chunks to determine if absolute)'
    )
    absolute_line_number: int = Field(
        default=-1, example=42, description='Absolute line number from file start (1-indexed), or -1 if unknown'
    )
    line_text: str | None = Field(None, example='error: something failed', description='The matched line')
    submatches: list[Submatch] | None = Field(None, description='Submatch details with positions')


class TraceResponse(BaseModel):
    """Response from trace endpoint using ID-based structure for multi-pattern support

    Attributes:
        request_id: Unique request identifier (UUID v7, time-sortable)
        path: Path(s) that were searched (list of paths)
        time: Search duration in seconds
        patterns: Mapping of pattern IDs to pattern strings
        files: Mapping of file IDs to file paths
        matches: List of matches with rich metadata
        scanned_files: List of files that were scanned
        skipped_files: List of files that were skipped
        file_chunks: Optional mapping of file IDs to number of chunks/workers used
        context_lines: Optional mapping of match composite keys to context lines
        before_context: Number of context lines shown before matches (if context requested)
        after_context: Number of context lines shown after matches (if context requested)
        max_results: Maximum number of results requested (None if not specified)
    """

    request_id: str = Field(
        ...,
        example='01936c8e-7b2a-7000-8000-000000000001',
        description='Unique request identifier (UUID v7, time-sortable)',
    )
    path: list[str] = Field(..., example=['/path/to/dir'], description='List of paths that were searched')
    time: float = Field(..., example=0.123)
    patterns: dict[str, str] = Field(..., example={'p1': 'error', 'p2': 'warning'})
    files: dict[str, str] = Field(..., example={'f1': '/path/file.log'})
    matches: list[Match] = Field(default=[], example=[])
    scanned_files: list[str] = Field(default=[], example=[])
    skipped_files: list[str] = Field(default=[], example=[])
    max_results: int | None = Field(None, description='Maximum number of results requested (None if not specified)')

    # File chunking metadata - shows how files were processed
    file_chunks: dict[str, int] | None = Field(
        None,
        description='Number of chunks/workers used per file (file_id -> num_chunks). '
        '1 = single worker (no chunking), >1 = parallel processing with multiple workers',
        example={'f1': 1, 'f2': 5, 'f3': 20},
    )

    # Context lines are stored with composite key: "p1:f1:100" -> [ContextLine, ...]
    context_lines: dict[str, list[ContextLine]] | None = Field(None, description='Context lines around matches')
    before_context: int | None = Field(None, example=3, description='Lines shown before matches')
    after_context: int | None = Field(None, example=3, description='Lines shown after matches')

    def to_cli(self, colorize: bool = False) -> str:
        """Format response for CLI output (human-readable, uses values instead of IDs)"""
        # ANSI color codes
        GREY = '\033[90m'
        CYAN = '\033[36m'
        BOLD_CYAN = '\033[1;36m'
        YELLOW = '\033[33m'
        MAGENTA = '\033[35m'
        BOLD_MAGENTA = '\033[1;35m'
        GREEN = '\033[32m'
        BOLD_GREEN = '\033[1;32m'
        BLUE = '\033[34m'
        RESET = '\033[0m'

        lines = []

        # Request ID in grey
        if colorize:
            lines.append(f'{GREY}Request ID:{RESET} {self.request_id}')
        else:
            lines.append(f'Request ID: {self.request_id}')

        # Path(s) in bold cyan - handle single or multiple paths
        path_display = ', '.join(self.path) if len(self.path) > 1 else self.path[0]
        if colorize:
            lines.append(f'{GREY}Path:{RESET} {BOLD_CYAN}{path_display}{RESET}')
        else:
            lines.append(f'Path: {path_display}')

        # Show patterns with magenta color
        if len(self.patterns) == 1:
            pattern_val = list(self.patterns.values())[0]
            if colorize:
                lines.append(f'{GREY}Pattern:{RESET} {BOLD_MAGENTA}{pattern_val}{RESET}')
            else:
                lines.append(f'Pattern: {pattern_val}')
        else:
            if colorize:
                lines.append(f'{GREY}Patterns ({len(self.patterns)}):{RESET}')
                for pid, pattern in sorted(self.patterns.items()):
                    lines.append(f'  {BLUE}{pid}{RESET}: {MAGENTA}{pattern}{RESET}')
            else:
                lines.append(f'Patterns ({len(self.patterns)}):')
                for pid, pattern in sorted(self.patterns.items()):
                    lines.append(f'  {pid}: {pattern}')

        # Time in yellow
        if colorize:
            lines.append(f'{GREY}Time:{RESET} {YELLOW}{self.time:.3f}s{RESET}')
        else:
            lines.append(f'Time: {self.time:.3f}s')

        # File stats in green
        if self.scanned_files:
            if colorize:
                lines.append(f'{GREY}Files scanned:{RESET} {GREEN}{len(self.scanned_files)}{RESET}')
            else:
                lines.append(f'Files scanned: {len(self.scanned_files)}')
        if self.skipped_files:
            if colorize:
                lines.append(f'{GREY}Files skipped:{RESET} {GREY}{len(self.skipped_files)}{RESET}')
            else:
                lines.append(f'Files skipped: {len(self.skipped_files)}')

        # File chunking info (show if any files were chunked)
        if self.file_chunks:
            chunked_files = [fid for fid, count in self.file_chunks.items() if count > 1]
            if chunked_files:
                total_chunks = sum(self.file_chunks.values())
                if colorize:
                    lines.append(
                        f'{GREY}Parallel workers:{RESET} {CYAN}{total_chunks}{RESET} '
                        f'{GREY}({len(chunked_files)} file(s) chunked){RESET}'
                    )
                else:
                    lines.append(f'Parallel workers: {total_chunks} ({len(chunked_files)} file(s) chunked)')

        # Matches count in bold green
        if colorize:
            lines.append(f'{GREY}Matches:{RESET} {BOLD_GREEN}{len(self.matches)}{RESET}')
        else:
            lines.append(f'Matches: {len(self.matches)}')

        if self.matches:
            lines.append('')
            if colorize:
                lines.append(f'{GREY}Matches (file:line:offset [pattern]):{RESET}')
            else:
                lines.append('Matches (file:line:offset [pattern]:')

            for match in self.matches:
                pattern_val = self.patterns.get(match.pattern, match.pattern)
                file_val = self.files.get(match.file, match.file)

                # Use absolute_line_number if available, otherwise relative_line_number, or -1
                line_num = (
                    match.absolute_line_number
                    if match.absolute_line_number != -1
                    else (match.relative_line_number or -1)
                )

                if colorize:
                    # Add ORANGE color for offset
                    ORANGE = '\033[38;5;214m'  # Orange color
                    LIGHT_GREY = '\033[37m'  # Light grey

                    lines.append(
                        f'  {CYAN}{file_val}{RESET}'
                        f'{GREY}:{RESET}'
                        f'{YELLOW}{line_num}{RESET}'
                        f'{GREY}:{RESET}'
                        f'{LIGHT_GREY}{match.offset}{RESET} '
                        f'{GREY}[{RESET}{MAGENTA}{pattern_val}{RESET}{GREY}]{RESET}'
                    )
                else:
                    lines.append(f'  {file_val}:{line_num}:{match.offset} [{pattern_val}]')

        return '\n'.join(lines)


class FileAnalysisResult(BaseModel):
    """Analysis result for a single file."""

    file: str = Field(..., description="File ID (e.g., 'f1')")
    size_bytes: int = Field(..., description='File size in bytes')
    size_human: str = Field(..., description='Human-readable file size')
    is_text: bool = Field(..., description='Whether the file is a text file')

    created_at: str | None = Field(None, description='File creation timestamp (ISO format)')
    modified_at: str | None = Field(None, description='File modification timestamp (ISO format)')
    permissions: str | None = Field(None, description='File permissions (octal)')
    owner: str | None = Field(None, description='File owner')

    line_count: int | None = Field(None, description='Total number of lines (text files only)')
    empty_line_count: int | None = Field(None, description='Number of empty lines')
    line_length_max: int | None = Field(None, description='Maximum line length')
    line_length_avg: float | None = Field(None, description='Average line length (excluding empty lines)')
    line_length_median: float | None = Field(None, description='Median line length')
    line_length_p95: float | None = Field(None, description='95th percentile of line lengths')
    line_length_p99: float | None = Field(None, description='99th percentile of line lengths')
    line_length_stddev: float | None = Field(None, description='Standard deviation of line lengths')
    line_length_max_line_number: int | None = Field(None, description='Line number of the longest line (1-indexed)')
    line_length_max_byte_offset: int | None = Field(None, description='Byte offset of the longest line')

    line_ending: str | None = Field(None, description='Line ending style: LF, CRLF, CR, or mixed')

    custom_metrics: dict = Field(default_factory=dict, description='Custom metrics from plugins')

    # Compression information
    is_compressed: bool = Field(default=False, description='Whether the file is compressed')
    compression_format: str | None = Field(None, description='Compression format (gzip, zstd, xz, bz2, etc.)')
    is_seekable_zstd: bool = Field(default=False, description='Whether file is seekable zstd format')
    compressed_size: int | None = Field(None, description='Compressed file size in bytes (if compressed)')
    decompressed_size: int | None = Field(None, description='Decompressed/original size in bytes (if compressed)')
    compression_ratio: float | None = Field(None, description='Compression ratio (if compressed)')

    # Index information
    has_index: bool = Field(default=False, description='Whether an index exists for this file')
    index_path: str | None = Field(None, description='Path to index file (if exists)')
    index_valid: bool = Field(default=False, description='Whether the index is valid (not stale)')
    index_checkpoint_count: int | None = Field(None, description='Number of checkpoints in index (if exists)')


class AnalyseResponse(BaseModel):
    """Response for file analysis endpoint."""

    path: str = Field(..., description='Analyzed path(s)')
    time: float = Field(..., description='Analysis time in seconds')
    files: dict[str, str] = Field(..., description='File ID to filepath mapping')
    results: list[FileAnalysisResult] = Field(..., description='Analysis results for each file')
    scanned_files: list[str] = Field(..., description='List of successfully scanned files')
    skipped_files: list[str] = Field(..., description='List of skipped files')

    def to_cli(self, colorize: bool = False) -> str:
        """Format analysis response for CLI output."""
        BOLD = '\033[1m'
        GREEN = '\033[32m'
        YELLOW = '\033[33m'
        CYAN = '\033[36m'
        GREY = '\033[90m'
        RESET = '\033[0m'

        lines = []

        # Header
        if colorize:
            lines.append(f'{BOLD}File Analysis{RESET}')
        else:
            lines.append('File Analysis')

        lines.append(f'Path: {self.path}')
        lines.append(f'Time: {self.time:.3f}s')
        lines.append(f'Files analyzed: {len(self.results)}')
        lines.append('')

        # Results for each file
        for result in self.results:
            filepath = self.files.get(result.file, result.file)

            if colorize:
                lines.append(f'{CYAN}{filepath}{RESET}')
            else:
                lines.append(filepath)

            lines.append(f'  Size: {result.size_human} ({result.size_bytes:,} bytes)')
            lines.append(f'  Type: {"Text" if result.is_text else "Binary"}')

            if result.modified_at:
                lines.append(f'  Modified: {result.modified_at}')
            if result.permissions:
                lines.append(f'  Permissions: {result.permissions}')
            if result.owner:
                lines.append(f'  Owner: {result.owner}')

            # Compression info
            if result.is_compressed:
                comp_info = f'  Compressed: {result.compression_format}'
                if result.is_seekable_zstd:
                    comp_info += ' (seekable)'
                if result.compression_ratio:
                    comp_info += f', ratio: {result.compression_ratio:.2f}x'
                if result.decompressed_size:
                    decompressed_human = human_readable_size(result.decompressed_size)
                    comp_info += f', decompressed: {decompressed_human}'
                lines.append(comp_info)

            # Index info
            if result.has_index:
                index_info = f'  Index: {"valid" if result.index_valid else "stale"}'
                if result.index_checkpoint_count:
                    index_info += f', {result.index_checkpoint_count} checkpoints'
                if result.index_path:
                    index_info += f' ({result.index_path})'
                lines.append(index_info)

            if result.is_text and result.line_count is not None:
                lines.append(f'  Lines: {result.line_count:,} total, {result.empty_line_count:,} empty')
                if result.line_ending:
                    lines.append(f'  Line ending: {result.line_ending}')
                if result.line_length_max:
                    line_stats = (
                        f'  Line length: max={result.line_length_max}, '
                        f'avg={result.line_length_avg:.1f}, '
                        f'median={result.line_length_median:.1f}'
                    )
                    if result.line_length_p95 is not None:
                        line_stats += f', p95={result.line_length_p95:.1f}'
                    if result.line_length_p99 is not None:
                        line_stats += f', p99={result.line_length_p99:.1f}'
                    if result.line_length_stddev is not None:
                        line_stats += f', stddev={result.line_length_stddev:.1f}'
                    lines.append(line_stats)
                    if result.line_length_max_line_number is not None:
                        lines.append(
                            f'  Longest line: line {result.line_length_max_line_number}, '
                            f'offset {result.line_length_max_byte_offset}'
                        )

            if result.custom_metrics:
                lines.append(f'  Custom metrics: {result.custom_metrics}')

            lines.append('')

        return '\n'.join(lines)


class RegexIssueDetail(BaseModel):
    """Detailed information about a single regex vulnerability issue"""

    type: str = Field(..., example='nested_quantifier', description='Issue type identifier')
    severity: str = Field(..., example='critical', description='Issue severity (critical, high, medium, low)')
    complexity_class: str = Field(
        ..., example='exponential', description='Complexity class (exponential, polynomial, linear)'
    )
    complexity_notation: str = Field(..., example='O(2^n)', description='Big-O notation')
    segment: str = Field('', example='(a+)+', description='The problematic pattern segment')
    explanation: str = Field(..., description='Human-readable explanation of the issue')
    fix_suggestions: list[str] = Field(default_factory=list, description='Specific recommendations to fix')


class PerformanceEstimate(BaseModel):
    """Estimated operations for different input sizes"""

    ops_at_100: int = Field(..., example=100, description='Estimated operations for 100-char input')
    ops_at_1000: int = Field(..., example=1000, description='Estimated operations for 1000-char input')
    ops_at_10000: int = Field(..., example=10000, description='Estimated operations for 10000-char input')
    safe_for_large_files: bool = Field(..., example=True, description='Whether pattern is safe for large files')


class ComplexityDetails(BaseModel):
    """Pattern analysis details"""

    star_height: int = Field(0, example=2, description='Maximum quantifier nesting depth')
    quantifier_count: int = Field(0, example=3, description='Total number of quantifiers')
    has_start_anchor: bool = Field(False, example=True, description='Pattern has ^ anchor')
    has_end_anchor: bool = Field(False, example=True, description='Pattern has $ anchor')
    issue_count: int = Field(0, example=1, description='Number of issues detected')


class ComplexityResponse(BaseModel):
    """
    Response from complexity analysis endpoint.

    Provides comprehensive analysis of regex pattern complexity including
    vulnerability detection, performance estimates, and actionable recommendations.

    Based on ReDoS research:
    - https://github.com/doyensec/regexploit
    - https://www.regular-expressions.info/catastrophic.html
    """

    regex: str = Field(..., example='(a+)+')
    score: float = Field(..., example=90.0, description='Complexity score (0-100)')

    # New primary fields
    risk_level: str = Field(
        ..., example='critical', description='Risk level: safe, caution, warning, dangerous, critical'
    )
    complexity_class: str = Field(
        ..., example='exponential', description='Complexity class: linear, polynomial, exponential'
    )
    complexity_notation: str = Field(..., example='O(2^n)', description='Big-O notation')

    issues: list[RegexIssueDetail] = Field(default_factory=list, description='Detected vulnerability issues')
    recommendations: list[str] = Field(default_factory=list, description='Actionable fix suggestions')
    performance: PerformanceEstimate = Field(..., description='Performance estimates')

    star_height: int = Field(0, example=2, description='Maximum quantifier nesting depth')
    pattern_length: int = Field(..., example=5)
    has_anchors: tuple[bool, bool] = Field((False, False), description='(has_start_anchor, has_end_anchor)')

    # Legacy fields for backwards compatibility
    level: str = Field(..., example='dangerous', description='Legacy level field')
    risk: str = Field(..., example='CRITICAL - ReDoS vulnerability', description='Risk description')
    warnings: list[str] = Field(default_factory=list, description='Legacy warnings list')
    details: ComplexityDetails = Field(default_factory=ComplexityDetails)

    def to_cli(self, colorize: bool = False) -> str:
        """Format complexity response for CLI output with detailed issue breakdown"""
        # ANSI color codes
        BOLD = '\033[1m'
        RED = '\033[91m'
        YELLOW = '\033[33m'
        GREEN = '\033[32m'
        CYAN = '\033[36m'
        MAGENTA = '\033[35m'
        GREY = '\033[90m'
        WHITE = '\033[97m'
        RESET = '\033[0m'

        lines = []

        # Determine colors based on risk level
        if self.risk_level in ['critical', 'dangerous']:
            level_color = RED if colorize else ''
            icon = '✗' if colorize else 'X'
        elif self.risk_level == 'warning':
            level_color = YELLOW if colorize else ''
            icon = '⚠' if colorize else '!'
        else:
            level_color = GREEN if colorize else ''
            icon = '✓' if colorize else '+'

        reset = RESET if colorize else ''
        bold = BOLD if colorize else ''
        grey = GREY if colorize else ''
        cyan = CYAN if colorize else ''

        # Header
        lines.append(f'{bold}COMPLEXITY ANALYSIS{reset}')
        lines.append('')

        # Risk level with color
        lines.append(f'{grey}Risk Level:{reset} {level_color}{self.risk_level.upper()}{reset}')

        # Complexity
        lines.append(
            f'{grey}Complexity:{reset} {level_color}{self.complexity_class.upper()} {self.complexity_notation}{reset}'
        )

        # Score
        lines.append(f'{grey}Score:{reset} {level_color}{self.score:.0f}/100{reset}')

        lines.append('')

        # Pattern
        lines.append(f'{grey}Pattern:{reset} {cyan}{self.regex}{reset}')
        lines.append('')

        # Issues
        if self.issues:
            lines.append(f'{bold}ISSUES FOUND:{reset}')
            lines.append('')

            for issue in self.issues:
                # Issue header with severity
                severity_colors = {
                    'critical': RED if colorize else '',
                    'high': YELLOW if colorize else '',
                    'medium': YELLOW if colorize else '',
                    'low': GREEN if colorize else '',
                }
                sev_color = severity_colors.get(issue.severity, '')

                issue_type_display = issue.type.replace('_', ' ').title()
                lines.append(
                    f'  {sev_color}{icon}{reset} [{sev_color}{issue.severity.upper()}{reset}] {issue_type_display}'
                )

                if issue.segment:
                    lines.append(f'    {grey}Pattern segment:{reset} {cyan}{issue.segment}{reset}')

                lines.append('')
                lines.append(f'    {bold}Explanation:{reset}')

                # Word wrap explanation
                explanation = issue.explanation
                words = explanation.split()
                current_line = '    '
                for word in words:
                    if len(current_line) + len(word) + 1 > 76:
                        lines.append(current_line)
                        current_line = '    ' + word
                    else:
                        current_line += ' ' + word if current_line.strip() else '    ' + word
                if current_line.strip():
                    lines.append(current_line)

                lines.append('')

                # Fix suggestions
                if issue.fix_suggestions:
                    lines.append(f'    {bold}Fix Suggestions:{reset}')
                    for i, suggestion in enumerate(issue.fix_suggestions, 1):
                        lines.append(f'      {i}. {suggestion}')
                    lines.append('')
        else:
            lines.append(f'{bold}PATTERN CHARACTERISTICS:{reset}')
            lines.append(f'  {GREEN if colorize else ""}{icon}{reset} No vulnerabilities detected')
            lines.append(f'  {grey}Complexity:{reset} {self.complexity_notation}')
            lines.append(f'  {grey}Star height:{reset} {self.star_height}')
            if self.has_anchors[0] and self.has_anchors[1]:
                lines.append(f'  {grey}Anchored:{reset} Yes (^ and $)')
            lines.append('')

        # Performance estimate
        lines.append(f'{bold}PERFORMANCE ESTIMATE:{reset}')

        perf = self.performance
        if self.complexity_class == 'exponential':
            lines.append(
                f'  {RED if colorize else ""}WARNING: Exponential complexity - unsafe for any non-trivial input{reset}'
            )
            lines.append('  Input of 25+ chars may cause severe slowdown or hang')
        else:
            lines.append(f'  {"Input Size":<15} {"Est. Operations":<20}')
            lines.append(f'  {"-" * 15} {"-" * 20}')
            lines.append(f'  {"100 chars":<15} {perf.ops_at_100:,}')
            lines.append(f'  {"1,000 chars":<15} {perf.ops_at_1000:,}')
            if perf.ops_at_10000 < 10**15:
                lines.append(f'  {"10,000 chars":<15} {perf.ops_at_10000:,}')
            else:
                lines.append(f'  {"10,000 chars":<15} > 10^15 (impractical)')

        lines.append('')

        # Final verdict
        if perf.safe_for_large_files:
            lines.append(f'{GREEN if colorize else ""}{icon}{reset} This pattern is safe for large files.')
        else:
            lines.append(
                f'{RED if colorize else ""}{icon}{reset} This pattern may cause performance issues on large files.'
            )

        return '\n'.join(lines)


class SamplesResponse(BaseModel):
    """Response from samples endpoint"""

    path: str = Field(..., example='/path/to/file.txt')
    offsets: dict[str, int] = Field(
        default_factory=dict, example={'123': 1, '456': 2}, description='Mapping of byte offset to line number'
    )
    lines: dict[str, int] = Field(
        default_factory=dict, example={'1': 0, '2': 123}, description='Mapping of line number to byte offset'
    )
    before_context: int = Field(..., example=3)
    after_context: int = Field(..., example=3)
    samples: dict[str, list[str]] = Field(
        ...,
        example={
            '123': ['Line before', 'Line with match', 'Line after'],
            '456': ['Another before', 'Another match', 'Another after'],
        },
    )
    is_compressed: bool = Field(default=False, description='Whether the source file is compressed')
    compression_format: str | None = Field(
        default=None, example='gzip', description='Compression format if file is compressed'
    )

    def to_cli(self, colorize: bool = False, regex: str = None) -> str:
        """Format response for CLI output

        Args:
            colorize: Whether to apply color formatting
            regex: Regex pattern to highlight in output
        """
        output_lines = []
        output_lines.append(f'File: {self.path}')
        if self.is_compressed and self.compression_format:
            output_lines.append(f'Compressed: {self.compression_format}')
        output_lines.append(f'Context: {self.before_context} before, {self.after_context} after')
        output_lines.append('')

        # ANSI color codes
        GREY = '\033[90m'
        CYAN = '\033[96m'
        YELLOW = '\033[93m'
        LIGHT_GREY = '\033[37m'
        RED = '\033[91m'
        RESET = '\033[0m'

        # Determine which mode we're in (offsets or lines)
        if self.offsets:
            for offset_str in self.offsets:
                offset = int(offset_str)
                key = str(offset)
                if key in self.samples:
                    # Get line number from offsets dict
                    line_num = self.offsets[offset_str]

                    # Format: file:line:offset
                    if colorize:
                        header = (
                            f'=== {CYAN}{self.path}{RESET}'
                            f'{GREY}:{RESET}'
                            f'{YELLOW}{line_num}{RESET}'
                            f'{GREY}:{RESET}'
                            f'{LIGHT_GREY}{offset}{RESET} ==='
                        )
                    else:
                        header = f'=== {self.path}:{line_num}:{offset} ==='
                    output_lines.append(header)

                    context_lines = self.samples[key]
                    for line in context_lines:
                        if colorize and regex:
                            try:
                                highlighted = re.sub(f'({regex})', f'{RED}\\1{RESET}', line)
                                output_lines.append(highlighted)
                            except re.error:
                                output_lines.append(line)
                        else:
                            output_lines.append(line)
                    output_lines.append('')
        elif self.lines:
            for line_num_str in self.lines:
                line_num = int(line_num_str)
                key = str(line_num)
                if key in self.samples:
                    # Get byte offset from lines dict
                    byte_offset = self.lines[line_num_str]

                    # Format: file:line:offset
                    if colorize:
                        header = (
                            f'=== {CYAN}{self.path}{RESET}'
                            f'{GREY}:{RESET}'
                            f'{YELLOW}{line_num}{RESET}'
                            f'{GREY}:{RESET}'
                            f'{LIGHT_GREY}{byte_offset}{RESET} ==='
                        )
                    else:
                        header = f'=== {self.path}:{line_num}:{byte_offset} ==='
                    output_lines.append(header)

                    context_lines = self.samples[key]
                    for line in context_lines:
                        if colorize and regex:
                            try:
                                highlighted = re.sub(f'({regex})', f'{RED}\\1{RESET}', line)
                                output_lines.append(highlighted)
                            except re.error:
                                output_lines.append(line)
                        else:
                            output_lines.append(line)
                    output_lines.append('')

        return '\n'.join(output_lines)


# Hook Payload Models


class FileScannedPayload(BaseModel):
    """Payload for on_file_scanned hook event.

    Sent when a file scan is completed during trace operation.
    """

    event: str = Field(default='file_scanned', description='Event type identifier')
    request_id: str = Field(..., description='Unique identifier for the trace request')
    file_path: str = Field(..., description='Path to the scanned file')
    file_size_bytes: int = Field(..., description='Size of the file in bytes')
    scan_time_ms: int = Field(..., description='Time taken to scan the file in milliseconds')
    matches_count: int = Field(..., description='Number of matches found in this file')


class MatchFoundPayload(BaseModel):
    """Payload for on_match_found hook event.

    Sent for each individual match found during trace operation.
    """

    event: str = Field(default='match_found', description='Event type identifier')
    request_id: str = Field(..., description='Unique identifier for the trace request')
    file_path: str = Field(..., description='Path to the file containing the match')
    pattern: str = Field(..., description='Regex pattern that was matched')
    offset: int = Field(..., description='Byte offset of the match in the file')
    line_number: int | None = Field(default=None, description='Line number of the match (1-based, optional)')


class TraceCompletePayload(BaseModel):
    """Payload for on_trace_complete hook event.

    Sent when the entire trace request completes.
    """

    event: str = Field(default='trace_complete', description='Event type identifier')
    request_id: str = Field(..., description='Unique identifier for the trace request')
    paths: str = Field(..., description='Comma-separated list of paths that were traced')
    patterns: str = Field(..., description='Comma-separated list of regex patterns used')
    total_files_scanned: int = Field(..., description='Total number of files scanned')
    total_files_skipped: int = Field(..., description='Total number of files skipped')
    total_matches: int = Field(..., description='Total number of matches found')
    total_time_ms: int = Field(..., description='Total time taken for the trace in milliseconds')


# Seekable Zstd / Compression Models


class SeekableFrameInfo(BaseModel):
    """Information about a single frame in a seekable zstd file."""

    index: int = Field(..., description='Frame index (0-based)')
    compressed_offset: int = Field(..., description='Byte offset of frame in compressed file')
    compressed_size: int = Field(..., description='Size of compressed frame in bytes')
    decompressed_offset: int = Field(..., description='Byte offset of frame in decompressed stream')
    decompressed_size: int = Field(..., description='Size of decompressed frame in bytes')
    first_line: int | None = Field(None, description='First line number in this frame (1-based)')
    last_line: int | None = Field(None, description='Last line number in this frame (1-based)')


class SeekableZstdInfoResponse(BaseModel):
    """Response with information about a seekable zstd file."""

    path: str = Field(..., description='Path to the seekable zstd file')
    compressed_size: int = Field(..., description='Compressed file size in bytes')
    compressed_size_human: str = Field(..., description='Human-readable compressed size')
    decompressed_size: int = Field(..., description='Decompressed content size in bytes')
    decompressed_size_human: str = Field(..., description='Human-readable decompressed size')
    compression_ratio: float = Field(..., description='Compression ratio (decompressed/compressed)')
    frame_count: int = Field(..., description='Number of frames in the file')
    frame_size_target: int = Field(..., description='Target frame size in bytes')
    total_lines: int | None = Field(None, description='Total line count (if index exists)')
    index_available: bool = Field(..., description='Whether line index is available')
    frames: list[SeekableFrameInfo] | None = Field(None, description='Frame details (if requested)')


class CompressRequest(BaseModel):
    """Request to compress a file to seekable zstd format."""

    input_path: str = Field(..., description='Path to input file')
    output_path: str | None = Field(None, description='Path for output .zst file (default: input_path.zst)')
    frame_size: str = Field(default='4M', description="Target frame size (e.g., '4M', '10MB')")
    compression_level: int = Field(default=3, ge=1, le=22, description='Zstd compression level (1-22)')
    build_index: bool = Field(default=True, description='Build line index after compression')
    force: bool = Field(default=False, description='Overwrite existing output file')


class CompressResponse(BaseModel):
    """Response from compress operation."""

    success: bool = Field(..., description='Whether compression succeeded')
    input_path: str = Field(..., description='Path to input file')
    output_path: str | None = Field(None, description='Path to output .zst file')
    compressed_size: int | None = Field(None, description='Compressed file size in bytes')
    decompressed_size: int | None = Field(None, description='Original/decompressed size in bytes')
    compression_ratio: float | None = Field(None, description='Compression ratio')
    frame_count: int | None = Field(None, description='Number of frames created')
    total_lines: int | None = Field(None, description='Total line count (if index built)')
    index_built: bool = Field(default=False, description='Whether line index was built')
    time_seconds: float | None = Field(None, description='Compression time in seconds')
    error: str | None = Field(None, description='Error message if failed')


# Index Request/Response Models


class IndexRequest(BaseModel):
    """Request to build a line index for a file."""

    path: str = Field(..., description='Path to file to index')
    force: bool = Field(default=False, description='Force rebuild even if valid index exists')
    threshold: int | None = Field(None, description='Minimum file size in MB to index (default: from env)')


class IndexResponse(BaseModel):
    """Response from index operation."""

    success: bool = Field(..., description='Whether indexing succeeded')
    path: str = Field(..., description='Path to indexed file')
    index_path: str | None = Field(None, description='Path to index file')
    line_count: int | None = Field(None, description='Total line count')
    file_size: int | None = Field(None, description='File size in bytes')
    checkpoint_count: int | None = Field(None, description='Number of index checkpoints')
    time_seconds: float | None = Field(None, description='Indexing time in seconds')
    error: str | None = Field(None, description='Error message if failed')


# Background Task Models


class TaskResponse(BaseModel):
    """Response when starting a background task."""

    task_id: str = Field(..., description='Unique task identifier')
    status: str = Field(..., description='Task status (queued, running, completed, failed)')
    message: str = Field(..., description='Human-readable message')
    path: str = Field(..., description='File path being processed')
    started_at: str | None = Field(None, description='ISO timestamp when task started')


class TaskStatusResponse(BaseModel):
    """Response when querying task status."""

    task_id: str = Field(..., description='Unique task identifier')
    status: str = Field(..., description='Task status (queued, running, completed, failed)')
    path: str = Field(..., description='File path being processed')
    operation: str = Field(..., description='Operation type (compress, index)')
    started_at: str | None = Field(None, description='ISO timestamp when task started')
    completed_at: str | None = Field(None, description='ISO timestamp when task completed')
    error: str | None = Field(None, description='Error message if failed')
    result: dict | None = Field(None, description='Task result if completed')

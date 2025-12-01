import logging
import os
import platform
from contextlib import asynccontextmanager
from time import time

import anyio
import psutil
import sh
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from rx import parse as parse_module

# Import real prometheus for server mode and swap it into parse module
from rx import prometheus as prom
from rx.__version__ import __version__
from rx.analyse import analyse_path, calculate_regex_complexity
from rx.models import (
    AnalyseResponse,
    ComplexityResponse,
    HealthResponse,
    Match,
    SamplesResponse,
    TraceResponse,
)
from rx.parse import get_context, get_context_by_lines, validate_file
from rx.parse_json import parse_paths_json

# Replace the noop prometheus in parse module with real one
parse_module.prom = prom

log_level_name = os.getenv('RX_LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: check for ripgrep
    try:
        rg = sh.Command('rg')
        rg_path = rg._path
        logger.info(f"ripgrep found at: {rg_path}")
        app.state.rg = rg
        app.state.rg_path = rg_path
    except sh.CommandNotFound:
        logger.warning(
            "ripgrep not found. Install it:\n"
            "  macOS: brew install ripgrep\n"
            "  Ubuntu/Debian: apt install ripgrep\n"
            "  Fedora: dnf install ripgrep\n"
            "  Or use Docker image with ripgrep pre-installed"
        )
        app.state.rg = None
        app.state.rg_path = None

    # Run app
    yield

    # Shutdown: cleanup if needed
    logger.info("Shutting down rx")


app = FastAPI(
    title='RX (Regex Tracer)',
    version=__version__,
    description="""
    A high-performance file search and analysis tool powered by ripgrep.

    ## Features

    * **Fast Pattern Matching**: Search files using powerful regex patterns with parallel processing
    * **Context Extraction**: Get surrounding lines for matched patterns
    * **Complexity Analysis**: Analyze regex patterns for performance characteristics and ReDoS risks
    * **Byte Offset Results**: Get precise byte offsets for all matches

    ## Endpoints

    * `/v1/trace` - Search for regex patterns in files with optional result limits
    * `/v1/samples` - Extract context lines around specific byte offsets
    * `/v1/complexity` - Analyze regex complexity and detect ReDoS vulnerabilities
    * `/health` - Check service health and ripgrep availability

    ## Performance

    - Handles files of any size using parallel processing
    - Splits large files (>25MB) into chunks (max 20 parallel processes)
    - Early termination support with `max_results` parameter
    - Line-aligned chunk boundaries to prevent pattern splitting

    ## Use Cases

    1. **Log Analysis**: Search multi-GB log files for error patterns
    2. **Code Search**: Find patterns across large codebases
    3. **Security Auditing**: Detect sensitive data patterns in files
    4. **Regex Testing**: Analyze regex complexity before production use
    """,
    contact={
        "name": "RxTracer API Support",
        "url": "https://github.com/wlame/rx",
    },
    license_info={"name": "MIT"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    # Serve the favicon (SVG format)
    import pathlib

    favicon_path = pathlib.Path(__file__).parent / 'favicon.svg'
    if favicon_path.exists():
        return Response(content=favicon_path.read_bytes(), media_type='image/svg+xml')
    return Response(status_code=204)


def get_os_info() -> dict:
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
    }


def get_system_resources() -> dict:
    mem = psutil.virtual_memory()
    return {
        'cpu_cores': psutil.cpu_count(logical=True),
        'cpu_cores_physical': psutil.cpu_count(logical=False),
        'ram_total_gb': round(mem.total / (1024**3), 2),
        'ram_available_gb': round(mem.available / (1024**3), 2),
        'ram_percent_used': mem.percent,
    }


def get_python_packages() -> dict:
    import importlib.metadata

    python_packages = {}
    key_packages = ['fastapi', 'pydantic', 'uvicorn', 'sh', 'psutil', 'prometheus-client']
    for package in key_packages:
        try:
            version = importlib.metadata.version(package)
            python_packages[package] = version
        except importlib.metadata.PackageNotFoundError:
            pass

    return python_packages


def get_constants() -> dict:
    # Collect application constants
    from rx import parse as parse_module
    from rx import parse_json
    from rx.utils import NEWLINE_SYMBOL

    return {
        'LOG_LEVEL': log_level_name,
        'DEBUG_MODE': parse_json.DEBUG_MODE,
        'LINE_SIZE_ASSUMPTION_KB': parse_module.LINE_SIZE_ASSUMPTION_KB,
        'MAX_SUBPROCESSES': parse_module.MAX_SUBPROCESSES,
        'MIN_CHUNK_SIZE_MB': parse_module.MIN_CHUNK_SIZE // (1024 * 1024),
        'MAX_FILES': parse_module.MAX_FILES,
        'NEWLINE_SYMBOL': repr(NEWLINE_SYMBOL),  # Show as repr to see escape sequences
    }


def get_app_env_variables() -> dict:
    env_vars = {}
    app_env_prefixes = ['RX_', 'UVICORN_', 'PROMETHEUS_', 'NEWLINE_SYMBOL']
    for key, value in os.environ.items():
        if key == 'NEWLINE_SYMBOL' or any(key.startswith(prefix) for prefix in app_env_prefixes):
            env_vars[key] = value

    return env_vars


@app.get('/', tags=['General'], response_model=HealthResponse)
async def health():
    """
    Health check and system introspection endpoint.

    Returns:
    - Service status
    - ripgrep availability
    - Application version
    - Operating system information
    - Application-related environment variables
    - Documentation URL
    """
    os_info = get_os_info()
    system_resources = get_system_resources()
    python_packages = get_python_packages()
    constants = get_constants()
    env_vars = get_app_env_variables()

    return {
        'status': 'ok',
        'ripgrep_available': app.state.rg_path is not None,
        'app_version': __version__,
        'python_version': platform.python_version(),
        'os_info': os_info,
        'system_resources': system_resources,
        'python_packages': python_packages,
        'constants': constants,
        'environment': env_vars,
        'docs_url': 'https://github.com/wlame/rx',
    }


@app.get('/metrics', tags=['Monitoring'], include_in_schema=True)
async def metrics():
    """
    Prometheus metrics endpoint.

    Exposes performance metrics, request counts, file processing statistics,
    and resource utilization for monitoring and observability.

    **Metrics Categories:**
    - Request metrics (counts, durations by endpoint)
    - File processing (sizes, counts, bytes processed)
    - Pattern matching (matches found, patterns per request)
    - Regex complexity (scores, levels)
    - Parallel processing (tasks, workers)
    - Errors (by type)

    Use with Prometheus to scrape and visualize RX performance.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get(
    '/v1/trace',
    tags=['Search'],
    summary="Search file for regex patterns (supports multiple patterns)",
    response_model=TraceResponse,
    responses={
        200: {"description": "Successfully found matches"},
        400: {"description": "Invalid regex pattern or binary file"},
        404: {"description": "File not found"},
        503: {"description": "ripgrep not available"},
    },
)
async def trace(
    path: list[str] = Query(
        ...,
        description="File or directory paths to search (can specify multiple)",
        examples=["/var/log/app.log", "/var/log/nginx"],
    ),
    regexp: list[str] = Query(
        ..., description="Regular expression patterns to search for (can specify multiple)", examples=["error.*failed"]
    ),
    max_results: int = Query(None, description="Maximum number of results to return (optional)", ge=1, examples=[100]),
) -> TraceResponse:
    """
    Search files or directories for regex pattern matches and return byte offsets with ID-based structure.

    This endpoint uses parallel processing with ripgrep to efficiently search large files or multiple files in a directory.
    Supports multiple paths and multiple patterns in a single request.

    - **path**: Absolute or relative path(s) to file or directory to search - can specify multiple
    - **regexp**: Regular expression pattern(s) (ripgrep syntax) - can specify multiple
    - **max_results**: Optional limit on number of results (stops early for performance)

    Returns ID-based structure:
    - **patterns**: Dict mapping pattern IDs (p1, p2, ...) to pattern strings
    - **files**: Dict mapping file IDs (f1, f2, ...) to file paths
    - **matches**: Array of {pattern, file, offset} objects using IDs
    - **scanned_files**: List of files that were scanned (for directories)
    - **skipped_files**: List of files that were skipped (binary files)

    Examples:
    ```
    GET /v1/trace?path=/var/log/app.log&regexp=error&regexp=warning
    GET /v1/trace?path=/var/log/app.log&path=/var/log/error.log&regexp=error
    ```

    Returns matches where any of the patterns were found across all paths.
    """
    if not app.state.rg:
        prom.record_error('service_unavailable')
        prom.record_http_response('GET', '/v1/trace', 503)
        raise HTTPException(status_code=503, detail="ripgrep is not available on this system")

    # Check if all paths exist
    for p in path:
        if not os.path.exists(p):
            prom.record_error('file_not_found')
            prom.record_http_response('GET', '/v1/trace', 404)
            raise HTTPException(status_code=404, detail=f"Path not found: {p}")

    # Parse files or directories with multiple patterns using JSON mode
    try:
        time_before = time()
        # Offload blocking I/O to thread pool to keep event loop responsive
        result = await anyio.to_thread.run_sync(
            parse_paths_json,
            path,
            regexp,
            max_results,
            None,  # rg_extra_args
            0,  # context_before - No context in API by default (use /v1/samples for that)
            0,  # context_after
        )
        parsing_time = time() - time_before

        # Calculate metrics
        num_files = len(result['files'])
        num_skipped = len(result['skipped_files'])
        num_patterns = len(result['patterns'])
        num_matches = len(result['matches'])

        # Calculate total bytes processed
        total_bytes = 0
        for filepath in result['files'].values():
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                total_bytes += file_size
                prom.record_file_size(file_size)

        # Record metrics
        hit_max_results = max_results is not None and num_matches >= max_results
        prom.record_trace_request(
            status='success',
            duration=parsing_time,
            num_files=num_files,
            num_skipped=num_skipped,
            num_patterns=num_patterns,
            num_matches=num_matches,
            total_bytes=total_bytes,
            hit_max_results=hit_max_results,
        )
        prom.record_http_response('GET', '/v1/trace', 200)

    except RuntimeError as e:
        prom.record_error('invalid_regex')
        prom.record_trace_request('error', 0, 0, 0, len(regexp), 0, 0)
        prom.record_http_response('GET', '/v1/trace', 400)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        prom.record_error('internal_error')
        prom.record_trace_request('error', 0, 0, 0, len(regexp), 0, 0)
        prom.record_http_response('GET', '/v1/trace', 500)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    # Build response using ID-based structure
    response = TraceResponse(
        path=path,  # Pass as list directly
        time=parsing_time,
        patterns=result['patterns'],
        files=result['files'],
        matches=[Match(**m) for m in result['matches']],
        scanned_files=result['scanned_files'],
        skipped_files=result['skipped_files'],
        file_chunks=result.get('file_chunks'),
        max_results=max_results,  # Include max_results in response
    )

    return response


@app.get(
    '/v1/complexity',
    tags=['Analysis'],
    summary="EXPERIMENTAL! Analyze regex complexity",
    response_model=ComplexityResponse,
    responses={
        200: {"description": "Complexity analysis completed"},
        500: {"description": "Internal error during analysis"},
    },
)
async def complexity(
    regex: str = Query(
        ..., description="Regular expression pattern to analyze", examples=["(a+)+", ".*.*", "^[a-z]+$"]
    ),
) -> dict:
    """
    Analyze regex pattern complexity and predict performance characteristics.

    This endpoint calculates a complexity score based on various regex features that
    can impact performance, particularly patterns that may cause catastrophic backtracking
    (ReDoS vulnerabilities).

    **Use this before running expensive searches to assess potential performance issues.**

    Returns:
    - **score**: Numeric complexity score
    - **level**: Complexity level (very_simple, simple, moderate, complex, very_complex, dangerous)
    - **risk**: Risk description
    - **warnings**: List of potential performance issues
    - **details**: Breakdown of scoring components
    - **regex**: The analyzed pattern

    Score ranges:
    - 0-10: Very Simple (substring search)
    - 11-30: Simple (basic patterns)
    - 31-60: Moderate (reasonable performance)
    - 61-100: Complex (monitor performance)
    - 101-200: Very Complex (significant impact)
    - 201+: Dangerous (ReDoS risk!)
    """
    try:
        time_before = time()
        # Offload CPU-bound regex analysis to thread pool
        result = await anyio.to_thread.run_sync(calculate_regex_complexity, regex)
        duration = time() - time_before

        result['regex'] = regex  # Include the regex pattern in response

        # Record metrics
        prom.record_complexity_request(duration=duration, score=result['score'], level=result['level'])
        prom.record_http_response('GET', '/v1/complexity', 200)

        return result
    except Exception as e:
        logger.error(f"Error analyzing regex complexity: {str(e)}")
        prom.record_error('internal_error')
        prom.record_http_response('GET', '/v1/complexity', 500)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get(
    '/v1/analyse',
    tags=['Analysis'],
    summary="Analyze files",
    response_model=AnalyseResponse,
    responses={
        200: {"description": "File analysis completed"},
        404: {"description": "Path not found"},
        500: {"description": "Internal error during analysis"},
    },
)
async def analyse(
    path: str | list[str] = Query(
        ..., description="File or directory path(s) to analyze", examples=["/var/log/app.log"]
    ),
    max_workers: int = Query(10, description="Maximum number of parallel workers", ge=1, le=50, examples=[10]),
) -> dict:
    """
    Analyze files to extract metadata and statistics.

    This endpoint analyzes text and binary files, providing:
    - File size (bytes and human-readable)
    - File metadata (creation time, modification time, permissions, owner)
    - For text files: line count, empty lines, line length statistics

    **Analysis is performed in parallel using multiple threads.**

    Returns:
    - **path**: Analyzed path(s)
    - **time**: Analysis time in seconds
    - **files**: File ID to filepath mapping (e.g., {"f1": "/path/to/file"})
    - **results**: List of analysis results for each file
    - **scanned_files**: List of successfully scanned files
    - **skipped_files**: List of files that failed analysis

    The plugin architecture allows easy extension with custom metrics.
    """
    try:
        # Convert single path to list
        paths = path if isinstance(path, list) else [path]

        # Check paths exist
        for p in paths:
            if not os.path.exists(p):
                prom.record_error('not_found')
                prom.record_http_response('GET', '/v1/analyse', 404)
                raise HTTPException(status_code=404, detail=f"Path not found: {p}")

        time_before = time()
        # Offload blocking file analysis to thread pool
        result = await anyio.to_thread.run_sync(analyse_path, paths, max_workers)
        duration = time() - time_before

        # Record metrics
        prom.record_http_response('GET', '/v1/analyse', 200)

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing files: {str(e)}")
        prom.record_error('internal_error')
        prom.record_http_response('GET', '/v1/analyse', 500)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get(
    '/v1/samples',
    tags=['Context'],
    summary="Get context lines around byte offsets or line numbers",
    response_model=SamplesResponse,
    responses={
        200: {"description": "Successfully retrieved context lines"},
        400: {"description": "Invalid offsets/lines format or negative context values"},
        404: {"description": "File not found"},
        503: {"description": "ripgrep not available"},
    },
)
async def samples(
    path: str = Query(..., description="File path to read from", examples=["/var/log/app.log"]),
    offsets: str = Query(None, description="Comma-separated list of byte offsets", examples=["123,456,789"]),
    lines: str = Query(None, description="Comma-separated list of line numbers (1-based)", examples=["100,200,300"]),
    context: int = Query(None, description="Number of context lines before and after (sets both)", ge=0, examples=[3]),
    before_context: int = Query(None, description="Number of lines before each offset", ge=0, examples=[2]),
    after_context: int = Query(None, description="Number of lines after each offset", ge=0, examples=[5]),
) -> dict:
    """
    Extract context lines around specific byte offsets or line numbers in a file.

    Use this endpoint to view the actual content around matches found by `/trace`.

    - **path**: Path to the file
    - **offsets**: Comma-separated byte offsets (e.g., "123,456,789") - mutually exclusive with lines
    - **lines**: Comma-separated line numbers (e.g., "100,200,300") - mutually exclusive with offsets
    - **context**: Set both before and after context (default: 3)
    - **before_context**: Override lines before (default: 3)
    - **after_context**: Override lines after (default: 3)

    Returns a dictionary mapping each offset/line to its context lines.

    Examples:
    ```
    GET /samples?path=data.txt&offsets=100,200&context=2
    GET /samples?path=data.txt&lines=50,100,150&context=3
    ```
    """
    if not app.state.rg:
        prom.record_error('service_unavailable')
        prom.record_http_response('GET', '/v1/samples', 503)
        raise HTTPException(status_code=503, detail="ripgrep is not available on this system")

    # Validate mutual exclusivity of offsets and lines
    if offsets and lines:
        prom.record_error('invalid_params')
        prom.record_http_response('GET', '/v1/samples', 400)
        raise HTTPException(status_code=400, detail="Cannot use both 'offsets' and 'lines'. Provide only one.")

    if not offsets and not lines:
        prom.record_error('invalid_params')
        prom.record_http_response('GET', '/v1/samples', 400)
        raise HTTPException(status_code=400, detail="Must provide either 'offsets' or 'lines' parameter.")

    # Parse offsets or lines
    offset_list: list[int] = []
    line_list: list[int] = []
    use_lines = False

    if offsets:
        try:
            offset_list = [int(o.strip()) for o in offsets.split(',')]
        except ValueError:
            prom.record_error('invalid_offsets')
            prom.record_http_response('GET', '/v1/samples', 400)
            raise HTTPException(status_code=400, detail="Invalid offsets format. Expected comma-separated integers.")
    else:
        use_lines = True
        try:
            line_list = [int(ln.strip()) for ln in lines.split(',')]
        except ValueError:
            prom.record_error('invalid_lines')
            prom.record_http_response('GET', '/v1/samples', 400)
            raise HTTPException(status_code=400, detail="Invalid lines format. Expected comma-separated integers.")

    before = before_context if before_context is not None else context if context is not None else 3
    after = after_context if after_context is not None else context if context is not None else 3

    # Validate file (offload blocking disk I/O)
    try:
        await anyio.to_thread.run_sync(validate_file, path)
    except FileNotFoundError as e:
        prom.record_error('file_not_found')
        prom.record_http_response('GET', '/v1/samples', 404)
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        prom.record_error('binary_file')
        prom.record_http_response('GET', '/v1/samples', 400)
        raise HTTPException(status_code=400, detail=str(e))

    # Get context (offload blocking file I/O)
    try:
        time_before = time()

        if use_lines:
            context_data: dict[int, list[str]] = await anyio.to_thread.run_sync(
                get_context_by_lines, path, line_list, before, after
            )
            num_items = len(line_list)
        else:
            context_data = await anyio.to_thread.run_sync(get_context, path, offset_list, before, after)
            num_items = len(offset_list)

        duration = time() - time_before

        # Record metrics
        prom.record_samples_request(
            status='success', duration=duration, num_offsets=num_items, before_ctx=before, after_ctx=after
        )
        prom.record_http_response('GET', '/v1/samples', 200)

        return {
            'path': path,
            'offsets': offset_list,
            'lines': line_list,
            'before_context': before,
            'after_context': after,
            'samples': {str(k): v for k, v in context_data.items()},
        }
    except ValueError as e:
        prom.record_error('invalid_context')
        num_items = len(line_list) if use_lines else len(offset_list)
        prom.record_samples_request('error', 0, num_items, before, after)
        prom.record_http_response('GET', '/v1/samples', 400)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        prom.record_error('internal_error')
        num_items = len(line_list) if use_lines else len(offset_list)
        prom.record_samples_request('error', 0, num_items, before, after)
        prom.record_http_response('GET', '/v1/samples', 500)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

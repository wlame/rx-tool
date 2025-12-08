"""CLI command for creating seekable zstd compressed files."""

import json
import os
import sys
from pathlib import Path

import click

from rx.compression import CompressionFormat, detect_compression, is_compound_archive
from rx.seekable_index import build_index, get_index_info
from rx.seekable_zstd import (
    DEFAULT_COMPRESSION_LEVEL,
    DEFAULT_FRAME_SIZE_BYTES,
    check_t2sz_available,
    create_seekable_zstd,
    get_seekable_zstd_info,
    is_seekable_zstd,
)


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f'{size_bytes:.2f} {unit}'
        size_bytes /= 1024
    return f'{size_bytes:.2f} PB'


def parse_size(size_str: str) -> int:
    """Parse human-readable size string to bytes.

    Supports: 1M, 1MB, 10m, 10mb, 1G, 1GB, etc.
    """
    size_str = size_str.strip().upper()

    multipliers = {
        'B': 1,
        'K': 1024,
        'KB': 1024,
        'M': 1024 * 1024,
        'MB': 1024 * 1024,
        'G': 1024 * 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
    }

    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if size_str.endswith(suffix):
            num_str = size_str[: -len(suffix)].strip()
            return int(float(num_str) * mult)

    # No suffix, assume bytes
    return int(size_str)


@click.command('compress')
@click.argument('paths', nargs=-1, required=False, type=click.Path(exists=True))
@click.option(
    '-o',
    '--output',
    type=click.Path(),
    help='Output file path (for single file input)',
)
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False),
    help='Output directory (uses source filename with .zst extension)',
)
@click.option(
    '-s',
    '--frame-size',
    default='4M',
    help='Target frame size (e.g., 4M, 10MB). Default: 4M',
)
@click.option(
    '-l',
    '--level',
    type=click.IntRange(1, 22),
    default=DEFAULT_COMPRESSION_LEVEL,
    help=f'Compression level 1-22. Default: {DEFAULT_COMPRESSION_LEVEL}',
)
@click.option(
    '-T',
    '--threads',
    type=int,
    default=None,
    help='Threads for compression (default: all CPUs, t2sz only)',
)
@click.option(
    '-f',
    '--force',
    is_flag=True,
    help='Overwrite existing output file',
)
@click.option(
    '--delete-source',
    is_flag=True,
    help='Delete source file after successful compression',
)
@click.option(
    '--no-index',
    is_flag=True,
    help='Skip building line index after compression',
)
@click.option(
    '--info',
    '-i',
    is_flag=True,
    help='Show info about existing seekable zstd file(s)',
)
@click.option(
    '--json',
    'json_output',
    is_flag=True,
    help='Output in JSON format',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Show detailed progress',
)
def compress_command(
    paths: tuple[str, ...],
    output: str | None,
    output_dir: str | None,
    frame_size: str,
    level: int,
    threads: int | None,
    force: bool,
    delete_source: bool,
    no_index: bool,
    info: bool,
    json_output: bool,
    verbose: bool,
):
    """Create seekable zstd compressed files for optimized rx-tool access.

    Seekable zstd files enable:

    \b
    - Parallel decompression for faster search
    - Random access without full decompression
    - Fast samples extraction

    \b
    Examples:
        rx compress input.log -o output.zst       # Compress to specific output
        rx compress input.log                     # Compress to input.log.zst
        rx compress input.gz -o output.zst        # Recompress from gzip
        rx compress /logs/*.log --output-dir /data/compressed/
        rx compress file.zst --info               # Show info about seekable zst
        rx compress --info                        # Check t2sz availability

    \b
    Frame Size:
        Smaller frames = more parallelism, slightly larger file
        Larger frames = less parallelism, slightly smaller file
        Default 4M gives ~20-60MB decompressed text per frame

    \b
    External Tools:
        For best performance, install t2sz: https://github.com/martinellimarco/t2sz
        Falls back to pyzstd library if t2sz is not available.
    """
    # Handle --info without paths (show tool availability)
    if not paths:
        if info:
            _show_tool_info(json_output)
            return
        else:
            click.echo("Error: Missing argument 'PATHS...'.", err=True)
            sys.exit(1)

    # Parse frame size
    try:
        frame_size_bytes = parse_size(frame_size)
    except ValueError:
        click.echo(f'Error: Invalid frame size: {frame_size}', err=True)
        sys.exit(1)

    # Validate output options
    if output and output_dir:
        click.echo('Error: Cannot specify both --output and --output-dir', err=True)
        sys.exit(1)

    if output and len(paths) > 1:
        click.echo('Error: --output can only be used with single input file', err=True)
        sys.exit(1)

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = []

    for input_path in paths:
        result = {'input': input_path}

        if info:
            # Info mode
            result['action'] = 'info'
            _show_file_info(input_path, result, json_output, verbose)
        else:
            # Compress mode
            result['action'] = 'compress'
            _compress_file(
                input_path=input_path,
                output=output,
                output_dir=output_dir,
                frame_size_bytes=frame_size_bytes,
                level=level,
                threads=threads,
                force=force,
                delete_source=delete_source,
                no_index=no_index,
                result=result,
                json_output=json_output,
                verbose=verbose,
            )

        results.append(result)

    if json_output:
        click.echo(json.dumps({'files': results}, indent=2))

    # Exit with error if any failures
    failures = [r for r in results if r.get('success') is False]
    if failures:
        sys.exit(1)


def _show_tool_info(json_output: bool) -> None:
    """Show information about available compression tools."""
    t2sz_available = check_t2sz_available()

    info = {
        't2sz_available': t2sz_available,
        'fallback': 'pyzstd' if not t2sz_available else None,
        'default_frame_size': DEFAULT_FRAME_SIZE_BYTES,
        'default_compression_level': DEFAULT_COMPRESSION_LEVEL,
    }

    if json_output:
        click.echo(json.dumps(info, indent=2))
    else:
        click.echo('Compression Tools:')
        if t2sz_available:
            click.echo('  t2sz: available (recommended)')
        else:
            click.echo('  t2sz: not found')
            click.echo('  Using fallback: pyzstd library')
            click.echo('')
            click.echo('  For better performance, install t2sz:')
            click.echo('    https://github.com/martinellimarco/t2sz')

        click.echo('')
        click.echo(f'Default frame size: {human_readable_size(DEFAULT_FRAME_SIZE_BYTES)}')
        click.echo(f'Default compression level: {DEFAULT_COMPRESSION_LEVEL}')


def _show_file_info(
    input_path: str,
    result: dict,
    json_output: bool,
    verbose: bool,
) -> None:
    """Show information about a seekable zstd file."""
    if not is_seekable_zstd(input_path):
        result['success'] = False
        result['error'] = 'Not a seekable zstd file'
        if not json_output:
            click.echo(f'{input_path}: not a seekable zstd file', err=True)
        return

    try:
        zst_info = get_seekable_zstd_info(input_path)
        result['success'] = True
        result['seekable_zstd'] = {
            'compressed_size': zst_info.compressed_size,
            'decompressed_size': zst_info.decompressed_size,
            'frame_count': zst_info.frame_count,
            'frame_size_target': zst_info.frame_size_target,
            'compression_ratio': round(zst_info.decompressed_size / zst_info.compressed_size, 2)
            if zst_info.compressed_size > 0
            else 0,
        }

        # Check for index
        idx_info = get_index_info(input_path)
        if idx_info:
            result['index'] = idx_info

        if not json_output:
            click.echo(f'\n{input_path}:')
            click.echo(f'  Compressed size: {human_readable_size(zst_info.compressed_size)}')
            click.echo(f'  Decompressed size: {human_readable_size(zst_info.decompressed_size)}')
            click.echo(f'  Compression ratio: {result["seekable_zstd"]["compression_ratio"]}:1')
            click.echo(f'  Frame count: {zst_info.frame_count}')
            click.echo(f'  Frame size target: {human_readable_size(zst_info.frame_size_target)}')

            if idx_info:
                click.echo(f'  Index: valid={idx_info["is_valid"]}, lines={idx_info["total_lines"]:,}')
            else:
                click.echo('  Index: not built (run without --info to build)')

            if verbose and zst_info.frames:
                click.echo('\n  Frames:')
                for i, frame in enumerate(zst_info.frames[:10]):
                    click.echo(
                        f'    [{i}] compressed={human_readable_size(frame.compressed_size)}, '
                        f'decompressed={human_readable_size(frame.decompressed_size)}'
                    )
                if len(zst_info.frames) > 10:
                    click.echo(f'    ... and {len(zst_info.frames) - 10} more frames')

    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        if not json_output:
            click.echo(f'{input_path}: error reading file: {e}', err=True)


def _compress_file(
    input_path: str,
    output: str | None,
    output_dir: str | None,
    frame_size_bytes: int,
    level: int,
    threads: int | None,
    force: bool,
    delete_source: bool,
    no_index: bool,
    result: dict,
    json_output: bool,
    verbose: bool,
) -> None:
    """Compress a single file to seekable zstd format."""
    input_path_obj = Path(input_path)

    # Skip compound archives
    if is_compound_archive(input_path):
        result['success'] = False
        result['error'] = 'Compound archives (tar.gz, etc.) are not supported'
        if not json_output:
            click.echo(f'{input_path}: skipped (compound archive not supported)', err=True)
        return

    # Skip if already seekable zstd (unless recompressing with different settings)
    if is_seekable_zstd(input_path) and not force:
        result['success'] = False
        result['error'] = 'Already a seekable zstd file (use --force to recompress)'
        if not json_output:
            click.echo(f'{input_path}: already seekable zstd (use --force to recompress)')
        return

    # Determine output path
    if output:
        output_path = Path(output)
    elif output_dir:
        # Use source filename with .zst extension in output_dir
        base_name = input_path_obj.name
        # Remove existing compression extension
        compression = detect_compression(input_path)
        if compression != CompressionFormat.NONE:
            # Remove compression suffix
            base_name = input_path_obj.stem
        output_path = Path(output_dir) / f'{base_name}.zst'
    else:
        # Default: same directory, add .zst extension
        base_name = input_path_obj.name
        compression = detect_compression(input_path)
        if compression != CompressionFormat.NONE:
            base_name = input_path_obj.stem
        output_path = input_path_obj.parent / f'{base_name}.zst'

    # Check if output exists
    if output_path.exists() and not force:
        result['success'] = False
        result['error'] = f'Output file exists: {output_path} (use --force to overwrite)'
        if not json_output:
            click.echo(f'{input_path}: output exists (use --force to overwrite)', err=True)
        return

    # Compress
    if not json_output:
        input_size = input_path_obj.stat().st_size
        click.echo(f'Compressing {input_path} ({human_readable_size(input_size)})...')
        if verbose:
            click.echo(f'  Output: {output_path}')
            click.echo(f'  Frame size: {human_readable_size(frame_size_bytes)}')
            click.echo(f'  Level: {level}')
            if check_t2sz_available():
                click.echo('  Using: t2sz')
            else:
                click.echo('  Using: pyzstd (fallback)')

    def progress_callback(bytes_done: int, total: int) -> None:
        if verbose and not json_output and total > 0:
            pct = bytes_done * 100 // total
            click.echo(f'\r  Progress: {pct}%', nl=False)

    try:
        zst_info = create_seekable_zstd(
            input_path=input_path,
            output_path=output_path,
            frame_size_bytes=frame_size_bytes,
            compression_level=level,
            threads=threads,
            progress_callback=progress_callback if verbose else None,
        )

        if verbose and not json_output:
            click.echo()  # Newline after progress

        result['success'] = True
        result['output'] = str(output_path)
        result['compressed_size'] = zst_info.compressed_size
        result['decompressed_size'] = zst_info.decompressed_size
        result['frame_count'] = zst_info.frame_count
        result['compression_ratio'] = (
            round(zst_info.decompressed_size / zst_info.compressed_size, 2) if zst_info.compressed_size > 0 else 0
        )

        if not json_output:
            ratio = result['compression_ratio']
            click.echo(
                f'  Created: {output_path} '
                f'({human_readable_size(zst_info.compressed_size)}, '
                f'{zst_info.frame_count} frames, {ratio}:1 ratio)'
            )

        # Build index unless --no-index
        if not no_index:
            if not json_output:
                click.echo('  Building line index...')

            def index_progress(frame_idx: int, total: int) -> None:
                if verbose and not json_output and total > 0:
                    pct = frame_idx * 100 // total
                    click.echo(f'\r  Indexing: {pct}%', nl=False)

            try:
                index = build_index(output_path, progress_callback=index_progress if verbose else None)

                if verbose and not json_output:
                    click.echo()  # Newline after progress

                result['index'] = {
                    'total_lines': index.total_lines,
                    'frame_count': index.frame_count,
                }

                if not json_output:
                    click.echo(f'  Index: {index.total_lines:,} lines indexed')

            except Exception as e:
                result['index_error'] = str(e)
                if not json_output:
                    click.echo(f'  Warning: failed to build index: {e}', err=True)

        # Delete source if requested
        if delete_source:
            try:
                os.remove(input_path)
                result['source_deleted'] = True
                if not json_output:
                    click.echo(f'  Deleted source: {input_path}')
            except OSError as e:
                result['source_deleted'] = False
                result['delete_error'] = str(e)
                if not json_output:
                    click.echo(f'  Warning: failed to delete source: {e}', err=True)

    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        if not json_output:
            click.echo(f'  Error: {e}', err=True)

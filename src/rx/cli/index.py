"""CLI command for large file indexing."""

import json
import os
import sys

import click

from rx import seekable_index, seekable_zstd
from rx.file_utils import is_text_file
from rx.index import (
    create_index_file,
    delete_index,
    get_index_info,
    get_index_path,
    get_large_file_threshold_bytes,
    is_index_valid,
)


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f'{size_bytes:.2f} {unit}'
        size_bytes /= 1024
    return f'{size_bytes:.2f} PB'


@click.command('index')
@click.argument('paths', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--force', '-f', is_flag=True, help='Force rebuild even if valid index exists')
@click.option('--info', '-i', is_flag=True, help='Show index info without rebuilding')
@click.option('--delete', '-d', is_flag=True, help='Delete index for file(s)')
@click.option('--json', 'json_output', is_flag=True, help='Output in JSON format')
@click.option('--recursive', '-r', is_flag=True, help='Recursively process directories')
@click.option(
    '--threshold',
    type=int,
    default=None,
    help='Only index files larger than this (MB). Default: RX_LARGE_TEXT_FILE_MB or 100',
)
def index_command(
    paths: tuple[str, ...],
    force: bool,
    info: bool,
    delete: bool,
    json_output: bool,
    recursive: bool,
    threshold: int | None,
):
    """Create or manage large file indexes.

    Indexes enable efficient line-based access to large text files.

    Examples:

        rx index /path/to/large.log          # Create index

        rx index /path/to/dir/ -r            # Index all large files in directory

        rx index /path/to/file.log --force   # Force rebuild

        rx index /path/to/file.log --info    # Show index info

        rx index /path/to/file.log --delete  # Delete index
    """
    # Collect all files to process
    files_to_process = []
    threshold_bytes = threshold * 1024 * 1024 if threshold else get_large_file_threshold_bytes()

    for path in paths:
        if os.path.isfile(path):
            files_to_process.append(path)
        elif os.path.isdir(path):
            if recursive:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        filepath = os.path.join(root, file)
                        files_to_process.append(filepath)
            else:
                for file in os.listdir(path):
                    filepath = os.path.join(path, file)
                    if os.path.isfile(filepath):
                        files_to_process.append(filepath)

    # Filter to text files or seekable zstd files (unless showing info or deleting)
    if not info and not delete:
        files_to_process = [f for f in files_to_process if is_text_file(f) or seekable_zstd.is_seekable_zstd(f)]

    if not files_to_process:
        click.echo('No files to process.', err=True)
        sys.exit(1)

    results = []

    for filepath in files_to_process:
        result = {'path': filepath}
        is_zst = seekable_zstd.is_seekable_zstd(filepath)

        if delete:
            # Delete mode
            if is_zst:
                success = seekable_index.delete_index(filepath)
            else:
                success = delete_index(filepath)
            result['action'] = 'delete'
            result['success'] = success
            result['type'] = 'seekable_zst' if is_zst else 'text'
            if not json_output:
                status = 'deleted' if success else 'failed'
                file_type = '.zst' if is_zst else 'text'
                click.echo(f'{filepath} ({file_type}): index {status}')

        elif info:
            # Info mode
            if is_zst:
                index_info = seekable_index.get_index_info(filepath)
                result['type'] = 'seekable_zst'
            else:
                index_info = get_index_info(filepath)
                result['type'] = 'text'

            result['action'] = 'info'
            if index_info:
                result['index'] = index_info
                if not json_output:
                    file_type = '.zst' if is_zst else 'text'
                    click.echo(f'\n{filepath} ({file_type}):')
                    click.echo(f'  Index path: {index_info["index_path"]}')
                    click.echo(f'  Valid: {index_info["is_valid"]}')
                    click.echo(f'  Created: {index_info.get("created_at", "N/A")}')

                    # Get source size (different field names for different types)
                    source_size = index_info.get('source_zst_size_bytes') or index_info.get('source_size_bytes', 0)
                    click.echo(f'  Source size: {human_readable_size(source_size)}')

                    if is_zst:
                        # Seekable zst specific info
                        click.echo(f'  Frames: {index_info.get("frame_count", "N/A")}')
                        click.echo(f'  Total lines: {index_info.get("total_lines", "N/A"):,}')
                        click.echo(
                            f'  Decompressed size: {human_readable_size(index_info.get("decompressed_size_bytes", 0))}'
                        )
                    else:
                        # Regular text file info
                        click.echo(f'  Index entries: {index_info["index_entries"]}')
                        click.echo(f'  Index step: {human_readable_size(index_info["index_step_bytes"])}')
                        if index_info.get('analysis'):
                            analysis = index_info['analysis']
                            click.echo(f'  Lines: {analysis.get("line_count", "N/A"):,}')
                            click.echo(f'  Line ending: {analysis.get("line_ending", "N/A")}')
            else:
                result['index'] = None
                if not json_output:
                    file_type = '.zst' if is_zst else 'text'
                    click.echo(f'\n{filepath} ({file_type}): no index exists')

        else:
            # Index/rebuild mode
            file_size = os.path.getsize(filepath)
            result['type'] = 'seekable_zst' if is_zst else 'text'

            # Check if file meets threshold (unless forcing specific file)
            # Skip threshold check for .zst files (they're always indexed)
            if not is_zst and file_size < threshold_bytes and len(paths) > 1:
                result['action'] = 'skipped'
                result['reason'] = f'File size ({human_readable_size(file_size)}) below threshold'
                if not json_output:
                    click.echo(f'{filepath}: skipped (below threshold)')
                results.append(result)
                continue

            # Check if valid index exists (unless forcing)
            has_valid_index = seekable_index.is_index_valid(filepath) if is_zst else is_index_valid(filepath)
            if not force and has_valid_index:
                result['action'] = 'exists'
                result['valid'] = True
                if not json_output:
                    file_type = '.zst' if is_zst else 'text'
                    click.echo(f'{filepath} ({file_type}): valid index exists (use --force to rebuild)')
                results.append(result)
                continue

            # Build index
            if not json_output:
                file_type = '.zst' if is_zst else 'text'
                click.echo(f'{filepath} ({file_type}): building index...', nl=False)

            if is_zst:
                # Build seekable zst index
                try:
                    index = seekable_index.build_index(filepath)
                    result['action'] = 'created'
                    result['success'] = True
                    result['index_path'] = str(seekable_index.get_index_path(filepath))
                    result['frame_count'] = index.frame_count
                    result['total_lines'] = index.total_lines
                    if not json_output:
                        click.echo(f' done ({index.frame_count} frames, {index.total_lines:,} lines)')
                except Exception as e:
                    result['action'] = 'failed'
                    result['success'] = False
                    result['error'] = str(e)
                    if not json_output:
                        click.echo(f' failed: {e}')
            else:
                # Build regular text file index
                index_data = create_index_file(filepath, force=force)
                if index_data:
                    result['action'] = 'created'
                    result['success'] = True
                    result['index_path'] = str(get_index_path(filepath))
                    result['index_entries'] = len(index_data.get('line_index', []))
                    if not json_output:
                        click.echo(f' done ({result["index_entries"]} entries)')
                else:
                    result['action'] = 'failed'
                    result['success'] = False
                    if not json_output:
                        click.echo(' failed')

        results.append(result)

    if json_output:
        click.echo(json.dumps({'files': results}, indent=2))

    # Exit with error if any failures
    failures = [r for r in results if r.get('success') is False]
    if failures:
        sys.exit(1)

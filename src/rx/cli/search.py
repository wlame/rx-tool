"""CLI search command for RX"""

import os
import re
import sys
import json
import click
from time import time
from typing import Dict, Tuple, Optional

from rx.parse_json import parse_paths_json
from rx.models import TraceResponse, Match, ContextLine


def format_context_header(file_val: str, offset_str: str, pattern_val: str, colorize: bool) -> str:
    """
    Format the header line for a context block.

    Args:
        file_val: File path
        offset_str: Byte offset as string
        pattern_val: Pattern that matched
        colorize: Whether to apply color styling

    Returns:
        Formatted header string
    """
    if colorize:
        return (
            click.style("=== ", fg="bright_black")
            + click.style(file_val, fg="cyan", bold=True)
            + click.style(":", fg="bright_black")
            + click.style(offset_str, fg="yellow")
            + " "
            + click.style("[", fg="bright_black")
            + click.style(pattern_val, fg="magenta", bold=True)
            + click.style("]", fg="bright_black")
            + " "
            + click.style("===", fg="bright_black")
        )
    else:
        return f"=== {file_val}:{offset_str} [{pattern_val}] ==="


def find_match_for_context(
    response: TraceResponse, pattern_id: str, file_id: str, offset_int: int
) -> Tuple[Optional[str], Optional[int]]:
    """
    Find the matched line and line number for a given context block.

    Args:
        response: The TraceResponse containing matches
        pattern_id: Pattern ID to search for
        file_id: File ID to search for
        offset_int: Byte offset to search for

    Returns:
        Tuple of (matched_line_text, line_number) or (None, None)
    """
    for match in response.matches:
        if match.pattern == pattern_id and match.file == file_id and match.offset == offset_int:
            return match.line_text, match.line_number
    return None, None


def build_lines_dict(ctx_lines: list, matched_line: Optional[str], match_line_number: Optional[int]) -> Dict[int, str]:
    """
    Build a dictionary of line numbers to line text from context lines.

    Args:
        ctx_lines: List of context line objects
        matched_line: The matched line text to include
        match_line_number: Line number of the matched line

    Returns:
        Dictionary mapping line_number -> line_text
    """
    lines_by_number = {}

    # Add context lines
    for ctx_line in ctx_lines:
        line_num = ctx_line.line_number if isinstance(ctx_line, ContextLine) else ctx_line.get('line_number')
        line_text = (
            ctx_line.line_text if isinstance(ctx_line, ContextLine) else ctx_line.get('line_text', str(ctx_line))
        )
        lines_by_number[line_num] = line_text

    # Add the matched line
    if matched_line and match_line_number:
        lines_by_number[match_line_number] = matched_line

    return lines_by_number


def highlight_pattern_in_line(line_text: str, pattern_val: str, colorize: bool) -> str:
    """
    Highlight pattern matches in a line of text.

    Args:
        line_text: The line to highlight
        pattern_val: The pattern to highlight
        colorize: Whether to apply color styling

    Returns:
        Highlighted line (or original if highlighting fails)
    """
    if not colorize:
        return line_text

    try:
        # Highlight the matched pattern in bold red
        parts = re.split(f'({pattern_val})', line_text)
        return ''.join(
            click.style(part, fg="bright_red", bold=True) if i % 2 == 1 else part for i, part in enumerate(parts)
        )
    except re.error:
        # If regex is invalid for highlighting, just return the original
        return line_text


def display_context_block(
    composite_key: str, response: TraceResponse, pattern_ids: Dict[str, str], file_ids: Dict[str, str], colorize: bool
) -> None:
    """
    Display a single context block (header + context lines).

    Args:
        composite_key: The composite key "pattern_id:file_id:offset"
        response: TraceResponse containing matches and context
        pattern_ids: Mapping of pattern IDs to patterns
        file_ids: Mapping of file IDs to file paths
        colorize: Whether to apply color styling
    """
    # Parse composite key
    parts = composite_key.split(':', 2)
    if len(parts) != 3:
        return

    pattern_id, file_id, offset_str = parts
    pattern_val = pattern_ids.get(pattern_id, pattern_id)
    file_val = file_ids.get(file_id, file_id)
    offset_int = int(offset_str)

    # Find the matched line
    matched_line, match_line_number = find_match_for_context(response, pattern_id, file_id, offset_int)

    # Format and display header
    header = format_context_header(file_val, offset_str, pattern_val, colorize)
    click.echo(header)

    # Get context lines for this match
    ctx_lines = response.context_lines[composite_key]

    # Build lines dictionary (line_number -> line_text)
    lines_by_number = build_lines_dict(ctx_lines, matched_line, match_line_number)

    # Display lines in order
    for line_num in sorted(lines_by_number.keys()):
        line_text = lines_by_number[line_num]
        highlighted = highlight_pattern_in_line(line_text, pattern_val, colorize)
        click.echo(highlighted)

    click.echo()  # Blank line after each context block


def display_samples_output(
    response: TraceResponse,
    pattern_ids: Dict[str, str],
    file_ids: Dict[str, str],
    before_ctx: int,
    after_ctx: int,
    colorize: bool,
) -> None:
    """
    Display sample output with context lines in CLI format.

    Args:
        response: TraceResponse containing matches and context
        pattern_ids: Mapping of pattern IDs to patterns
        file_ids: Mapping of file IDs to file paths
        before_ctx: Number of context lines before
        after_ctx: Number of context lines after
        colorize: Whether to apply color styling
    """
    # Display basic match summary
    click.echo(response.to_cli(colorize=colorize))
    click.echo()
    click.echo(f"Samples (context: {before_ctx} before, {after_ctx} after):")
    click.echo()

    # Display context blocks
    if response.context_lines:
        for composite_key in sorted(response.context_lines.keys()):
            display_context_block(composite_key, response, pattern_ids, file_ids, colorize)
    else:
        click.echo("No context available (context may not have been requested or no matches found)")


def handle_samples_output(
    response: TraceResponse,
    pattern_ids: Dict[str, str],
    file_ids: Dict[str, str],
    before_ctx: int,
    after_ctx: int,
    output_json: bool,
    no_color: bool,
) -> None:
    """
    Handle output when --samples flag is enabled.

    Args:
        response: TraceResponse containing matches and context
        pattern_ids: Mapping of pattern IDs to patterns
        file_ids: Mapping of file IDs to file paths
        before_ctx: Number of context lines before
        after_ctx: Number of context lines after
        output_json: Whether to output as JSON
        no_color: Whether to disable color
    """
    try:
        if output_json:
            # JSON output with samples
            output_data = response.model_dump()
            click.echo(json.dumps(output_data, indent=2))
        else:
            # CLI output with samples (human-readable)
            colorize = not no_color and sys.stdout.isatty()
            display_samples_output(response, pattern_ids, file_ids, before_ctx, after_ctx, colorize)
    except Exception as e:
        click.echo(f"❌ Error displaying samples: {e}", err=True)
        sys.exit(1)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.argument('path_arg', type=click.Path(exists=True), required=False, metavar='PATH')
@click.argument('regex_arg', type=str, required=False, metavar='REGEX')
@click.option(
    '--path',
    '--file',
    type=click.Path(exists=True),
    multiple=True,
    help="File or directory path to search (can be specified multiple times)",
)
@click.option(
    '--regexp',
    '--regex',
    '-e',
    'regexp',
    type=str,
    multiple=True,
    help="Regex pattern to search (can be specified multiple times)",
)
@click.option('--max-results', type=int, help="Maximum number of results to return")
@click.option('--samples', is_flag=True, help="Show context lines around matches")
@click.option('--context', type=int, help="Number of lines before and after (for --samples)")
@click.option('--before', '-B', type=int, help="Number of lines before match (for --samples)")
@click.option('--after', '-A', type=int, help="Number of lines after match (for --samples)")
@click.option('--json', 'output_json', is_flag=True, help="Output results as JSON")
@click.option('--no-color', is_flag=True, help="Disable colored output")
@click.option('--debug', is_flag=True, help="Enable debug mode (creates .debug_* files)")
@click.pass_context
def search_command(
    ctx, path_arg, regex_arg, path, regexp, max_results, samples, context, before, after, output_json, no_color, debug
):
    """
    Search files and directories for regex patterns using ripgrep.

    \b
    Basic Examples:
        rx /path/to/file.txt "error.*"              # Search for pattern
        rx file.log "error" --max-results=10        # Limit results
        rx file.log "error" --samples               # Show context lines
        rx file.log "error" --samples --context=5   # Custom context size

    \b
    Ripgrep Passthrough:
        Any unrecognized options are passed directly to ripgrep:
        rx file.log "error" -i                      # Case-insensitive search
        rx file.log "error" --case-sensitive        # Case-sensitive search
        rx file.log "error" -w                      # Match whole words only
        rx file.log "error" -A 3                    # Show 3 lines after match

    \b
    Requirements:
        - ripgrep must be installed on your system
          macOS: brew install ripgrep
          Ubuntu/Debian: apt install ripgrep
          Fedora: dnf install ripgrep
    """

    # Resolve paths and regexps from positional or named params
    # Handle multiple paths - always treat as list internally
    final_paths = []
    if path:
        # Named parameter --path/--file (tuple from multiple=True)
        final_paths.extend(list(path))
    if path_arg:
        # Positional PATH argument
        final_paths.append(path_arg)

    # Handle regexp patterns - always treat as list internally
    final_regexps = []
    if regexp:
        # Named parameter --regexp/-e (tuple from multiple=True)
        final_regexps.extend(list(regexp))
    if regex_arg:
        # Positional REGEX argument
        final_regexps.append(regex_arg)

    # Extract extra ripgrep arguments from unknown options
    rg_extra_args = ctx.args if ctx.args else []

    # Enable debug mode if --debug flag is set
    if debug:
        os.environ['RX_DEBUG'] = '1'
        # Reload the parse_json module to pick up the new DEBUG_MODE setting
        import importlib
        from rx import parse_json

        importlib.reload(parse_json)
        click.echo("Debug mode enabled - will create .debug_* files", err=True)

    if final_paths and final_regexps:
        # Calculate context parameters if --samples is requested
        if samples:
            before_ctx = before if before is not None else context if context is not None else 3
            after_ctx = after if after is not None else context if context is not None else 3
        else:
            before_ctx = 0
            after_ctx = 0

        # Parse files or directories for matches using JSON mode
        try:
            time_before = time()
            result = parse_paths_json(
                final_paths,
                final_regexps,
                max_results=max_results,
                rg_extra_args=rg_extra_args,
                context_before=before_ctx,
                context_after=after_ctx,
            )
            parsing_time = time() - time_before

            # Extract data from ID-based result structure
            pattern_ids = result['patterns']  # {'p1': 'error', 'p2': 'warning'}
            file_ids = result['files']  # {'f1': '/path/file.log'}
            matches = result['matches']  # [{'pattern': 'p1', 'file': 'f1', 'offset': 100, ...}]
            scanned_files = result['scanned_files']
            skipped_files = result['skipped_files']
            context_lines_dict = result.get('context_lines')  # Optional context from ripgrep

        except FileNotFoundError as e:
            click.echo(f"❌ Error: {e}", err=True)
            sys.exit(1)
        except RuntimeError as e:
            click.echo(f"❌ Error: {e}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"❌ Unexpected error: {e}", err=True)
            sys.exit(1)

        # Build response object
        # For display, show all paths as comma-separated string if multiple
        display_path = ", ".join(final_paths) if len(final_paths) > 1 else final_paths[0]

        # Convert context_lines to proper format if present
        converted_context = None
        if context_lines_dict:
            converted_context = {}
            for key, ctx_lines in context_lines_dict.items():
                # Convert ContextLine objects to dict for serialization
                converted_context[key] = ctx_lines

        response = TraceResponse(
            path=display_path,
            time=parsing_time,
            patterns=pattern_ids,
            files=file_ids,
            matches=[Match(**m) for m in matches],
            scanned_files=scanned_files,
            skipped_files=skipped_files,
            context_lines=converted_context,
            before_context=before_ctx if samples else None,
            after_context=after_ctx if samples else None,
        )

        # Handle --samples flag
        if samples:
            handle_samples_output(response, pattern_ids, file_ids, before_ctx, after_ctx, output_json, no_color)
        else:
            # No samples - just show matches
            if output_json:
                click.echo(response.model_dump_json(indent=2))
            else:
                colorize = not no_color
                click.echo(response.to_cli(colorize=colorize))

        sys.exit(0)

    # No valid mode provided - show help
    ctx = click.get_current_context()
    click.echo(ctx.get_help())
    sys.exit(0)


if __name__ == '__main__':
    search_command()

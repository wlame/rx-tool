"""CLI command for getting file samples around byte offsets or line numbers."""

import json
import sys

import click

from rx.file_utils import get_context, get_context_by_lines, is_text_file
from rx.index import calculate_exact_line_for_offset, calculate_exact_offset_for_line, get_index_path, load_index
from rx.models import SamplesResponse


@click.command("samples")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--byte-offset",
    "-b",
    multiple=True,
    type=int,
    help="Byte offset(s) to get context for. Can be specified multiple times.",
)
@click.option(
    "--line-offset",
    "-l",
    multiple=True,
    type=int,
    help="Line number(s) to get context for (1-based). Can be specified multiple times.",
)
@click.option(
    "--context",
    "-c",
    type=int,
    default=None,
    help="Number of context lines before and after (default: 3)",
)
@click.option(
    "--before",
    "-B",
    type=int,
    default=None,
    help="Number of context lines before offset",
)
@click.option(
    "--after",
    "-A",
    type=int,
    default=None,
    help="Number of context lines after offset",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output in JSON format",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable colored output",
)
@click.option(
    "--regex",
    "-r",
    type=str,
    default=None,
    help="Regex pattern to highlight in output",
)
def samples_command(
    path: str,
    byte_offset: tuple[int, ...],
    line_offset: tuple[int, ...],
    context: int | None,
    before: int | None,
    after: int | None,
    json_output: bool,
    no_color: bool,
    regex: str | None,
):
    """Get file content around specified byte offsets or line numbers.

    This command reads lines of context around one or more byte offsets
    or line numbers in a text file. Useful for examining specific locations
    in large files.

    Use -b/--byte-offset for byte offsets, or -l/--line-offset for line numbers.
    These options are mutually exclusive.

    Examples:

        rx samples /var/log/app.log -b 1234

        rx samples /var/log/app.log -b 1234 -b 5678 -c 5

        rx samples /var/log/app.log -l 100 -l 200

        rx samples /var/log/app.log -l 100 --before=2 --after=10

        rx samples /var/log/app.log -b 1234 --json
    """
    # Validate mutual exclusivity
    if byte_offset and line_offset:
        click.echo("Error: Cannot use both --byte-offset and --line-offset. Choose one.", err=True)
        sys.exit(1)

    if not byte_offset and not line_offset:
        click.echo("Error: Must provide either --byte-offset (-b) or --line-offset (-l)", err=True)
        sys.exit(1)

    # Validate file is text
    if not is_text_file(path):
        click.echo(f"Error: {path} is not a text file", err=True)
        sys.exit(1)

    # Determine context lines
    before_context = before if before is not None else context if context is not None else 3
    after_context = after if after is not None else context if context is not None else 3

    if before_context < 0 or after_context < 0:
        click.echo("Error: Context values must be non-negative", err=True)
        sys.exit(1)

    try:
        if byte_offset:
            # Byte offset mode
            offset_list = list(byte_offset)
            context_data = get_context(path, offset_list, before_context, after_context)

            # Calculate line numbers for each byte offset (only if JSON output)
            offset_to_line = {}
            if json_output:
                index_data = load_index(get_index_path(path))
                for offset in offset_list:
                    line_num = calculate_exact_line_for_offset(path, offset, index_data)
                    offset_to_line[str(offset)] = line_num
            else:
                # For CLI mode, populate with offsets as keys (no line mapping needed)
                for offset in offset_list:
                    offset_to_line[str(offset)] = -1

            response = SamplesResponse(
                path=path,
                offsets=offset_to_line,
                lines={},
                before_context=before_context,
                after_context=after_context,
                samples={str(k): v for k, v in context_data.items()},
            )
        else:
            # Line offset mode
            line_list = list(line_offset)
            context_data = get_context_by_lines(path, line_list, before_context, after_context)

            # Calculate byte offsets for each line number (only if JSON output)
            line_to_offset = {}
            if json_output:
                index_data = load_index(get_index_path(path))
                for line_num in line_list:
                    byte_offset = calculate_exact_offset_for_line(path, line_num, index_data)
                    line_to_offset[str(line_num)] = byte_offset
            else:
                # For CLI mode, populate with line numbers as keys (no offset mapping needed)
                for line_num in line_list:
                    line_to_offset[str(line_num)] = -1

            response = SamplesResponse(
                path=path,
                offsets={},
                lines=line_to_offset,
                before_context=before_context,
                after_context=after_context,
                samples={str(k): v for k, v in context_data.items()},
            )

        if json_output:
            click.echo(json.dumps(response.model_dump(), indent=2))
        else:
            colorize = not no_color and sys.stdout.isatty()
            click.echo(response.to_cli(colorize=colorize, regex=regex))

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

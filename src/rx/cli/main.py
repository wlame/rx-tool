"""Main CLI entry point with command groups"""

import multiprocessing

import click

from rx.__version__ import __version__
from rx.cli.analyse import analyse_command
from rx.cli.check import check_command
from rx.cli.index import index_command
from rx.cli.search import search_command
from rx.cli.serve import serve_command


class DefaultCommandGroup(click.Group):
    """Custom Click Group that allows a default command"""

    def parse_args(self, ctx, args):
        # If --help or --version is requested, show group help/version
        if args and args[0] in ('--help', '-h', '--version'):
            return super().parse_args(ctx, args)

        # Check if first arg is a known command
        if args and args[0] in self.commands:
            return super().parse_args(ctx, args)

        # Otherwise, treat as search command (default)
        # Insert 'search' as the command name
        return super().parse_args(ctx, ['search'] + args)


@click.group(cls=DefaultCommandGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name='RX')
@click.pass_context
def cli(ctx):
    """
    RX (Regex Tracer) - High-performance file search and analysis tool.

    \b
    Commands:
      rx <path> <pattern>     Search files (default command)
      rx analyse <path>       Analyze files (metadata, statistics)
      rx check <pattern>      Analyze regex complexity
      rx index <path>         Create/manage large file indexes
      rx serve               Start web API server

    \b
    Examples:
      rx /var/log/app.log "error"
      rx /var/log/ "error.*failed" -i
      rx analyse /var/log/app.log
      rx check "(a+)+"
      rx serve --port 8000

    \b
    For more help on each command:
      rx --help
      rx analyse --help
      rx check --help
      rx serve --help
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)


# Register subcommands (search is the default command)
cli.add_command(search_command, name='search')
cli.add_command(analyse_command, name='analyse')
cli.add_command(check_command, name='check')
cli.add_command(index_command, name='index')
cli.add_command(serve_command, name='serve')


def main():
    """Entry point for the CLI"""
    # Support for multiprocessing in frozen binaries (PyInstaller)
    # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.freeze_support
    multiprocessing.freeze_support()

    cli()


if __name__ == '__main__':
    main()

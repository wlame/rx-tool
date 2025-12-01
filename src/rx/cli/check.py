"""Check command for regex complexity analysis"""

import sys

import click

from rx.models import ComplexityResponse
from rx.regex import calculate_regex_complexity


@click.command()
@click.argument('pattern', type=str)
@click.option('--json', 'output_json', is_flag=True, help="Output as JSON")
@click.option('--no-color', is_flag=True, help="Disable colored output")
def check_command(pattern, output_json, no_color):
    """
    Analyze regex pattern complexity and detect ReDoS vulnerabilities.

    Uses AST-based analysis to detect patterns that can cause catastrophic
    backtracking (ReDoS vulnerabilities). Provides detailed explanations
    and fix suggestions for each detected issue.

    \b
    Complexity Classes:
      LINEAR O(n):       Safe for any input size
      POLYNOMIAL O(n^k): Caution with large inputs (k=2,3,...)
      EXPONENTIAL O(2^n): DANGEROUS - avoid these patterns!

    \b
    Risk Levels:
      safe:      Pattern is safe for production use
      caution:   Minor concerns, monitor on large files
      warning:   Polynomial complexity, may be slow
      dangerous: High risk, significant performance impact
      critical:  ReDoS vulnerability, exponential backtracking

    \b
    Examples:
      rx check "error.*"           # Simple pattern - safe
      rx check "(a+)+"             # Nested quantifier - CRITICAL
      rx check ".*error.*failed.*" # Multiple greedy - warning
      rx check "^[a-z]+$" --json   # Output as JSON

    \b
    References:
      - https://github.com/doyensec/regexploit
      - https://www.regular-expressions.info/catastrophic.html
    """
    try:
        # Calculate complexity
        result = calculate_regex_complexity(pattern)
        result['regex'] = pattern

        # Create response model
        response = ComplexityResponse(**result)

        if output_json:
            # JSON output
            click.echo(response.model_dump_json(indent=2))
        else:
            # Human-readable output
            colorize = not no_color and sys.stdout.isatty()
            output = response.to_cli(colorize=colorize)
            click.echo(output)

        # Exit with non-zero code if dangerous or critical
        if result['risk_level'] in ['dangerous', 'critical']:
            sys.exit(2)  # Warning exit code

    except Exception as e:
        click.echo(f"Error analyzing pattern: {e}", err=True)
        sys.exit(1)

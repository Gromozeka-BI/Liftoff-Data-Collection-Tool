"""dct validate <session> — re-run validation on an existing session."""
from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from dct.validator import validate_session

console = Console()


@click.command()
@click.argument("session_path", type=click.Path(exists=True, file_okay=False))
def validate(session_path: str):
    """Validate a recorded session."""
    p = Path(session_path)
    result = validate_session(p)
    if result.passed:
        console.print(f"[green]PASSED[/green]  {result.stats}")
    else:
        console.print("[red]FAILED[/red]")
        for issue in result.issues:
            console.print(f"  [red]•[/red] {issue}")
        console.print(f"  Stats: {result.stats}")

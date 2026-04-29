"""dct list — list all recorded sessions."""
from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from dct.config import settings
from dct.session import load_meta

console = Console()


@click.command()
@click.option("--sessions-dir", default=None, type=click.Path(), help="Override sessions directory")
def list_cmd(sessions_dir: str | None):
    """List all recorded sessions."""
    base = Path(sessions_dir) if sessions_dir else settings.sessions_dir
    if not base.exists():
        console.print(f"[yellow]No sessions directory found at {base}[/yellow]")
        return

    dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    if not dirs:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    t = Table(title=f"Sessions in {base}")
    t.add_column("#", style="dim", width=4)
    t.add_column("Session ID")
    t.add_column("Pilot", width=8)
    t.add_column("Drone", width=12)
    t.add_column("Track", width=12)
    t.add_column("Packets", justify="right", width=8)
    t.add_column("Laps", justify="right", width=5)
    t.add_column("Duration", justify="right", width=9)
    t.add_column("Valid", width=6)

    for i, d in enumerate(dirs, 1):
        try:
            m = load_meta(d)
            valid = "[green]✓[/green]" if m.get("validated") else "[dim]—[/dim]"
            dur = f"{m.get('duration_s', 0):.0f}s" if m.get("duration_s") else "—"
            t.add_row(
                str(i), m["session_id"], m.get("pilot", "—"), m.get("drone", "—"),
                m.get("track", "—"), str(m.get("total_packets", 0)),
                str(m.get("total_laps", 0)), dur, valid,
            )
        except Exception:
            t.add_row(str(i), d.name, *["?"] * 7)

    console.print(t)

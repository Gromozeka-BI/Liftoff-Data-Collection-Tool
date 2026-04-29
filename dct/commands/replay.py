"""dct replay <session> — replay a recorded session."""
from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from dct.replay import ReplayEngine

console = Console()


@click.command()
@click.argument("session_path", type=click.Path(exists=True, file_okay=False))
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=9001, show_default=True)
@click.option("--speed", default=1.0, show_default=True, help="Playback speed multiplier")
def replay(session_path: str, host: str, port: int, speed: float):
    """Replay a recorded session over UDP."""
    p = Path(session_path)
    console.print(f"[green]Replaying[/green] {p.name} → {host}:{port}  speed={speed}x")
    engine = ReplayEngine(p, target_host=host, target_port=port, speed=speed)
    engine.run()
    console.print("[green]Replay finished.[/green]")

"""dct align <session> -- show video <-> telemetry alignment statistics."""
from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from dct.align import alignment_stats

console = Console()


@click.command()
@click.argument("session_path", type=click.Path(exists=True, file_okay=False))
def align_cmd(session_path: str):
    """Show video <-> telemetry synchronisation quality."""
    p = Path(session_path)
    ts_file = p / "video_timestamps.parquet"
    if not ts_file.exists():
        console.print("[red]video_timestamps.parquet not found.[/red]")
        console.print("[yellow]Re-record the session to generate timestamps.[/yellow]")
        return

    stats = alignment_stats(p)
    console.print(f"\n[bold]Alignment stats for[/bold] {p.name}")
    console.print(f"  Frames:       {stats['frames']}")
    console.print(f"  Mean  Δt:     {stats['dt_mean_ms']} ms")
    console.print(f"  Median Δt:    {stats['dt_median_ms']} ms")
    console.print(f"  P95 Δt:       {stats['dt_p95_ms']} ms")
    console.print(f"  Max Δt:       {stats['dt_max_ms']} ms")

    p95 = stats["dt_p95_ms"]
    if p95 <= 10:
        console.print("[green]Excellent — P95 < 10 ms[/green]")
    elif p95 <= 33:
        console.print("[yellow]Good — P95 < 33 ms (< 1 video frame)[/yellow]")
    else:
        console.print(f"[red]Poor — P95 {p95} ms, check system load[/red]")

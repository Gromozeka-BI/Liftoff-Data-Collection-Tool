"""dct inspect <session> — show session summary."""
from __future__ import annotations

from pathlib import Path

import click
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

from dct.session import load_meta

console = Console()


@click.command()
@click.argument("session_path", type=click.Path(exists=True, file_okay=False))
def inspect_cmd(session_path: str):
    """Inspect a recorded session."""
    p = Path(session_path)
    meta = load_meta(p)

    console.print(f"\n[bold]Session:[/bold] {meta['session_id']}")
    console.rule()

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim", width=18)
    t.add_column(style="bold")
    for k in ("pilot", "drone", "track", "purpose", "created_at", "finished_at",
              "duration_s", "total_packets", "total_laps", "validated"):
        t.add_row(k, str(meta.get(k, "—")))
    console.print(t)

    # Telemetry stats
    telem = p / "telemetry.parquet"
    if telem.exists():
        table = pq.read_table(telem)
        n = len(table)
        ts = table.column("ts_wall").to_pylist()
        dur = ts[-1] - ts[0] if n > 1 else 0
        hz = (n - 1) / dur if dur > 0 else 0
        console.rule("Telemetry")
        console.print(f"  Rows: {n}   Duration: {dur:.1f}s   Sample rate: {hz:.1f} Hz")

    # Events
    events = p / "events.parquet"
    if events.exists():
        etable = pq.read_table(events)
        from collections import Counter
        types = Counter(etable.column("event_type").to_pylist())
        console.rule("Events")
        for etype, cnt in sorted(types.items()):
            console.print(f"  {etype}: {cnt}")

    # Video
    video = p / "video.mp4"
    if video.exists():
        size_mb = video.stat().st_size / 1e6
        console.print(f"\n[green]Video:[/green] video.mp4  {size_mb:.1f} MB")

"""dct record command — main session recording loop."""
from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path
from queue import Empty

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

from dct.config import settings
from dct.log import setup_logging, get_logger

_log = get_logger("record")
from dct.receivers.liftoff_udp import LiftoffUDPReceiver
from dct.receivers.button_api import ButtonAPI
from dct.screen_recorder import ScreenRecorder
from dct.rh_simulator import RHSimulator
from dct.session import create_session, finalize_meta, copy_track, load_track
from dct.storage.writer import TelemetryWriter, EventsWriter
from dct.validator import validate_session

console = Console()


def _find_track(track_id: str) -> Path | None:
    candidates = [
        Path("tracks") / f"{track_id}.json",
        Path(track_id),
        Path(track_id).with_suffix(".json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _build_status_table(stats: dict) -> Table:
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="bold green")
    t.add_row("Packets", str(stats.get("packets", 0)))
    t.add_row("Laps", str(stats.get("laps", 0)))
    t.add_row("Dropped UDP", str(stats.get("dropped", 0)))
    t.add_row("Duration", f"{stats.get('duration', 0):.1f}s")
    if stats.get("rec_frames"):
        t.add_row("Video frames", str(stats["rec_frames"]))
    return t


@click.command()
@click.option("--pilot", "-p", required=True, help="Pilot identifier (e.g. A, pilot1)")
@click.option("--drone", "-d", required=True, help="Drone config name (e.g. cinemarc)")
@click.option("--track", "-t", required=True, help="Track ID or path to track.json")
@click.option("--purpose", default="training", show_default=True, help="Session purpose label")
@click.option("--no-video", is_flag=True, default=False, help="Disable screen recording")
@click.option("--no-rh-sim", is_flag=True, default=False, help="Disable mock RH simulator")
def record(pilot: str, drone: str, track: str, purpose: str, no_video: bool, no_rh_sim: bool):
    """Record a new telemetry session."""
    setup_logging()
    _log.info("dct record start: pilot=%s drone=%s track=%s purpose=%s", pilot, drone, track, purpose)
    # --- Resolve track ---
    track_path = _find_track(track)
    track_data: dict | None = None
    if track_path:
        with open(track_path, encoding="utf-8") as f:
            track_data = json.load(f)
        console.print(f"[green]Track:[/green] {track_data.get('name', track_path.name)}")
    else:
        console.print(f"[yellow]Warning:[/yellow] track '{track}' not found, gate detection disabled")

    # --- Create session dir ---
    session_dir = create_session(pilot, drone, track, purpose)
    console.print(f"[green]Session:[/green] {session_dir}")

    if track_path:
        copy_track(session_dir, track_path)

    # --- Storage ---
    telem_writer = TelemetryWriter(
        session_dir,
        flush_rows=settings.parquet_flush_rows,
        flush_interval=settings.parquet_flush_interval,
    )
    events_writer = EventsWriter(session_dir)
    events_writer.write_event("session_start", time.time(), source="dct")

    # --- UDP receiver ---
    udp = LiftoffUDPReceiver(settings.udp_host, settings.udp_port)
    try:
        udp.start()
    except OSError as e:
        console.print(f"[red]Cannot bind UDP port {settings.udp_port}: {e}[/red]")
        console.print("[yellow]Hint: another dct process may be running. Kill it first.[/yellow]")
        raise SystemExit(1)
    console.print(f"[green]UDP:[/green] listening {settings.udp_host}:{settings.udp_port}")

    # --- REST API ---
    api = ButtonAPI(settings.api_host, settings.api_port)
    api.start()
    console.print(f"[green]API:[/green] http://{settings.api_host}:{settings.api_port}")

    # --- Screen recorder ---
    recorder: ScreenRecorder | None = None
    if not no_video:
        recorder = ScreenRecorder(
            session_dir / "video.mp4",
            settings.screen_window_title,
            fps=settings.screen_fps,
            target_w=settings.screen_width,
            target_h=settings.screen_height,
        )
        recorder.start()
        console.print(f"[green]Video:[/green] {settings.screen_width}x{settings.screen_height}@{settings.screen_fps}fps → video.mp4")

    # --- Mock RH simulator ---
    rh_sim: RHSimulator | None = None
    if not no_rh_sim and track_data:
        gates = track_data.get("gates", [])
        sf_id = next((g["id"] for g in gates if g.get("is_start_finish")), 0)
        api_base = f"http://127.0.0.1:{settings.api_port}"
        rh_sim = RHSimulator(api_base, gates, sf_id, settings.rh_gate_radius)
        rh_sim.start()
        console.print(f"[green]RH-sim:[/green] monitoring {len(gates)} gates")

    # --- Main loop ---
    start_time = time.time()
    stats = {"packets": 0, "laps": 0, "dropped": 0, "duration": 0.0}
    running = True

    def _stop(sig=None, frame=None):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    console.print("\n[bold]Recording... press Ctrl+C to stop[/bold]\n")

    with Live(refresh_per_second=4, console=console) as live:
        while running:
            # drain UDP queue
            drained = 0
            while drained < 200:
                try:
                    frame = udp.queue.get_nowait()
                except Empty:
                    break
                telem_writer.write(frame)
                if rh_sim:
                    rh_sim.feed(frame)
                stats["packets"] += 1
                drained += 1

            # drain API events queue
            while True:
                try:
                    ev = api.events.get_nowait()
                except Empty:
                    break
                events_writer.write_event(
                    ev["event_type"],
                    ev["ts_wall"],
                    gate_id=ev.get("gate_id"),
                    lap_num=ev.get("lap_num"),
                    source="api",
                )
                if "lap" in ev["event_type"]:
                    stats["laps"] += 1

            stats["dropped"] = udp.dropped
            stats["duration"] = time.time() - start_time
            if recorder:
                stats["rec_frames"] = recorder.frames_written

            live.update(_build_status_table(stats))
            time.sleep(0.05)

    # --- Shutdown ---
    console.print("\n[yellow]Stopping...[/yellow]")

    udp.stop()
    if rh_sim:
        rh_sim.stop()
    if recorder:
        recorder.stop()
        if recorder.has_error:
            console.print("[red]Screen recorder error — video may be incomplete[/red]")
        else:
            console.print(f"[green]Video:[/green] {recorder.frames_written} frames @ {recorder.actual_fps} fps")
    api.stop()

    telem_writer.close()
    events_writer.write_event("session_stop", time.time(), source="dct")
    events_writer.close()

    finalize_meta(session_dir, stats["packets"], stats["laps"], start_time)

    # --- Validate ---
    console.print("[yellow]Validating session...[/yellow]")
    result = validate_session(session_dir)
    if result.passed:
        console.print(f"[green]Validation PASSED[/green]  {result.stats}")
    else:
        console.print(f"[red]Validation FAILED[/red]")
        for issue in result.issues:
            console.print(f"  [red]•[/red] {issue}")
        console.print(f"  Stats: {result.stats}")

    _log.info("Session saved: %s | packets=%d laps=%d", session_dir, stats["packets"], stats["laps"])
    console.print(f"\n[bold green]Session saved:[/bold green] {session_dir}")

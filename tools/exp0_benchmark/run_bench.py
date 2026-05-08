"""Run benchmark_v2 on both tracks and save to exp0_benchmark subfolders."""
import subprocess
import sys
from pathlib import Path

BENCH = Path(r"C:\Users\Gromozeka\source\repos\stick_localizer_bench")
OUT_BASE = Path(__file__).parent

# Find Part_5 dynamically
liftoff = Path(r"D:\DroneTrackerDB\Liftoff")
part5 = sorted([d for d in liftoff.iterdir() if d.name.startswith("Part_5")])[0]

part1_sessions = sorted(
    [d for d in Path(r"D:\DroneTrackerDB\Liftoff\Part_1").iterdir()
     if d.is_dir() and "session-009" in d.name]
)
track001_session = str(part1_sessions[0])

RUNS = [
    {
        "label": "track-001 (MadTrainer, session-009, 20 laps)",
        "session": track001_session,
        "out": str(OUT_BASE / "track001"),
    },
    {
        "label": "track-002 (MadTrainer+LiftOff200, cross-drone, 20 laps)",
        "session": str(part5),
        "out": str(OUT_BASE / "track002"),
    },
]

for run in RUNS:
    print(f"\n{'='*60}")
    print(f"Starting: {run['label']}")
    print(f"{'='*60}")
    cmd = [
        sys.executable,
        str(BENCH / "benchmark_v2.py"),
        "--session", run["session"],
        "--mode", "loo",
        "--deltas", "0",
        "--jobs", "-1",
        "--out-dir", run["out"],
    ]
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(BENCH))
    print(f"\nExit code: {result.returncode}")

print("\nAll done.")

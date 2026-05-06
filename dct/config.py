from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DCT_", env_file=".env", extra="ignore")

    # UDP receiver
    udp_host: str = "127.0.0.1"
    udp_port: int = 9001
    udp_buffer: int = 1024

    # REST API
    api_host: str = "0.0.0.0"
    api_port: int = 8765

    # Storage
    sessions_dir: Path = Path("sessions")
    parquet_flush_rows: int = 500       # flush row group every N rows
    parquet_flush_interval: float = 2.0 # or every N seconds (whichever comes first)

    # Screen recording
    screen_fps: int = 30
    screen_width: int = 1280
    screen_height: int = 720
    screen_window_title: str = "Liftoff"
    # Capture backend: "auto" tries DXGI (dxcam) on Windows, falls back to mss.
    # Force one with: DCT_SCREEN_CAPTURE_BACKEND=mss   or   =dxgi
    screen_capture_backend: str = "auto"

    # Mock RH gate detection
    rh_gate_radius: float = 2.0  # metres — matches track check_radius


settings = Settings()

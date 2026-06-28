"""Modal remote runner for soccer AI pipeline.

Usage:
    # (1) One-time weight upload — only needed when weights change:
    modal run modal_runner.py::seed_weights

    # (2) Run pipeline:
    python main.py models.device=modal video.source_path=data/0bfacc_0.mp4

Setup:
    uv sync --extra modal
    modal token new
"""
from __future__ import annotations

from pathlib import Path
import modal

TIMEOUT = 3600

app = modal.App("soccer-ai")

# Persistent volume — weights stored once on Modal, reused across all runs.
weights_vol = modal.Volume.from_name("soccer-ai-weights", create_if_missing=True)

_WEIGHT_FILES = [
    ("weights/player_detection.pt",            "player_detection.pt"),
    ("weights/SV_kp",                          "SV_kp"),
    ("weights/SV_lines",                       "SV_lines"),
    ("weights/osnet_x1_0_sportsreid.pth.tar",  "osnet_x1_0_sportsreid.pth.tar"),
]

# ---------------------------------------------------------------------------
# Image: pip/apt deps + source code only (no weights — they live in the volume).
# Image is rebuilt only when source code or deps change.
# ---------------------------------------------------------------------------
_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0", "libgl1", "git")
    .pip_install(
        "gdown>=5.2.1",
        "hydra-core>=1.3.2",
        "ultralytics>=8.0.0",
        "opencv-python-headless>=4.8.0",
        "supervision>=0.23.0",
        "tqdm>=4.66.0",
        "omegaconf",
        "pyyaml",
        "scipy",
        "Pillow",
        "shapely",
    )
    .run_commands(
        "pip install git+https://github.com/roboflow/sports.git",
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
        "pip install --force-reinstall opencv-python-headless",
    )
    .add_local_file("main.py",    remote_path="/app/main.py")
    .add_local_dir("conf",        remote_path="/app/conf")
    .add_local_dir("soccer_ai",   remote_path="/app/soccer_ai")
)


# ---------------------------------------------------------------------------
# Seed: upload weights to the volume (run once from local machine).
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def seed_weights():
    """Upload weights to Modal volume — run once, or after weights change.

    modal run modal_runner.py::seed_weights
    """
    with weights_vol.batch_upload(force=True) as batch:
        for local, remote in _WEIGHT_FILES:
            p = Path(local)
            if not p.exists():
                print(f"  SKIP (not found locally): {local}")
                continue
            print(f"  Uploading {local}  ({p.stat().st_size / 1_000_000:.0f} MB)...")
            batch.put_file(str(p), remote)
    print("Done. Weights stored in Modal volume 'soccer-ai-weights'.")


# ---------------------------------------------------------------------------
# Pipeline function — weights mounted from volume, never re-uploaded.
# ---------------------------------------------------------------------------

@app.function(
    gpu="H100",
    image=_image,
    volumes={"/app/weights": weights_vol},
    timeout=TIMEOUT,
)
def run_pipeline(video_bytes: bytes, extra_overrides: list[str] = []) -> dict[str, bytes | None]:
    import os, sys, subprocess, tempfile
    os.chdir("/app")
    sys.path.insert(0, "/app")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src       = tmp / "input.mp4"
        out       = tmp / "output.mp4"
        stats_dir = tmp / "player_stats"

        src.write_bytes(video_bytes)

        cmd = [
            sys.executable, "/app/main.py",
            f"video.source_path={src}",
            f"video.output_path={out}",
            f"player_stats.output_dir={stats_dir}",
            "models.device=0",
            "reid.enabled=true",
            "hydra.run.dir=/tmp/hydra",
        ] + extra_overrides

        proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Pipeline failed:\n{proc.stderr[-4000:]}")

        stats_json = stats_dir / "player_stats.json"
        return {
            "video":        out.read_bytes()        if out.exists()        else None,
            "player_stats": stats_json.read_bytes() if stats_json.exists() else None,
        }


@app.local_entrypoint()
def main(video_path: str, output_path: str, stats_path: str = ""):
    """Called by: modal run modal_runner.py --video-path=... --output-path=..."""
    video_bytes = Path(video_path).read_bytes()
    result = run_pipeline.remote(video_bytes)

    if result.get("video"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(result["video"])
        print(f"Saved video: {output_path}")

    if result.get("player_stats") and stats_path:
        Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
        Path(stats_path).write_bytes(result["player_stats"])
        print(f"Saved stats: {stats_path}")

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

import gdown

log = logging.getLogger(__name__)

VIDEO_FILES = [
    ("0bfacc_0.mp4", "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"),
    ("2e57b9_0.mp4", "19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf"),
    ("08fd33_0.mp4", "1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-"),
    ("573e61_0.mp4", "1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU"),
    ("121364_0.mp4", "1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"),
]

RUNS_ZIP_URL = "https://drive.google.com/file/d/1s_M_Jz4SK7LVG_se_ApNuwbmo3qNnBT7/view?usp=sharing"


def _to_gdown_url(url_or_id: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url_or_id):
        return f"https://drive.google.com/uc?id={url_or_id}"
    match = re.search(r"/d/([A-Za-z0-9_-]+)", url_or_id)
    if match:
        return f"https://drive.google.com/uc?id={match.group(1)}"
    return url_or_id


def download_videos(project_root: Path | None = None) -> None:
    root = project_root or Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, file_id in VIDEO_FILES:
        output_path = data_dir / filename
        if output_path.exists():
            continue
        gdown.download(_to_gdown_url(file_id), str(output_path), quiet=False)


def ensure_runs_weights(
    project_root: Path,
    required_paths: list[Path],
    zip_url: str = RUNS_ZIP_URL,
) -> bool:
    if all(path.exists() for path in required_paths):
        return False

    research_dir = project_root / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    zip_path = research_dir / "runs.zip"
    if not zip_path.exists():
        log.warning("Missing model weights, downloading runs.zip from Google Drive...")
        gdown.download(_to_gdown_url(zip_url), str(zip_path), quiet=False)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(research_dir)
    return True


def download_runs() -> None:
    root = Path(__file__).resolve().parent.parent
    ensure_runs_weights(
        project_root=root,
        required_paths=[
            root / "research/runs/detect/train/weights/best.pt",
            root / "research/runs/pose/train/weights/best.pt",
        ],
        zip_url=RUNS_ZIP_URL,
    )


def main() -> None:
    download_videos()


if __name__ == "__main__":
    main()

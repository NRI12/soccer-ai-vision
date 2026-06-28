"""Real-time soccer analysis with adjustable NBJW calibration interval.

Usage:
    python realtime.py --source data/08fd33_0.mp4
    python realtime.py --source data/08fd33_0.mp4 --interval 10 --output output/rt.mp4
    python realtime.py --source 0   # webcam

Keyboard controls:
    q / ESC   quit
    p         pause / resume
    +/-       increase / decrease NBJW interval (1–60)
    s         save current frame as screenshot
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from omegaconf import OmegaConf
from sports.configs.soccer import SoccerPitchConfiguration

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _load_cfg():
    base = OmegaConf.load("conf/config.yaml")
    for name in ("detect", "filter", "track", "team", "pitch", "annotate", "reid"):
        base = OmegaConf.merge(base, OmegaConf.load(f"conf/pipeline/{name}.yaml"))
    return base


class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._times: list[float] = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


def _draw_hud(frame: np.ndarray, fps: float, interval: int, paused: bool) -> None:
    lines = [f"FPS: {fps:.1f}", f"NBJW interval: {interval}  (+/- to change)"]
    if paused:
        lines.append("PAUSED")
    y = 30
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/08fd33_0.mp4")
    parser.add_argument("--output", default="")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    cfg = _load_cfg()

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from soccer_ai.download_data import RUNS_ZIP_URL, ensure_runs_weights
    from ultralytics import YOLO

    cwd = Path.cwd()
    ensure_runs_weights(cwd, [cwd / cfg.models.player_detection.path], RUNS_ZIP_URL)
    player_model = YOLO(str(cwd / cfg.models.player_detection.path))

    from soccer_ai.calibration import NBJWCalibrator
    calibrator = NBJWCalibrator(
        weights_kp=str(cwd / cfg.pitch.nbjw_weights_kp),
        weights_lines=str(cwd / cfg.pitch.nbjw_weights_lines),
        device=device,
        kp_threshold=float(cfg.pitch.nbjw_kp_threshold),
        line_threshold=float(cfg.pitch.nbjw_line_threshold),
    )

    from soccer_ai.detector import build_tracker
    from soccer_ai.visualizer import build_annotators
    from soccer_ai.pipeline import process_frame
    from soccer_ai.calibration import PositionSmoother

    pitch_config = SoccerPitchConfiguration()
    annotators = build_annotators(cfg.annotate)
    last_transformer_ref = [None]
    pos_smoother = PositionSmoother(alpha=float(cfg.pitch.ema_alpha), max_age=int(cfg.pitch.pos_max_age))

    # --- Video source ---
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Source: {args.source}  {fw}x{fh} @ {src_fps:.1f} fps")

    tracker = None
    if cfg.track.enabled:
        tracker = build_tracker(cfg.track, frame_rate=max(1, int(round(src_fps))))
        tracker.reset()

    reid = None
    if cfg.reid.enabled:
        from soccer_ai.reid import PlayerReID
        reid = PlayerReID(
            weights=str(cwd / cfg.reid.weights),
            device=device,
            img_size=tuple(cfg.reid.img_size),
            threshold=float(cfg.reid.threshold),
            max_age=int(cfg.reid.max_age),
            update_interval=int(cfg.reid.update_interval),
            emb_ema=float(cfg.reid.emb_ema),
        )

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = sv.VideoSink(args.output, sv.VideoInfo(width=fw, height=fh, fps=src_fps))
        writer.__enter__()
        print(f"Saving to: {args.output}")

    fps_counter = FPSCounter()
    interval = args.interval
    frame_idx = 0
    paused = False
    screenshot_idx = 0
    out_frame = np.zeros((fh, fw, 3), dtype=np.uint8)  # blank until first frame

    print("\nControls: q/ESC=quit  p=pause  +/-=interval  s=screenshot\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            cfg.pitch.nbjw_interval = interval
            data = process_frame(
                frame=frame,
                player_model=player_model,
                tracker=tracker,
                annotators=annotators,
                pitch_config=pitch_config,
                cfg=cfg,
                device=device,
                frame_idx=frame_idx,
                last_transformer_ref=last_transformer_ref,
                pos_smoother=pos_smoother,
                calibrator=calibrator,
                reid=reid,
            )
            out_frame = data["frame"]
            fps = fps_counter.tick()
            frame_idx += 1
            if writer:
                writer.write_frame(out_frame)
        else:
            fps = 0.0

        display_frame = out_frame.copy()
        _draw_hud(display_frame, fps, interval, paused)

        if not args.no_display:
            cv2.imshow("Soccer AI — realtime", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("p"):
            paused = not paused
        elif key in (ord("+"), ord("=")) and interval < 60:
            interval += 1
            print(f"Interval -> {interval}")
        elif key == ord("-") and interval > 1:
            interval -= 1
            print(f"Interval -> {interval}")
        elif key == ord("s"):
            path = f"output/screenshot_{screenshot_idx:03d}.jpg"
            cv2.imwrite(path, display_frame)
            print(f"Screenshot -> {path}")
            screenshot_idx += 1

    cap.release()
    if writer:
        writer.__exit__(None, None, None)
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

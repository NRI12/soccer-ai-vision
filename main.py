"""Hydra entry point for soccer AI video analysis pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import hydra
import supervision as sv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

log = logging.getLogger(__name__)


def patch_aattn_compat(model: object) -> int:
    """Backfill missing AAttn attributes for old serialized checkpoints."""
    try:
        import ultralytics.nn.modules.block as block
    except Exception:
        return 0
    patched = 0
    for module in getattr(model, "modules", lambda: [])():
        if isinstance(module, block.AAttn) and not hasattr(module, "all_head_dim"):
            head_dim = getattr(module, "head_dim", None)
            num_heads = getattr(module, "num_heads", None)
            if head_dim is not None and num_heads is not None:
                module.all_head_dim = int(head_dim) * int(num_heads)
                patched += 1
    return patched


def _run_on_modal(source_path: str, output_path: str, cfg: DictConfig, orig_cwd: Path) -> None:
    """Delegate to Modal GPU via CLI subprocess."""
    import os, subprocess
    cmd = [
        sys.executable, "-m", "modal", "run", "modal_runner.py::main",
        f"--video-path={source_path}",
        f"--output-path={output_path}",
    ]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    if sys.platform == "win32":
        subprocess.run(["chcp", "65001"], shell=True, capture_output=True)
    log.info("Submitting to Modal GPU (building image on first run)...")
    if subprocess.run(cmd, cwd=str(orig_cwd), env=env).returncode != 0:
        raise RuntimeError("Modal run failed — check output above.")


def ensure_model_weights(orig_cwd: Path, cfg: DictConfig) -> None:
    from soccer_ai.download_data import RUNS_ZIP_URL, ensure_runs_weights
    if ensure_runs_weights(
        project_root=orig_cwd,
        required_paths=[orig_cwd / Path(cfg.models.player_detection.path)],
        zip_url=RUNS_ZIP_URL,
    ):
        log.info("Downloaded/extracted weights archive to research/runs")


def _resolve_device(cfg: DictConfig) -> str:
    device = str(cfg.models.device)
    if device in ("auto", "0"):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    orig_cwd = Path(hydra.utils.get_original_cwd())
    source_path = str(orig_cwd / cfg.video.source_path)
    output_path = str(orig_cwd / cfg.video.output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if str(cfg.models.device) == "modal":
        _run_on_modal(source_path, output_path, cfg, orig_cwd)
        return

    ensure_model_weights(orig_cwd, cfg)
    device = _resolve_device(cfg)

    # --- Models ---
    from ultralytics import YOLO
    player_model = YOLO(str(orig_cwd / cfg.models.player_detection.path))
    patched = patch_aattn_compat(player_model.model)
    if patched:
        log.warning("Patched %d AAttn layers on player model for compatibility", patched)

    calibrator = None
    if cfg.pitch.enabled:
        from soccer_ai.calibration import NBJWCalibrator
        calibrator = NBJWCalibrator(
            weights_kp=str(orig_cwd / cfg.pitch.get("nbjw_weights_kp", "weights/SV_kp")),
            weights_lines=str(orig_cwd / cfg.pitch.get("nbjw_weights_lines", "weights/SV_lines")),
            device=device,
            kp_threshold=float(cfg.pitch.get("nbjw_kp_threshold", 0.1486)),
            line_threshold=float(cfg.pitch.get("nbjw_line_threshold", 0.3880)),
        )

    # --- Pipeline objects ---
    from soccer_ai.pipeline import process_frame
    from soccer_ai.visualizer import build_annotators
    from soccer_ai.calibration import PositionSmoother
    from sports.configs.soccer import SoccerPitchConfiguration

    annotators = build_annotators(cfg.annotate) if cfg.annotate.enabled else None
    pitch_config = SoccerPitchConfiguration()

    last_transformer_ref = [None] if cfg.pitch.enabled else None
    pos_smoother = (
        PositionSmoother(alpha=cfg.pitch.get("ema_alpha", 0.4), max_age=cfg.pitch.get("pos_max_age", 30))
        if cfg.pitch.enabled else None
    )

    # --- Video info + tracker (built once with correct effective fps) ---
    from soccer_ai.detector import build_tracker
    from soccer_ai.stats import PlayerStatsTracker

    video_info = sv.VideoInfo.from_video_path(source_path)

    tracker = None
    if cfg.track.enabled:
        effective_fps = max(1, int(round(video_info.fps / cfg.video.stride)))
        tracker = build_tracker(cfg.track, frame_rate=effective_fps)
        tracker.reset()
        log.info("Tracking: sv.ByteTrack @ %d fps", effective_fps)

    player_stats_tracker = (
        PlayerStatsTracker(fps=video_info.fps, stride=cfg.video.stride)
        if cfg.player_stats.enabled else None
    )

    reid = None
    if cfg.reid.enabled:
        from soccer_ai.reid import PlayerReID
        reid = PlayerReID(
            weights=str(orig_cwd / cfg.reid.weights),
            device=device,
            img_size=tuple(cfg.reid.img_size),
            threshold=float(cfg.reid.threshold),
            max_age=int(cfg.reid.max_age),
            update_interval=int(cfg.reid.update_interval),
            emb_ema=float(cfg.reid.emb_ema),
            min_crop_h=int(cfg.reid.min_crop_h),
        )

    # --- Video loop ---
    frames = sv.get_video_frames_generator(source_path=source_path, stride=cfg.video.stride)
    total = video_info.total_frames
    if total and cfg.video.stride > 1:
        total = total // cfg.video.stride

    log.info("Processing: %s", source_path)
    with sv.VideoSink(target_path=output_path, video_info=video_info) as sink:
        for i, frame in enumerate(tqdm(frames, total=total, desc="processing")):
            data = process_frame(
                frame=frame,
                player_model=player_model,
                tracker=tracker,
                annotators=annotators,
                pitch_config=pitch_config,
                cfg=cfg,
                device=device,
                frame_idx=i,
                last_transformer_ref=last_transformer_ref,
                pos_smoother=pos_smoother,
                calibrator=calibrator,
                reid=reid,
            )
            if player_stats_tracker is not None:
                player_stats_tracker.update(data, i, frame)
            sink.write_frame(data["frame"])

    log.info("Output: %s", output_path)

    # --- Post-process: player stats ---
    if player_stats_tracker is not None:
        stats_output = str(orig_cwd / cfg.player_stats.output_dir)
        player_stats_tracker.export_all(output_dir=stats_output, pitch_config=pitch_config, cfg=cfg)
        if cfg.player_stats.get("export_videos", False):
            player_stats_tracker.export_player_videos(
                source_path=source_path, output_dir=stats_output,
                pitch_config=pitch_config, cfg=cfg,
            )



if __name__ == "__main__":
    main()

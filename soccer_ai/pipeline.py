"""Soccer AI pipeline orchestrator.

Runs all enabled stages on a single frame in order:
    1. detect       — YOLO → sv.Detections
    2. filter       — separate ball, NMS, drop staff, remap 8→3 classes
    3. track        — ByteTrack assigns persistent tracker IDs
    4. team         — split players/refs, remap to pipeline convention (0/1/2)
    5. calibrate    — NBJW homography → pitch coordinates
    6. annotate     — draw ellipses/labels, radar/Voronoi minimaps

All stages can be disabled via cfg.<stage>.enabled: false.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import supervision as sv
from omegaconf import DictConfig

from soccer_ai.detector import (
    detect,
    filter_detections,
    track,
    classify_teams,
    classify_teams_kmeans,
    KMeansTeamClassifier,
    _REFEREE_CLASS,
)
from soccer_ai.calibration import NBJWCalibrator, transform_detections, PositionSmoother
from soccer_ai.visualizer import build_annotators, annotate_frame
from soccer_ai.reid import PlayerReID

log = logging.getLogger(__name__)

FrameData = dict[str, Any]


def process_frame(
    frame: np.ndarray,
    player_model: Any,
    tracker: Any | None,
    annotators: dict[str, Any] | None,
    pitch_config: Any,
    cfg: DictConfig,
    device: str = "cpu",
    frame_idx: int = 0,
    last_transformer_ref: list | None = None,
    pos_smoother: PositionSmoother | None = None,
    calibrator: NBJWCalibrator | None = None,
    reid: PlayerReID | None = None,
    team_classifier: KMeansTeamClassifier | None = None,
) -> FrameData:
    """Run all enabled pipeline stages on a single frame."""
    if not cfg.detect.enabled:
        return {"frame": frame}

    data = detect(frame, player_model, cfg.detect, device)

    if cfg.filter.enabled:
        data = filter_detections(data, cfg)

    if cfg.track.enabled and tracker is not None:
        if not cfg.filter.enabled:
            log.warning("Track requires filter stage. Skipping.")
        else:
            data = track(data, tracker)

    if reid is not None and "all_detections" in data:
        data["all_detections"] = reid.update(frame, data["all_detections"])

    if cfg.team.enabled:
        if not cfg.filter.enabled:
            log.warning("Team classification requires filter stage. Skipping.")
        elif getattr(cfg.team, "mode", "from_model") == "from_color" and team_classifier is not None:
            data = classify_teams_kmeans(data, team_classifier)
        else:
            data = classify_teams(data)
    elif cfg.filter.enabled and "all_detections" in data:
        all_det = data["all_detections"]
        data["players_detections"] = all_det[all_det.class_id != _REFEREE_CLASS]
        data["referees_detections"] = all_det[all_det.class_id == _REFEREE_CLASS]

    nbjw_interval = int(cfg.pitch.get("nbjw_interval", 1))
    if cfg.pitch.enabled and calibrator is not None:
        if frame_idx % nbjw_interval == 0:
            transformer = calibrator.get_transformer(frame)
        else:
            transformer = None

        if transformer is not None:
            if last_transformer_ref is not None:
                last_transformer_ref[0] = transformer
            data["transformer"] = transformer
            data = transform_detections(data, transformer)
        elif last_transformer_ref is not None and last_transformer_ref[0] is not None:
            data = transform_detections(data, last_transformer_ref[0])

        if pos_smoother is not None:
            if "pitch_players_xy" in data:
                data["pitch_players_xy"] = pos_smoother.smooth_players(
                    data["pitch_players_xy"],
                    data.get("pitch_players_tracker_id"),
                )
            if "pitch_ball_xy" in data:
                data["pitch_ball_xy"] = pos_smoother.smooth_ball(data["pitch_ball_xy"])

    if cfg.annotate.enabled and annotators is not None:
        data = annotate_frame(data, annotators, pitch_config, cfg)

    return data

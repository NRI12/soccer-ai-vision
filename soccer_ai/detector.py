"""Detection, filtering, tracking, and team classification.

Pipeline order
--------------
detect()             YOLO → sv.Detections
filter_detections()  separate ball, NMS, drop staff, remap 8→3 classes
track()              ByteTrack → tracker IDs
classify_teams()     split players / refs, remap to pipeline convention (0/1/2)

Class schema
------------
After filter_detections():
    1 = team_left   (players + goalkeepers left)
    2 = team_right  (players + goalkeepers right)
    3 = referee     (main + side)
    Staff (7) are dropped.

After classify_teams():
    0 = team_left
    1 = team_right
    2 = referee
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import supervision as sv
from omegaconf import DictConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect(
    frame: np.ndarray,
    model: Any,
    cfg: DictConfig,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run YOLO on a single frame."""
    result = model(frame, conf=cfg.confidence, device=device, verbose=False)[0]
    return {"frame": frame, "detections": sv.Detections.from_ultralytics(result)}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_detections(data: dict[str, Any], cfg: DictConfig) -> dict[str, Any]:
    """Separate ball, apply NMS to non-ball detections, remap 8→3 classes.

    Remapping before ByteTrack is critical: sv.ByteTrack is class-aware and
    only matches within the same class_id.  Without this remap, a goalkeeper
    oscillating between class 1 (player_left) and class 3 (goalkeeper_left)
    would receive a new tracker ID on every class switch.
    """
    detections = data["detections"]
    ids = cfg.class_ids
    ball_id = int(ids.ball)

    ball_detections = detections[detections.class_id == ball_id]
    ball_detections.xyxy = sv.pad_boxes(
        xyxy=ball_detections.xyxy, px=cfg.filter.ball_pad_px
    )

    # Drop staff before NMS so they don't suppress real players
    all_detections = detections[detections.class_id != ball_id]
    all_detections = all_detections[all_detections.class_id != int(ids.staff)]
    all_detections = all_detections.with_nms(
        threshold=cfg.filter.nms_threshold,
        class_agnostic=cfg.filter.nms_class_agnostic,
    )

    _remap = {
        int(ids.player_left):      1,
        int(ids.goalkeeper_left):  1,
        int(ids.player_right):     2,
        int(ids.goalkeeper_right): 2,
        int(ids.main_referee):     3,
        int(ids.side_referee):     3,
    }
    cid = all_detections.class_id.astype(int)
    all_detections.class_id = np.vectorize(lambda c: _remap.get(c, c))(cid).astype(int)

    data["ball_detections"] = ball_detections
    data["all_detections"] = all_detections
    return data


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

_TrackerCls = getattr(sv, "ByteTracker", sv.ByteTrack)


def build_tracker(cfg: DictConfig, frame_rate: int | None = None) -> sv.ByteTrack:
    """Build a ByteTracker instance from config.

    Pass frame_rate=effective_fps (video_fps / stride) so the Kalman filter
    and lost_track_buffer are calibrated to the actual inter-frame interval.
    Uses sv.ByteTracker (supervision >= 0.28) with fallback to sv.ByteTrack.
    """
    return _TrackerCls(
        track_activation_threshold=cfg.track_activation_threshold,
        lost_track_buffer=cfg.lost_track_buffer,
        minimum_matching_threshold=cfg.minimum_matching_threshold,
        frame_rate=frame_rate if frame_rate is not None else cfg.frame_rate,
        minimum_consecutive_frames=cfg.minimum_consecutive_frames,
    )


def track(data: dict[str, Any], tracker: sv.ByteTrack) -> dict[str, Any]:
    """Update ByteTrack with current all_detections."""
    all_detections = tracker.update_with_detections(detections=data["all_detections"])
    data["all_detections"] = all_detections
    data["labels"] = [f"#{tid}" for tid in all_detections.tracker_id]
    return data


# ---------------------------------------------------------------------------
# Team classification
# ---------------------------------------------------------------------------

# Remapped class IDs: referee after filter stage
_REFEREE_CLASS = 3


def classify_teams(data: dict[str, Any]) -> dict[str, Any]:
    """Split all_detections into players / referees and remap class_id.

    Input class_ids from filter stage: 1=team_left, 2=team_right, 3=referee.
    Output pipeline convention: 0=team_left, 1=team_right, 2=referee.
    """
    all_detections = data["all_detections"]
    cid = all_detections.class_id.astype(int)

    players_mask = (cid == 1) | (cid == 2)
    ref_mask = cid == _REFEREE_CLASS

    players_detections = all_detections[players_mask]
    referees_detections = all_detections[ref_mask]

    def _remap(dets: sv.Detections) -> sv.Detections:
        c = dets.class_id.astype(int)
        dets.class_id = np.where(c == 2, 1, np.where(c == 3, 2, 0)).astype(int)
        return dets

    players_detections = _remap(players_detections)
    referees_detections = _remap(referees_detections)

    merged = sv.Detections.merge([players_detections, referees_detections])
    merged.class_id = merged.class_id.astype(int)

    if merged.tracker_id is not None:
        data["labels"] = [str(tid) for tid in merged.tracker_id]

    data["all_detections"] = merged
    data["players_detections"] = players_detections
    data["referees_detections"] = referees_detections
    return data

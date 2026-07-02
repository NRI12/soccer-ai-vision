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
from collections import defaultdict
from typing import Any

import cv2
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


class KMeansTeamClassifier:
    """Phân loại team cầu thủ bằng màu áo (KMeans k=2 trên không gian LAB).

    Giai đoạn warmup: tích lũy đặc trưng màu từ crop vùng áo.
    Sau warmup: fit KMeans một lần, sau đó dùng tracker_id để giữ nhãn
    ổn định; track mới được gán về cluster center gần nhất.
    """

    _MAX_BUFFER = 4000

    def __init__(self, warmup_frames: int = 60, min_crop_h: int = 30) -> None:
        self.warmup_frames = warmup_frames
        self.min_crop_h = min_crop_h
        self._buffer: list[tuple[int, np.ndarray]] = []
        self._tracker_teams: dict[int, int] = {}
        self._centers: np.ndarray | None = None  # shape (2, 3) LAB
        self._fitted = False

    def _jersey_feature(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray | None:
        h, w = y2 - y1, x2 - x1
        if h < self.min_crop_h or w < 10:
            return None
        yt = y1 + int(h * 0.15)
        yb = y1 + int(h * 0.55)
        xl = x1 + int(w * 0.20)
        xr = x1 + int(w * 0.80)
        crop = frame[yt:yb, xl:xr]
        if crop.size == 0:
            return None
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        return lab.reshape(-1, 3).mean(axis=0).astype(np.float32)

    def _fit(self) -> None:
        from sklearn.cluster import KMeans
        feats = np.stack([f for _, f in self._buffer])
        tids = [t for t, _ in self._buffer]
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(feats)
        self._centers = km.cluster_centers_
        votes: dict[int, list[int]] = defaultdict(list)
        for tid, lbl in zip(tids, km.labels_):
            votes[tid].append(int(lbl))
        self._tracker_teams = {
            tid: int(np.bincount(np.array(vs)).argmax())
            for tid, vs in votes.items()
        }
        self._fitted = True
        log.info(
            "KMeansTeamClassifier fitted: %d samples, %d trackers, centres=%s",
            len(self._buffer),
            len(self._tracker_teams),
            self._centers.round(1).tolist(),
        )

    def classify(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Trả về mảng nhãn team (0 hoặc 1) tương ứng với mỗi detection."""
        n = len(detections)
        if n == 0:
            return np.zeros(0, dtype=int)

        tids = (
            detections.tracker_id.tolist()
            if detections.tracker_id is not None
            else [None] * n
        )

        # Tích lũy đặc trưng
        for i, (xyxy, tid) in enumerate(zip(detections.xyxy, tids)):
            if len(self._buffer) >= self._MAX_BUFFER:
                break
            x1, y1, x2, y2 = map(int, xyxy)
            feat = self._jersey_feature(frame, x1, y1, x2, y2)
            if feat is not None:
                self._buffer.append((tid if tid is not None else -(i + 1), feat))

        if not self._fitted and len(self._buffer) >= max(10, self.warmup_frames // 2):
            self._fit()

        teams = np.zeros(n, dtype=int)
        if not self._fitted:
            return teams

        for i, (xyxy, tid) in enumerate(zip(detections.xyxy, tids)):
            if tid is not None and tid in self._tracker_teams:
                teams[i] = self._tracker_teams[tid]
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            feat = self._jersey_feature(frame, x1, y1, x2, y2)
            if feat is not None:
                teams[i] = int(np.argmin(np.linalg.norm(self._centers - feat, axis=1)))
                if tid is not None:
                    self._tracker_teams[tid] = teams[i]

        return teams


def classify_teams_kmeans(
    data: dict[str, Any], classifier: KMeansTeamClassifier
) -> dict[str, Any]:
    """Phân loại team bằng KMeans màu áo; referees vẫn giữ class_id=2."""
    frame = data["frame"]
    all_detections = data["all_detections"]
    cid = all_detections.class_id.astype(int)

    players_mask = (cid == 1) | (cid == 2)
    ref_mask = cid == _REFEREE_CLASS

    players_detections = all_detections[players_mask]
    referees_detections = all_detections[ref_mask]

    if len(players_detections) > 0:
        players_detections.class_id = classifier.classify(frame, players_detections)

    if len(referees_detections) > 0:
        referees_detections.class_id = np.full(len(referees_detections), 2, dtype=int)

    merged = sv.Detections.merge([players_detections, referees_detections])
    merged.class_id = merged.class_id.astype(int)

    if merged.tracker_id is not None:
        data["labels"] = [str(tid) for tid in merged.tracker_id]

    data["all_detections"] = merged
    data["players_detections"] = players_detections
    data["referees_detections"] = referees_detections
    return data


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

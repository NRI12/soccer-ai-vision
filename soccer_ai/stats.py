"""Per-player tracking statistics and highlight video export."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# SoccerPitchConfiguration: 12000 units = 105 m, 7000 units = 68 m
_UNIT_TO_M = np.array([105.0 / 12000.0, 68.0 / 7000.0])
_MAX_SPEED_MS = 10.0       # clamp unrealistic speed spikes (elite sprint max)
_SMOOTH_WINDOW = 9         # rolling-average window to suppress homography jitter
_MAX_CROPS = 8             # max player crops stored in RAM per player
_CROP_INTERVAL = 60        # store a crop every N frames (~2 s at 30 fps)


class PlayerStatsTracker:
    """Accumulates pitch trajectories and crops for every tracked player.

    Call ``update()`` once per frame after the pitch-transform stage.
    Call ``export_all()`` once after the video loop.
    """

    def __init__(self, fps: float, stride: int = 1) -> None:
        self.fps = fps
        self.stride = stride
        self._positions: dict[int, list[dict]] = {}
        self._crops: dict[int, list[np.ndarray]] = {}
        self._next_crop_frame: dict[int, int] = {}
        # {tracker_id: {team_id: vote_count}} — majority vote across frames
        self._team_votes: dict[int, dict[int, int]] = {}
        # {frame_idx: [{tid, xyxy, team, pitch_xy}]} — used for per-player video export
        self._frame_players: dict[int, list[dict]] = {}

    # ------------------------------------------------------------------
    def update(self, data: dict[str, Any], frame_idx: int, original_frame: np.ndarray) -> None:
        """Accumulate tracking data for this frame."""
        pitch_xy = data.get("pitch_players_xy")
        tracker_ids = data.get("pitch_players_tracker_id")
        class_ids = data.get("players_class_id")
        players_det = data.get("players_detections")

        if pitch_xy is None or tracker_ids is None or len(pitch_xy) == 0:
            return

        frame_entries: list[dict] = []

        for i in range(len(pitch_xy)):
            raw_tid = tracker_ids[i]
            if raw_tid is None:
                continue
            tid = int(raw_tid)
            team = int(class_ids[i]) if class_ids is not None and i < len(class_ids) else 0
            if team not in (0, 1):  # skip referees and staff
                continue

            xy = pitch_xy[i].copy()
            if tid not in self._positions:
                self._positions[tid] = []
                self._crops[tid] = []
                self._next_crop_frame[tid] = 0
                self._team_votes[tid] = {}

            self._positions[tid].append({"frame_idx": frame_idx, "pitch_xy": xy})
            self._team_votes[tid][team] = self._team_votes[tid].get(team, 0) + 1

            entry: dict = {"tid": tid, "team": team, "pitch_xy": xy}
            if players_det is not None and i < len(players_det.xyxy):
                entry["xyxy"] = players_det.xyxy[i]
            frame_entries.append(entry)

            # Store player crop at regular intervals
            if (
                players_det is not None
                and i < len(players_det.xyxy)
                and frame_idx >= self._next_crop_frame[tid]
                and len(self._crops[tid]) < _MAX_CROPS
            ):
                try:
                    x1, y1, x2, y2 = map(int, players_det.xyxy[i])
                    fh, fw = original_frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(fw, x2), min(fh, y2)
                    if x2 > x1 and y2 > y1:
                        self._crops[tid].append(original_frame[y1:y2, x1:x2].copy())
                        self._next_crop_frame[tid] = frame_idx + _CROP_INTERVAL
                except Exception:
                    pass

        if frame_entries:
            self._frame_players[frame_idx] = frame_entries

    # ------------------------------------------------------------------
    def compute_stats(self) -> dict[int, dict]:
        """Return per-player stats dict keyed by tracker_id."""
        stats: dict[int, dict] = {}
        for tid, frames in self._positions.items():
            votes = self._team_votes.get(tid, {})
            team = max(votes, key=votes.get) if votes else -1
            base = {
                "tracker_id": tid, "team": team, "num_frames": len(frames),
                "total_distance_m": 0.0,
                "max_speed_ms": 0.0, "avg_speed_ms": 0.0,
                "max_speed_kmh": 0.0, "avg_speed_kmh": 0.0,
            }
            if len(frames) < 2:
                stats[tid] = base
                continue

            raw_m = np.array([f["pitch_xy"] for f in frames]) * _UNIT_TO_M
            frame_idx_arr = np.array([f["frame_idx"] for f in frames], dtype=float)

            positions_m = _rolling_mean(raw_m, _SMOOTH_WINDOW)
            dt_arr = np.diff(frame_idx_arr) / self.fps
            dt_arr = np.where(dt_arr > 0, dt_arr, 1.0 / self.fps)

            diffs = np.diff(positions_m, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            max_dist_per_step = _MAX_SPEED_MS * dt_arr
            valid = dists <= max_dist_per_step

            velocities = np.where(valid, dists / dt_arr, 0.0)
            valid_v = velocities[valid]

            base.update({
                "total_distance_m": round(float(np.sum(dists[valid])), 1),
                "max_speed_ms": round(float(np.max(valid_v)) if len(valid_v) else 0.0, 2),
                "avg_speed_ms": round(float(np.mean(valid_v)) if len(valid_v) else 0.0, 2),
                "max_speed_kmh": round(float(np.max(valid_v) * 3.6) if len(valid_v) else 0.0, 1),
                "avg_speed_kmh": round(float(np.mean(valid_v) * 3.6) if len(valid_v) else 0.0, 1),
            })
            stats[tid] = base
        return stats

    # ------------------------------------------------------------------
    def export_player_videos(
        self,
        source_path: str | Path,
        output_dir: str | Path,
        pitch_config: Any,
        cfg: DictConfig,
    ) -> None:
        """Export one highlight video per tracked player (single source pass)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = self.compute_stats()
        min_frames = int(cfg.player_stats.get("min_frames_for_video", 30))
        eligible = {tid: s for tid, s in stats.items() if s["num_frames"] >= min_frames}
        if not eligible:
            log.warning("No players meet min_frames_for_video=%d; skipping video export", min_frames)
            return

        for old_file in output_dir.glob("*_highlight.mp4"):
            try:
                old_file.unlink(missing_ok=True)
            except PermissionError:
                log.warning("Cannot delete locked file %s; will attempt to overwrite", old_file.name)

        colors = cfg.annotate.colors
        team_hex = {0: colors.team_1, 1: colors.team_2, 2: colors.referee}
        target_bgrs: dict[int, tuple[int, int, int]] = {
            tid: _hex_to_bgr(team_hex.get(s["team"], "#FFFFFF"))
            for tid, s in eligible.items()
        }

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            log.error("Cannot open source video: %s", source_path)
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writers: dict[int, cv2.VideoWriter] = {}
        for tid, s in eligible.items():
            path = str(output_dir / f"player_{tid:04d}_team{s['team']}_highlight.mp4")
            w = cv2.VideoWriter(path, fourcc, fps, (fw, fh))
            if w.isOpened():
                writers[tid] = w
            else:
                log.warning("Cannot write highlight for player %d (file locked?); skipping", tid)

        frame_real, frame_strided = -1, -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_real += 1
            if frame_real % self.stride != 0:
                continue
            frame_strided += 1

            entries = self._frame_players.get(frame_strided)
            for tid, writer in writers.items():
                writer.write(_render_highlight_frame(
                    frame=frame, target_tid=tid,
                    target_bgr=target_bgrs[tid],
                    entries=entries, pitch_config=pitch_config,
                ))

        cap.release()
        for writer in writers.values():
            writer.release()

        log.info("Per-player highlight videos -> %s (%d players)", output_dir, len(writers))

    # ------------------------------------------------------------------
    def export_all(self, output_dir: str | Path, pitch_config: Any, cfg: DictConfig) -> None:
        """Export per-player overlay PNGs and a summary JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = self.compute_stats()

        json_path = output_dir / "player_stats.json"
        json_path.write_text(
            json.dumps(list(stats.values()), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Player stats -> %s", json_path)

        colors = cfg.annotate.colors
        team_hex = {0: colors.team_1, 1: colors.team_2, 2: colors.referee}

        for tid, s in stats.items():
            frames = self._positions.get(tid, [])
            if len(frames) < 2:
                continue
            img = _render_player_overlay(
                tid=tid, positions=frames, stats=s,
                crops=self._crops.get(tid, []),
                pitch_config=pitch_config, team_hex=team_hex,
            )
            cv2.imwrite(str(output_dir / f"player_{tid:04d}_team{s['team']}.png"), img)

        log.info("Per-player overlays -> %s (%d players)", output_dir, len(stats))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    if n < window:
        return arr
    half = window // 2
    out = np.empty_like(arr, dtype=float)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = arr[lo:hi].mean(axis=0)
    return out


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _pitch_scale(pitch_config: Any, canvas_shape: tuple[int, int]) -> tuple[float, float, int]:
    canvas_h, canvas_w = canvas_shape
    p = 50
    sx = (canvas_w - 2 * p) / getattr(pitch_config, "length", 12000)
    sy = (canvas_h - 2 * p) / getattr(pitch_config, "width", 7000)
    return sx, sy, p


def _to_pixel(x: float, y: float, sx: float, sy: float, p: int) -> tuple[int, int]:
    return int(x * sx) + p, int(y * sy) + p


def _render_player_overlay(
    tid: int,
    positions: list[dict],
    stats: dict,
    crops: list[np.ndarray],
    pitch_config: Any,
    team_hex: dict[int, str],
) -> np.ndarray:
    from sports.annotators.soccer import draw_pitch

    pitch_img = draw_pitch(pitch_config)
    sx, sy, p = _pitch_scale(pitch_config, pitch_img.shape[:2])
    team = stats.get("team", 0)
    bgr = _hex_to_bgr(team_hex.get(team, "#FFFFFF"))

    raw_cm = np.array([f["pitch_xy"] for f in positions])
    smooth_cm = _rolling_mean(raw_cm, _SMOOTH_WINDOW)
    pts = np.array([_to_pixel(xy[0], xy[1], sx, sy, p) for xy in smooth_cm])

    n = len(pts)
    for i in range(1, n):
        alpha = 0.25 + 0.75 * (i / n)
        color = tuple(int(c * alpha) for c in bgr)
        cv2.line(pitch_img, tuple(pts[i - 1]), tuple(pts[i]), color, 3, cv2.LINE_AA)

    if n > 0:
        cv2.circle(pitch_img, tuple(pts[0]), 7, (0, 220, 0), -1, cv2.LINE_AA)
        cv2.circle(pitch_img, tuple(pts[-1]), 10, bgr, -1, cv2.LINE_AA)
        cv2.circle(pitch_img, tuple(pts[-1]), 10, (255, 255, 255), 2, cv2.LINE_AA)
        lx, ly = pts[-1]
        for col, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(pitch_img, f"#{tid}", (lx + 12, ly - 4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, col, thick, cv2.LINE_AA)

    canvas_w = pitch_img.shape[1]
    panel = np.zeros((130, canvas_w, 3), dtype=np.uint8)
    team_label = {0: "Team A", 1: "Team B", 2: "Referee"}.get(team, f"Team {team}")
    for j, (text, brightness) in enumerate([
        (f"Player #{tid}  |  {team_label}", 1.0),
        (f"Distance:  {stats['total_distance_m']:.1f} m", 0.85),
        (f"Max speed: {stats['max_speed_kmh']:.1f} km/h    Avg: {stats['avg_speed_kmh']:.1f} km/h", 0.85),
    ]):
        c = int(255 * brightness)
        cv2.putText(panel, text, (12, 30 + j * 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (c, c, c), 1, cv2.LINE_AA)

    overlay = np.vstack([pitch_img, panel])
    if crops:
        collage = _build_crop_collage(crops, overlay.shape[0], max(160, canvas_w // 5))
        overlay = np.hstack([overlay, collage])

    return overlay


def _draw_ellipse_bottom(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
    color: tuple[int, int, int], thickness: int,
) -> None:
    cx, cy = (x1 + x2) // 2, y2
    a = max(4, (x2 - x1) // 2)
    cv2.ellipse(img, (cx, cy), (a, max(3, a // 4)), 0, -180, 0, color, thickness, cv2.LINE_AA)


def _draw_player_minimap(pitch_config: Any, xy: np.ndarray | None, bgr: tuple[int, int, int]) -> np.ndarray:
    from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
    import supervision as sv

    canvas = draw_pitch(pitch_config)
    if xy is not None:
        b, g, r = bgr
        canvas = draw_points_on_pitch(
            config=pitch_config, xy=np.array([xy]),
            face_color=sv.Color(r=int(r), g=int(g), b=int(b)),
            edge_color=sv.Color.WHITE, radius=10, thickness=2, pitch=canvas,
        )
    return canvas


def _overlay_minimap_no_label(frame: np.ndarray, minimap: np.ndarray,
                               scale: float = 0.36, margin: int = 12) -> np.ndarray:
    fh, fw = frame.shape[:2]
    mw = int(fw * scale)
    mh = int(minimap.shape[0] * (mw / minimap.shape[1]))
    m = cv2.resize(minimap, (mw, mh), interpolation=cv2.INTER_LANCZOS4)
    border = 3
    m = cv2.copyMakeBorder(m, border, border, border, border, cv2.BORDER_CONSTANT, value=(220, 220, 220))
    mh, mw = m.shape[:2]

    y, x = max(0, fh - mh - margin), max(0, fw - mw - margin)
    ye, xe = min(fh, y + mh), min(fw, x + mw)

    sp = 6
    frame[max(0, y-sp):min(fh, ye+sp), max(0, x-sp):min(fw, xe+sp)] = (
        frame[max(0, y-sp):min(fh, ye+sp), max(0, x-sp):min(fw, xe+sp)] * 0.45
    ).astype(np.uint8)
    frame[y:ye, x:xe] = m[:ye - y, :xe - x]
    return frame


def _render_highlight_frame(
    frame: np.ndarray, target_tid: int,
    target_bgr: tuple[int, int, int],
    entries: list[dict] | None, pitch_config: Any,
) -> np.ndarray:
    out = frame.copy()
    target_xy: np.ndarray | None = None

    if entries:
        for entry in entries:
            xyxy = entry.get("xyxy")
            if xyxy is not None:
                _draw_ellipse_bottom(out, *map(int, xyxy), (55, 55, 55), 2)

        for entry in entries:
            if entry["tid"] != target_tid:
                continue
            target_xy = entry.get("pitch_xy")
            xyxy = entry.get("xyxy")
            if xyxy is not None:
                x1, y1, x2, y2 = map(int, xyxy)
                _draw_ellipse_bottom(out, x1, y1, x2, y2, target_bgr, 4)
                _draw_ellipse_bottom(out, x1, y1, x2, y2, (255, 255, 255), 2)
                lx = (x1 + x2) // 2
                cv2.putText(out, f"#{target_tid}", (lx - 20, y1 - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, f"#{target_tid}", (lx - 20, y1 - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            break

    minimap = _draw_player_minimap(pitch_config, target_xy, target_bgr)
    return _overlay_minimap_no_label(out, minimap)


def _build_crop_collage(crops: list[np.ndarray], target_h: int, target_w: int) -> np.ndarray:
    n = len(crops)
    if n == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    cell_h = target_h // n
    panels: list[np.ndarray] = []
    for crop in crops:
        ch, cw = crop.shape[:2]
        if ch == 0 or cw == 0:
            panels.append(np.zeros((cell_h, target_w, 3), dtype=np.uint8))
            continue
        scale = min(cell_h / ch, target_w / cw)
        nh, nw = int(ch * scale), int(cw * scale)
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        cell = np.zeros((cell_h, target_w, 3), dtype=np.uint8)
        y0, x0 = (cell_h - nh) // 2, (target_w - nw) // 2
        cell[y0:y0 + nh, x0:x0 + nw] = resized
        panels.append(cell)

    collage = np.vstack(panels)
    if collage.shape[0] < target_h:
        collage = np.vstack([collage, np.zeros((target_h - collage.shape[0], target_w, 3), dtype=np.uint8)])
    return collage[:target_h]

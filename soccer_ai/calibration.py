"""Pitch calibration: NBJW camera models, homography, coordinate transforms, EMA smoothing.

Coordinate chain
----------------
NBJW → cam_params → 3×4 projection matrix P
    → H_sn (pitch SN-meters center-origin → image px)
    → invert → image px → SN meters
    → _SN_TO_SPORTS affine → sports canvas (0–12000, 0–7000 units, TL origin)

Unit conversion (SoccerPitchConfiguration):
    x_m = x * 105 / 12000
    y_m = y * 68  / 7000
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv

log = logging.getLogger(__name__)

# SN world: center origin, x∈[-52.5, 52.5] m, y∈[-34, 34] m
# Sports canvas: top-left origin, 12000 × 7000 units
_SN_TO_SPORTS = np.array([
    [12000.0 / 105.0, 0.0,              52.5 * 12000.0 / 105.0],
    [0.0,             7000.0 / 68.0,    34.0 *  7000.0 /  68.0],
    [0.0,             0.0,              1.0                     ],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Homography wrapper
# ---------------------------------------------------------------------------

class _SimpleTransformer:
    """ViewTransformer-compatible wrapper around a 3×3 homography matrix."""

    def __init__(self, m: np.ndarray) -> None:
        self.m = m.astype(np.float64)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points
        # cv2.perspectiveTransform requires src dtype == matrix dtype
        pts = points.reshape(-1, 1, 2).astype(np.float64)
        return cv2.perspectiveTransform(pts, self.m).reshape(-1, 2).astype(np.float32)


# ---------------------------------------------------------------------------
# NBJW calibrator
# ---------------------------------------------------------------------------

_NBJW_CONFIG_DIR = Path(__file__).parent / "nbjw" / "config"


class NBJWCalibrator:
    """Camera calibration via NBJW dual-HRNet (keypoints + lines).

    Original repo: https://github.com/mguti97/No-Bells-Just-Whistles
    Weights: weights/SV_kp and weights/SV_lines (~252 MB each).
    """

    def __init__(
        self,
        weights_kp: str,
        weights_lines: str,
        device: str = "cpu",
        kp_threshold: float = 0.1486,
        line_threshold: float = 0.3880,
    ) -> None:
        import torch
        import yaml
        import torchvision.transforms as T

        from soccer_ai.nbjw.cls_hrnet import get_cls_net
        from soccer_ai.nbjw.cls_hrnet_l import get_cls_net as get_cls_net_l
        from soccer_ai.nbjw.utils_calib import FramebyFrameCalib
        from soccer_ai.nbjw.utils_heatmap import (
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
            complete_keypoints,
            coords_to_dict,
        )

        cfg_kp   = yaml.safe_load((_NBJW_CONFIG_DIR / "hrnetv2_w48.yaml").read_text())
        cfg_line = yaml.safe_load((_NBJW_CONFIG_DIR / "hrnetv2_w48_l.yaml").read_text())

        model_kp = get_cls_net(cfg_kp)
        model_kp.load_state_dict(torch.load(weights_kp, map_location=device, weights_only=False))
        model_kp.to(device).eval()

        model_line = get_cls_net_l(cfg_line)
        model_line.load_state_dict(torch.load(weights_lines, map_location=device, weights_only=False))
        model_line.to(device).eval()

        self._model_kp    = model_kp
        self._model_line  = model_line
        self._FrameCalib  = FramebyFrameCalib
        self._get_kp      = get_keypoints_from_heatmap_batch_maxpool
        self._get_lines   = get_keypoints_from_heatmap_batch_maxpool_l
        self._complete_kp = complete_keypoints
        self._coords_dict = coords_to_dict
        self._resize      = T.Resize((540, 960))
        self._device      = device
        self._torch       = torch
        self._kp_thresh   = kp_threshold
        self._line_thresh = line_threshold
        self._cam         = None
        self._cam_size    = (-1, -1)

        log.info("NBJWCalibrator ready (device=%s)", device)

    def _infer(self, frame: np.ndarray):
        import torchvision.transforms.functional as tvf
        from PIL import Image as PILImage

        fh, fw = frame.shape[:2]
        if self._cam is None or self._cam_size != (fw, fh):
            self._cam = self._FrameCalib(iwidth=fw, iheight=fh, denormalize=True)
            self._cam_size = (fw, fh)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = tvf.to_tensor(PILImage.fromarray(rgb)).float().unsqueeze(0)
        if t.size()[-1] != 960:
            t = self._resize(t)
        t = t.to(self._device)
        _, _, h, w = t.size()

        with self._torch.no_grad():
            heatmaps   = self._model_kp(t)
            heatmaps_l = self._model_line(t)

        kp    = self._get_kp(heatmaps[:, :-1, :, :])
        lines = self._get_lines(heatmaps_l[:, :-1, :, :])
        kd    = self._coords_dict(kp,    threshold=self._kp_thresh)
        ld    = self._coords_dict(lines, threshold=self._line_thresh)
        final = self._complete_kp(kd, ld, w=w, h=h, normalize=True)

        self._cam.update(final[0])
        return self._cam.heuristic_voting()

    def get_transformer(self, frame: np.ndarray) -> _SimpleTransformer | None:
        """Run NBJW on a frame; return image→sports-canvas transformer or None."""
        try:
            result = self._infer(frame)
        except Exception as exc:
            log.debug("NBJW infer error: %s", exc)
            return None

        if result is None:
            return None

        try:
            p = result["cam_params"]
            K = np.array([
                [p["x_focal_length"], 0,                   p["principal_point"][0]],
                [0,                   p["y_focal_length"],  p["principal_point"][1]],
                [0,                   0,                    1                      ],
            ], dtype=np.float64)
            pos = np.array(p["position_meters"])
            rot = np.array(p["rotation_matrix"])
            It  = np.eye(4)[:-1]
            It[:, -1] = -pos
            P = K @ (rot @ It)

            H_sn = P[:, [0, 1, 3]].astype(np.float64)
            if abs(H_sn[2, 2]) < 1e-9:
                return None
            H_sn /= H_sn[2, 2]

            ok, H_inv = cv2.invert(H_sn)
            if not ok:
                return None

            H_sports = _SN_TO_SPORTS @ H_inv
        except Exception as exc:
            log.debug("Homography error: %s", exc)
            return None

        fh, fw = frame.shape[:2]
        probe  = np.array([[[fw / 2.0, fh / 2.0]]], dtype=np.float64)
        tx, ty = cv2.perspectiveTransform(probe, H_sports)[0, 0]
        if not (-2000 <= tx <= 14000 and -2000 <= ty <= 9000):
            log.debug("Sanity fail: center → (%.1f, %.1f)", tx, ty)
            return None

        return _SimpleTransformer(H_sports)


# ---------------------------------------------------------------------------
# Pitch projection
# ---------------------------------------------------------------------------

def transform_detections(data: dict[str, Any], transformer: _SimpleTransformer) -> dict[str, Any]:
    """Project BOTTOM_CENTER foot-points of all detections onto the pitch canvas."""
    ball = data.get("ball_detections")
    if ball is not None and len(ball) > 0:
        data["pitch_ball_xy"] = transformer.transform_points(
            ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        )

    players = data.get("players_detections")
    if players is not None and len(players) > 0:
        data["pitch_players_xy"]        = transformer.transform_points(
            players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        )
        data["players_class_id"]         = players.class_id
        data["pitch_players_tracker_id"] = players.tracker_id

    refs = data.get("referees_detections")
    if refs is not None and len(refs) > 0:
        data["pitch_referees_xy"] = transformer.transform_points(
            refs.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        )

    return data


# ---------------------------------------------------------------------------
# Position smoother
# ---------------------------------------------------------------------------

class PositionSmoother:
    """Per-entity EMA smoothing on 2-D pitch coordinates.

    Smooths in output (R²) space keyed by tracker_id, preventing
    homography jitter from causing position oscillation on the minimap.
    """

    def __init__(self, alpha: float = 0.4, max_age: int = 30) -> None:
        self.alpha   = alpha   # weight of new observation (higher = more responsive)
        self.max_age = max_age
        self._players: dict[int, dict] = {}
        self._ball: np.ndarray | None  = None

    def smooth_players(
        self,
        pitch_xy: np.ndarray,
        tracker_ids: np.ndarray | None,
    ) -> np.ndarray:
        if pitch_xy is None or len(pitch_xy) == 0:
            self._expire(set())
            return pitch_xy

        seen: set[int] = set()
        out = pitch_xy.copy().astype(np.float32)

        for i, raw in enumerate(tracker_ids if tracker_ids is not None else []):
            if raw is None:
                continue
            tid = int(raw)
            seen.add(tid)
            obs = pitch_xy[i].astype(np.float32)
            if tid in self._players:
                prev = self._players[tid]["pos"]
                pos  = self.alpha * obs + (1.0 - self.alpha) * prev
            else:
                pos = obs
            self._players[tid] = {"pos": pos, "age": 0}
            out[i] = pos

        self._expire(seen)
        return out

    def smooth_ball(self, pitch_ball_xy: np.ndarray | None) -> np.ndarray | None:
        if pitch_ball_xy is None or len(pitch_ball_xy) == 0:
            self._ball = None
            return pitch_ball_xy
        obs = pitch_ball_xy[0].astype(np.float32)
        self._ball = self.alpha * obs + (1.0 - self.alpha) * self._ball if self._ball is not None else obs
        out    = pitch_ball_xy.copy().astype(np.float32)
        out[0] = self._ball
        return out

    def _expire(self, seen: set[int]) -> None:
        expired = []
        for tid, s in self._players.items():
            if tid not in seen:
                s["age"] += 1
                if s["age"] > self.max_age:
                    expired.append(tid)
        for tid in expired:
            del self._players[tid]

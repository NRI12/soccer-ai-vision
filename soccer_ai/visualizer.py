"""Annotation, radar minimap rendering, and frame overlay.

Public API
----------
build_annotators(cfg)            → dict of sv annotator objects
annotate_frame(data, ...)        → draws ellipses/labels/triangles + radar minimap
overlay_minimap(frame, minimap)  → composites a minimap onto a corner
draw_radar(data, ...)            → pitch canvas with player/ball/referee dots
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import supervision as sv
from omegaconf import DictConfig

from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
)


# ---------------------------------------------------------------------------
# Radar
# ---------------------------------------------------------------------------

_RADAR_SCALE = 0.1
_RADAR_PAD = 50


def _pitch_to_pixel(x: float, y: float) -> tuple[int, int]:
    return int(x * _RADAR_SCALE) + _RADAR_PAD, int(y * _RADAR_SCALE) + _RADAR_PAD


def _draw_id_labels(
    canvas: np.ndarray,
    pitch_xy: np.ndarray,
    tracker_ids: np.ndarray | None,
    radius: int,
) -> np.ndarray:
    if tracker_ids is None or len(tracker_ids) == 0:
        return canvas
    offset_y = radius + 14
    for i, tid_raw in enumerate(tracker_ids):
        if tid_raw is None:
            continue
        px, py = _pitch_to_pixel(*pitch_xy[i])
        label = str(int(tid_raw))
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tx, ty = px - tw // 2, py - offset_y
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def draw_radar(data: dict[str, Any], pitch_config: Any, cfg: DictConfig) -> np.ndarray:
    """Draw player/ball/referee positions on a pitch diagram."""
    colors = cfg.annotate.colors
    radar_cfg = cfg.annotate.radar

    canvas = draw_pitch(pitch_config)

    pitch_ball_xy = data.get("pitch_ball_xy")
    if pitch_ball_xy is not None and len(pitch_ball_xy) > 0:
        canvas = draw_points_on_pitch(
            config=pitch_config, xy=pitch_ball_xy,
            face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK,
            radius=radar_cfg.ball_radius, thickness=2, pitch=canvas,
        )

    pitch_players_xy = data.get("pitch_players_xy")
    players_class_id = data.get("players_class_id")
    players_tracker_id = data.get("pitch_players_tracker_id")

    if pitch_players_xy is not None and players_class_id is not None:
        for team_id, hex_color in ((0, colors.team_1), (1, colors.team_2)):
            mask = players_class_id == team_id
            if np.any(mask):
                canvas = draw_points_on_pitch(
                    config=pitch_config, xy=pitch_players_xy[mask],
                    face_color=sv.Color.from_hex(hex_color), edge_color=sv.Color.WHITE,
                    radius=radar_cfg.player_radius, thickness=2, pitch=canvas,
                )
        if getattr(radar_cfg, "show_ids", True):
            canvas = _draw_id_labels(canvas, pitch_players_xy, players_tracker_id, radar_cfg.player_radius)

    return canvas


# ---------------------------------------------------------------------------
# Minimap overlay
# ---------------------------------------------------------------------------

def overlay_minimap(
    frame: np.ndarray,
    minimap: np.ndarray,
    position: str = "bottom_right",
    scale: float = 0.25,
    margin: int = 10,
    border: int = 2,
    border_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Composite a minimap image onto a corner of the video frame."""
    fh, fw = frame.shape[:2]
    mw = int(fw * scale)
    mh = int(minimap.shape[0] * (mw / minimap.shape[1]))
    m = cv2.resize(minimap, (mw, mh), interpolation=cv2.INTER_LANCZOS4)

    if border > 0:
        m = cv2.copyMakeBorder(m, border, border, border, border,
                               cv2.BORDER_CONSTANT, value=border_color)
        mh, mw = m.shape[:2]

    if position == "bottom_right":
        y, x = fh - mh - margin, fw - mw - margin
    elif position == "bottom_left":
        y, x = fh - mh - margin, margin
    elif position == "top_right":
        y, x = margin, fw - mw - margin
    else:  # top_left
        y, x = margin, margin

    y, x = max(0, y), max(0, x)
    ye, xe = min(fh, y + mh), min(fw, x + mw)

    sp = 6
    frame[max(0, y-sp):min(fh, ye+sp), max(0, x-sp):min(fw, xe+sp)] = (
        frame[max(0, y-sp):min(fh, ye+sp), max(0, x-sp):min(fw, xe+sp)] * 0.45
    ).astype(np.uint8)

    frame[y:ye, x:xe] = m[:ye - y, :xe - x]
    return frame


# ---------------------------------------------------------------------------
# Annotators
# ---------------------------------------------------------------------------

def build_annotators(cfg: DictConfig) -> dict[str, Any]:
    """Construct sv annotator instances from config."""
    colors = cfg.colors
    player_palette = sv.ColorPalette.from_hex([colors.team_1, colors.team_2, colors.referee])

    # 8-class raw palette (basic mode, uses SoccerNet class IDs directly)
    raw_palette = sv.ColorPalette.from_hex([
        "#FF8C00",      # 0 ball
        colors.team_1,  # 1 player_left
        colors.team_2,  # 2 player_right
        colors.team_1,  # 3 goalkeeper_left
        colors.team_2,  # 4 goalkeeper_right
        colors.referee, # 5 main_referee
        colors.referee, # 6 side_referee
        "#808080",      # 7 staff
    ])

    return {
        "ellipse": sv.EllipseAnnotator(color=player_palette, thickness=cfg.ellipse.thickness),
        "triangle": sv.TriangleAnnotator(
            color=sv.Color.from_hex(colors.ball),
            base=cfg.triangle.base, height=cfg.triangle.height,
            outline_thickness=cfg.triangle.outline_thickness,
        ),
        "label": sv.LabelAnnotator(
            color=player_palette,
            text_color=sv.Color.from_hex(colors.text),
            text_position=sv.Position[cfg.label.text_position],
        ),
        "box": sv.BoxAnnotator(color=raw_palette, thickness=cfg.box.thickness),
        "box_label": sv.LabelAnnotator(color=raw_palette, text_color=sv.Color.from_hex(colors.text)),
    }


def annotate_frame(
    data: dict[str, Any],
    annotators: dict[str, Any],
    pitch_config: Any,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Draw player/ball annotations and minimap overlays onto the frame."""
    frame = data["frame"].copy()
    all_detections = data.get("all_detections", data.get("detections"))
    ball_detections = data.get("ball_detections")
    mode = cfg.annotate.mode

    if mode == "basic":
        detections = data.get("detections")
        if detections is not None:
            basic_labels = [
                f"{name} {conf:.2f}"
                for name, conf in zip(detections["class_name"], detections.confidence)
            ]
            frame = annotators["box"].annotate(scene=frame, detections=detections)
            frame = annotators["box_label"].annotate(scene=frame, detections=detections, labels=basic_labels)
    elif mode == "advanced":
        players = data.get("players_detections", all_detections)
        if players is not None and len(players) > 0:
            frame = annotators["ellipse"].annotate(scene=frame, detections=players)
            if players.tracker_id is not None:
                player_labels = [str(tid) for tid in players.tracker_id]
                frame = annotators["label"].annotate(scene=frame, detections=players, labels=player_labels)
        if ball_detections is not None and len(ball_detections) > 0:
            frame = annotators["triangle"].annotate(scene=frame, detections=ball_detections)

    data["frame"] = frame

    if cfg.annotate.radar.enabled and "pitch_players_xy" in data:
        radar = draw_radar(data, pitch_config, cfg)
        data["radar"] = radar
        data["frame"] = overlay_minimap(
            data["frame"], radar,
            position="bottom_right", scale=0.36, margin=12,
            border=3, border_color=(220, 220, 220),
        )

    return data

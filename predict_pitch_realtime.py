"""Realtime pitch keypoint/line detection trên test.mp4 dùng NBJW weights (SV_kp + SV_lines).

Usage:
    python predict_pitch_realtime.py
    python predict_pitch_realtime.py --source test.mp4 --output output/test_pitch.mp4
    python predict_pitch_realtime.py --kp-thres 0.15 --line-thres 0.40
    python predict_pitch_realtime.py --source 0   # webcam

Phím tắt:
    q / ESC   thoát
    p         pause / resume
    k         bật / tắt vẽ keypoints
    l         bật / tắt vẽ lines
    n         bật / tắt số keypoint
    s         lưu screenshot
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as tvf
import yaml
from PIL import Image as PILImage

from soccer_ai.nbjw.cls_hrnet import get_cls_net
from soccer_ai.nbjw.cls_hrnet_l import get_cls_net as get_cls_net_l
from soccer_ai.nbjw.utils_heatmap import (
    coords_to_dict,
    complete_keypoints,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

NBJW_CONFIG_DIR = Path("soccer_ai") / "nbjw" / "config"

# 23 lines của SoccerNet calibration (xem complete_keypoints trong utils_heatmap.py)
LINE_NAMES = [
    "BigRect L bot", "BigRect L main", "BigRect L top",
    "BigRect R bot", "BigRect R main", "BigRect R top",
    "Goal L cross",  "Goal L post L", "Goal L post R",
    "Goal R cross",  "Goal R post L", "Goal R post R",
    "Middle line",
    "Side bottom",   "Side left",     "Side right",    "Side top",
    "SmallRect L bot", "SmallRect L main", "SmallRect L top",
    "SmallRect R bot", "SmallRect R main", "SmallRect R top",
]


class PitchKeypointDetector:
    def __init__(self, weights_kp: str, weights_lines: str, device: str,
                 kp_threshold: float, line_threshold: float) -> None:
        cfg_kp   = yaml.safe_load((NBJW_CONFIG_DIR / "hrnetv2_w48.yaml").read_text())
        cfg_line = yaml.safe_load((NBJW_CONFIG_DIR / "hrnetv2_w48_l.yaml").read_text())

        m_kp = get_cls_net(cfg_kp)
        m_kp.load_state_dict(torch.load(weights_kp, map_location=device, weights_only=False))
        m_kp.to(device).eval()

        m_line = get_cls_net_l(cfg_line)
        m_line.load_state_dict(torch.load(weights_lines, map_location=device, weights_only=False))
        m_line.to(device).eval()

        self.m_kp = m_kp
        self.m_line = m_line
        self.resize = T.Resize((540, 960))
        self.device = device
        self.kp_thresh = kp_threshold
        self.line_thresh = line_threshold

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray) -> tuple[dict, dict]:
        """Return (kp_dict, lines_dict) với toạ độ đã chuẩn về pixel của khung hình gốc."""
        fh, fw = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = tvf.to_tensor(PILImage.fromarray(rgb)).float().unsqueeze(0)
        if t.size(-1) != 960:
            t = self.resize(t)
        t = t.to(self.device)
        _, _, h, w = t.size()

        heatmaps   = self.m_kp(t)
        heatmaps_l = self.m_line(t)

        kp    = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        lines = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
        kd    = coords_to_dict(kp,    threshold=self.kp_thresh)
        ld    = coords_to_dict(lines, threshold=self.line_thresh)
        # normalize=True -> toạ độ trong [0,1]; nhân lại với kích thước khung gốc
        final = complete_keypoints(kd, ld, w=w, h=h, normalize=True)

        kp_pix: dict[int, tuple[int, int, float]] = {}
        for k, v in final[0].items():
            x, y = v["x"] * fw, v["y"] * fh
            kp_pix[k] = (int(round(x)), int(round(y)), float(v.get("p", 1.0)))

        line_pix: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for k, v in ld[0].items():
            # ld trong toạ độ ảnh 960x540, chuyển về (fw, fh)
            x1 = v["x_1"] * fw / w
            y1 = v["y_1"] * fh / h
            x2 = v["x_2"] * fw / w
            y2 = v["y_2"] * fh / h
            line_pix[k] = ((int(round(x1)), int(round(y1))),
                           (int(round(x2)), int(round(y2))))

        return kp_pix, line_pix


def _color_for(i: int) -> tuple[int, int, int]:
    rng = np.random.RandomState(i * 9973 + 7)
    return tuple(int(c) for c in rng.randint(64, 256, size=3))


def draw_lines(frame: np.ndarray, line_dict: dict, label: bool) -> None:
    for k, (p1, p2) in line_dict.items():
        color = _color_for(k + 200)
        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
        if label and 1 <= k <= len(LINE_NAMES):
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            name = LINE_NAMES[k - 1]
            cv2.putText(frame, name, (mx + 4, my - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, name, (mx + 4, my - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_keypoints(frame: np.ndarray, kp_dict: dict, label_num: bool) -> int:
    for k, (x, y, _p) in kp_dict.items():
        color = _color_for(k)
        cv2.circle(frame, (x, y), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
        if label_num:
            cv2.putText(frame, str(k), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, str(k), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return len(kp_dict)


def draw_hud(frame: np.ndarray, fps: float, n_kp: int, n_lines: int,
             show_kp: bool, show_lines: bool, paused: bool) -> None:
    lines = [
        f"FPS: {fps:5.2f}   kp: {n_kp}   lines: {n_lines}",
        f"[k] kp={'ON' if show_kp else 'off'}  [l] lines={'ON' if show_lines else 'off'}  [n] numbers  [p] pause",
    ]
    if paused:
        lines.append("PAUSED")
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="test.mp4")
    ap.add_argument("--weights-kp", default="weights/SV_kp")
    ap.add_argument("--weights-lines", default="weights/SV_lines")
    ap.add_argument("--output", default="")
    ap.add_argument("--kp-thres", type=float, default=0.1486)
    ap.add_argument("--line-thres", type=float, default=0.3880)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[info] device: {device}")

    for p in (args.weights_kp, args.weights_lines):
        if not Path(p).is_file():
            raise FileNotFoundError(f"Không tìm thấy weights: {p}")

    print(f"[info] loading {args.weights_kp}")
    print(f"[info] loading {args.weights_lines}")
    det = PitchKeypointDetector(
        weights_kp=args.weights_kp,
        weights_lines=args.weights_lines,
        device=device,
        kp_threshold=args.kp_thres,
        line_threshold=args.line_thres,
    )

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn video: {args.source}")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[info] source: {args.source}  {fw}x{fh} @ {src_fps:.1f}fps")

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps, (fw, fh))
        print(f"[info] saving -> {args.output}")

    show_kp = True
    show_lines = True
    label_num = True
    paused = False
    shot_idx = 0
    last_out = None
    n_kp = n_lines = 0
    fps_window: list[float] = []

    print("\nControls: q/ESC quit | p pause | k toggle kp | l toggle lines | n toggle numbers | s screenshot\n")
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            kp_dict, line_dict = det.infer(frame)
            dt = time.perf_counter() - t0
            fps_window.append(dt)
            if len(fps_window) > 20:
                fps_window.pop(0)
            cur_fps = len(fps_window) / sum(fps_window) if fps_window else 0.0

            out = frame.copy()
            if show_lines:
                draw_lines(out, line_dict, label=label_num)
            n_kp = draw_keypoints(out, kp_dict, label_num) if show_kp else 0
            n_lines = len(line_dict)
            last_out = out

            if writer is not None:
                writer.write(out)
        else:
            cur_fps = 0.0

        if last_out is None:
            continue

        disp = last_out.copy()
        draw_hud(disp, cur_fps, n_kp, n_lines, show_kp, show_lines, paused)

        if not args.no_display:
            cv2.imshow("NBJW pitch keypoints", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                paused = not paused
            elif key == ord("k"):
                show_kp = not show_kp
            elif key == ord("l"):
                show_lines = not show_lines
            elif key == ord("n"):
                label_num = not label_num
            elif key == ord("s"):
                Path("output").mkdir(exist_ok=True)
                p = f"output/pitch_screenshot_{shot_idx:03d}.jpg"
                cv2.imwrite(p, disp)
                print(f"[info] screenshot -> {p}")
                shot_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[done]")


if __name__ == "__main__":
    main()

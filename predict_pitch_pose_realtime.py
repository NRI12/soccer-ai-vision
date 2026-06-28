"""Realtime pitch keypoint detection trên test.mp4 dùng YOLOv8-pose
(weights/research/results/pose_keypoint/train/weights/best.pt — 32 keypoints sân).

Nhẹ hơn HRNet (SV_kp/SV_lines) nhiều, chạy được trên CPU thường.

Usage:
    python predict_pitch_pose_realtime.py
    python predict_pitch_pose_realtime.py --weights research/results/pose_keypoint/train/weights/best.pt
    python predict_pitch_pose_realtime.py --output output/test_pitch_pose.mp4
    python predict_pitch_pose_realtime.py --source 0   # webcam

Phím tắt:
    q / ESC   thoát
    p         pause
    k         ẩn/hiện keypoints
    e         ẩn/hiện skeleton edges
    n         ẩn/hiện số keypoint
    b         ẩn/hiện bounding box pitch
    s         screenshot
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_WEIGHTS = "research/results/pose_keypoint/train/weights/best.pt"


def _build_skeleton() -> list[tuple[int, int]]:
    """Trả về các cạnh sân theo chỉ số keypoint 0–31 của model."""
    try:
        from sports.configs.soccer import SoccerPitchConfiguration
    except ImportError:
        return []
    c = SoccerPitchConfiguration()
    # labels[i] = số SoccerNet của vertex thứ i (1-based, vd '01', '15', '14', ...)
    label_to_idx = {int(lbl): i for i, lbl in enumerate(c.labels)}
    edges: list[tuple[int, int]] = []
    for a, b in c.edges:
        if a in label_to_idx and b in label_to_idx:
            edges.append((label_to_idx[a], label_to_idx[b]))
    return edges


def _color_for(i: int) -> tuple[int, int, int]:
    rng = np.random.RandomState(i * 9973 + 7)
    return tuple(int(c) for c in rng.randint(64, 256, size=3))


def draw_skeleton(frame: np.ndarray, kps: np.ndarray, conf: np.ndarray,
                  edges: list[tuple[int, int]], kp_thres: float) -> None:
    for a, b in edges:
        if conf[a] < kp_thres or conf[b] < kp_thres:
            continue
        pa = (int(kps[a, 0]), int(kps[a, 1]))
        pb = (int(kps[b, 0]), int(kps[b, 1]))
        cv2.line(frame, pa, pb, (0, 255, 200), 2, cv2.LINE_AA)


def draw_keypoints(frame: np.ndarray, kps: np.ndarray, conf: np.ndarray,
                   kp_thres: float, label_num: bool) -> int:
    n = 0
    for i, ((x, y), p) in enumerate(zip(kps, conf)):
        if p < kp_thres:
            continue
        x, y = int(x), int(y)
        color = _color_for(i)
        cv2.circle(frame, (x, y), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
        if label_num:
            cv2.putText(frame, str(i), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, str(i), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        n += 1
    return n


def draw_bbox(frame: np.ndarray, xyxy: np.ndarray, conf: float) -> None:
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
    cv2.putText(frame, f"pitch {conf:.2f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f"pitch {conf:.2f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, fps: float, n_kp: int,
             show_kp: bool, show_edge: bool, show_bbox: bool, paused: bool) -> None:
    lines = [
        f"FPS: {fps:5.2f}   kp: {n_kp}/32",
        f"[k] kp={'ON' if show_kp else 'off'}  [e] edges={'ON' if show_edge else 'off'}  "
        f"[b] bbox={'ON' if show_bbox else 'off'}  [n] numbers  [p] pause",
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
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--output", default="")
    ap.add_argument("--conf", type=float, default=0.30, help="bbox conf threshold")
    ap.add_argument("--kp-thres", type=float, default=0.50, help="visibility threshold cho mỗi keypoint")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[info] device: {device}")

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy weights: {weights_path}")
    print(f"[info] loading {weights_path}")
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    if model.task != "pose":
        raise ValueError(f"Weights không phải pose model (task={model.task})")

    edges = _build_skeleton()
    print(f"[info] skeleton edges: {len(edges)}")

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
    show_edge = True
    show_bbox = True
    label_num = True
    paused = False
    shot_idx = 0
    last_out = None
    n_kp = 0
    fps_window: list[float] = []

    print("\nControls: q/ESC quit | p pause | k kp | e edges | b bbox | n numbers | s screenshot\n")
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            res = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]
            dt = time.perf_counter() - t0
            fps_window.append(dt)
            if len(fps_window) > 20:
                fps_window.pop(0)
            cur_fps = len(fps_window) / sum(fps_window) if fps_window else 0.0

            out = frame.copy()
            n_kp = 0

            if res.keypoints is not None and res.boxes is not None and len(res.boxes) > 0:
                # chọn instance có conf cao nhất (sân chỉ có 1)
                best = int(res.boxes.conf.argmax().item())
                kps_xy = res.keypoints.xy[best].cpu().numpy()           # (32, 2)
                kps_conf = res.keypoints.conf[best].cpu().numpy()       # (32,)
                box_xyxy = res.boxes.xyxy[best].cpu().numpy()
                box_conf = float(res.boxes.conf[best].item())

                if show_bbox:
                    draw_bbox(out, box_xyxy, box_conf)
                if show_edge:
                    draw_skeleton(out, kps_xy, kps_conf, edges, args.kp_thres)
                if show_kp:
                    n_kp = draw_keypoints(out, kps_xy, kps_conf, args.kp_thres, label_num)

            last_out = out
            if writer is not None:
                writer.write(out)
        else:
            cur_fps = 0.0

        if last_out is None:
            continue

        disp = last_out.copy()
        draw_hud(disp, cur_fps, n_kp, show_kp, show_edge, show_bbox, paused)

        if not args.no_display:
            cv2.imshow("YOLO-pose pitch keypoints", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                paused = not paused
            elif key == ord("k"):
                show_kp = not show_kp
            elif key == ord("e"):
                show_edge = not show_edge
            elif key == ord("b"):
                show_bbox = not show_bbox
            elif key == ord("n"):
                label_num = not label_num
            elif key == ord("s"):
                Path("output").mkdir(exist_ok=True)
                p = f"output/pitch_pose_screenshot_{shot_idx:03d}.jpg"
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

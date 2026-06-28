"""Realtime player prediction trên test.mp4 dùng SoccerNet pretrain weights.

Usage:
    python predict_realtime.py
    python predict_realtime.py --source test.mp4 --weights weights/player_detection.pt
    python predict_realtime.py --weights research/results/detection_soccernet/train/weights/best.pt
    python predict_realtime.py --output output/test_pred.mp4
    python predict_realtime.py --source 0   # webcam

Phím tắt:
    q / ESC   thoát
    p         pause / resume
    s         lưu screenshot
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2

from soccer_ai.minimap_mask import MinimapMasker

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# SoccerNet 8 classes (xem conf/config.yaml)
CLASS_NAMES = {
    0: "ball",
    1: "player_L",
    2: "player_R",
    3: "GK_L",
    4: "GK_R",
    5: "ref_main",
    6: "ref_side",
    7: "staff",
}
CLASS_COLORS = {
    0: (0, 255, 255),    # ball - yellow
    1: (255, 80, 80),    # player_L - blue
    2: (80, 80, 255),    # player_R - red
    3: (255, 200, 0),    # GK_L
    4: (0, 200, 255),    # GK_R
    5: (0, 255, 0),      # ref_main - green
    6: (0, 180, 0),      # ref_side
    7: (180, 180, 180),  # staff - gray
}


def draw_detections(frame, boxes, confs, clss, conf_thres: float) -> int:
    n = 0
    for (x1, y1, x2, y2), c, k in zip(boxes, confs, clss):
        if c < conf_thres:
            continue
        k = int(k)
        color = CLASS_COLORS.get(k, (200, 200, 200))
        name = CLASS_NAMES.get(k, str(k))
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        n += 1
    return n


def draw_hud(frame, fps: float, n_det: int, paused: bool, weights_name: str) -> None:
    lines = [
        f"FPS: {fps:5.1f}   detections: {n_det}",
        f"weights: {weights_name}",
    ]
    if paused:
        lines.append("PAUSED")
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="test.mp4")
    ap.add_argument("--weights", default="weights/player_detection.pt",
                    help="SoccerNet YOLO weights (mặc định weights/player_detection.pt; "
                         "có thể dùng research/results/detection_soccernet/train/weights/best.pt)")
    ap.add_argument("--output", default="", help="Đường dẫn xuất video (vd output/test_pred.mp4)")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--minimap", default="minimap.png",
                    help="template minimap để che trước khi predict; '' để tắt")
    ap.add_argument("--minimap-score", type=float, default=0.55)
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    # device
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[info] device: {device}")

    # weights
    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy weights: {weights_path}\n"
            "Thử: weights/player_detection.pt "
            "hoặc research/results/detection_soccernet/train/weights/best.pt"
        )
    print(f"[info] loading {weights_path}")
    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    # video
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

    masker: MinimapMasker | None = None
    if args.minimap and Path(args.minimap).is_file():
        masker = MinimapMasker(args.minimap, score_lock=args.minimap_score)
        print(f"[info] minimap template: {args.minimap}")
    elif args.minimap:
        print(f"[warn] minimap template không tồn tại: {args.minimap}")

    fps_window: list[float] = []
    shot_idx = 0
    paused = False
    last_out = None
    last_n = 0
    weights_name = weights_path.name

    print("\nControls: q/ESC quit | p pause | s screenshot\n")
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            in_frame = masker.apply(frame) if masker is not None else frame

            t0 = time.perf_counter()
            res = model.predict(
                in_frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]
            dt = time.perf_counter() - t0
            fps_window.append(dt)
            if len(fps_window) > 30:
                fps_window.pop(0)
            cur_fps = len(fps_window) / sum(fps_window) if fps_window else 0.0

            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
            else:
                xyxy, confs, clss = [], [], []

            out = in_frame.copy()
            last_n = draw_detections(out, xyxy, confs, clss, args.conf)
            last_out = out

            if writer is not None:
                writer.write(out)
        else:
            cur_fps = 0.0

        if last_out is None:
            continue

        disp = last_out.copy()
        draw_hud(disp, cur_fps, last_n, paused, weights_name)
        if masker is not None:
            cv2.putText(disp, masker.info, (12, disp.shape[0] - 12),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, masker.info, (12, disp.shape[0] - 12),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if not args.no_display:
            cv2.imshow("SoccerNet realtime predict", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                paused = not paused
            elif key == ord("s"):
                Path("output").mkdir(exist_ok=True)
                p = f"output/predict_screenshot_{shot_idx:03d}.jpg"
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

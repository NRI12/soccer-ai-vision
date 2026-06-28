"""Unified SoccerNet → YOLO fine-tuning pipeline.

Sub-commands:
    download  Download SoccerNet v3 frames + labels.
    convert   Convert SoccerNet annotations to YOLO detection format.
    weights   Download a YOLO pretrained checkpoint.
    train     Fine-tune YOLO on the converted dataset.
    all       Run the full pipeline end-to-end.

Examples:
    python finetune_soccernet.py all \\
        --soccernet-root path/to/SoccerNet \\
        --data-root       data \\
        --data-yaml       soccer_detect.yaml \\
        --weights         weights/yolo11m.pt

    python finetune_soccernet.py train \\
        --data soccer_detect.yaml --weights weights/yolo11m.pt --device 0
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
BBOX_CLASSES = [
    "Ball",
    "Player team left",
    "Player team right",
    "Goalkeeper team left",
    "Goalkeeper team right",
    "Main referee",
    "Side referee",
    "Staff members",
]
CLASS_TO_ID = {name: i for i, name in enumerate(BBOX_CLASSES)}
VALID_SPLITS = {"train", "valid", "test"}

DEFAULT_WEIGHTS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
)


# ---------------------------------------------------------------------------
# Step 1 — download SoccerNet frames + labels
# ---------------------------------------------------------------------------
def download_soccernet(local_dir: str | Path) -> None:
    from SoccerNet.Downloader import SoccerNetDownloader

    downloader = SoccerNetDownloader(LocalDirectory=str(local_dir))
    downloader.downloadGames(
        files=["Labels-v3.json", "Frames-v3.zip"],
        split=["train", "valid", "test"],
        task="frames",
    )
    print(f"[download] SoccerNet frames saved under: {local_dir}")


# ---------------------------------------------------------------------------
# Step 2 — convert SoccerNet labels → YOLO detection format
# ---------------------------------------------------------------------------
def _box_to_yolo(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> Optional[Tuple[float, float, float, float]]:
    x1 = max(0.0, min(float(width), float(x1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y1 = max(0.0, min(float(height), float(y1)))
    y2 = max(0.0, min(float(height), float(y2)))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 1.0 or bh <= 1.0:
        return None
    xc = (x1 + x2) / 2.0 / width
    yc = (y1 + y2) / 2.0 / height
    return xc, yc, bw / width, bh / height


class _ZipMemberCache:
    def __init__(self, members: Iterable[str]) -> None:
        self.by_name: Dict[str, str] = {}
        for member in members:
            name = Path(member).name
            self.by_name.setdefault(name, member)

    def get(self, image_name: str) -> Optional[str]:
        return self.by_name.get(image_name)


def _safe_split(value: str) -> str:
    return value if value in VALID_SPLITS else "train"


def convert_soccernet_to_yolo(
    soccernet_root: str | Path, output_root: str | Path
) -> None:
    soccernet_root = Path(soccernet_root)
    output_root = Path(output_root)

    for split in VALID_SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    label_files = sorted(soccernet_root.rglob("Labels-v3.json"))
    print(f"[convert] found {len(label_files)} label files")

    num_images = 0
    num_boxes = 0

    for label_path in label_files:
        game_dir = label_path.parent
        frames_zip = game_dir / "Frames-v3.zip"
        if not frames_zip.exists():
            print(f"[convert] skip (missing zip): {frames_zip}")
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        game_tag = game_dir.as_posix().replace("/", "__")

        with zipfile.ZipFile(frames_zip, "r") as zf:
            cache = _ZipMemberCache(zf.namelist())

            for section in ("actions", "replays"):
                for image_name, ann in data.get(section, {}).items():
                    meta = ann.get("imageMetadata", {})
                    split = _safe_split(meta.get("set", "train"))
                    width = int(meta.get("width", 1920))
                    height = int(meta.get("height", 1080))

                    member = cache.get(image_name)
                    if member is None:
                        print(f"[convert] image missing in zip: {image_name}")
                        continue

                    out_stem = f"{game_tag}__{Path(image_name).stem}"
                    out_img = output_root / "images" / split / f"{out_stem}.png"
                    out_lbl = output_root / "labels" / split / f"{out_stem}.txt"

                    if not out_img.exists():
                        with zf.open(member) as src, open(out_img, "wb") as dst:
                            dst.write(src.read())

                    lines = []
                    for bbox in ann.get("bboxes", []):
                        cls_name = bbox.get("class")
                        if cls_name not in CLASS_TO_ID:
                            continue
                        pts = bbox.get("points", {})
                        converted = _box_to_yolo(
                            pts.get("x1", 0.0),
                            pts.get("y1", 0.0),
                            pts.get("x2", 0.0),
                            pts.get("y2", 0.0),
                            width,
                            height,
                        )
                        if converted is None:
                            continue
                        xc, yc, bw, bh = converted
                        cls_id = CLASS_TO_ID[cls_name]
                        lines.append(
                            f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                        )
                        num_boxes += 1

                    with open(out_lbl, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines))

                    num_images += 1

    print(f"[convert] exported {num_images} images, {num_boxes} boxes → {output_root}")


# ---------------------------------------------------------------------------
# Step 3 — download pretrained YOLO weights
# ---------------------------------------------------------------------------
def download_weights(url: str, out: str | Path) -> None:
    import requests

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"[weights] saved: {out}")


# ---------------------------------------------------------------------------
# Step 4 — fine-tune YOLO
# ---------------------------------------------------------------------------
def train_yolo(
    data: str,
    model_path: str = "yolo11m.pt",
    epochs: int = 100,
    imgsz: int = 1024,
    batch: int = 8,
    workers: int = 2,
    device: str = "0",
    project: str = "runs/detect",
    name: str = "soccernet_yolo11_detect",
    cache: bool = False,
    # Geometry: broadcast cameras tilt rất nhỏ, không flip
    rect: bool = False,
    degrees: float = 3.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 1.5,
    perspective: float = 0.0001,
    fliplr: float = 0.0,
    flipud: float = 0.0,
    # Color: hue cỏ không nên đổi nhiều, brightness đổi mạnh (ngày/đêm/đèn)
    hsv_h: float = 0.01,
    hsv_s: float = 0.6,
    hsv_v: float = 0.5,
    # Mixing: mosaic + copy_paste rất hiệu quả cho soccer (bóng nhỏ, cầu thủ che nhau)
    mosaic: float = 1.0,
    mixup: float = 0.15,
    cutmix: float = 0.0,
    copy_paste: float = 0.4,
    copy_paste_mode: str = "flip",
    erasing: float = 0.4,
    close_mosaic: int = 30,
    # Strategy
    multi_scale: bool = True,
    optimizer: str = "AdamW",
    cos_lr: bool = True,
    warmup_epochs: float = 5.0,
    lr0: float = 0.001,
    lrf: float = 0.01,
    weight_decay: float = 0.0005,
    label_smoothing: float = 0.1,
    dropout: float = 0.1,
    amp: bool = True,
    patience: int = 50,
) -> None:
    from ultralytics import YOLO

    model = YOLO(model_path)
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        cache=cache,
        device=device,
        pretrained=True,
        project=project,
        name=name,
        rect=rect,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        fliplr=fliplr,
        flipud=flipud,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        mosaic=mosaic,
        mixup=mixup,
        cutmix=cutmix,
        copy_paste=copy_paste,
        copy_paste_mode=copy_paste_mode,
        erasing=erasing,
        close_mosaic=close_mosaic,
        multi_scale=multi_scale,
        optimizer=optimizer,
        cos_lr=cos_lr,
        warmup_epochs=warmup_epochs,
        lr0=lr0,
        lrf=lrf,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        dropout=dropout,
        amp=amp,
        patience=patience,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SoccerNet YOLO fine-tuning pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # download
    pd = sub.add_parser("download", help="Download SoccerNet frames+labels")
    pd.add_argument("--soccernet-root", required=True, help="Local SoccerNet root dir")

    # convert
    pc = sub.add_parser("convert", help="Convert SoccerNet → YOLO format")
    pc.add_argument("--soccernet-root", required=True)
    pc.add_argument("--output-root", required=True, help="YOLO dataset output dir")

    # weights
    pw = sub.add_parser("weights", help="Download pretrained YOLO weights")
    pw.add_argument("--url", default=DEFAULT_WEIGHTS_URL)
    pw.add_argument("--out", required=True)

    # train
    pt = sub.add_parser("train", help="Fine-tune YOLO")
    pt.add_argument("--data", required=True, help="Path to dataset YAML")
    pt.add_argument("--weights", required=True, help="Pretrained .pt path")
    pt.add_argument("--epochs", type=int, default=100)
    pt.add_argument("--imgsz", type=int, default=1024)
    pt.add_argument("--batch", type=int, default=8)
    pt.add_argument("--workers", type=int, default=2)
    pt.add_argument("--device", default="0")
    pt.add_argument("--project", default="runs/detect")
    pt.add_argument("--name", default="soccernet_yolo11_detect")
    pt.add_argument("--cache", action="store_true")

    # all
    pa = sub.add_parser("all", help="Run full pipeline: download → convert → weights → train")
    pa.add_argument("--soccernet-root", required=True)
    pa.add_argument("--data-root", required=True, help="Converted YOLO dataset root")
    pa.add_argument("--data-yaml", required=True, help="Existing dataset YAML for training")
    pa.add_argument("--weights", required=True, help="Where to save pretrained .pt")
    pa.add_argument("--weights-url", default=DEFAULT_WEIGHTS_URL)
    pa.add_argument("--skip-download", action="store_true")
    pa.add_argument("--epochs", type=int, default=100)
    pa.add_argument("--imgsz", type=int, default=1024)
    pa.add_argument("--batch", type=int, default=8)
    pa.add_argument("--workers", type=int, default=2)
    pa.add_argument("--device", default="0")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.cmd == "download":
        download_soccernet(args.soccernet_root)

    elif args.cmd == "convert":
        convert_soccernet_to_yolo(args.soccernet_root, args.output_root)

    elif args.cmd == "weights":
        download_weights(args.url, args.out)

    elif args.cmd == "train":
        train_yolo(
            data=args.data,
            model_path=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=args.project,
            name=args.name,
            cache=args.cache,
        )

    elif args.cmd == "all":
        if not args.skip_download:
            download_soccernet(args.soccernet_root)
        convert_soccernet_to_yolo(args.soccernet_root, args.data_root)
        if not Path(args.weights).exists():
            download_weights(args.weights_url, args.weights)
        train_yolo(
            data=args.data_yaml,
            model_path=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
        )


if __name__ == "__main__":
    main()

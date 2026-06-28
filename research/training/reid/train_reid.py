"""Train OSNet_x1_0 for soccer player re-identification.

Dataset layout expected:
    <data_root>/
        train/
            <identity_0001>/   img_001.jpg  img_002.jpg ...
            <identity_0002>/   ...
        val/                   (optional, same layout — used for Rank-1 / mAP eval)

Example:
    python train_reid.py --data data/reid_dataset --epochs 60
    python train_reid.py --data data/reid_dataset --pretrained weights/osnet_x1_0_sportsreid.pth.tar
    python train_reid.py --data data/reid_dataset --epochs 80 --batch-p 32 --batch-k 4 --device cuda

Output (all written to --output dir, default output/reid_train/):
    best.pth.tar    ← best val mAP (or lowest train loss when no val/)
    last.pth.tar    ← checkpoint after every epoch
    train.log       ← full training log (also printed to console)
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from soccer_ai.osnet import osnet_x1_0, load_weights


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_dir / "train.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers)
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ReIDDataset(Dataset):
    """ImageFolder-style ReID dataset.  Each sub-directory = one identity."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str | Path, transform=None) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.labels:  list[int] = []

        root = Path(root)
        identity_dirs = sorted(p for p in root.iterdir() if p.is_dir())
        for pid, id_dir in enumerate(identity_dirs):
            imgs = [f for f in sorted(id_dir.iterdir())
                    if f.suffix.lower() in self.IMG_EXTS]
            for img_path in imgs:
                self.samples.append((img_path, pid))
                self.labels.append(pid)

        self.num_classes = len(identity_dirs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# PK Sampler
# ---------------------------------------------------------------------------

class PKSampler(Sampler):
    """Yield P identities × K images per iteration.

    Guarantees each mini-batch contains K samples per identity so that
    BatchHard triplet mining finds meaningful hard positives and negatives.
    Sampling is with replacement when an identity has fewer than K images.
    """

    def __init__(self, labels: list[int], P: int, K: int) -> None:
        self.P = P
        self.K = K
        self.id2idx: dict[int, list[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.id2idx[lbl].append(i)
        self.pids = list(self.id2idx.keys())

    def __len__(self) -> int:
        return len(self.pids) * self.K

    def __iter__(self) -> Iterator[int]:
        pids = self.pids.copy()
        random.shuffle(pids)
        batch: list[int] = []
        for pid in pids:
            idxs = self.id2idx[pid]
            batch.extend(random.choices(idxs, k=self.K))
            if len(batch) >= self.P * self.K:
                yield from batch
                batch = []
        if batch:
            yield from batch


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def batch_hard_triplet_loss(
    emb: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """BatchHard triplet loss (Hermans et al., 2017).

    For each anchor:
      - hardest positive  = same identity, largest L2 distance
      - hardest negative  = different identity, smallest L2 distance
    """
    dist = torch.cdist(emb, emb, p=2)                          # (N, N)
    same = labels.unsqueeze(1) == labels.unsqueeze(0)          # (N, N) bool

    # Hardest positive: mask out cross-identity, take max
    pos_dist, _ = (dist * same.float()).max(dim=1)

    # Hardest negative: mask out same-identity with large value, take min
    neg_dist, _ = (dist + same.float() * 1e6).min(dim=1)

    return F.relu(pos_dist - neg_dist + margin).mean()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def _train_transform(h: int, w: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), value=0),
    ])


def _val_transform(h: int, w: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_emb, all_lbl = [], []
    for imgs, lbls in loader:
        feats = model.embed(imgs.to(device)).cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.where(norms > 1e-8, norms, 1.0)
        all_emb.append(feats)
        all_lbl.append(lbls.numpy())
    return np.concatenate(all_emb), np.concatenate(all_lbl)


def compute_rank1_mAP(emb: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Brute-force Rank-1 accuracy and mean Average Precision for identity retrieval."""
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)  # exclude self-match

    rank1_ok = 0
    ap_sum   = 0.0
    N = len(emb)
    for i in range(N):
        order = np.argsort(-sim[i])
        gt    = labels[order] == labels[i]
        rank1_ok += int(gt[0])
        hits = np.where(gt)[0] + 1  # 1-indexed hit positions
        if len(hits):
            ap_sum += np.mean([(k / pos) for k, pos in enumerate(hits, 1)])

    return rank1_ok / N, ap_sum / N


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save(model: nn.Module, out_dir: Path, name: str) -> None:
    torch.save({"state_dict": model.state_dict()}, out_dir / name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _download_soccernet_reid(dest: Path) -> None:
    """Download SoccerNet Re-ID dataset and restructure into ReIDDataset layout.

    Downloaded layout (SoccerNet):
        <dest>/reid/<split>/<sequence>/<tracklet_id>/<frame>.jpg

    Restructured layout (ReIDDataset):
        <dest>/train/<sequence>_<tracklet_id>/<frame>.jpg
        <dest>/val/  <sequence>_<tracklet_id>/<frame>.jpg

    Each (sequence, tracklet) pair is treated as one identity.
    """
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        raise SystemExit(
            "SoccerNet SDK not installed.\n"
            "Run:  pip install SoccerNet"
        )

    print(f"Downloading SoccerNet Re-ID → {dest} ...")
    dl = SoccerNetDownloader(LocalDirectory=str(dest))
    dl.downloadDataTask(task="reid", split=["train", "valid", "test"])

    raw_root = dest / "reid"
    if not raw_root.exists():
        raise FileNotFoundError(
            f"Expected download at {raw_root} — check SoccerNet SDK output above."
        )

    split_map = {"train": "train", "valid": "val", "test": "val"}
    moved = 0
    for split_src, split_dst in split_map.items():
        src_split = raw_root / split_src
        if not src_split.exists():
            continue
        dst_split = dest / split_dst
        for seq_dir in sorted(src_split.iterdir()):
            if not seq_dir.is_dir():
                continue
            for tracklet_dir in sorted(seq_dir.iterdir()):
                if not tracklet_dir.is_dir():
                    continue
                identity_name = f"{seq_dir.name}_{tracklet_dir.name}"
                out_dir = dst_split / identity_name
                out_dir.mkdir(parents=True, exist_ok=True)
                for img in tracklet_dir.iterdir():
                    if img.suffix.lower() in ReIDDataset.IMG_EXTS:
                        img.rename(out_dir / img.name)
                        moved += 1

    print(f"Restructured {moved} images → {dest}/train  {dest}/val")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train OSNet_x1_0 for soccer player re-identification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",     required=True,
                   help="Root dir containing train/ (and optionally val/). "
                        "Pass an empty dir together with --download to auto-populate.")
    p.add_argument("--download", action="store_true",
                   help="Download SoccerNet Re-ID dataset into --data before training")
    p.add_argument("--output",   default="output/reid_train",
                   help="Output dir for checkpoints and train.log")
    p.add_argument("--pretrained", default=None,
                   help="Optional pretrained weights path (warm-start)")

    # Training hyper-params
    p.add_argument("--epochs",         type=int,   default=60)
    p.add_argument("--batch-p",        type=int,   default=16,   dest="batch_p",
                   help="Identities per batch (P in PK sampling)")
    p.add_argument("--batch-k",        type=int,   default=4,    dest="batch_k",
                   help="Images per identity per batch (K in PK sampling)")
    p.add_argument("--img-h",          type=int,   default=256,  dest="img_h")
    p.add_argument("--img-w",          type=int,   default=128,  dest="img_w")
    p.add_argument("--lr",             type=float, default=3.5e-4)
    p.add_argument("--weight-decay",   type=float, default=5e-4, dest="weight_decay")
    p.add_argument("--triplet-margin", type=float, default=0.3,  dest="triplet_margin")
    p.add_argument("--triplet-weight", type=float, default=1.0,  dest="triplet_weight",
                   help="Scale factor on triplet loss term")
    p.add_argument("--label-smoothing",type=float, default=0.1,  dest="label_smoothing")
    p.add_argument("--warmup-epochs",  type=int,   default=10,   dest="warmup_epochs",
                   help="Linear LR warm-up before cosine decay")
    p.add_argument("--eval-every",     type=int,   default=5,    dest="eval_every",
                   help="Run val evaluation every N epochs")

    # System
    p.add_argument("--device",  default="auto", help="cuda | cpu | auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = _parse()
    out_dir = Path(args.output)
    log    = _setup_logging(out_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    log.info("Device: %s", device)

    # ── Dataset download (optional) ───────────────────────────────────────────
    data_root = Path(args.data)
    if args.download:
        _download_soccernet_reid(data_root)
    train_ds  = ReIDDataset(data_root / "train", _train_transform(args.img_h, args.img_w))
    log.info("Train: %d images, %d identities", len(train_ds), train_ds.num_classes)

    sampler      = PKSampler(train_ds.labels, P=args.batch_p, K=args.batch_k)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_p * args.batch_k,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    val_dir    = data_root / "val"
    val_loader = None
    if val_dir.exists():
        val_ds     = ReIDDataset(val_dir, _val_transform(args.img_h, args.img_w))
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                                num_workers=args.workers, pin_memory=(device == "cuda"))
        log.info("Val:   %d images, %d identities", len(val_ds), val_ds.num_classes)
    else:
        log.info("No val/ folder — skipping periodic evaluation")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = osnet_x1_0(num_classes=train_ds.num_classes).to(device)
    if args.pretrained:
        load_weights(model, args.pretrained, device)
        log.info("Warm-start from: %s", args.pretrained)

    # ── Optimizer + LR schedule ───────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def _lr_lambda(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return 0.1 + 0.9 * epoch / max(1, args.warmup_epochs)
        t = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_metric = -1.0
    best_epoch  = 0

    log.info("=" * 64)
    log.info(
        "Start  epochs=%d  P=%d  K=%d  lr=%.1e  margin=%.2f  device=%s",
        args.epochs, args.batch_p, args.batch_k,
        args.lr, args.triplet_margin, device,
    )
    log.info("CE(smooth=%.2f) + %.1f × Triplet(BatchHard)", args.label_smoothing, args.triplet_weight)
    log.info("=" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        sum_ce = sum_tri = n_total = 0

        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)                              # (N, num_classes) — train mode
            embs   = model.embed(imgs)                        # (N, 512)

            loss_ce  = ce_loss_fn(logits, labels)
            loss_tri = batch_hard_triplet_loss(embs, labels, margin=args.triplet_margin)
            loss     = loss_ce + args.triplet_weight * loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bsz      = imgs.size(0)
            sum_ce  += loss_ce.item()  * bsz
            sum_tri += loss_tri.item() * bsz
            n_total += bsz

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        log.info(
            "Epoch %3d/%d  CE=%.4f  Tri=%.4f  lr=%.2e  %.0fs",
            epoch, args.epochs,
            sum_ce / n_total, sum_tri / n_total,
            lr_now, elapsed,
        )

        _save(model, out_dir, "last.pth.tar")

        # Validation
        if val_loader is not None and (epoch % args.eval_every == 0 or epoch == args.epochs):
            emb, lbl  = extract_embeddings(model, val_loader, device)
            rank1, mAP = compute_rank1_mAP(emb, lbl)
            log.info("  Val  Rank-1=%.2f%%  mAP=%.2f%%", rank1 * 100, mAP * 100)
            model.train()
            if mAP > best_metric:
                best_metric = mAP
                best_epoch  = epoch
                _save(model, out_dir, "best.pth.tar")
                log.info("  ✓ best.pth.tar  (mAP=%.2f%% @ epoch %d)", best_metric * 100, best_epoch)
        elif val_loader is None:
            combined = (sum_ce + sum_tri) / n_total
            if best_metric < 0 or combined < best_metric:
                best_metric = combined
                best_epoch  = epoch
                _save(model, out_dir, "best.pth.tar")

    log.info("=" * 64)
    if val_loader is not None:
        log.info("Done.  Best mAP=%.2f%% at epoch %d", best_metric * 100, best_epoch)
    else:
        log.info("Done.  Best combined loss=%.4f at epoch %d", best_metric, best_epoch)
    log.info("Checkpoints → %s", out_dir)


if __name__ == "__main__":
    main()

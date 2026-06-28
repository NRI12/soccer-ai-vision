"""Player Re-Identification using OSNet_x1_0 (sportsreid weights).

Sits on top of ByteTrack as a post-processing step.  When the tracker assigns
a new ID to a detection, this module checks whether the appearance matches a
recently-lost track via cosine similarity.  If so, the old stable ID is restored.

Setup
-----
1. Enable:  reid.enabled: true  in conf/pipeline/reid.yaml  (or CLI override)
   Weights auto-download from Google Drive on first run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from soccer_ai.osnet import osnet_x1_0, load_weights, ensure_weights

log = logging.getLogger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class PlayerReID:
    """Appearance-based ID recovery on top of any tracker.

    Gallery layout
    --------------
    _active[stable_id]  → L2-normalised embedding (EMA-updated)
    _lost[stable_id]    → {"emb": ndarray, "age": int}
    """

    def __init__(
        self,
        weights: str,
        device: str = "cpu",
        img_size: tuple[int, int] = (256, 128),   # H × W
        threshold: float = 0.65,
        max_age: int = 120,
        update_interval: int = 30,
        emb_ema: float = 0.15,
        min_crop_h: int = 40,
    ) -> None:
        import torch
        self._torch  = torch
        self._device = device
        self._H, self._W = img_size
        self.threshold       = threshold
        self.max_age         = max_age
        self.update_interval = update_interval
        self.emb_ema         = emb_ema
        self.min_crop_h      = min_crop_h

        weights_path = Path(weights)
        if not weights_path.exists():
            weights_path = ensure_weights(weights_path.parent)
        model = osnet_x1_0(num_classes=1)
        load_weights(model, str(weights_path), device)
        model.to(device).eval()
        self._model = model

        self._active: dict[int, np.ndarray] = {}
        self._lost:   dict[int, dict]       = {}
        self._seen:   set[int]              = set()
        self._tick    = 0

        log.info("PlayerReID ready  device=%s  threshold=%.2f", device, threshold)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, detections: sv.Detections) -> sv.Detections:
        """Remap tracker IDs; return (possibly modified) detections."""
        self._tick += 1

        if detections.tracker_id is None or len(detections) == 0:
            self._age_lost(set())
            return detections

        bt_ids      = detections.tracker_id.astype(int).tolist()
        current_set = set(bt_ids)
        new_ids     = [t for t in bt_ids if t not in self._seen]

        # --- 1. Match new IDs against lost gallery -------------------------
        # Only process new IDs whose crop is tall enough for reliable embeddings.
        new_embs_by_btid: dict[int, np.ndarray] = {}
        id_remap: dict[int, int] = {}
        if new_ids:
            new_set     = set(new_ids)
            new_mask    = np.array([t in new_set for t in bt_ids])
            new_xyxy    = detections.xyxy[new_mask]
            new_ordered = [t for t in bt_ids if t in new_set]
            size_ok     = (new_xyxy[:, 3] - new_xyxy[:, 1]) >= self.min_crop_h
            large_xyxy    = new_xyxy[size_ok]
            large_ordered = [t for t, ok in zip(new_ordered, size_ok) if ok]
            if len(large_xyxy) > 0:
                embs  = self._extract(frame, large_xyxy)
                used: set[int] = set()
                for i, bt_id in enumerate(large_ordered):
                    new_embs_by_btid[bt_id] = embs[i]
                    matched = self._best_match(embs[i], exclude=used)
                    if matched is not None:
                        id_remap[bt_id] = matched
                        used.add(matched)
                        log.debug("ReID: bt=%d → stable=%d", bt_id, matched)

        # --- 2. Apply remap ------------------------------------------------
        if id_remap:
            tid = detections.tracker_id.copy()
            for bt_id, stable_id in id_remap.items():
                tid[tid == bt_id] = stable_id
                self._lost.pop(stable_id, None)
            detections.tracker_id = tid
            current_set = set(detections.tracker_id.astype(int).tolist())

        # --- 3. Init gallery for new IDs using already-extracted embeddings
        for bt_id, emb in new_embs_by_btid.items():
            sid = id_remap.get(bt_id, bt_id)
            if sid not in self._active:
                self._active[sid] = emb

        # --- 4. Periodic gallery refresh — only large-enough crops
        if self._tick % self.update_interval == 0:
            sids     = detections.tracker_id.astype(int).tolist()
            size_ok  = (detections.xyxy[:, 3] - detections.xyxy[:, 1]) >= self.min_crop_h
            lg_xyxy  = detections.xyxy[size_ok]
            lg_sids  = [s for s, ok in zip(sids, size_ok) if ok]
            if len(lg_xyxy) > 0:
                embs = self._extract(frame, lg_xyxy)
                for i, sid in enumerate(lg_sids):
                    if sid in self._active:
                        self._active[sid] = (
                            self.emb_ema * embs[i] + (1 - self.emb_ema) * self._active[sid]
                        )
                    else:
                        self._active[sid] = embs[i]

        self._seen.update(current_set)

        # --- 5. Move disappeared tracks to lost gallery --------------------
        for sid in list(self._active):
            if sid not in current_set:
                self._lost[sid] = {"emb": self._active.pop(sid), "age": 0}

        self._age_lost(current_set)
        return detections

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray, xyxy: np.ndarray):
        fh, fw = frame.shape[:2]
        crops  = []
        mean   = _IMAGENET_MEAN
        std    = _IMAGENET_STD

        for x1, y1, x2, y2 in xyxy.astype(int):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            if x2 <= x1 or y2 <= y1:
                crops.append(self._torch.zeros(3, self._H, self._W))
                continue
            crop = cv2.resize(
                cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB),
                (self._W, self._H), interpolation=cv2.INTER_LINEAR,
            )
            t = self._torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
            for c in range(3):
                t[c] = (t[c] - mean[c]) / std[c]
            crops.append(t)

        return self._torch.stack(crops)

    def _extract(self, frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        """Return L2-normalised embeddings, shape (N, 512)."""
        if len(xyxy) == 0:
            return np.empty((0, 512), dtype=np.float32)
        batch = self._preprocess(frame, xyxy).to(self._device)
        with self._torch.no_grad():
            feats = self._model(batch).cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.where(norms > 1e-8, norms, 1.0)

    def _best_match(self, emb: np.ndarray, exclude: set[int]) -> int | None:
        best_id, best_sim = None, self.threshold
        for sid, v in self._lost.items():
            if sid in exclude:
                continue
            sim = float(np.dot(emb, v["emb"]))
            if sim > best_sim:
                best_sim, best_id = sim, sid
        return best_id

    def _age_lost(self, current_set: set[int]) -> None:
        expired = [sid for sid, v in self._lost.items()
                   if sid in current_set or v["age"] >= self.max_age]
        for sid in expired:
            del self._lost[sid]
        for v in self._lost.values():
            v["age"] += 1

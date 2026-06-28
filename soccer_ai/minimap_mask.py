"""Che minimap trong frame qua template matching trước khi đưa vào model.

Khi minimap nằm cố định ở 1 góc, ta chỉ cần định vị nó một lần rồi tô đè màu xám
trung tính (114,114,114 — đúng bằng letterbox của YOLO) để model không bắt nhầm
cầu thủ/bóng trong minimap.

Cách dùng:
    masker = MinimapMasker("minimap.png")
    masked = masker.apply(frame)  # gọi mỗi frame; tự khoá ROI sau khi match đủ tốt
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class MinimapMasker:
    def __init__(
        self,
        template_path: str | Path,
        score_lock: float = 0.55,
        try_first_n: int = 30,
        fill_color: tuple[int, int, int] = (114, 114, 114),
        scales: tuple[float, ...] = (1.0, 0.85, 0.7, 1.2, 1.5),
        padding: int = 4,
    ) -> None:
        tpl = cv2.imread(str(template_path))
        if tpl is None:
            raise FileNotFoundError(f"Không đọc được template: {template_path}")
        self.template_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        self.template_size = tpl.shape[:2]  # (h, w)
        self.score_lock = score_lock
        self.try_first_n = try_first_n
        self.fill = fill_color
        self.scales = scales
        self.padding = padding

        self._roi: Optional[tuple[int, int, int, int]] = None
        self._tried = 0
        self._locked = False
        self._best_score: float = -1.0

    def _try_locate(self, frame_gray: np.ndarray) -> Optional[tuple[float, int, int, int, int]]:
        fh, fw = frame_gray.shape[:2]
        th0, tw0 = self.template_size
        best: Optional[tuple[float, int, int, int, int]] = None
        for s in self.scales:
            th, tw = int(th0 * s), int(tw0 * s)
            if th < 20 or tw < 20 or th >= fh or tw >= fw:
                continue
            t = cv2.resize(self.template_gray, (tw, th))
            r = cv2.matchTemplate(frame_gray, t, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(r)
            if best is None or max_val > best[0]:
                x1, y1 = max_loc
                best = (float(max_val), x1, y1, x1 + tw, y1 + th)
        return best

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Trả về frame đã che minimap. Mutates a copy, không sửa frame gốc."""
        if not self._locked:
            if self._tried < self.try_first_n:
                self._tried += 1
                res = self._try_locate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if res is not None:
                    score, x1, y1, x2, y2 = res
                    if score > self._best_score:
                        self._best_score = score
                        self._roi = (x1, y1, x2, y2)
                    if score >= self.score_lock:
                        self._locked = True
            else:
                # hết quota thử — chốt với ROI tốt nhất đã thấy (nếu có)
                self._locked = True

        if self._roi is None:
            return frame

        out = frame.copy()
        x1, y1, x2, y2 = self._roi
        p = self.padding
        x1 = max(0, x1 - p); y1 = max(0, y1 - p)
        x2 = min(frame.shape[1], x2 + p); y2 = min(frame.shape[0], y2 + p)
        cv2.rectangle(out, (x1, y1), (x2, y2), self.fill, thickness=-1)
        return out

    @property
    def roi(self) -> Optional[tuple[int, int, int, int]]:
        return self._roi

    @property
    def info(self) -> str:
        if self._roi is None:
            return "minimap: not located"
        x1, y1, x2, y2 = self._roi
        state = "LOCKED" if self._locked else "trying"
        return f"minimap: ({x1},{y1})-({x2},{y2}) score={self._best_score:.2f} [{state}]"

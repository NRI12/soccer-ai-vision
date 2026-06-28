# auto_game_dls

Realtime detect cầu thủ + bóng trên video bóng đá, dùng YOLO finetune trên
SoccerNet (8 class). Bản tối giản tách ra từ project `SOCCER_FOOTBALL_FINAL`,
chỉ giữ phần phát hiện đối tượng.

Entry point duy nhất: `predict_realtime.py`.

---

## 1. Cấu trúc thư mục

```
auto_game_dls/
├── predict_realtime.py          # entrypoint
├── minimap.png                  # template để che minimap khỏi frame
├── test.mp4                     # video mẫu
├── weights/
│   └── player_detection.pt      # YOLO finetune trên SoccerNet (8 class)
├── soccer_ai/
│   ├── __init__.py
│   └── minimap_mask.py          # template matching che minimap
└── output/                      # nơi lưu video / screenshot xuất ra
```

## 2. Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\activate

pip install ultralytics opencv-python numpy torch
```

GPU: cài `torch` đúng phiên bản CUDA theo
<https://pytorch.org/get-started/locally/>.

## 3. Cách chạy

```powershell
# Mặc định
python predict_realtime.py

# Xuất video
python predict_realtime.py --output output/test_pred.mp4

# Webcam
python predict_realtime.py --source 0

# Đổi weights
python predict_realtime.py --weights weights/player_detection.pt

# Tắt mask minimap nếu video không có minimap
python predict_realtime.py --minimap ""
```

**Phím tắt (cửa sổ OpenCV cần focus):**

| Phím | Hành động |
|------|-----------|
| `q` / `ESC` | Thoát |
| `p` | Pause / resume |
| `s` | Lưu screenshot vào `output/` |

---

## 4. Cơ chế

### 4.1 SoccerNet 8-class detector — `weights/player_detection.pt`

YOLO finetune trên dataset SoccerNet với 8 class:

| ID | Tên | Màu khung |
|----|-----|-----------|
| 0 | `ball` | vàng |
| 1 | `player_L` | xanh dương |
| 2 | `player_R` | đỏ |
| 3 | `GK_L` (thủ môn đội trái) | cam nhạt |
| 4 | `GK_R` (thủ môn đội phải) | xanh dương nhạt |
| 5 | `ref_main` (trọng tài chính) | xanh lá |
| 6 | `ref_side` (trọng tài biên) | xanh lá đậm |
| 7 | `staff` | xám |

Class trái/phải xác định **đội** trực tiếp từ output model — không cần
team classifier phụ.

### 4.2 `soccer_ai/minimap_mask.py` — `MinimapMasker`

**Vấn đề:** Phát sóng bóng đá thường overlay minimap chiến thuật ở 1 góc.
Minimap đó chứa hình cầu thủ thu nhỏ → YOLO dễ bắt nhầm thành "player" và
"ball" giả. Cần che trước khi đưa vào model.

**Logic:**

1. Đọc `minimap.png` làm template.
2. Mỗi frame trong N frame đầu (`try_first_n=30`):
   - `cv2.matchTemplate` với `TM_CCOEFF_NORMED` ở nhiều scale (0.7×, 0.85×, 1.0×, 1.2×, 1.5×).
   - Lưu vị trí có score cao nhất từng thấy.
3. Khi score ≥ `score_lock` (mặc định `0.55`) → khoá ROI, không match nữa.
4. Mỗi frame sau khi khoá: vẽ đè rectangle màu `(114, 114, 114)` (đúng màu
   YOLO letterbox — không kích hoạt detection nào) lên ROI → trả về frame "sạch".

Tham số CLI:

```
--minimap minimap.png      # đường dẫn template; "" để tắt mask
--minimap-score 0.55       # ngưỡng chốt vị trí
```

HUD hiển thị: `minimap: (x1,y1)-(x2,y2) score=0.78 [LOCKED]`.

### 4.3 `predict_realtime.py` — pipeline

```
read frame ─► minimap mask ─► YOLO predict ─► draw bbox + label ─► display/write
```

CLI quan trọng:

| Flag | Mặc định | Ý nghĩa |
|------|----------|---------|
| `--source` | `test.mp4` | Video hoặc số webcam |
| `--weights` | `weights/player_detection.pt` | Đường dẫn .pt |
| `--conf` | 0.30 | Ngưỡng confidence |
| `--iou` | 0.50 | Ngưỡng NMS IoU |
| `--imgsz` | 1280 | Kích thước inference (960 để chạy CPU mượt hơn) |
| `--device` | auto | "auto" / "cpu" / "0" / "cuda" |
| `--output` | "" | Path mp4 để lưu (rỗng = không lưu) |
| `--minimap` | `minimap.png` | Template che minimap; "" để tắt |
| `--minimap-score` | 0.55 | Ngưỡng score template-matching để khoá ROI |

---

## 5. Sơ đồ tổng

```
   ┌───────────────┐
   │ test.mp4 / cam│
   └──────┬────────┘
          │ frame BGR
          ▼
   ┌──────────────────┐
   │  MinimapMasker   │  ← minimap.png (template, lock 1 lần)
   └──────┬───────────┘
          │ frame đã che minimap
          ▼
   ┌──────────────────┐
   │ YOLO predict     │  ← weights/player_detection.pt
   │ (8 class)        │
   └──────┬───────────┘
          │ boxes, conf, cls
          ▼
   ┌──────────────────┐
   │ draw bbox + HUD  │
   └──────┬───────────┘
          ▼
     display + write
```

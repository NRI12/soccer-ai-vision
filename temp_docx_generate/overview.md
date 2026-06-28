# Soccer AI — Tổng Quan Dự Án

## 1. Giới Thiệu

Hệ thống phân tích video bóng đá tự động sử dụng AI, xử lý video trận đấu để phát hiện cầu thủ, theo dõi quỹ đạo, phân loại đội, hiệu chỉnh góc nhìn camera, và xuất thống kê chi tiết cho từng cầu thủ. Pipeline được cấu hình qua Hydra, hỗ trợ chạy local GPU/CPU và cloud (Modal.com).

---

## 2. Tổng Quan Dữ Liệu

### 2.1 Dữ Liệu Video Đầu Vào (Inference)

5 video mẫu từ trận đấu bóng đá thực tế, tải tự động từ Google Drive:

| File | Mặc định | Ghi chú |
|------|----------|---------|
| `data/08fd33_0.mp4` | ✅ | Video dùng trong demo & thống kê mẫu |
| `data/0bfacc_0.mp4` | | |
| `data/2e57b9_0.mp4` | | |
| `data/573e61_0.mp4` | | |
| `data/121364_0.mp4` | | |

**Đặc điểm video:** Góc quay broadcast chuẩn (camera chính), độ phân giải cao (1080p), ~30fps, cắt ngắn ~25–30 giây/đoạn để test pipeline.

---

### 2.2 Dữ Liệu Training

Dự án sử dụng **4 bộ dữ liệu** cho các bài toán khác nhau:

---

#### 2.2.1 SoccerNet Tracking Dataset — dùng train YOLO Detection

> **Paper:** *SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos*
> **Repo:** [github.com/SoccerNet/sn-tracking](https://github.com/SoccerNet/sn-tracking)

| Thuộc tính | Giá trị |
|-----------|---------|
| Số sequences | 200 đoạn video × 30 giây |
| Độ phân giải | 1080p, camera chính |
| Nhãn | Bounding box + track ID (CSV: frame, id, x, y, w, h) |
| Đối tượng | Cầu thủ, trọng tài, bóng |
| Thách thức | 50 clip bổ sung (SoccerNet 2023 Challenge, CVPR) |
| Nhãn dự án | 8 lớp: ball, player_left/right, goalkeeper_left/right, main_referee, side_referee, staff |

**Pipeline nội bộ remap về 3 lớp:** `team_0`, `team_1`, `referee`

```
SoccerNet class → Internal class
─────────────────────────────────
player_left  (1) ┐
goalkeeper_left (3) ┘ → team_0

player_right (2) ┐
goalkeeper_right (4) ┘ → team_1

main_referee (5) ┐
side_referee (6) ┘ → referee

staff (7)  → bị loại bỏ khỏi pipeline
ball  (0)  → xử lý riêng
```

---

#### 2.2.2 Custom Soccer Pose Dataset — dùng train YOLO Pose

| Thuộc tính | Giá trị |
|-----------|---------|
| Nguồn | Tự xây dựng (annotate từ ảnh cắt cầu thủ) |
| Keypoints | 17 điểm khớp chuẩn COCO Pose |
| Mục đích | Phát hiện tư thế cầu thủ |
| Kết quả | mAP50-95 (Pose) = **0.956** sau 120 epochs |

---

#### 2.2.3 SoccerNet Re-ID Dataset — dùng train OSNet Re-ID

> **Repo:** [github.com/SoccerNet/sn-reid](https://github.com/SoccerNet/sn-reid)

| Thuộc tính | Giá trị |
|-----------|---------|
| Tổng số ảnh | **340,993** thumbnail cầu thủ |
| Nguồn | 400 trận đấu từ 6 giải lớn châu Âu |
| Train split | 248,234 ảnh |
| Query (test) | 11,777 ảnh |
| Gallery (test) | 34,989 ảnh |
| Thách thức | Đội mặc đồng phục giống nhau, ảnh đa độ phân giải, ít ảnh/identity |
| Đánh giá | mAP + CMC curves, trong phạm vi cùng action |

**Weights đang dùng:** `osnet_x1_0_sportsreid.pth.tar` — pretrained trên bộ SoccerNet Re-ID này.

---

#### 2.2.4 NBJW Camera Calibration Dataset — dùng train HRNet-W48

> **Paper:** *No Bells Just Whistles: Sports Field Registration by Leveraging Geometric Properties* — CVPR Workshop 2024
> **Repo:** [github.com/mguti97/No-Bells-Just-Whistles](https://github.com/mguti97/No-Bells-Just-Whistles)

| Thuộc tính | Giá trị |
|-----------|---------|
| Tập train | SoccerNet-Calibration annotations |
| Tập eval | SoccerNet-Calibration, WorldCup 2014, TS-WorldCup |
| Nhãn | Keypoints sân bóng + đường kẻ field line extremities |
| 2 mô hình | `SV_kp` (keypoint detection) + `SV_lines` (line detection) |
| Weights trong dự án | `weights/SV_kp` (~252 MB), `weights/SV_lines` (~252 MB) |

---

### 2.3 Dữ Liệu Weights Đang Sử Dụng (Inference)

| File Weights | Nguồn | Kích thước | Dùng cho |
|-------------|-------|-----------|---------|
| `weights/player_detection.pt` | SoccerNet finetuned YOLO11 | ~auto-download | Object detection |
| `weights/SV_kp` | NBJW pretrained HRNet-W48 | ~252 MB | Keypoint detection sân |
| `weights/SV_lines` | NBJW pretrained HRNet-W48 | ~252 MB | Line detection sân |
| `weights/osnet_x1_0_sportsreid.pth.tar` | SoccerNet Re-ID pretrained | ~auto-download | Player Re-ID |

---

## 3. Kiến Trúc Pipeline

Hệ thống xử lý **6 giai đoạn tuần tự** mỗi frame:

```
Video Input
    │
    ▼
┌─────────────────────────────────────────────────┐
│ [1] DETECT  — YOLO11 (8 lớp)                    │
│     → Phát hiện bóng, cầu thủ, trọng tài        │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ [2] FILTER  — Supervision NMS                   │
│     → Tách bóng, loại bỏ nhiễu, remap lớp      │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ [3] TRACK   — ByteTrack + Kalman Filter         │
│     → Gán ID liên tục xuyên suốt video          │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ [4] TEAM    — Class ID Parsing                  │
│     → Phân loại team 0 / team 1 / referee       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ [5] PITCH   — NBJW Dual HRNet-W48               │
│     → Hiệu chỉnh camera, homography             │
│     → Chiếu tọa độ ảnh → tọa độ sân thực       │
│     → EMA smoothing (alpha=0.4)                  │
│     → OSNet Re-ID (khôi phục ID bị mất)         │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ [6] ANNOTATE — Supervision + Sports             │
│     → Vẽ ellipse/nhãn, radar minimap, Voronoi   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
              Output Video (MP4)
              Player Stats (JSON)
```

---

## 4. Các Mô Hình & Thuật Toán

### 4.1 YOLO11 — Object Detection

> **Framework:** [Ultralytics](https://github.com/ultralytics/ultralytics)
> YOLO11 là thế hệ tiếp theo của dòng YOLO, cải tiến từ YOLOv8 với backbone hiệu quả hơn, anchor-free detection head, và tốc độ inference nhanh hơn.

| Thông số | Giá trị |
|----------|---------|
| Architecture | YOLO11 (Ultralytics, anchor-free) |
| Input | Video frame (RGB) |
| Output | Bounding boxes + class ID + confidence score |
| Confidence threshold | 0.3 |
| NMS threshold | 0.5 |
| Số lớp | 8 (SoccerNet schema) |
| Weights | `weights/player_detection.pt` |
| Finetuned trên | SoccerNet 2023 Tracking Dataset |

**Vai trò trong pipeline:** Bước đầu tiên — phát hiện tất cả đối tượng trong frame, cung cấp bounding box cho tracker.

---

### 4.2 ByteTrack — Multi-Object Tracking

> **Paper:** *ByteTrack: Multi-Object Tracking by Associating Every Detection Box* — ECCV 2022
> **Framework:** [Roboflow Supervision](https://github.com/roboflow/supervision) (tích hợp sẵn)

**Ý tưởng cốt lõi:** Thay vì chỉ match detection có confidence cao, ByteTrack associate *tất cả* detection box (kể cả confidence thấp) với existing tracks qua Kalman Filter + IoU matching. Điều này giúp không mất track khi cầu thủ bị che khuất một phần.

| Thông số | Giá trị |
|----------|---------|
| Algorithm | ByteTrack + Kalman Filter |
| Activation threshold | 0.35 |
| Lost buffer (frames) | 90 (~3 giây @ 30fps) |
| Min confirmed frames | 3 |
| Matching strategy | IoU-based, class-aware |

**Vai trò trong pipeline:** Gán tracker ID liên tục cho mỗi đối tượng qua các frame.

---

### 4.3 NBJW — Camera Calibration

> **Paper:** *No Bells, Just Whistles: Sports Field Registration by Leveraging Geometric Properties*
> **Tác giả:** Marc Gutiérrez-Pérez, Antonio Agudo
> **Venue:** CVPR Workshop (CVsports), June 2024, trang 3325–3334
> **Repo:** [github.com/mguti97/No-Bells-Just-Whistles](https://github.com/mguti97/No-Bells-Just-Whistles)

**Ý tưởng cốt lõi:** Hiệu chỉnh camera sân thể thao *không cần* marker đặc biệt hay post-processing phức tạp. Pipeline đơn giản:
1. Dùng encoder-decoder HRNet-W48 để detect keypoints (giao điểm) và line extremities trên sân
2. Fit ellipse cho vòng tròn trung tâm
3. Áp dụng DLT (Direct Linear Transform) để tính ma trận homography
4. Chiếu tọa độ pixel → tọa độ thực tế trên sân (mét)

| Thông số | Giá trị |
|----------|---------|
| Backbone | HRNet-W48 (High-Resolution Network) |
| Mô hình 1 — Keypoints | `weights/SV_kp` (~252 MB) — detect giao điểm sân |
| Mô hình 2 — Lines | `weights/SV_lines` (~252 MB) — detect đường kẻ sân |
| Keypoint threshold | 0.1486 |
| Line threshold | 0.3880 |
| Calibration interval | Mỗi 15 frames (cache homography giữa các lần) |
| EMA alpha | 0.4 (làm mượt tọa độ sân theo thời gian) |
| Output | Ma trận homography 3×3 → ánh xạ pixel → tọa độ sân (m) |
| Evaluated on | SoccerNet-Calibration, WorldCup 2014, TS-WorldCup |

**Vai trò trong pipeline:** Chuyển đổi vị trí cầu thủ từ pixel ảnh sang tọa độ thực tế trên sân để tính quãng đường & tốc độ chính xác.

---

### 4.4 OSNet — Player Re-Identification

> **Paper:** *Omni-Scale Feature Learning for Person Re-Identification*
> **Tác giả:** Kaiyang Zhou, Yongxin Yang, Andrea Cavallaro, Tao Xiang
> **Affiliations:** University of Surrey, Queen Mary University of London, Samsung AI Center Cambridge
> **Venue:** ICCV 2019
> **Repo:** [github.com/KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) (Torchreid)

**Ý tưởng cốt lõi — Omni-Scale Feature Learning:**
- Học đặc trưng ngoại hình ở **nhiều tỉ lệ không gian** đồng thời (omni-scale)
- **Unified Aggregation Gate (UAG):** Cổng tổng hợp channel-wise động, thích ứng theo đầu vào để fuse features đa tỉ lệ
- Dùng **depthwise + pointwise convolution** để giảm tham số nhưng vẫn học spatial-channel correlation tốt
- Cực kỳ nhẹ — nhỏ hơn ~10× so với ResNet50 nhưng accuracy tương đương hoặc vượt trội

| Thông số | Giá trị |
|----------|---------|
| Architecture | OSNet_x1_0 |
| Input size | 256 × 128 pixels (H × W) |
| Similarity metric | Cosine similarity |
| Match threshold | 0.65 |
| Max gallery age | 120 frames (~4 giây) |
| Weights | `osnet_x1_0_sportsreid.pth.tar` |
| Pretrained trên | SoccerNet Re-ID Dataset (340,993 thumbnails) |
| Đánh giá | State-of-the-art trên 6 bộ dữ liệu Re-ID |

**Vai trò trong pipeline:** Khi ByteTrack mất track một cầu thủ (bị che khuất, ra ngoài frame), OSNet so sánh embedding ngoại hình với gallery để khôi phục đúng tracker ID — tránh tạo ID mới sai.

---

### 4.5 YOLO11-Pose — Keypoint Detection (Research)

| Thông số | Giá trị |
|----------|---------|
| Architecture | YOLO11-Pose (Ultralytics) |
| Keypoints | 17 điểm khớp chuẩn COCO Pose (vai, khuỷu, gối, ...) |
| Weights | `research/runs/pose/train/weights/best.pt` |
| Mục đích | Phân tích tư thế cầu thủ, dùng trong nghiên cứu |

---

### 4.6 Bảng Tổng Hợp Tất Cả Mô Hình

| Mô hình | Bài toán | Nguồn gốc | Weights |
|---------|----------|-----------|---------|
| YOLO11 | Object detection (8 lớp) | Ultralytics + SoccerNet finetuned | `player_detection.pt` |
| ByteTrack | Multi-object tracking | ECCV 2022, via Supervision | Built-in |
| NBJW HRNet-W48 | Camera calibration | CVPR Workshop 2024 | `SV_kp`, `SV_lines` |
| OSNet_x1_0 | Player Re-ID | ICCV 2019, SoccerNet pretrained | `osnet_x1_0_sportsreid.pth.tar` |
| YOLO11-Pose | Pose keypoints | Ultralytics (research) | `best.pt` (pose) |

---

## 5. Kết Quả Training

### 5.1 Detection Baseline (YOLO11 — 120 Epochs)

> Dataset: Bộ dữ liệu tùy chỉnh 8 lớp | Platform: HPC cluster (SLURM)

| Metric | Epoch 1 | Epoch 60 | Epoch 120 (Final) | Best |
|--------|---------|----------|-------------------|------|
| mAP50 (B) | 0.0 | ~0.81 | 0.8268 | 0.8317 |
| **mAP50-95 (B)** | 0.0 | ~0.57 | **0.5822** | **0.5848** |
| Precision | 0.0 | ~0.93 | 0.9013 | 0.9320 |
| Recall | 0.0 | ~0.74 | 0.7583 | 0.8115 |
| Train box_loss | 1.622 | — | 0.780 | — |
| Train cls_loss | 1.873 | — | 0.354 | — |

**Thời gian:** ~1542 giây tổng (120 epoch) trên HPC.

---

### 5.2 Detection SoccerNet Finetuned (YOLO11 — 50+ Epochs)

> Dataset: SoccerNet 2023 Tracking (dataset lớn hơn, tinh chỉnh thêm)

| Metric | Epoch 50 | So sánh baseline |
|--------|----------|-----------------|
| mAP50-95 (B) | 0.5817 | Tương đương |
| Precision | **0.9491** | +0.048 |
| Recall | **0.7976** | +0.039 |
| Val box_loss | 0.8689 | Ổn định, không overfit |
| Thời gian/epoch | ~706 giây | Lâu hơn do dataset lớn |

---

### 5.3 Pose Keypoint Detection (YOLO11-Pose — 120 Epochs)

> Dataset: Custom soccer pose với 17 keypoints

| Metric | Epoch 1 | Epoch 120 (Final) | Best |
|--------|---------|-------------------|------|
| mAP50 (B) | ~0.60 | 0.9983 | 0.9983 |
| **mAP50-95 (B)** | ~0.50 | **0.8724** | **0.9586** |
| mAP50 (Pose) | 0.0 | 0.9667 | 0.9667 |
| **mAP50-95 (Pose)** | 0.0 | **0.9560** | **0.9811** |
| Precision (B) | — | 0.8725 | 0.9983 |
| Recall (B) | — | 0.9983 | 0.9983 |

**Nhận xét:** Pose model đạt chất lượng **xuất sắc** (mAP50-95 Pose = 0.956), hội tụ ổn định từ epoch 80+. Thời gian training chỉ ~932 giây tổng — rất hiệu quả.

---

### 5.4 Tổng Hợp So Sánh

| Mô hình | Task | mAP50-95 | Precision | Recall | Epochs |
|---------|------|----------|-----------|--------|--------|
| YOLO11 Baseline | Detection 8 lớp | 0.582 | 0.901 | 0.758 | 120 |
| YOLO11 SoccerNet | Detection 8 lớp | 0.582 | **0.949** | **0.798** | 50+ |
| YOLO11-Pose | Keypoint 17 pts | **0.956** (Pose) | 0.998 | 0.998 | 120 |

---

## 6. Kết Quả Inference — Thống Kê Cầu Thủ

### 6.1 Output Mẫu (`output/player_stats/player_stats.json`)

Kết quả từ video `data/08fd33_0.mp4` — **30 tracker IDs** được theo dõi:

| Tracker ID | Team | Frames | Quãng Đường (m) | Tốc Độ Max (km/h) | Tốc Độ TB (km/h) |
|-----------|------|--------|-----------------|-------------------|-------------------|
| 16 | Team 0 | 746 | **106.2** | 35.5 | 13.3 |
| 11 | Team 0 | 747 | 100.8 | 32.2 | 12.6 |
| 12 | Team 1 | 745 | 99.4 | 33.5 | 12.3 |
| 26 | Team 0 | 661 | 93.6 | 35.1 | 12.9 |
| 9 | Team 1 | 746 | 89.1 | 35.5 | 11.1 |
| 4 | Team 1 | 742 | 88.2 | 35.8 | 11.1 |
| 28 | Team 1 | 641 | 87.8 | 33.0 | 12.5 |
| 5 | Team 1 | 733 | 87.1 | 35.3 | 10.9 |
| 27 | Team 0 | 661 | 85.5 | 28.6 | 11.7 |
| 14 | Team 1 | 747 | 85.2 | **35.8** | 10.4 |

### 6.2 Thống Kê Tổng Hợp

| Chỉ Số | Giá Trị |
|--------|---------|
| Tổng tracker IDs | 30 |
| Cầu thủ active (≥600 frames) | 10 |
| Quãng đường lớn nhất | **106.2 m** (tracker #16, Team 0) |
| Tốc độ cao nhất | **36.0 km/h** (tracker #1, 9.99 m/s) |
| Tốc độ trung bình nhóm | ~11.2 km/h |
| Thời lượng video | ~25 giây (747 frames @ ~30fps) |
| Team 0 players | 10 trackers |
| Team 1 players | 14 trackers |
| Referee/staff | 6 trackers |

---

## 7. Output Artifacts

| Artifact | Đường Dẫn | Mô Tả |
|----------|-----------|-------|
| Video phân tích | `output/result.mp4` | Annotated: ellipse, nhãn, radar minimap, Voronoi |
| Thống kê JSON | `output/player_stats/player_stats.json` | Per-player: distance, speed max/avg, frames, team |
| Ảnh trajectory | `output/player_stats/player_XXXX_teamY.png` | Quỹ đạo trên sân + crop collage cầu thủ |
| Highlight video | `output/player_stats/player_XXXX.mp4` | Clip highlight từng cầu thủ (nếu ≥280 frames) |
| Hydra config | `outputs/YYYY-MM-DD/HH-MM-SS/` | Config đầy đủ mỗi lần chạy |

---

## 8. Cấu Hình & Tham Số Chính

```yaml
# conf/config.yaml
video:
  source_path: data/08fd33_0.mp4
  output_path: output/result.mp4
  stride: 1                    # xử lý mỗi N frames

models:
  device: "auto"               # auto / cpu / cuda / 0 / modal

player_stats:
  enabled: true
  output_dir: output/player_stats
  export_videos: true
  min_frames_for_video: 280    # bỏ qua cầu thủ có ít frames

# conf/pipeline/detect.yaml
confidence: 0.3

# conf/pipeline/track.yaml
activation_threshold: 0.35
lost_buffer: 90
min_frames: 3

# conf/pipeline/pitch.yaml
calibration_interval: 15
ema_alpha: 0.4

# conf/pipeline/reid.yaml
threshold: 0.65
max_age: 120
```

---

## 9. Cách Chạy

```bash
# Cài đặt
pip install -e .

# Chạy mặc định (video 08fd33_0.mp4)
python main.py

# Chạy với video khác
python main.py video.source_path=data/573e61_0.mp4

# Chạy realtime (cửa sổ cv2)
python realtime.py --source data/08fd33_0.mp4

# Chạy trên cloud GPU H100 (Modal.com)
python main.py models.device=modal

# Tùy chỉnh tham số
python main.py detect.confidence=0.4 pitch.enabled=false
```

---

## 10. Dependencies Chính

| Package | Version | Mục Đích |
|---------|---------|----------|
| torch | ≥2.0.0 | Deep learning backend (CUDA 12.1) |
| ultralytics | ≥8.0.0 | YOLO11 inference + training |
| supervision | ≥0.23.0 | ByteTrack, NMS, visualization primitives |
| opencv-python | ≥4.8.0 | Xử lý video/ảnh, homography |
| hydra-core | ≥1.3.2 | Quản lý cấu hình |
| sports | latest (Roboflow) | Pitch rendering, minimap |
| gdown | ≥5.2.1 | Tải weights & video từ Google Drive |
| modal | ≥0.73.0 | Cloud GPU execution (optional) |

---

## 11. Tài Liệu Tham Khảo

| Mô hình | Paper | Venue |
|---------|-------|-------|
| **NBJW** | *No Bells, Just Whistles: Sports Field Registration by Leveraging Geometric Properties* — Gutiérrez-Pérez & Agudo | CVPR Workshop 2024 |
| **OSNet** | *Omni-Scale Feature Learning for Person Re-Identification* — Zhou et al. | ICCV 2019 |
| **ByteTrack** | *ByteTrack: Multi-Object Tracking by Associating Every Detection Box* — Zhang et al. | ECCV 2022 |
| **YOLO11** | Ultralytics YOLO11 | 2024 |
| **HRNet** | *Deep High-Resolution Representation Learning for Visual Recognition* — Wang et al. | TPAMI 2020 |
| **SoccerNet Tracking** | *SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark* | CVPR 2022 |
| **SoccerNet Re-ID** | SoccerNet Re-Identification Dataset | [soccernet.org](https://www.soccer-net.org/tasks/re-identification) |

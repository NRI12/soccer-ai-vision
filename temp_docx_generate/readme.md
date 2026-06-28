# Soccer AI — Video Analysis Pipeline

Phân tích video bóng đá với YOLO detection, ByteTrack tracking, NBJW camera calibration, OSNet Re-ID và annotation đầy đủ (radar minimap, Voronoi, per-player stats).

---

## Cài đặt

**Yêu cầu:** Python 3.11, Windows 10+

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
cd D:\kho_du_an\folder\SOCCER_FOOTBALL_FINAL
uv venv .venv --python 3.11
.\.venv\Scripts\activate
uv sync
```

**Tải video mẫu:**

```powershell
python -m soccer_ai.download_data
```

---

## Weights

Tất cả weights đặt trong `weights/`. File `player_detection.pt` tự tải lần đầu chạy. Hai file NBJW cần tải thủ công:

| File | Mô tả | Tải về |
|---|---|---|
| `weights/player_detection.pt` | SoccerNet finetuned YOLO — 8 class, encode team | Tự tải khi chạy |
| `weights/SV_kp` | NBJW HRNet-W48 — keypoint detection | [GitHub release v1.0.0](https://github.com/mguti97/No-Bells-Just-Whistles/releases/tag/v1.0.0) |
| `weights/SV_lines` | NBJW HRNet-W48 — line detection | [GitHub release v1.0.0](https://github.com/mguti97/No-Bells-Just-Whistles/releases/tag/v1.0.0) |
| `weights/osnet_x1_0_sportsreid.pth.tar` | OSNet Re-ID — SoccerNet finetuned | Tự tải khi bật reid |

```powershell
# Tải NBJW weights (~252 MB mỗi file)
python -c "
import urllib.request, pathlib
root = pathlib.Path('weights'); root.mkdir(exist_ok=True)
for w in ('SV_kp', 'SV_lines'):
    urllib.request.urlretrieve(
        f'https://github.com/mguti97/No-Bells-Just-Whistles/releases/download/v1.0.0/{w}',
        root / w)
print('done')
"
```

---

## Luồng pipeline

Mỗi frame đi qua 6 stage theo thứ tự. Disable bất kỳ stage nào bằng `<stage>.enabled=false`.

```
frame
  │
  ▼
[1] detect       YOLO → sv.Detections (8 class)
  │
  ▼
[2] filter       Tách ball / non-ball, NMS, drop staff, remap 8→3 class
  │
  ▼
[3] track        sv.ByteTrack → tracker_id bền vững
  │
  ▼
[4] team         Split players / referees, remap → team_0 / team_1 / referee
  │
  ▼
[5] pitch        NBJW calibration → homography → tọa độ canvas sân
  │              (chạy mỗi N frame, cache transformer giữa các frame)
  ▼
[6] annotate     Vẽ ellipse, label, radar minimap, Voronoi overlay
```

**Team classification** không dùng KMeans — weights đã encode `player_left` / `player_right` trực tiếp trong class ID (class 1–4). Referee ở class 5–6.

**Re-ID (tuỳ chọn)** — nếu `reid.enabled: true`, OSNet chạy sau ByteTrack, khớp ID mới với gallery các track đã mất dựa trên cosine similarity embedding, phục hồi ID ổn định khi player ra khỏi khung hình rồi quay lại.

---

## Chạy local

**Mặc định** (video `data/08fd33_0.mp4` → `output/result.mp4`):

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python main.py
```

**Chỉ định video / output:**

```powershell
python main.py video.source_path="data/0bfacc_0.mp4" video.output_path="output/0bfacc_result.mp4"
```

**Override config (Hydra syntax):**

```powershell
# Tăng confidence, giảm interval pitch calibration
python main.py detect.confidence=0.4 pitch.nbjw_interval=5

# Bỏ pitch/radar để debug tracking nhanh
python main.py pitch.enabled=false annotate.radar.enabled=false

# Bật Re-ID
python main.py reid.enabled=true

# Stride 2 để xử lý nhanh hơn (bỏ qua frame chẵn)
python main.py video.stride=2
```

**Real-time mode** (hiển thị cv2 window, hỗ trợ webcam):

```powershell
python realtime.py --source data/08fd33_0.mp4
python realtime.py --source 0               # webcam
# Phím: q=thoát  p=pause  +/-=interval  s=screenshot
```

**Player stats** được xuất sau khi video kết thúc vào `output/player_stats/`:
- `player_stats.json` — distance, max/avg speed per player
- `player_XXXX_teamY.png` — trajectory overlay + crop collage

---

## Chạy trên Modal.com (GPU cloud)

**Setup một lần:**

```powershell
uv sync --extra modal
modal token new
```

**Chạy:**

```powershell
python main.py models.device=modal video.source_path=data/08fd33_0.mp4
```

Lệnh trên sẽ:
1. Đóng gói source code + weights vào Modal image
2. Chạy pipeline trên GPU H100 (Re-ID bật tự động)
3. Tải kết quả về `output/result.mp4` và `output/player_stats/player_stats.json`

**Tham số bổ sung:**

```powershell
# Chỉ định output path
python main.py models.device=modal \
  video.source_path=data/0bfacc_0.mp4 \
  video.output_path=output/0bfacc_result.mp4
```

> Image Modal tự build lần đầu (~5 phút). Các lần sau cache lại, chỉ upload video.

---

## Cấu trúc project

```
soccer_ai/
  pipeline.py          orchestrator — gọi các stage theo thứ tự
  detector.py          detect, filter, track, classify_teams
  calibration.py       NBJWCalibrator, PositionSmoother, transform_detections
  visualizer.py        build_annotators, annotate_frame
  radar.py             radar minimap, Voronoi overlay
  stats.py             PlayerStatsTracker
  reid.py              PlayerReID (OSNet wrapper)
  osnet.py             OSNet_x1_0 architecture + weight loader
  nbjw/                NBJW modules (cls_hrnet, utils_calib, utils_heatmap, config/)

conf/
  config.yaml          root config
  pipeline/            detect, filter, track, team, pitch, annotate, reid

weights/
  player_detection.pt
  SV_kp / SV_lines
  osnet_x1_0_sportsreid.pth.tar

research/
  training/
    detection/         finetune YOLO SoccerNet + notebooks
    pitch/             notebooks calibration approach cũ
    reid/              train_reid.py
  results/
    detection_baseline/
    detection_soccernet/
    pose_keypoint/
```

---

## Tham chiếu

- **NBJW calibration** — [No-Bells-Just-Whistles](https://github.com/mguti97/No-Bells-Just-Whistles) — Gutierrez et al., dual HRNet-W48 camera calibration từ field markings
- **OSNet Re-ID** — [Omni-Scale Feature Learning for Person Re-ID](https://arxiv.org/abs/1905.00953) — Zhou et al., ICCV 2019; weights finetuned trên SoccerNet Re-ID dataset
- **ByteTrack** — [supervision](https://github.com/roboflow/supervision) `sv.ByteTrack`
- **SoccerNet detection dataset** — weights `player_detection.pt` finetuned từ SoccerNet 2023 Tracking

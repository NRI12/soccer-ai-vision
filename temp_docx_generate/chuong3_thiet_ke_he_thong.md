# CHƯƠNG 3: THIẾT KẾ VÀ HIỆN THỰC HỆ THỐNG

---

## 3.1 Kiến Trúc Tổng Thể Hệ Thống

### 3.1.1 Tổng Quan Pipeline

Hệ thống phân tích video bóng đá được tổ chức theo kiến trúc **pipeline tuần tự** gồm sáu giai đoạn xử lý (stages), mỗi giai đoạn đảm nhiệm một nhiệm vụ độc lập và có thể bật/tắt riêng biệt qua cấu hình. Toàn bộ pipeline được điều phối bởi hàm `process_frame()` trong module `soccer_ai/pipeline.py`, hàm này nhận đầu vào là một frame ảnh RGB dưới dạng `numpy.ndarray` và trả về một từ điển `FrameData` chứa kết quả của tất cả các giai đoạn.

Thiết kế theo mô hình *data-flow pipeline* mang lại hai lợi ích chính: (1) mỗi giai đoạn được kiểm thử và phát triển độc lập mà không ảnh hưởng đến các giai đoạn khác; (2) việc bật/tắt từng giai đoạn cho phép so sánh hiệu năng và debug dễ dàng — ví dụ, chạy pipeline không có Re-ID để đo tốc độ thuần của tracking, hoặc tắt module Pitch để kiểm tra annotation mà không cần tính toán homography.

Sáu giai đoạn của pipeline theo thứ tự thực thi là:

| Thứ tự | Giai đoạn | Module | Chức năng |
|--------|-----------|--------|-----------|
| 1 | **Detect** | `soccer_ai/detector.py` | YOLO11 phát hiện đối tượng (8 lớp) |
| 2 | **Filter** | `soccer_ai/detector.py` | Lọc nhiễu, NMS, remap lớp 8→3 |
| 3 | **Track** | `soccer_ai/detector.py` | ByteTrack gán và duy trì tracker ID |
| 4 | **Re-ID** | `soccer_ai/reid.py` | OSNet khôi phục ID bị mất |
| 5 | **Pitch** | `soccer_ai/calibration.py` | NBJW homography + EMA smoothing |
| 6 | **Annotate** | `soccer_ai/visualizer.py` | Vẽ ellipse, nhãn, radar minimap |

Sau khi video loop kết thúc, module thống kê (`soccer_ai/stats.py`) thực hiện xử lý hậu kỳ (post-processing) để tính quãng đường, tốc độ và xuất file JSON cùng ảnh quỹ đạo cho từng cầu thủ.

### 3.1.2 Cấu Trúc Dữ Liệu Liên Module

Các giai đoạn trong pipeline giao tiếp với nhau thông qua từ điển `FrameData` — một cấu trúc dữ liệu mutable được truyền và bổ sung qua từng giai đoạn. Thiết kế này ưu tiên tính linh hoạt: mỗi giai đoạn chỉ đọc các key mà nó cần và ghi thêm các key mới mà không làm mất kết quả của giai đoạn trước. Bảng 3.1 liệt kê các key chính trong `FrameData` và giai đoạn tạo ra/tiêu thụ chúng.

**Bảng 3.1: Các key chính trong FrameData và luồng dữ liệu giữa các giai đoạn**

| Key | Kiểu dữ liệu | Tạo bởi | Tiêu thụ bởi | Nội dung |
|-----|-------------|---------|-------------|----------|
| `frame` | `np.ndarray` | Input | Annotate, Stats | Frame ảnh gốc (BGR) |
| `detections` | `sv.Detections` | Detect | Filter | Toàn bộ detection từ YOLO |
| `ball_detections` | `sv.Detections` | Filter | Pitch, Annotate | Detection bóng (đã pad) |
| `all_detections` | `sv.Detections` | Filter | Track, Re-ID, Team | Tất cả (trừ bóng, staff) |
| `players_detections` | `sv.Detections` | Team | Pitch, Annotate, Stats | Cầu thủ đã phân loại team |
| `referees_detections` | `sv.Detections` | Team | Pitch, Annotate | Trọng tài |
| `labels` | `list[str]` | Track | Annotate | Nhãn `#ID` cho mỗi detection |
| `pitch_players_xy` | `np.ndarray` | Pitch | Stats, Annotate | Tọa độ sân (canvas units) |
| `pitch_ball_xy` | `np.ndarray` | Pitch | Annotate | Tọa độ bóng trên sân |
| `pitch_players_tracker_id` | `np.ndarray` | Pitch | Stats | Tracker ID tương ứng |
| `players_class_id` | `np.ndarray` | Pitch | Stats, Annotate | Team ID (0 hoặc 1) |

---

## 3.2 Module Phát Hiện Đối Tượng (Detection)

### 3.2.1 Mô Hình YOLO11

Giai đoạn phát hiện đối tượng sử dụng mô hình **YOLO11** — thế hệ mới nhất của dòng YOLO do Ultralytics phát triển — được fine-tune trên SoccerNet Tracking Dataset để nhận diện 8 lớp đối tượng trong video bóng đá. YOLO11 kế thừa kiến trúc anchor-free từ YOLOv8 với các cải tiến quan trọng: khối C3k2 (Cross Stage Partial với kernel 2) thay thế C2f, mô-đun chú ý không gian C2PSA (Cross Stage Partial with Spatial Attention) ở tầng đặc trưng sâu, và backbone tối ưu cho inference nhanh hơn.

Mô hình được tải thông qua thư viện Ultralytics và thực hiện inference trong hàm `detect()`:

```python
def detect(frame, model, cfg, device):
    result = model(frame, conf=cfg.confidence, device=device, verbose=False)[0]
    return {"frame": frame, "detections": sv.Detections.from_ultralytics(result)}
```

Kết quả YOLO được chuyển đổi sang định dạng `sv.Detections` của thư viện Supervision — một cấu trúc dữ liệu chuẩn hóa lưu trữ bounding box (`xyxy`), độ tin cậy (`confidence`), và nhãn lớp (`class_id`) dưới dạng các mảng NumPy, tối ưu cho xử lý batch.

### 3.2.2 Schema Lớp 8 Đầu Ra

Mô hình phát hiện đối tượng được fine-tune để phân biệt 8 lớp theo schema SoccerNet, phản ánh sự khác biệt về đội bóng (trái/phải trong ảnh broadcast):

| ID Lớp | Tên lớp | Ghi chú |
|--------|---------|---------|
| 0 | `ball` | Bóng |
| 1 | `player_left` | Cầu thủ đội trái |
| 2 | `player_right` | Cầu thủ đội phải |
| 3 | `goalkeeper_left` | Thủ môn đội trái |
| 4 | `goalkeeper_right` | Thủ môn đội phải |
| 5 | `main_referee` | Trọng tài chính |
| 6 | `side_referee` | Trọng tài biên |
| 7 | `staff` | Nhân viên hỗ trợ |

### 3.2.3 Tham Số Detection

Ngưỡng confidence được đặt tại $\tau_{conf} = 0.3$ — thấp hơn mức mặc định 0.5 của YOLO — để đảm bảo không bỏ sót cầu thủ bị che khuất hoặc ở xa camera. Các detection có confidence thấp vẫn được giữ lại ở giai đoạn này vì ByteTrack ở giai đoạn sau sẽ xử lý chúng theo hai luồng riêng biệt (high-confidence và low-confidence) theo thuật toán BYTE.

### 3.2.4 Tương Thích Checkpoint Cũ

Một vấn đề kỹ thuật phát sinh khi tải checkpoint YOLO11 cũ là thiếu thuộc tính `all_head_dim` trong lớp `AAttn` (Attention Attention) do thay đổi schema giữa các phiên bản Ultralytics. Hàm `patch_aattn_compat()` được triển khai để tự động phát hiện và backfill thuộc tính còn thiếu:

```python
def patch_aattn_compat(model):
    for module in model.modules():
        if isinstance(module, block.AAttn) and not hasattr(module, "all_head_dim"):
            module.all_head_dim = int(module.head_dim) * int(module.num_heads)
```

Kỹ thuật này đảm bảo backward compatibility mà không cần serialize lại checkpoint — tránh rủi ro thay đổi trọng số trong quá trình lưu/tải.

---

## 3.3 Module Lọc và Phân Loại (Filter)

### 3.3.1 Xử Lý Bóng

Bóng được tách ra khỏi luồng xử lý chính ngay sau giai đoạn detection và xử lý riêng biệt vì hai lý do: (1) bóng không cần gán tracker ID theo cùng cơ chế với cầu thủ; (2) bóng thường nhỏ và bị che khuất một phần bởi cầu thủ nên cần padding để tăng diện tích bounding box trước khi chiếu tọa độ.

```python
ball_detections = detections[detections.class_id == ball_id]
ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=cfg.filter.ball_pad_px)
```

### 3.3.2 Loại Bỏ Staff và Áp Dụng NMS

Lớp `staff` (class ID = 7) bị loại bỏ hoàn toàn khỏi pipeline trước khi áp dụng NMS. Thứ tự này có chủ đích: nếu staff được đưa vào NMS trước, các bounding box của họ có thể suppress detection của cầu thủ đứng gần, dẫn đến mất track không đáng có.

NMS (Non-Maximum Suppression) được áp dụng với ngưỡng IoU $\tau_{nms} = 0.5$ theo chế độ class-agnostic (không phân biệt lớp khi tính IoU), cho phép loại bỏ các detection trùng lặp giữa các lớp khác nhau — ví dụ, cùng một cầu thủ bị detect vừa là `player_left` vừa là `goalkeeper_left`.

### 3.3.3 Remap Lớp 8→3 (Bước Quan Trọng Cho ByteTrack)

Sau NMS, 6 lớp đối tượng (loại trừ bóng và staff) được gộp lại thành 3 lớp theo quy ước pipeline nội bộ:

$$\text{player\_left}(1), \text{goalkeeper\_left}(3) \rightarrow \text{team\_left}(1)$$
$$\text{player\_right}(2), \text{goalkeeper\_right}(4) \rightarrow \text{team\_right}(2)$$
$$\text{main\_referee}(5), \text{side\_referee}(6) \rightarrow \text{referee}(3)$$

Bước remapping này **phải được thực hiện trước khi đưa detection vào ByteTrack** — đây là yêu cầu kỹ thuật quan trọng. ByteTrack trong thư viện Supervision hoạt động theo chế độ *class-aware*: nó chỉ matching detection với track thuộc cùng class_id. Nếu không remap, một thủ môn (goalkeeper) bị detection dao động giữa class 1 (`player_left`) và class 3 (`goalkeeper_left`) sẽ nhận tracker ID mới mỗi khi class thay đổi, gây ra hiện tượng ID switching không mong muốn.

Phép remap được thực hiện hiệu quả bằng vectorized lookup với NumPy:

```python
_remap = {
    int(ids.player_left):      1,
    int(ids.goalkeeper_left):  1,
    int(ids.player_right):     2,
    int(ids.goalkeeper_right): 2,
    int(ids.main_referee):     3,
    int(ids.side_referee):     3,
}
all_detections.class_id = np.vectorize(lambda c: _remap.get(c, c))(cid).astype(int)
```

---

## 3.4 Module Theo Dõi Đa Đối Tượng (Tracking)

### 3.4.1 Tích Hợp ByteTrack Qua Supervision

ByteTrack được tích hợp thông qua thư viện Supervision (`sv.ByteTrack` hoặc `sv.ByteTracker` tùy phiên bản), đóng gói toàn bộ logic thuật toán BYTE và Kalman Filter bên trong. Hàm `build_tracker()` khởi tạo tracker với frame rate thực tế của video được tính theo công thức:

$$\text{fps\_effective} = \left\lfloor \frac{\text{fps\_video}}{\text{stride}} \right\rfloor$$

trong đó `stride` là số frame bỏ qua giữa hai lần xử lý. Việc truyền `frame_rate` chính xác vào tracker là quan trọng: ByteTrack dùng giá trị này để hiệu chỉnh ma trận nhiễu quá trình (process noise) của Kalman Filter và tính thời gian tối đa để giữ track trong trạng thái "lost" (`lost_track_buffer` frames).

### 3.4.2 Tham Số Tracker

**Bảng 3.2: Tham số cấu hình ByteTrack**

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `track_activation_threshold` | 0.35 | Ngưỡng confidence tối thiểu để một detection được xét là "high confidence" trong thuật toán BYTE |
| `lost_track_buffer` | 90 | Số frame tối đa một track được duy trì ở trạng thái "lost" trước khi bị xóa (~3 giây @ 30fps) |
| `minimum_consecutive_frames` | 3 | Số frame liên tiếp tối thiểu một detection phải xuất hiện trước khi được xác nhận (tránh ID rác từ false positive) |
| `minimum_matching_threshold` | 0.1 | Ngưỡng IoU tối thiểu để tính là một match hợp lệ trong Hungarian Algorithm |

### 3.4.3 Luồng Xử Lý Tracking

Hàm `track()` thực hiện một thao tác duy nhất: cập nhật trạng thái ByteTrack với detections hiện tại và nhận về detections đã được gán tracker ID:

```python
def track(data, tracker):
    all_detections = tracker.update_with_detections(detections=data["all_detections"])
    data["all_detections"] = all_detections
    data["labels"] = [f"#{tid}" for tid in all_detections.tracker_id]
    return data
```

Sau bước này, mỗi detection trong `all_detections` được gán một `tracker_id` là số nguyên dương duy nhất và nhất quán xuyên suốt toàn bộ video. Nhãn dạng chuỗi `#ID` được tạo sẵn để sử dụng trong giai đoạn annotation.

### 3.4.4 Reset Tracker Giữa Các Video

Tracker được khởi tạo một lần duy nhất và reset (`tracker.reset()`) trước khi bắt đầu xử lý video mới. Việc reset giải phóng toàn bộ lịch sử track, đảm bảo tracker ID bắt đầu lại từ 1 cho mỗi video — tránh tình trạng tracker ID của video trước "rò rỉ" sang video sau khi tái sử dụng tracker object.

---

## 3.5 Module Nhận Diện Lại Cầu Thủ (Re-Identification)

### 3.5.1 Vị Trí Trong Pipeline

Module Re-ID (lớp `PlayerReID` trong `soccer_ai/reid.py`) được đặt **sau** giai đoạn Track và **trước** giai đoạn phân loại team — một thiết kế có chủ đích. Khi ByteTrack gặp trường hợp cầu thủ bị che khuất và quay lại frame, nó thường gán tracker ID mới. Re-ID module can thiệp ngay tại điểm này để phát hiện ID mới này thực chất là một cầu thủ quen thuộc và khôi phục ID ổn định cũ trước khi các giai đoạn sau nhận dữ liệu.

### 3.5.2 Kiến Trúc Gallery Hai Tầng

`PlayerReID` duy trì hai cấu trúc lưu trữ song song:

**Gallery active** (`_active`): ánh xạ `stable_id → embedding` cho tất cả cầu thủ đang hiện diện trong frame hiện tại. Embedding được cập nhật theo cơ chế EMA (Exponential Moving Average) mỗi `update_interval = 30` frame để phản ánh sự thay đổi ngoại hình theo thời gian (thay đổi góc nhìn, chiếu sáng):

$$\mathbf{e}_{t} = \alpha_{emb} \cdot \mathbf{e}_{new} + (1 - \alpha_{emb}) \cdot \mathbf{e}_{t-1}$$

với $\alpha_{emb} = 0.15$ — trọng số nhỏ đảm bảo embedding mới không thay đổi quá nhanh so với lịch sử.

**Gallery lost** (`_lost`): ánh xạ `stable_id → {"emb": embedding, "age": int}` cho các cầu thủ vừa biến mất khỏi frame. Mỗi frame, `age` của track trong gallery lost tăng thêm 1. Track bị xóa khi `age ≥ max_age = 120` frame (~4 giây @ 30fps).

### 3.5.3 Quy Trình Matching Re-ID

Khi ByteTrack gán một tracker ID mới (ID chưa từng xuất hiện), Re-ID module thực hiện quy trình 5 bước:

**Bước 1 — Trích xuất embedding:** Crop vùng ảnh tương ứng với bounding box, resize về $256 \times 128$ pixel (H × W), chuẩn hóa theo ImageNet mean/std, và đưa qua OSNet để lấy vector đặc trưng 512 chiều.

**Bước 2 — L2 normalization:** Vector đặc trưng được chuẩn hóa L2 để phép đo cosine similarity tương đương với tích vô hướng:

$$\hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2}$$

**Bước 3 — So sánh với gallery lost:** Tính cosine similarity giữa embedding của ID mới và tất cả embedding trong `_lost`:

$$\text{sim}(i, j) = \hat{\mathbf{e}}_i \cdot \hat{\mathbf{e}}_j$$

**Bước 4 — Chọn match tốt nhất:** ID ổn định có similarity cao nhất vượt ngưỡng $\tau_{reid} = 0.65$ được chọn làm match:

$$j^* = \arg\max_{j \notin \text{used}} \text{sim}(i, j), \quad \text{nếu } \text{sim}(i, j^*) > \tau_{reid}$$

**Bước 5 — Remap ID:** Tracker ID mới từ ByteTrack được thay thế bằng stable ID cũ trong toàn bộ `detections.tracker_id` array, đồng thời xóa stable ID này khỏi `_lost` gallery.

### 3.5.4 Bộ Lọc Crop Chất Lượng Thấp

Để tránh trích xuất embedding từ ảnh crop quá nhỏ (không đủ chi tiết ngoại hình để phân biệt cầu thủ), Re-ID áp dụng ngưỡng chiều cao tối thiểu `min_crop_h = 40` pixel. Detection có bounding box nhỏ hơn ngưỡng này bị bỏ qua trong cả hai bước matching và cập nhật gallery — đây là bảo đảm chất lượng embedding quan trọng, đặc biệt với cầu thủ ở xa camera.

---

## 3.6 Module Hiệu Chỉnh Camera và Ánh Xạ Tọa Độ (Pitch Calibration)

### 3.6.1 Sơ Đồ Chuyển Đổi Tọa Độ

Module hiệu chỉnh camera là thành phần kỹ thuật phức tạp nhất trong pipeline, thực hiện chuỗi chuyển đổi tọa độ 4 bước từ pixel ảnh sang tọa độ canvas sân:

$$\underbrace{\text{pixel ảnh}}_{\text{(u, v)}} \xrightarrow{\text{NBJW}} \underbrace{\text{cam\_params}}_{\text{K, R, t}} \xrightarrow{\text{P = K[R|t]}} \underbrace{\text{H}_{sn}}_{\text{3×3}} \xrightarrow{\text{invert}} \underbrace{\text{H}_{sn}^{-1}}_{\text{pixel→SN}} \xrightarrow{\mathbf{A}_{SN}} \underbrace{\text{canvas}}_{\text{(0–12000, 0–7000)}}$$

trong đó $\mathbf{A}_{SN}$ là ma trận affine chuyển từ hệ tọa độ SoccerNet (gốc tọa độ ở tâm sân, $x \in [-52.5, 52.5]$ m, $y \in [-34, 34]$ m) sang hệ canvas (gốc tọa độ ở góc trên-trái, 12000 × 7000 đơn vị):

$$\mathbf{A}_{SN} = \begin{pmatrix} \frac{12000}{105} & 0 & 52.5 \cdot \frac{12000}{105} \\ 0 & \frac{7000}{68} & 34 \cdot \frac{7000}{68} \\ 0 & 0 & 1 \end{pmatrix}$$

Ma trận biến đổi toàn phần từ pixel sang canvas được tính là:

$$\mathbf{H}_{sports} = \mathbf{A}_{SN} \cdot \mathbf{H}_{sn}^{-1}$$

### 3.6.2 Khởi Tạo NBJWCalibrator

Lớp `NBJWCalibrator` tải và khởi tạo hai mô hình HRNet-W48 song song khi khởi động pipeline:

- **Model kp** (`get_cls_net`): phát hiện giao điểm (keypoints) trên sân — ngưỡng $\tau_{kp} = 0.1486$
- **Model lines** (`get_cls_net_l`): phát hiện đầu cuối đường kẻ sân (line extremities) — ngưỡng $\tau_{lines} = 0.3880$

Cả hai mô hình nhận đầu vào là ảnh RGB được resize về $540 \times 960$ pixel và trả về tensor heatmap. Ngưỡng phát hiện được chọn dựa trên kết quả đánh giá của tác giả NBJW trên SoccerNet-Calibration — đây là các giá trị tối ưu hóa theo $\text{acc}_{5}$ (tỷ lệ keypoints phát hiện đúng với sai số ≤5 pixel).

### 3.6.3 Quy Trình Tính Homography

Với mỗi frame hiệu chỉnh (cứ mỗi `nbjw_interval` frame), phương thức `_infer()` thực hiện:

1. Chuyển đổi frame BGR → RGB và đưa về tensor PyTorch float32
2. Chạy inference song song hai model HRNet, thu được heatmap keypoints và heatmap lines
3. Trích xuất đỉnh heatmap bằng max-pooling (`get_keypoints_from_heatmap_batch_maxpool`)
4. Hoàn thiện bộ keypoints qua `complete_keypoints()` — kết hợp keypoints và line endpoints theo 5 nhóm hình học của NBJW
5. Cập nhật bộ voter `FramebyFrameCalib` và gọi `heuristic_voting()` để lấy camera parameters

Từ camera parameters, ma trận chiếu $3 \times 4$ được xây dựng:

$$\mathbf{P} = \mathbf{K} \cdot [\mathbf{R} \mid -\mathbf{R}\mathbf{t}] = \mathbf{K} \cdot \mathbf{R} \cdot [\mathbf{I} \mid -\mathbf{t}]$$

trong đó $\mathbf{K}$ là ma trận nội tham số camera (focal length, principal point), $\mathbf{R}$ là ma trận quay, và $\mathbf{t}$ là vector vị trí camera. Homography $\mathbf{H}_{sn}$ được trích từ $\mathbf{P}$ bằng cách lấy 3 cột (0, 1, 3) — bỏ cột tương ứng với trục Z vuông góc với mặt phẳng sân:

$$\mathbf{H}_{sn} = \mathbf{P}[:, [0, 1, 3]]$$

### 3.6.4 Kiểm Tra Sanity và Cache Transformer

Trước khi trả về transformer, một kiểm tra sanity được thực hiện: chiếu tâm ảnh qua $\mathbf{H}_{sports}$ và kiểm tra tọa độ kết quả có nằm trong phạm vi hợp lý của canvas sân ($x \in [-2000, 14000]$, $y \in [-2000, 9000]$). Nếu không thỏa mãn, transformer bị từ chối và kết quả trả về `None`.

Khi NBJW thất bại (do góc camera bị cắt, nhiều người che khuất vạch sân), pipeline sử dụng transformer từ lần hiệu chỉnh trước đó được lưu trong `last_transformer_ref`. Cơ chế caching này đảm bảo pipeline không bị gián đoạn khi hiệu chỉnh không thành công ở một số frame nhất định.

### 3.6.5 Chiếu Tọa Độ Detection

Hàm `transform_detections()` chiếu tọa độ điểm chân (BOTTOM_CENTER của bounding box) của từng detection lên bề mặt sân. Điểm chân được chọn thay vì tâm bounding box vì nó gần với điểm tiếp xúc thực tế của cầu thủ với mặt sân, giảm sai số chiếu tọa độ do góc nhìn phối cảnh (perspective distortion):

```python
data["pitch_players_xy"] = transformer.transform_points(
    players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
)
```

Phép chiếu được thực hiện bằng `cv2.perspectiveTransform()` với ma trận $\mathbf{H}_{sports}$ đã tính, cho kết quả tọa độ canvas $(x, y)$ trong khoảng $[0, 12000] \times [0, 7000]$.

### 3.6.6 EMA Smoothing Tọa Độ Sân

Homography thay đổi nhẹ giữa các frame do nhiều yếu tố: camera rung, thay đổi góc nhìn khi track bóng, biến động nhỏ trong phát hiện keypoints. Sự thay đổi này tạo ra "jitter" trên minimap — các điểm cầu thủ nhảy loạn mặc dù cầu thủ thực tế di chuyển mượt mà.

Lớp `PositionSmoother` áp dụng bộ lọc EMA per-entity: mỗi tracker ID có trạng thái EMA riêng, cập nhật theo:

$$\hat{\mathbf{p}}_t^{(i)} = \alpha \cdot \mathbf{p}_t^{(i)} + (1-\alpha) \cdot \hat{\mathbf{p}}_{t-1}^{(i)}$$

với $\alpha = 0.4$ (trọng số cho quan sát mới) và $\mathbf{p}_t^{(i)}$ là tọa độ sân thô của cầu thủ $i$ tại frame $t$. Giá trị $\alpha = 0.4$ là điểm cân bằng giữa phản hồi nhanh (cần thiết để minimap theo kịp di chuyển thực tế) và làm mượt đủ mạnh (giảm jitter homography). Track không xuất hiện trong frame sẽ bị xóa sau `max_age = 30` frame không active để giải phóng bộ nhớ.

---

## 3.7 Module Phân Loại Đội (Team Classification)

### 3.7.1 Phân Tách Cầu Thủ và Trọng Tài

Sau khi tracking, hàm `classify_teams()` thực hiện phân tách `all_detections` thành hai nhóm riêng biệt: cầu thủ (`players_detections`) và trọng tài (`referees_detections`). Phân tách dựa trên `class_id` sau bước remap 8→3: class 1 và 2 là cầu thủ, class 3 là trọng tài.

### 3.7.2 Remap Lần Hai: 1,2,3 → 0,1,2

Sau phân tách, class ID được chuẩn hóa lần cuối về quy ước pipeline:

$$\text{team\_left}(1) \rightarrow \text{team\_0}(0), \quad \text{team\_right}(2) \rightarrow \text{team\_1}(1), \quad \text{referee}(3) \rightarrow \text{referee}(2)$$

Quy ước 0-indexed này nhất quán với cách thư viện Supervision và sports library định nghĩa team ID trong các visualizer.

### 3.7.3 Gán Team Dựa Trên Class YOLO

Đáng lưu ý là hệ thống **không** sử dụng phân cụm màu áo (K-Means clustering) để phân loại đội — một phương pháp phổ biến trong các hệ thống tracking bóng đá khác. Thay vào đó, phân loại đội được lấy trực tiếp từ class ID của mô hình YOLO11 đã được fine-tune. Mô hình đã học phân biệt `player_left` và `player_right` trong quá trình training trên SoccerNet Dataset — nơi "trái" và "phải" tương ứng với hai đội trong cách bố trí camera broadcast chuẩn. Phương pháp này nhanh hơn, ổn định hơn với đồng phục tương tự màu, và không cần bước xử lý thêm.

---

## 3.8 Module Thống Kê Cầu Thủ (Player Statistics)

### 3.8.1 Tích Lũy Quỹ Đạo

Lớp `PlayerStatsTracker` tích lũy dữ liệu tọa độ sân của từng cầu thủ qua toàn bộ video. Phương thức `update()` được gọi mỗi frame sau giai đoạn Pitch, lưu tọa độ canvas và frame index vào danh sách `_positions[tracker_id]`. Song song đó, hệ thống thực hiện **majority voting** để xác định team ID ổn định: mỗi frame, team ID quan sát được cho tracker ID tương ứng được cộng vào bộ đếm vote; cuối video, team có số vote cao nhất được chọn làm team chính thức.

### 3.8.2 Tính Tốc Độ và Quãng Đường

Sau khi video loop kết thúc, `compute_stats()` tính các chỉ số cho từng cầu thủ theo quy trình:

**Bước 1 — Chuyển đổi đơn vị:** Tọa độ canvas được chuyển sang mét thực tế:

$$x_m = x_{canvas} \cdot \frac{105}{12000}, \quad y_m = y_{canvas} \cdot \frac{68}{7000}$$

với 105m và 68m là kích thước chuẩn sân bóng đá FIFA.

**Bước 2 — Làm mượt quỹ đạo:** Áp dụng rolling mean window = 9 frame để loại bỏ nhiễu homography:

$$\bar{\mathbf{p}}_i = \frac{1}{|W_i|} \sum_{j \in W_i} \mathbf{p}_j$$

trong đó $W_i = [\max(0, i-4), \min(n, i+5)]$ là cửa sổ trung tâm quanh frame $i$.

**Bước 3 — Tính tốc độ tức thời:** Tốc độ giữa hai frame liên tiếp:

$$v_i = \frac{\|\bar{\mathbf{p}}_{i+1} - \bar{\mathbf{p}}_i\|_2}{\Delta t_i}$$

trong đó $\Delta t_i = \frac{\text{frame\_idx}_{i+1} - \text{frame\_idx}_i}{\text{fps}}$ tính theo giây thực.

**Bước 4 — Lọc spike tốc độ:** Tốc độ tức thời vượt $v_{max} = 10$ m/s (~36 km/h — ngưỡng sprint tối đa của vận động viên chuyên nghiệp) bị loại bỏ, tương ứng khoảng cách dịch chuyển bị loại khỏi tổng quãng đường:

$$\text{valid}_i = \mathbf{1}[v_i \leq v_{max}]$$

**Bước 5 — Tổng hợp thống kê cuối:**

$$d_{total} = \sum_{i: \text{valid}_i} \|\Delta\mathbf{p}_i\|_2, \quad v_{max\_actual} = \max_{i: \text{valid}_i} v_i, \quad v_{avg} = \frac{\sum_{i: \text{valid}_i} v_i}{|\{i : \text{valid}_i\}|}$$

### 3.8.3 Xuất Kết Quả

Kết quả thống kê được xuất ra hai dạng:

**File JSON** (`player_stats.json`): mảng các đối tượng JSON, mỗi đối tượng chứa `tracker_id`, `team`, `num_frames`, `total_distance_m`, `max_speed_ms`, `avg_speed_ms`, `max_speed_kmh`, `avg_speed_kmh`.

**Ảnh overlay PNG** (`player_XXXX_teamY.png`): ảnh quỹ đạo trên sân với đường di chuyển màu dần theo thời gian (từ xanh lúc bắt đầu đến màu đội bóng lúc kết thúc), bảng thông số tóm tắt ở cuối, và collage tối đa 8 crop ảnh cầu thủ ở góc phải.

**Video highlight** (tùy chọn): một clip MP4 riêng cho từng cầu thủ với minimap vị trí theo thời gian thực, chỉ xuất cho cầu thủ có ít nhất `min_frames_for_video = 280` frame được phát hiện.

---

## 3.9 Module Trực Quan Hóa (Annotation)

### 3.9.1 Chú Thích Trên Frame

Giai đoạn cuối của pipeline vẽ các yếu tố trực quan lên frame video:

- **Ellipse dưới chân:** Thay vì bounding box vuông, hệ thống vẽ nửa ellipse dưới chân mỗi cầu thủ — thể hiện điểm tiếp xúc thực tế với mặt sân, dễ nhìn hơn và ít che khuất cầu thủ hơn. Ellipse của cầu thủ team A và team B được tô màu khác nhau theo cấu hình.
- **Nhãn ID:** Số tracker ID hiển thị trên đầu mỗi cầu thủ bằng font DUPLEX với outline đen để đọc được trên mọi nền.
- **Tam giác cho trọng tài:** Trọng tài được đánh dấu bằng tam giác thay vì ellipse để phân biệt trực quan.
- **Marker bóng:** Bóng được đánh dấu riêng biệt (thường là hình tròn trắng).

### 3.9.2 Radar Minimap

Ở góc phải dưới frame, hệ thống vẽ một minimap radar thể hiện vị trí tất cả cầu thủ và bóng trên sơ đồ sân được thu nhỏ. Minimap sử dụng thư viện `sports` của Roboflow để render sân (`draw_pitch`) và hiển thị điểm (`draw_points_on_pitch`). Cầu thủ team A và B được tô màu khác nhau; bóng được hiển thị màu trắng với viền đen; nhãn tracker ID được vẽ phía trên mỗi điểm. Minimap được composited lên frame chính với tỷ lệ chiều rộng 36% và làm mờ vùng nền phía sau để tăng khả năng đọc.

---

## 3.10 Cấu Hình Hệ Thống (Hydra Configuration)

### 3.10.1 Hệ Thống Cấu Hình Phân Cấp

Toàn bộ pipeline được cấu hình thông qua **Hydra** — framework quản lý cấu hình của Facebook Research cho ứng dụng Python phức tạp. Hydra cho phép tổ chức cấu hình theo cấu trúc phân cấp thư mục, compose tự động các file cấu hình con, và override bất kỳ tham số nào từ dòng lệnh mà không cần sửa code.

Cấu trúc cấu hình của dự án:

```
conf/
├── config.yaml          # Entry point, compose tất cả config con
├── pipeline/
│   ├── detect.yaml      # Tham số YOLO detection
│   ├── filter.yaml      # NMS threshold, ball pad, class_agnostic
│   ├── track.yaml       # ByteTrack parameters
│   ├── team.yaml        # Team classification settings
│   ├── pitch.yaml       # NBJW calibration, EMA, interval
│   ├── annotate.yaml    # Colors, radar settings, visual options
│   └── reid.yaml        # OSNet weights, threshold, gallery settings
```

### 3.10.2 Cơ Chế Auto-Detection Thiết Bị

Pipeline hỗ trợ ba chế độ thực thi:

```yaml
models:
  device: "auto"   # Tự động: CUDA nếu có, fallback CPU
                   # "cpu": ép buộc CPU
                   # "0": GPU đầu tiên
                   # "modal": chạy trên cloud Modal.com
```

Hàm `_resolve_device()` sử dụng `torch.cuda.is_available()` để kiểm tra GPU ở chế độ `auto`. Chế độ `modal` kích hoạt một subprocess chạy `modal run modal_runner.py` — delegate toàn bộ xử lý lên GPU H100 trên Modal.com thông qua CLI, cho phép chạy inference nhanh mà không cần GPU local.

### 3.10.3 Override Tham Số Từ Dòng Lệnh

Hydra cho phép override tham số bất kỳ khi chạy mà không cần sửa file cấu hình:

```bash
# Đổi video đầu vào
python main.py video.source_path=data/573e61_0.mp4

# Tắt module pitch calibration (nhanh hơn nhưng không có thống kê)
python main.py pitch.enabled=false

# Tăng ngưỡng confidence detection
python main.py detect.confidence=0.5

# Chạy trên cloud H100
python main.py models.device=modal
```

Cơ chế này đặc biệt hữu ích khi thực nghiệm với nhiều cấu hình: mỗi lần chạy tạo ra một thư mục `outputs/YYYY-MM-DD/HH-MM-SS/` chứa file `.hydra/config.yaml` ghi lại toàn bộ cấu hình đã dùng — đảm bảo tính tái hiện (reproducibility) của thực nghiệm.

---

## 3.11 Xử Lý Video và Luồng Chính

### 3.11.1 Cấu Trúc Video Loop

Vòng lặp xử lý video sử dụng `sv.get_video_frames_generator()` của Supervision — một generator lazy-loading frame theo yêu cầu, không tải toàn bộ video vào RAM. Tham số `stride` cho phép bỏ qua $s-1$ frame giữa mỗi hai frame được xử lý, giảm workload khi cần inference nhanh hơn mà không cần độ chính xác thống kê cao:

```python
frames = sv.get_video_frames_generator(source_path=source_path, stride=cfg.video.stride)
with sv.VideoSink(target_path=output_path, video_info=video_info) as sink:
    for i, frame in enumerate(tqdm(frames, total=total)):
        data = process_frame(frame, ..., frame_idx=i)
        if player_stats_tracker is not None:
            player_stats_tracker.update(data, i, frame)
        sink.write_frame(data["frame"])
```

`sv.VideoSink` ghi frame kết quả vào file MP4 theo stream — không buffer toàn bộ video trong RAM trước khi ghi, quan trọng khi xử lý video dài.

### 3.11.2 Quản Lý Bộ Nhớ

Pipeline được thiết kế để chạy trong RAM giới hạn:

- Generator lazy-loading: chỉ một frame trong RAM tại một thời điểm
- `PlayerStatsTracker` lưu tối đa `_MAX_CROPS = 8` crop ảnh mỗi cầu thủ (mỗi `_CROP_INTERVAL = 60` frame lưu một crop)
- Gallery EMA lost tracks bị xóa sau `max_age = 120` frame inactive
- EMA position smoother xóa track cũ sau `max_age = 30` frame không active

### 3.11.3 Tải Trọng Số Tự Động

Hệ thống tích hợp cơ chế tự động tải weights khi chạy lần đầu. Hàm `ensure_runs_weights()` kiểm tra sự tồn tại của các file weights cần thiết và tải về từ Google Drive nếu thiếu. Tương tự, `ensure_weights()` trong module Re-ID xử lý việc tải `osnet_x1_0_sportsreid.pth.tar`. Cơ chế này cho phép người dùng mới chạy hệ thống chỉ bằng `python main.py` mà không cần tải thủ công các file lớn.

---

## 3.12 Kết Quả Thực Nghiệm

### 3.12.1 Môi Trường Thực Nghiệm

Toàn bộ thực nghiệm được thực hiện trên video mẫu `data/08fd33_0.mp4` — đoạn clip 25 giây (747 frame) từ góc quay broadcast chuẩn độ phân giải 1080p @ 30fps. Pipeline được chạy với đầy đủ 6 giai đoạn trên thiết bị có GPU CUDA.

### 3.12.2 Kết Quả Tracking và Re-ID

Sau khi xử lý toàn bộ video, hệ thống theo dõi được tổng cộng **30 tracker ID** khác nhau (bao gồm cầu thủ, trọng tài, và các ID ngắn ngủi từ false positive). Trong số đó, **10 cầu thủ** duy trì hiện diện trong ít nhất 600/747 frame — tương đương hơn 80% thời lượng video — cho thấy khả năng tracking liên tục ổn định của ByteTrack + Re-ID.

**Bảng 3.3: Kết quả thống kê cầu thủ (top 10 theo quãng đường di chuyển)**

| Tracker ID | Đội | Số Frame | Quãng Đường (m) | Tốc Độ Max (km/h) | Tốc Độ TB (km/h) |
|-----------|-----|----------|-----------------|-------------------|------------------|
| 16 | Team 0 | 746 | **106.2** | 35.5 | 13.3 |
| 11 | Team 0 | 747 | 100.8 | 32.2 | 12.6 |
| 12 | Team 1 | 745 | 99.4 | 33.5 | 12.3 |
| 26 | Team 0 | 661 | 93.6 | 35.1 | 12.9 |
| 9 | Team 1 | 746 | 89.1 | 35.5 | 11.1 |
| 4 | Team 1 | 742 | 88.2 | **35.8** | 11.1 |
| 28 | Team 1 | 641 | 87.8 | 33.0 | 12.5 |
| 5 | Team 1 | 733 | 87.1 | 35.3 | 10.9 |
| 27 | Team 0 | 661 | 85.5 | 28.6 | 11.7 |
| 14 | Team 1 | 747 | 85.2 | 35.8 | 10.4 |

### 3.12.3 Phân Tích Kết Quả

Kết quả cho thấy hệ thống đạt được các chỉ số phù hợp với đặc tính thực tế của bóng đá chuyên nghiệp:

**Tốc độ tối đa:** Tracker #1 đạt tốc độ cao nhất là **36.0 km/h** (9.99 m/s) — ngưỡng sát với sprint tối đa của cầu thủ chuyên nghiệp. Tốc độ max của nhóm top 10 dao động trong khoảng 32–36 km/h, phù hợp với dải tốc độ sprint thực tế (~28–35 km/h).

**Quãng đường:** Trong 25 giây, cầu thủ di chuyển nhiều nhất đạt **106.2 m**. Ngoại suy tuyến tính lên 90 phút tương đương ~14.5 km — xấp xỉ quãng đường trung bình của tiền vệ chạy nhiều trong trận đấu thực tế (10–13 km/90 phút). Sự sai lệch nhỏ này phản ánh bản chất đoạn clip được chọn lọc (giai đoạn cường độ cao) thay vì toàn trận.

**Tốc độ trung bình:** Dao động 9.9–13.3 km/h trong nhóm cầu thủ chính, cao hơn tốc độ đi bộ (4–6 km/h) và chạy nhẹ (8–10 km/h) — phù hợp với giai đoạn thi đấu tích cực.

**Tracker ID ngắn ngủi:** Một số tracker ID có rất ít frame (tracker #19: 2 frame, #22: 11 frame, #35: 11 frame) — đây là các false positive hoặc đối tượng đi qua frame trong thời gian ngắn. Hệ thống không tự động lọc bỏ các tracker này trong pipeline mà để người dùng quyết định ngưỡng `min_frames_for_video` khi xuất highlight video.

### 3.12.4 Output Artifacts

Sau mỗi lần chạy pipeline, hệ thống tạo ra các file sau:

**Bảng 3.4: Các file kết quả được tạo ra**

| File | Đường dẫn | Mô tả |
|------|-----------|-------|
| Video phân tích | `output/result.mp4` | Video với ellipse, nhãn tracker ID, radar minimap |
| Thống kê JSON | `output/player_stats/player_stats.json` | Dữ liệu thống kê tất cả cầu thủ |
| Ảnh quỹ đạo | `output/player_stats/player_XXXX_teamY.png` | Quỹ đạo trên sân + crop collage |
| Highlight video | `output/player_stats/player_XXXX_teamY_highlight.mp4` | Clip highlight từng cầu thủ |
| Cấu hình Hydra | `outputs/YYYY-MM-DD/HH-MM-SS/` | Lịch sử cấu hình đầy đủ mỗi lần chạy |

---

## 3.13 Tóm Tắt Chương

Chương này đã trình bày thiết kế và hiện thực chi tiết của hệ thống phân tích video bóng đá theo kiến trúc pipeline 6 giai đoạn. Hệ thống tích hợp năm mô hình học sâu (YOLO11, ByteTrack, HRNet-W48/NBJW, OSNet) thành một pipeline mạch lạc, trong đó mỗi module đảm nhiệm một nhiệm vụ rõ ràng và giao tiếp thông qua cấu trúc `FrameData` thống nhất.

Các đặc điểm thiết kế nổi bật bao gồm: (1) remap lớp 8→3 trước ByteTrack để tránh ID switching do dao động class; (2) tích hợp Re-ID gallery hai tầng (active/lost) để khôi phục ID bị mất; (3) chuỗi chuyển đổi tọa độ đa bước từ pixel sang tọa độ sân thực tế qua homography NBJW; (4) EMA smoothing per-entity để giảm jitter homography trên minimap; (5) lọc spike tốc độ dựa trên giới hạn sinh lý để đảm bảo thống kê đáng tin cậy. Kết quả thực nghiệm trên video 25 giây cho thấy hệ thống theo dõi được 30 đối tượng, đạt các chỉ số quãng đường và tốc độ phù hợp với đặc tính thực tế của thi đấu bóng đá chuyên nghiệp.

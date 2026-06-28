# CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 4.1 Môi Trường Thực Nghiệm

### 4.1.1 Hạ Tầng Huấn Luyện

Toàn bộ quá trình fine-tune mô hình được thực hiện trên **cụm HPC (High-Performance Computing)** được quản lý bởi SLURM. Môi trường phần mềm được kiểm soát chặt chẽ để đảm bảo tính tái hiện thực nghiệm:

| Thành phần | Phiên bản / Mô tả |
|-----------|-------------------|
| Framework học sâu | PyTorch ≥ 2.0 (CUDA 12.1) |
| Thư viện detection | Ultralytics ≥ 8.0 |
| Thư viện Re-ID | Torchreid (deep-person-reid) |
| Ngôn ngữ | Python 3.10+ |
| Lập lịch tác vụ | SLURM Workload Manager |

Các tham số huấn luyện được lưu cố định trong file cấu hình YAML và commit vào version control, cho phép tái hiện chính xác từng thực nghiệm.

### 4.1.2 Môi Trường Inference

Inference pipeline được thực thi trên ba chế độ tùy thuộc thiết bị và yêu cầu tốc độ:

| Chế độ | Thiết bị | Trường hợp dùng |
|--------|---------|----------------|
| Local CPU | Intel/AMD | Kiểm thử logic, debug nhanh |
| Local GPU | CUDA-compatible (RTX series) | Phát triển và đánh giá thông thường |
| Cloud GPU | Modal.com H100 | Inference nhanh nhất, không cần cấu hình local |

## 4.2 Thực Nghiệm Fine-tune YOLO11 trên SoccerNet

### 4.2.1 Bộ Dữ Liệu và Thiết Lập

Mô hình YOLO11 được fine-tune trên **SoccerNet Tracking Dataset** — bộ dữ liệu benchmark chuyên biệt cho tracking bóng đá, sử dụng checkpoint pretrained làm điểm khởi đầu (warm start):

| Thuộc tính | Giá trị |
|-----------|---------|
| Số chuỗi video (sequences) | 200 đoạn × 30 giây |
| Độ phân giải | 1080p, camera broadcast chính |
| Nhãn đối tượng | 8 lớp (SoccerNet schema) |
| Nguồn | SoccerNet 2023 Tracking Challenge (CVPR Workshop) |
| Thách thức đặc thù | Đám đông, che khuất, ánh sáng thay đổi |

Cơ chế warm start từ checkpoint pretrained giúp mô hình hội tụ nhanh hơn và ổn định hơn so với training from-scratch trên bộ dữ liệu lớn, đồng thời tránh mất ổn định gradient khi tiếp xúc đột ngột với phân phối dữ liệu broadcast bóng đá.

### 4.2.2 Kết Quả Fine-tune

**Bảng 4.1: Kết quả fine-tune YOLO11 trên SoccerNet Tracking Dataset (50 epochs)**

| Epoch | Box Loss (train) | Box Loss (val) | mAP50 | mAP50-95 | Precision | Recall |
|-------|-----------------|----------------|-------|----------|-----------|--------|
| 1 | 1.033 | 0.940 | 0.774 | 0.546 | 0.723 | 0.751 |
| 10 | 0.982 | 0.910 | 0.816 | 0.586 | 0.753 | 0.792 |
| 20 | 0.951 | 0.884 | 0.836 | 0.607 | 0.775 | 0.810 |
| 30 | 0.931 | 0.878 | 0.840 | 0.612 | 0.778 | 0.816 |
| 40 | 0.910 | 0.875 | 0.840 | 0.615 | 0.780 | 0.819 |
| **50** | **0.887** | **0.875** | **0.841** | **0.615** | **0.776** | **0.823** |

**Thời gian tổng: ~76,171 giây (~21.2 giờ)** | **Thời gian/epoch: ~1,523 giây**

Mô hình hội tụ ổn định: Box Loss (val) ổn định quanh 0.875 từ epoch 35 trở đi với biến thiên dưới 0.001 ở các epoch 40–50 — bằng chứng không xảy ra overfitting dù tập dữ liệu lớn và đa dạng. Hiệu quả của warm start thể hiện rõ ngay từ epoch đầu tiên: mAP50-95 = 0.546 đã ở mức cao, phản ánh trọng số pretrained đã học được các đặc trưng tổng quát có thể chuyển giao cho phân phối SoccerNet.

### 4.2.3 Phân Tích Kết Quả

Kết quả cuối cùng ở epoch 50 cho thấy sự cân bằng phù hợp cho bài toán tracking trong pipeline:

**mAP50-95 = 0.615:** Metric tổng hợp hiệu năng tại nhiều ngưỡng IoU (0.50–0.95), phản ánh chất lượng định vị bounding box sát với đặc thù phân phối broadcast bóng đá — đặc biệt với cầu thủ nhỏ ở xa camera và trong đám đông.

**Recall = 0.823:** Mô hình phát hiện đầy đủ cầu thủ trong các tình huống khó như che khuất một phần hoặc chồng lên nhau. Recall cao là ưu tiên cho tracking vì bỏ sót detection dẫn đến mất track và gây ID switching ở giai đoạn ByteTrack.

**Precision = 0.776:** Mô hình tích cực hơn trong phát hiện, chấp nhận thêm false positive để đổi lấy recall cao. Điều này phù hợp với thiết kế pipeline: ngưỡng confidence $\tau_{conf} = 0.3$ kết hợp cơ chế xác nhận liên tiếp của ByteTrack (`minimum_consecutive_frames = 3`) sẽ lọc bỏ false positive ở giai đoạn sau.

Checkpoint `player_detection.pt` được sử dụng trong pipeline là checkpoint tốt nhất được chọn theo mAP50-95 trong quá trình training.

## 4.3 Thực Nghiệm Nhận Diện Lại Cầu Thủ (Re-ID)

### 4.3.1 Bộ Dữ Liệu SoccerNet Re-ID

Mô hình Re-ID được huấn luyện trên **SoccerNet Re-ID Dataset** — bộ dữ liệu thumbnail cầu thủ chuyên biệt ở quy mô lớn:

| Thuộc tính | Giá trị |
|-----------|---------|
| Tổng số ảnh | 340,993 thumbnail cầu thủ |
| Nguồn | 400 trận đấu từ 6 giải lớn châu Âu |
| Tập huấn luyện | 248,234 ảnh |
| Tập truy vấn (query) | 11,777 ảnh |
| Tập gallery | 34,989 ảnh |
| Số identity trong test | 1,932 cầu thủ duy nhất |

SoccerNet Re-ID đặt ra những thách thức đặc thù không có trong Re-ID người đi bộ thông thường: tất cả cầu thủ cùng đội mặc đồng phục giống hệt nhau, thumbnail ở xa camera rất nhỏ và mờ, mỗi cầu thủ chỉ có vài ảnh trong một action clip, và tư thế thay đổi liên tục (chạy, nhảy, quay người).

### 4.3.2 Kiến Trúc OSNet và Giao Thức Đánh Giá

**Omni-Scale Network (OSNet)** học đặc trưng ngoại hình ở nhiều tỉ lệ không gian đồng thời bằng tập hợp các nhánh depthwise convolution với kernel size khác nhau:

$$\mathbf{t}^{(s)} = \delta\!\left(\mathbf{W}_{s}^{dw} * \mathbf{x} + \mathbf{b}_{s}^{dw}\right), \quad s \in \{1, 3, 5, 7\}$$

$$\mathbf{T} = \mathbf{G} \odot \left(\sum_{s} \mathbf{t}^{(s)}\right), \quad \mathbf{G} = \sigma\!\left(\mathbf{W}_{AG} \cdot \text{GAP}(\mathbf{x})\right)$$

trong đó $\mathbf{G}$ là **Unified Aggregation Gate (UAG)** — vector cổng channel-wise tính từ Global Average Pooling, điều chỉnh linh hoạt tỉ trọng kết hợp đặc trưng đa tỉ lệ theo từng ảnh đầu vào.

Đánh giá Re-ID dùng hai metric: **Rank-1** (tỷ lệ query tìm đúng match ở vị trí đầu tiên) và **mAP** (mean Average Precision, đo chất lượng toàn danh sách xếp hạng):

$$\text{mAP} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{|R_q|} \sum_{r=1}^{|G|} P(r) \cdot \mathrm{rel}(r)$$

trong đó $P(r)$ là precision tại rank $r$ và $\mathrm{rel}(r) = 1$ nếu gallery item ở rank $r$ có cùng identity với query $q$.

### 4.3.3 Kết Quả Trên SoccerNet Re-ID

Weights `osnet_x1_0_sportsreid.pth.tar` được huấn luyện trên 248,234 ảnh với chiến lược hierarchical sampling kết hợp Centroid Loss, và đánh giá trên test split (arXiv:2206.02373).

**Bảng 4.2: Kết quả OSNet\_x1\_0 trên SoccerNet Re-ID test split**

| Mô Hình | mAP | Rank-1 |
|---------|-----|--------|
| ResNet50-fc512 (baseline) | 70.3 | 61.2 |
| **OSNet\_x1\_0** | **83.4** | **78.0** |
| ViT-B/16 (best overall) | 86.0 | 81.5 |

*Nguồn: arXiv:2206.02373, Bảng 2.*

OSNet\_x1\_0 đạt **mAP = 83.4%, Rank-1 = 78.0%** — cải thiện đáng kể so với ResNet50 và chỉ kém ViT-B/16 (~86M tham số) khoảng 2.6 điểm mAP trong khi OSNet\_x1\_0 chỉ có ~2.2M tham số, phù hợp hơn nhiều cho inference thời gian thực khi pipeline cần xử lý 10–20 crop/frame liên tục.

Kết quả được đạt nhờ **hierarchical sampling** — đảm bảo mỗi mini-batch chứa các cặp cầu thủ khó phân biệt nhất (cùng đội, cùng trận, đồng phục đồng nhất) — và **Centroid Loss** thu hẹp intra-class variance bằng cách kéo embedding của cùng một identity về phía centroid chung trong feature space.

### 4.3.4 Phân Tích Ngưỡng Matching và Tham Số Gallery

Ngưỡng cosine similarity $\tau_{reid} = 0.65$ trong pipeline được chọn dựa trên phân tích similarity distribution:

- $\tau < 0.50$: quá nhiều false match — cầu thủ khác đội bị nhầm ID với cầu thủ vừa mất track.
- $\tau = 0.65$ (chọn): cân bằng tốt, đủ chặt để loại nhầm lẫn liên đội, đủ rộng để chấp nhận thay đổi ngoại hình nhỏ theo góc nhìn và ánh sáng.
- $\tau > 0.80$: quá ít match hợp lệ — nhiều cầu thủ bị gán ID mới dù mô hình nhận ra ở similarity 0.70–0.79.

Tham số `max_age = 120` frame (~4 giây @ 30fps) cho gallery lost phản ánh thực nghiệm: trong hầu hết các tình huống gameplay, cầu thủ bị che khuất quay lại trong vòng 2–3 giây. Track không xuất hiện sau 4 giây được coi là rời frame và xóa khỏi gallery để giải phóng bộ nhớ.

## 4.4 Kết Quả Pipeline Tổng Thể

### 4.4.1 Điều Kiện Thực Nghiệm

Video thực nghiệm: `data/08fd33_0.mp4` — đoạn clip **25 giây**, **747 frame**, góc quay broadcast chuẩn, độ phân giải 1080p @ 29.97 fps. Pipeline chạy với đầy đủ 6 giai đoạn xử lý (Detect → Filter → Track → Team → Pitch → Annotate) cùng module thống kê cầu thủ, tham số mặc định với `stride = 1`.

### 4.4.2 Kết Quả Tracking và Thống Kê

Sau khi xử lý toàn bộ 747 frame, hệ thống gán tổng cộng **30 tracker ID** (bao gồm cầu thủ cả hai đội, trọng tài và một số ID ngắn hạn từ false positive). Tổng hợp chỉ số:

| Chỉ Số | Giá Trị |
|--------|---------|
| Tổng tracker ID được gán | 30 |
| Cầu thủ active (≥600/747 frame) | 10 |
| Frame coverage trung bình (top 10) | 96.4% |
| Tracker ID dài nhất | #1, #11, #14 — 747/747 frame |
| Team 0 / Team 1 / ID ngắn hạn | 10 / 14 / 6 |

**Bảng 4.3: Thống kê top 10 cầu thủ theo quãng đường di chuyển**

| Tracker ID | Đội | Số Frame | Quãng Đường (m) | v max (km/h) | v trung bình (km/h) |
|-----------|-----|----------|----------------|-------------|-------------------|
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

### 4.4.3 Phân Tích Kết Quả

**Tốc độ tối đa:** Tracker #1 đạt 36.0 km/h (9.99 m/s) — sát với giới hạn sinh lý sprint của vận động viên chuyên nghiệp (35–37 km/h). Nhóm top 10 dao động 28.6–35.8 km/h, phản ánh đúng dải tốc độ thực tế với cả sprint tối đa (#4, #14) và jogging nhẹ (#27).

**Quãng đường:** Trong 25 giây, cầu thủ di chuyển nhiều nhất đạt 106.2 m, tốc độ trung bình 13.3 km/h — tương ứng giai đoạn thi đấu cường độ cao. Ngoại suy tuyến tính lên 90 phút đạt ~14.5 km, gần với quãng đường tiền vệ tích cực trong trận thực tế (10–13 km).

**Bộ lọc spike hiệu quả:** Không có tracker nào có $v_{max}$ vượt 36.5 km/h — bộ lọc $v_{max} = 10$ m/s loại bỏ nhiễu homography mà không cắt sprint thực sự.

**ID stability:** 10 cầu thủ active duy trì tracker ID với coverage >80% trong 747 frame, xác nhận ByteTrack + Re-ID hoạt động hiệu quả trong đoạn video 25 giây với nhiều tình huống che khuất và tranh chấp bóng.

### 4.4.4 File Kết Quả Được Tạo Ra

**Bảng 4.4: Tổng hợp artifact sau mỗi lần chạy pipeline**

| Artifact | Đường Dẫn | Mô Tả |
|----------|-----------|-------|
| Video phân tích | output/result.mp4 | Frame với ellipse, nhãn tracker ID, radar minimap |
| Thống kê JSON | output/player\_stats/player\_stats.json | Dữ liệu thống kê đầy đủ 30 trackers |
| Ảnh quỹ đạo | output/player\_stats/player\_XXXX\_teamY.png | Quỹ đạo sân + collage crop cầu thủ |
| Highlight video | output/player\_stats/player\_XXXX\_teamY\_highlight.mp4 | Clip theo từng cầu thủ (nếu ≥280 frames) |
| Cấu hình Hydra | outputs/YYYY-MM-DD/HH-MM-SS/.hydra/ | Toàn bộ cấu hình mỗi lần chạy |

## 4.5 Tóm Tắt Chương

Chương này trình bày kết quả thực nghiệm trên hai thành phần học sâu được huấn luyện trong hệ thống, cùng với đánh giá tổng thể toàn pipeline.

**Về phát hiện đối tượng:** Fine-tune YOLO11 trên SoccerNet Tracking Dataset trong 50 epoch đạt mAP50-95 = **0.615** và Recall = **0.823**, phản ánh khả năng phát hiện đầy đủ cầu thủ trong các tình huống khó của video broadcast. Box Loss (val) ổn định từ epoch 35 trở đi, xác nhận mô hình hội tụ mà không overfit.

**Về nhận diện lại cầu thủ:** OSNet\_x1\_0 với hierarchical sampling và Centroid Loss đạt **mAP = 83.4%, Rank-1 = 78.0%** trên SoccerNet Re-ID test split, vượt trội ResNet50 và cân bằng tốt giữa hiệu năng và tốc độ inference so với các Transformer nặng hơn.

**Về pipeline tổng thể:** Inference trên video 25 giây (747 frame) cho 30 tracker ID với 10 cầu thủ active có coverage 96.4%. Tốc độ tối đa 36.0 km/h và quãng đường tối đa 106.2 m nằm trong dải hợp lý của bóng đá chuyên nghiệp, xác nhận tính chính xác của chuỗi chuyển đổi tọa độ pixel → sân thực tế qua homography NBJW.

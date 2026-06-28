# CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 5.1 Kết Luận

### 5.1.1 Tóm Tắt Công Trình

Luận văn này trình bày thiết kế, hiện thực và đánh giá thực nghiệm một hệ thống phân tích video bóng đá tự động dựa trên học sâu. Hệ thống được xây dựng theo kiến trúc **pipeline tuần tự sáu giai đoạn** — Phát hiện, Lọc, Theo dõi, Nhận diện lại, Hiệu chỉnh camera, Trực quan hóa — tích hợp liền mạch nhiều mô hình học sâu chuyên biệt thành một luồng xử lý thống nhất, từ frame video thô đến thống kê cầu thủ có ý nghĩa thực tiễn.

Điểm khác biệt cốt lõi so với các hệ thống phân tích video đơn lẻ là khả năng **khép kín từ đầu đến cuối**: đầu vào là video broadcast chuẩn không cần thiết bị đặc biệt, đầu ra là số liệu định lượng ở không gian sân thực tế (mét, km/h) được xuất dưới nhiều định dạng (video có chú thích, JSON, ảnh quỹ đạo, video highlight cầu thủ). Toàn bộ tham số được quản lý qua Hydra với khả năng tái hiện thực nghiệm đầy đủ, và hệ thống hỗ trợ inference trên CPU, GPU local và GPU cloud (Modal.com H100) mà không thay đổi code.

### 5.1.2 Kết Quả Đạt Được

**Về phát hiện đối tượng:** Mô hình YOLO11 được fine-tune trên SoccerNet Tracking Dataset (200 chuỗi × 30 giây, 1080p broadcast) trong 50 epoch đạt **mAP50-95 = 0.615** và **Recall = 0.823**. Recall cao phản ánh ưu tiên thiết kế đúng đắn cho bài toán tracking: phát hiện đầy đủ cầu thủ dù có che khuất để tránh mất track ở giai đoạn ByteTrack. Box Loss (val) ổn định từ epoch 35 trở đi xác nhận mô hình hội tụ mà không overfit dù bộ dữ liệu đa dạng và quy mô lớn.

**Về nhận diện lại cầu thủ:** OSNet\_x1\_0 với chiến lược hierarchical sampling và Centroid Loss đạt **mAP = 83.4%, Rank-1 = 78.0%** trên SoccerNet Re-ID test split (arXiv:2206.02373). Mô hình cải thiện 13.1 điểm mAP so với ResNet50 baseline, trong khi chỉ có ~2.2M tham số — phù hợp cho inference thời gian thực. Ngưỡng cosine similarity $\tau = 0.65$ và gallery max\_age = 120 frame cho phép khôi phục tracker ID trong phần lớn tình huống che khuất thực tế.

**Về hiệu chỉnh camera:** Module NBJW dựa trên HRNet-W48 ước lượng ma trận homography từ keypoints và đường kẻ sân, ánh xạ tọa độ pixel sang tọa độ sân thực tế (mét) với EMA smoothing (α = 0.4) để loại bỏ rung lắc. Kết quả tốc độ tối đa 36.0 km/h và khoảng cách tối đa 106.2 m trong 25 giây nằm trong dải hợp lý của bóng đá chuyên nghiệp, xác nhận chuỗi chuyển đổi tọa độ hoạt động chính xác.

**Về pipeline tổng thể:** Thực nghiệm trên video `08fd33_0.mp4` (747 frame, 25 giây) gán được **30 tracker ID** với **10 cầu thủ active** có frame coverage trung bình 96.4%. Module thống kê xuất đầy đủ JSON, ảnh quỹ đạo và video highlight cho từng cầu thủ, sẵn sàng sử dụng cho phân tích chiến thuật thực tế.

### 5.1.3 Đóng Góp Chính

Luận văn đóng góp các thành phần sau:

1. **Pipeline tích hợp đầy đủ** kết hợp YOLO11, ByteTrack, OSNet Re-ID và NBJW calibration trong một hệ thống cấu hình thống nhất, có thể mở rộng và tái sử dụng.

2. **Fine-tune YOLO11 trên SoccerNet** với chiến lược warm start từ checkpoint pretrained, đạt kết quả tốt (mAP50-95 = 0.615) trên bộ dữ liệu broadcast quy mô lớn với thời gian training hợp lý (~21 giờ).

3. **Tích hợp Re-ID thời gian thực** vào pipeline tracking sử dụng OSNet\_x1\_0 — giải quyết bài toán ID switching đặc thù của bóng đá khi nhiều cầu thủ mặc đồng phục đồng nhất.

4. **Module thống kê định lượng** tự động tính quãng đường, tốc độ tối đa và tốc độ trung bình từ tọa độ sân thực tế, xuất đa định dạng phục vụ phân tích hậu trận.

## 5.2 Hạn Chế

Mặc dù đạt được các kết quả khả quan, hệ thống hiện tại vẫn còn một số hạn chế cần nhìn nhận:

**Phụ thuộc góc quay đơn:** Pipeline được thiết kế và kiểm nghiệm trên video broadcast một camera chính. Khi camera pan quá nhanh hoặc zoomed quá gần (mất đường kẻ sân), module NBJW không thể ước lượng homography đáng tin cậy và cache homography cũ, dẫn đến sai số tọa độ tích lũy trong khoảng thời gian đó.

**Phân loại đội dựa vào vị trí ảnh:** Hệ thống phân biệt team 0 và team 1 dựa vào class ID của YOLO11 (player\_left / player\_right trong SoccerNet schema) thay vì màu sắc jersey. Khi cầu thủ di chuyển sang nửa sân đối diện, nhãn đội có thể bị đổi nếu mô hình không ổn định.

**Chưa có nhận diện số áo:** Re-ID dựa trên đặc trưng ngoại hình tổng thể (màu áo, vóc dáng) không phân biệt được hai cầu thủ cùng đội có đặc trưng rất giống nhau. Nhận diện số áo (jersey number recognition) sẽ cung cấp định danh tuyệt đối nhưng chưa được tích hợp.

**Chưa được kiểm chứng với GPS thực tế:** Các chỉ số quãng đường và tốc độ tính từ homography chưa được đối chiếu với dữ liệu GPS chính xác từ thiết bị đeo của cầu thủ. Sai số homography có thể tích lũy theo thời gian, đặc biệt với các clip dài hơn 25 giây.

**Hiệu năng inference trên CPU:** Pipeline đầy đủ (YOLO11 + ByteTrack + OSNet + NBJW) không đạt real-time trên CPU thông thường do chi phí tính toán của hai mô hình HRNet-W48 trong module calibration. Triển khai trên môi trường không có GPU đòi hỏi tắt module Pitch hoặc tăng calibration interval.

**Phạm vi thực nghiệm:** Toàn bộ thực nghiệm được thực hiện trên 5 đoạn clip ngắn (~25–30 giây), chưa đánh giá hiệu năng trên trận đấu đầy đủ 90 phút với các biến đổi ánh sáng, camera góc rộng và thay người.

## 5.3 Hướng Phát Triển

Dựa trên kết quả và hạn chế hiện tại, các hướng phát triển tiếp theo được đề xuất theo thứ tự ưu tiên:

### 5.3.1 Nhận Diện Số Áo

Tích hợp mô-đun nhận diện số áo (jersey number recognition) sử dụng OCR hoặc mạng phân loại nhỏ trên crop cầu thủ sẽ cung cấp định danh tuyệt đối, độc lập với Re-ID ngoại hình. Kết hợp số áo với embedding OSNet sẽ tăng đáng kể độ chính xác tracking dài hạn, đặc biệt trong các tình huống thay người hoặc cầu thủ quay lại sau khoảng thời gian dài ra ngoài frame.

### 5.3.2 Phân Tích Chiến Thuật

Với tọa độ sân đã có, bước tiếp theo tự nhiên là phân tích chiến thuật tự động: heatmap vùng hoạt động, formation recognition (xác định sơ đồ đội hình 4-3-3, 4-4-2...), phân tích pressing intensity và Voronoi dominance area theo thời gian. Các phân tích này có giá trị thực tiễn cao cho ban huấn luyện và hoàn toàn khả thi với dữ liệu tọa độ hiện có.

### 5.3.3 Phát Hiện Sự Kiện

Huấn luyện mô hình phát hiện sự kiện (event detection) — sút bóng, chuyền bóng, tranh bóng, phạm lỗi — từ chuỗi tọa độ cầu thủ và bóng. Kết hợp với thống kê cá nhân sẽ tạo ra bản tóm tắt trận đấu tự động có cấu trúc (shot map, pass network, tackle heatmap).

### 5.3.4 Hỗ Trợ Đa Camera

Mở rộng pipeline để fuse thông tin từ nhiều góc quay (camera phụ, camera VAR) bằng cách hiệu chỉnh homography từng camera về cùng hệ tọa độ sân. Điều này giải quyết điểm mù của camera chính khi cầu thủ bị che khuất hoàn toàn và cho phép theo dõi liên tục suốt trận.

### 5.3.5 Tối Ưu Hóa Real-time

Thay thế hai mô hình HRNet-W48 của NBJW bằng kiến trúc nhẹ hơn (MobileNet-V3 hoặc EfficientViT) để giảm thời gian calibration từ ~150ms/lần xuống dưới 30ms, cho phép tăng calibration\_interval nhỏ hơn mà không ảnh hưởng tổng throughput. Kết hợp với TensorRT quantization cho YOLO11 và OSNet, pipeline có thể đạt real-time (≥25fps) trên GPU consumer-grade.

### 5.3.6 Kiểm Chứng Với Dữ Liệu GPS

Đối chiếu các chỉ số quãng đường và tốc độ tính từ homography với dữ liệu GPS thực tế từ thiết bị wearable (Catapult, STATSports) trên cùng đoạn video để định lượng sai số tích lũy và hiệu chỉnh hệ số chuyển đổi pixel–mét. Kết quả kiểm chứng này sẽ nâng cao độ tin cậy của hệ thống trong ứng dụng phân tích chuyên nghiệp.

## 5.4 Nhận Xét Cuối

Hệ thống phân tích video bóng đá được trình bày trong luận văn này chứng minh rằng việc tích hợp các mô hình học sâu chuyên biệt — phát hiện đối tượng, tracking đa đối tượng, nhận diện lại và hiệu chỉnh camera — vào một pipeline thống nhất là hoàn toàn khả thi với tài nguyên tính toán có sẵn (HPC cluster + GPU cloud). Kết quả thực nghiệm đạt được ở mức đủ để phục vụ phân tích chiến thuật sơ bộ và mở ra nền tảng cho các hướng nghiên cứu và ứng dụng sâu hơn trong lĩnh vực thể thao thông minh.

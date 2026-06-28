# MỞ ĐẦU

## 1. Lý Do Chọn Đề Tài

Bóng đá là môn thể thao được theo dõi rộng rãi nhất thế giới, thu hút hàng tỷ người xem mỗi mùa giải và tạo ra một ngành công nghiệp phân tích thể thao trị giá hàng chục tỷ đô la. Trong môi trường thi đấu chuyên nghiệp hiện đại, lợi thế cạnh tranh ngày càng phụ thuộc vào khả năng thu thập và phân tích dữ liệu chuyển động của cầu thủ: tốc độ sprint, quãng đường di chuyển, cường độ hoạt động theo từng giai đoạn trận đấu. Những chỉ số này cung cấp nền tảng khách quan để ban huấn luyện đánh giá thể lực, xây dựng chiến thuật và phòng ngừa chấn thương.

Truyền thống, dữ liệu chuyển động cầu thủ được thu thập bằng thiết bị GPS wearable hoặc hệ thống multi-camera chuyên dụng như Tracab hay ChyronHego — những giải pháp có chi phí hạ tầng rất cao (hàng chục nghìn đến hàng triệu đô la mỗi giải đấu) và đòi hỏi lắp đặt phần cứng phức tạp. Điều này tạo ra khoảng cách lớn giữa câu lạc bộ chuyên nghiệp hàng đầu và phần còn lại của hệ sinh thái bóng đá — nơi phần lớn các đội bóng hạng dưới, giải nghiệp dư và trường học vẫn phụ thuộc vào quan sát thủ công của huấn luyện viên.

Sự phát triển vượt bậc của học sâu trong lĩnh vực thị giác máy tính mở ra khả năng thu hẹp khoảng cách này: thay vì lắp đặt cảm biến chuyên dụng, có thể trích xuất thông tin chuyển động cầu thủ trực tiếp từ video broadcast sẵn có bằng các mô hình phát hiện và theo dõi đối tượng. Tuy nhiên, video bóng đá đặt ra những thách thức kỹ thuật đặc thù — nhiều cầu thủ mặc đồng phục giống hệt nhau, che khuất liên tục trong các pha tranh chấp, camera di chuyển và zoom khiến tọa độ pixel không ổn định, tốc độ cao yêu cầu phát hiện chính xác trong thời gian thực.

Xuất phát từ bối cảnh đó, đề tài **"Xây Dựng Hệ Thống Phân Tích Video Bóng Đá Tự Động Sử Dụng Học Sâu"** được thực hiện nhằm xây dựng một pipeline tích hợp, kết hợp các kỹ thuật phát hiện đối tượng, theo dõi đa đối tượng, nhận diện lại và hiệu chỉnh camera hình học để trích xuất thống kê cầu thủ định lượng từ video broadcast thông thường — không cần thiết bị chuyên dụng.

## 2. Mục Tiêu Nghiên Cứu

Luận văn đặt ra các mục tiêu cụ thể sau:

**(1)** Xây dựng pipeline phát hiện và theo dõi đa đối tượng cho video bóng đá broadcast, tích hợp mô hình YOLO11 fine-tune trên SoccerNet và thuật toán ByteTrack để duy trì tracker ID liên tục qua các frame.

**(2)** Tích hợp mô-đun nhận diện lại cầu thủ (Re-ID) dựa trên OSNet để khôi phục tracker ID khi cầu thủ bị mất track do che khuất hoặc ra ngoài khung hình, giải quyết bài toán đặc thù của bóng đá là nhiều cầu thủ mặc đồng phục đồng nhất.

**(3)** Tích hợp mô-đun hiệu chỉnh camera tự động dựa trên phương pháp NBJW để ước lượng ma trận homography và ánh xạ tọa độ pixel sang không gian tọa độ sân thực tế (đơn vị mét), làm nền tảng cho các chỉ số định lượng có ý nghĩa vật lý.

**(4)** Xây dựng module thống kê tự động tính quãng đường di chuyển và tốc độ (tối đa, trung bình) cho từng cầu thủ được theo dõi, xuất kết quả dưới dạng JSON, ảnh quỹ đạo và video highlight.

**(5)** Đánh giá thực nghiệm từng thành phần và toàn pipeline trên dữ liệu SoccerNet, kiểm chứng tính hợp lý của các chỉ số đầu ra so với thực tế bóng đá chuyên nghiệp.

## 3. Đối Tượng và Phạm Vi Nghiên Cứu

**Đối tượng nghiên cứu:** Video bóng đá được quay từ góc camera broadcast chính (camera cố định hoặc pan nhẹ, bao quát phần lớn sân), độ phân giải 1080p, tốc độ khung hình 25–30fps — định dạng phổ biến nhất trong các giải đấu từ nghiệp dư đến chuyên nghiệp.

**Phạm vi nghiên cứu:**

- Phát hiện và theo dõi cầu thủ, thủ môn và trọng tài trong video broadcast.
- Phân loại đội dựa trên thông tin lớp từ mô hình detection (không bao gồm nhận dạng số áo hay màu jersey).
- Hiệu chỉnh camera đơn (single-camera homography) — không hỗ trợ đa camera.
- Thống kê tốc độ và quãng đường; không bao gồm phân tích chiến thuật hay phát hiện sự kiện.
- Clip thực nghiệm có độ dài 25–30 giây, đủ để kiểm chứng pipeline; không đánh giá trên trận đấu đầy đủ 90 phút.

## 4. Phương Pháp Nghiên Cứu

Luận văn áp dụng phương pháp nghiên cứu ứng dụng kết hợp thực nghiệm:

- **Fine-tuning mô hình:** Áp dụng transfer learning, khởi động từ checkpoint pretrained và tinh chỉnh trên bộ dữ liệu SoccerNet chuyên biệt để thích ứng mô hình với phân phối video bóng đá broadcast.

- **Tích hợp hệ thống:** Kết hợp các mô hình và thuật toán đã được công bố (YOLO11, ByteTrack, OSNet, NBJW) thành pipeline có cấu trúc rõ ràng, đánh giá từng giai đoạn độc lập và tổng thể.

- **Đánh giá định lượng:** Sử dụng các metric chuẩn trong lĩnh vực — mAP50-95 cho detection, mAP và Rank-1 cho Re-ID — kết hợp kiểm chứng định tính các chỉ số đầu ra (tốc độ, quãng đường) theo kiến thức bóng đá chuyên nghiệp.

## 5. Cấu Trúc Luận Văn

Luận văn được tổ chức thành năm chương:

**Chương 1 — Tổng Quan:** Trình bày bối cảnh bài toán, tổng quan các nghiên cứu liên quan, xác định thách thức kỹ thuật đặc thù của phân tích video bóng đá và đề xuất hướng tiếp cận.

**Chương 2 — Cơ Sở Lý Thuyết:** Trình bày nền tảng lý thuyết của các kỹ thuật học sâu được sử dụng: mạng nơ-ron tích chập, phát hiện đối tượng (YOLO11), theo dõi đa đối tượng (ByteTrack), nhận diện lại người (OSNet), mạng độ phân giải cao (HRNet) và phép biến đổi homography.

**Chương 3 — Thiết Kế và Hiện Thực Hệ Thống:** Mô tả kiến trúc pipeline sáu giai đoạn, thiết kế các module, cấu trúc dữ liệu liên module và các quyết định kỹ thuật trong quá trình hiện thực.

**Chương 4 — Thực Nghiệm và Đánh Giá:** Trình bày môi trường thực nghiệm, kết quả fine-tune detection, kết quả Re-ID, kết quả inference pipeline tổng thể và phân tích các chỉ số đạt được.

**Chương 5 — Kết Luận và Hướng Phát Triển:** Tổng kết đóng góp, nhận xét hạn chế hiện tại và đề xuất các hướng nghiên cứu tiếp theo.

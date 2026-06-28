# CHƯƠNG 1: TỔNG QUAN

## 1.1 Bối Cảnh và Động Lực

### 1.1.1 Phân Tích Thể Thao Trong Kỷ Nguyên Dữ Liệu

Bóng đá hiện đại đang trải qua một cuộc chuyển đổi sâu sắc dưới tác động của khoa học dữ liệu và trí tuệ nhân tạo. Các câu lạc bộ hàng đầu như FC Barcelona, Manchester City hay RB Leipzig đã xây dựng bộ phận phân tích hiệu suất riêng, sử dụng dữ liệu chuyển động cầu thủ như yếu tố đầu vào chính trong quyết định chiến thuật, tuyển dụng và quản lý thể lực. Theo báo cáo của Deloitte, thị trường phân tích thể thao toàn cầu dự kiến vượt 5 tỷ USD vào năm 2025, với phần lớn mức tăng trưởng đến từ các ứng dụng thị giác máy tính và học sâu.

Nguồn dữ liệu phổ biến nhất trong phân tích bóng đá chuyên nghiệp hiện nay bao gồm:

- **Thiết bị GPS wearable** (Catapult, STATSports): Cung cấp dữ liệu vị trí 10-18 Hz với sai số dưới 0.3m, nhưng không được phép sử dụng trong thi đấu chính thức theo quy định FIFA.
- **Hệ thống quang học đa camera** (Tracab, ChyronHego, Second Spectrum): Sử dụng 10-18 camera góc cố định lắp đặt quanh sân, tracking cầu thủ với độ chính xác cao nhưng chi phí hạ tầng rất lớn.
- **Semi-automated offside technology (SAOT)**: Hệ thống 12 camera chuyên dụng được FIFA tích hợp từ World Cup 2022, hiện chỉ khả dụng ở các giải đấu hàng đầu.

Điểm chung của các giải pháp trên là chi phí cao và phụ thuộc vào hạ tầng cố định, khiến chúng không khả thi cho đại đa số câu lạc bộ ở giải hạng dưới, trường học, hoặc các quốc gia đang phát triển. Phân tích video từ camera broadcast thông thường, vốn đã tồn tại ở hầu hết các giải đấu từ cấp tỉnh trở lên, mở ra hướng tiếp cận dân chủ hóa dữ liệu thể thao với chi phí triển khai thấp hơn nhiều bậc.

### 1.1.2 Cơ Hội Từ Học Sâu

Các đột phá trong học sâu thị giác máy tính, đặc biệt từ năm 2017 đến nay, đã tạo ra nền tảng kỹ thuật cho phân tích video thể thao tự động. Các mô hình phát hiện đối tượng thế hệ mới liên tục cải thiện tốc độ và độ chính xác, đạt real-time trên phần cứng GPU thông thường. Các thuật toán tracking đa đối tượng đạt mức tin cậy cao trên benchmark MOT chuẩn. Kỹ thuật nhận diện lại người mang lại khả năng khôi phục danh tính sau khi đối tượng bị mất khỏi tầm nhìn. Đặc biệt, sự ra đời của các bộ dữ liệu benchmark chuyên dụng, trong đó có SoccerNet với các tác vụ tracking, nhận diện lại và hiệu chỉnh camera, cung cấp dữ liệu huấn luyện quy mô lớn và đặc thù bóng đá, thúc đẩy sự phát triển các mô hình ngày càng chuyên biệt cho lĩnh vực này.

## 1.2 Tổng Quan Nghiên Cứu Liên Quan

### 1.2.1 Phát Hiện Đối Tượng Trong Video Thể Thao

Phát hiện đối tượng trong video thể thao là bài toán được nghiên cứu tích cực từ cuối những năm 2000. Các phương pháp truyền thống dựa trên đặc trưng thủ công như HOG (Histogram of Oriented Gradients) kết hợp bộ phân lớp SVM (Dalal và Triggs, 2005) đã đạt kết quả khả quan trong điều kiện nền phức tạp nhưng gặp khó khăn nghiêm trọng với môi trường đông người và che khuất lẫn nhau đặc thù của bóng đá.

Bước ngoặt đến với sự ra đời của các kiến trúc học sâu. Girshick et al. đề xuất R-CNN (2014) và Faster R-CNN (2015), thiết lập hướng tiếp cận hai giai đoạn (two-stage) với vùng đề xuất (region proposal network) và mạng phân loại riêng biệt, đạt độ chính xác cao trên PASCAL VOC và COCO. Liu et al. đề xuất SSD (Single Shot MultiBox Detector, 2016), thực hiện phát hiện đối tượng trực tiếp tại nhiều tỉ lệ đặc trưng trong một lượt forward pass. Redmon et al. với dòng YOLO (You Only Look Once, 2016) và các phiên bản kế tiếp theo đuổi tốc độ real-time bằng cách hợp nhất phân loại và định vị vào một mạng đơn duy nhất. Carion et al. đề xuất DETR (2020), ứng dụng cơ chế attention của Transformer vào phát hiện đối tượng, loại bỏ hoàn toàn NMS và anchor-based design.

Trong lĩnh vực bóng đá cụ thể, Istasse et al. (2021) fine-tune mô hình YOLO trên SoccerNet để phát hiện cầu thủ và bóng, chứng minh rằng transfer learning từ checkpoint pretrained trên COCO sang phân phối broadcast bóng đá giúp hội tụ nhanh hơn đáng kể. Vaufreydaz et al. (2022) kết hợp detection với instance segmentation để phân tách các cầu thủ che khuất nhau, cải thiện chất lượng bounding box trong các pha tranh chấp. SoccerNet-Tracking Challenge tại CVPR Workshop 2023 chuẩn hóa benchmark đánh giá detector trong video bóng đá và công bố nhiều kết quả so sánh hệ thống, cho thấy mô hình detection fine-tune trên dữ liệu SoccerNet đặc thù vượt trội đáng kể so với mô hình pretrained COCO thuần túy.

### 1.2.2 Theo Dõi Đa Đối Tượng

Theo dõi đa đối tượng (Multi-Object Tracking, MOT) nhằm duy trì danh tính nhất quán cho từng đối tượng qua các frame liên tiếp. Bài toán này thường được giải theo phương thức tracking-by-detection: detector chạy trên từng frame, sau đó bộ tracker thực hiện data association để ghép nối detection hiện tại với track đang tồn tại.

Bewley et al. đề xuất SORT (Simple Online and Realtime Tracking, 2016), kết hợp bộ lọc Kalman dự đoán vị trí và thuật toán Hungarian để giải bài toán gán khớp dựa trên IoU. SORT đặt nền tảng cho hầu hết các tracker hiện đại nhờ thiết kế đơn giản và tốc độ cao. Wojke et al. mở rộng thành DeepSORT (2017) bằng cách thêm đặc trưng ngoại hình từ mạng CNN vào metric gán khớp, giảm đáng kể tỷ lệ ID switching khi đối tượng bị che khuất và xuất hiện trở lại. Zhang et al. đề xuất FairMOT (2021), hợp nhất detection và Re-ID thành một mạng duy nhất chia sẻ backbone, cải thiện tốc độ inference so với pipeline hai giai đoạn. Du et al. phát triển StrongSORT (2022), tăng cường SORT bằng nhiều cải tiến nhỏ nhưng hiệu quả: camera motion compensation, ECC alignment, và cơ chế association kết hợp IoU với embedding. Cao et al. đề xuất OC-SORT (Observation-Centric SORT, 2022), giải quyết vấn đề error accumulation trong Kalman Filter khi đối tượng bị mất dài hạn bằng cách điều chỉnh hướng cập nhật dựa trên observation thực tế thay vì dự đoán.

Trong video bóng đá, thách thức đặc thù là ID switching xảy ra thường xuyên hơn do cầu thủ cùng đội mặc đồng phục giống nhau, khiến đặc trưng ngoại hình không đủ để phân biệt. Kết quả SoccerNet-Tracking Challenge 2023 cho thấy các hệ thống dẫn đầu đều kết hợp tracker mạnh với detector fine-tune chuyên biệt bóng đá, thay vì dùng detector generic.

### 1.2.3 Nhận Diện Lại Người

Nhận diện lại người (Person Re-Identification) là bài toán tìm ảnh của cùng một người trong các camera hoặc thời điểm khác nhau, đóng vai trò then chốt trong hệ thống giám sát và tracking dài hạn. Các benchmark phổ biến gồm Market-1501 (Zheng et al., 2015) với 32,668 ảnh từ 6 camera, DukeMTMC-ReID (Ristani et al., 2016) với 36,411 ảnh từ 8 camera, và MSMT17 (Wei et al., 2018) với 126,441 ảnh đa điều kiện ánh sáng.

Luo et al. đề xuất BoT (Bag of Tricks, 2019), tập hợp nhiều kỹ thuật nhỏ nhưng hiệu quả (random erasing, label smoothing, last stride = 1, BNNeck) để cải thiện Rank-1 trên Market-1501 từ khoảng 88% lên 94% mà không thay đổi kiến trúc backbone. Wang et al. đề xuất MGN (Multiple Granularity Network, 2018), sử dụng nhiều nhánh đặc trưng tương ứng các mức độ chi tiết khác nhau của thân người (toàn thân, nửa trên, nửa dưới) và tổng hợp tại inference. He et al. đề xuất AGW (Attentive Generalized Mean pooling with Weighted regularization, 2021), một baseline mạnh kết hợp Generalized Mean Pooling và Weighted Regularization Triplet Loss, đạt kết quả cạnh tranh trên nhiều benchmark. Dosovitskiy et al. với Vision Transformer (ViT, 2021) mở ra hướng attention-based Re-ID; TransReID (He et al., 2021) ứng dụng ViT vào Re-ID và đạt state-of-the-art trên nhiều benchmark, tuy nhiên với chi phí tính toán cao hơn đáng kể.

Metric đánh giá Re-ID chuẩn gồm Cumulative Matching Characteristic (CMC) tập trung vào Rank-1 accuracy và mean Average Precision (mAP) đo chất lượng toàn danh sách xếp hạng, trong đó mAP phản ánh đầy đủ hơn hiệu năng thực tế khi mỗi query có nhiều gallery match đúng.

### 1.2.4 Nhận Diện Lại Cầu Thủ Trong Bóng Đá

Re-ID trong bóng đá đặt ra thách thức khác về bản chất so với Re-ID người đi bộ thông thường: thay vì phân biệt trang phục đa dạng, bài toán là phân biệt các cầu thủ cùng đội mặc đồng phục đồng nhất, thumbnail nhỏ và mờ do ở xa camera, và mỗi cầu thủ chỉ có vài ảnh trong mỗi action clip ngắn. Các phương pháp Re-ID người đi bộ state-of-the-art tuy đạt Rank-1 trên 95% trên Market-1501 nhưng hiệu năng suy giảm mạnh trong môi trường đồng phục đồng nhất này.

Girone et al. (arXiv:2206.02373) công bố SoccerNet Re-ID Dataset gồm 340,993 thumbnail cầu thủ từ 400 trận đấu thuộc 6 giải lớn châu Âu, với phân chia train/query/gallery chuẩn và giao thức đánh giá trong phạm vi cùng action clip. Benchmark này lần đầu chuẩn hóa so sánh các phương pháp Re-ID dành riêng cho bóng đá. Kết quả trên benchmark cho thấy chiến lược lấy mẫu (sampling strategy) trong quá trình huấn luyện ảnh hưởng đáng kể đến hiệu năng: hierarchical sampling, đảm bảo mỗi mini-batch chứa các cặp cầu thủ khó phân biệt nhất (cùng đội, cùng trận), cải thiện đáng kể so với random sampling. Việc bổ sung Centroid Loss, kéo embedding của cùng một identity về phía centroid chung trong feature space, giúp giảm intra-class variance và tăng inter-class margin trong điều kiện đồng phục đồng nhất.

### 1.2.5 Hiệu Chỉnh Camera Sân Thể Thao

Hiệu chỉnh camera (sports field registration) nhằm ước lượng ma trận homography ánh xạ tọa độ pixel sang tọa độ sân thực tế, là bước thiết yếu để tính các chỉ số vật lý có ý nghĩa như quãng đường và tốc độ. Bài toán này được nghiên cứu từ lâu trong phân tích video thể thao nhưng trở nên thực tiễn hơn nhờ học sâu.

Các phương pháp truyền thống dựa vào template matching hoặc edge detection thủ công để xác định đường kẻ sân và tính homography. Farin et al. (2004) sử dụng Hough Transform để phát hiện đường thẳng sân bóng và fit homography. Tuy nhiên các phương pháp này nhạy cảm với bóng của cầu thủ, ánh sáng không đều, và phần sân bị che khuất bởi quảng cáo hay khán đài.

Hướng tiếp cận dựa trên học sâu đạt độ tin cậy cao hơn đáng kể. Nie et al. (2021) đề xuất mô hình CNN học ánh xạ trực tiếp từ ảnh sân đến tập tham số homography. Citraro et al. (2022) sử dụng keypoint detection bằng heatmap regression để phát hiện các giao điểm đặc trưng sân bóng, sau đó áp dụng RANSAC để loại nhiễu trước khi tính DLT. Chen et al. (2022) đề xuất kết hợp semantic segmentation đường kẻ sân với template matching trên sơ đồ sân chuẩn để ước lượng homography robust hơn trong điều kiện camera pan nhanh. Nguyen et al. (2022) tổng hợp nhiều hướng tiếp cận keypoint-based và công bố SoccerNet-Calibration benchmark gồm 476 video từ 12 giải đấu, chuẩn hóa đánh giá cho lĩnh vực này. Phương pháp sử dụng dual model phát hiện đồng thời keypoints giao điểm và extremities đường kẻ sân đạt kết quả tốt nhất, tận dụng thông tin bổ sung từ hai loại đặc trưng hình học khác nhau của sân để tăng số điểm tham chiếu đáng tin cậy khi ước lượng homography.

### 1.2.6 Hệ Thống Phân Tích Video Thể Thao Tích Hợp

Trong khi từng kỹ thuật thành phần đã có nhiều nghiên cứu chuyên sâu, việc tích hợp chúng thành pipeline hoàn chỉnh từ đầu đến cuối còn tương đối ít trong tài liệu học thuật mở. Phần lớn các hệ thống tích hợp hoàn chỉnh tập trung ở dạng giải pháp thương mại đóng như Wyscout, InStat hay Hudl Sportscode, không công bố chi tiết kỹ thuật.

Một số công trình học thuật đáng chú ý: Huang et al. (2019) đề xuất TrackNet, mạng CNN đặc thù để track quỹ đạo bóng nhỏ tốc độ cao trong cầu lông và bóng bàn từ video broadcast, sử dụng ảnh liên tiếp làm đầu vào để tận dụng thông tin chuyển động. Scott et al. (2022) xây dựng SoccerTrack, pipeline tracking đa camera cho bóng đá kết hợp Re-ID để khôi phục track khi cầu thủ di chuyển giữa vùng phủ của các camera. Bochinski et al. (2017) đề xuất IOU Tracker, một tracker cực đơn giản dựa thuần túy trên IoU mà không dùng đặc trưng ngoại hình, hoạt động tốt trong môi trường detection frame rate cao.

Nhìn chung, các công trình học thuật hiện tại thường giải quyết từng bài toán riêng lẻ (detection, hoặc tracking, hoặc Re-ID, hoặc calibration) và thiếu các pipeline kết hợp đầy đủ có khả năng cấu hình linh hoạt, tái hiện thực nghiệm, và xuất thống kê định lượng ở không gian sân thực tế.

## 1.3 Thách Thức Kỹ Thuật Đặc Thù

Phân tích video bóng đá broadcast đặt ra nhiều thách thức kỹ thuật không có trong các bài toán thị giác máy tính thông thường:

**Đồng phục đồng nhất (Uniform Similarity):** Toàn bộ cầu thủ cùng đội mặc trang phục giống hệt nhau, gây khó khăn cho cả tracking (ID switching khi cầu thủ va chạm) và Re-ID (phân biệt cầu thủ cùng đội dựa trên ngoại hình tổng thể thay vì trang phục đặc trưng).

**Che khuất dày đặc (Heavy Occlusion):** Trong các pha tranh chấp, nhiều cầu thủ che khuất nhau tạo thành các blob phức tạp mà detector cần phân tách. Tracker dễ mất track và gán sai ID khi cầu thủ xuất hiện trở lại sau thời gian bị che khuất.

**Biến thiên kích thước đối tượng (Scale Variation):** Cầu thủ ở xa camera chỉ chiếm 20-50 pixel chiều cao trong khung hình 1080p, trong khi cầu thủ gần camera có thể cao đến 200 pixel. Mô hình detection cần hoạt động chính xác trên toàn dải kích thước này.

**Camera chuyển động và zoom (Camera Motion):** Camera broadcast thường pan và zoom theo bóng, làm thay đổi liên tục tọa độ pixel của cầu thủ không liên quan đến chuyển động thực của họ trên sân. Điều này ảnh hưởng đến chất lượng homography và làm nhiễu thuật toán tracking dựa trên vận tốc pixel.

**Tốc độ di chuyển cao (High Speed):** Cầu thủ có thể sprint đạt 10 m/s (khoảng 36 km/h), gây ra motion blur và biến thiên vị trí lớn giữa các frame liên tiếp, thách thức cho cả detector lẫn tracker.

**Phân phối dữ liệu đặc thù:** Dữ liệu broadcast bóng đá có phân phối rất khác so với ảnh tổng quát dùng để pretrain các mô hình lớn: góc nhìn từ trên cao xuống, nền là mặt sân cỏ xanh đồng nhất, mật độ đối tượng cao trong khung hình. Fine-tune trên dữ liệu chuyên biệt là bước thiết yếu để thích ứng mô hình với phân phối này.

## 1.4 Hướng Tiếp Cận Đề Xuất

Để giải quyết các thách thức trên, luận văn đề xuất kiến trúc **pipeline tuần tự sáu giai đoạn** với các nguyên tắc thiết kế chính:

**Tách biệt mối quan tâm (Separation of Concerns):** Mỗi giai đoạn xử lý một nhiệm vụ cụ thể (phát hiện, lọc, tracking, nhận diện lại, hiệu chỉnh camera, trực quan hóa), có thể bật/tắt và thay thế độc lập mà không ảnh hưởng các giai đoạn khác. Thiết kế này cho phép so sánh hiệu năng từng giai đoạn và debug có hệ thống.

**Ưu tiên Recall trong Detection:** Ngưỡng confidence được đặt thấp để ưu tiên không bỏ sót cầu thủ hơn là tránh false positive, sau đó bù đắp bằng cơ chế xác nhận liên tiếp của tracker. Bỏ sót detection ở giai đoạn đầu dẫn đến mất track không thể phục hồi, trong khi false positive có thể loại bỏ ở giai đoạn sau.

**Re-ID như lớp bảo vệ danh tính:** Mô-đun nhận diện lại không thay thế tracker mà bổ sung thêm một lớp khôi phục danh tính sau khi tracker mất track, giải quyết trực tiếp vấn đề ID switching đặc thù của môi trường đồng phục đồng nhất.

**Làm mượt tọa độ sân theo thời gian:** Áp dụng Exponential Moving Average cho tọa độ sân sau homography để loại bỏ rung lắc do camera chuyển động mà không thêm độ trễ đáng kể, đảm bảo các chỉ số tốc độ và quãng đường không bị ảnh hưởng bởi nhiễu homography từng frame.

**Cấu hình linh hoạt và tái hiện thực nghiệm:** Toàn bộ tham số pipeline được quản lý qua hệ thống cấu hình tập trung, lưu cố định mỗi lần chạy để đảm bảo kết quả tái hiện chính xác và so sánh thực nghiệm có hệ thống.

## 1.5 Bộ Dữ Liệu Sử Dụng

Hệ thống được huấn luyện và đánh giá trên các bộ dữ liệu chuẩn thuộc bộ SoccerNet:

**SoccerNet Tracking Dataset** (CVPR Workshop 2023): 200 chuỗi video, mỗi chuỗi 30 giây, độ phân giải 1080p, góc quay broadcast chính, được gán nhãn 8 lớp đối tượng bao gồm cầu thủ hai đội, thủ môn và trọng tài. Bộ dữ liệu này được dùng để fine-tune mô hình phát hiện đối tượng cho phân phối bóng đá broadcast.

**SoccerNet Re-ID Dataset** (arXiv:2206.02373): 340,993 thumbnail cầu thủ từ 400 trận đấu thuộc 6 giải lớn châu Âu, phân chia thành tập huấn luyện (248,234 ảnh), tập truy vấn (11,777 ảnh) và gallery (34,989 ảnh) với 1,932 identity riêng biệt trong tập kiểm tra. Bộ dữ liệu này được dùng để huấn luyện và đánh giá mô-đun nhận diện lại.

**SoccerNet-Calibration** (CVPR 2022): Bộ dữ liệu benchmark cho hiệu chỉnh camera sân bóng, cùng với WorldCup 2014 và TS-WorldCup, được sử dụng trong các nghiên cứu gốc để đánh giá chất lượng homography.

**Video thực nghiệm**: 5 đoạn clip broadcast (~25-30 giây mỗi đoạn) từ các trận đấu chuyên nghiệp dùng để kiểm chứng pipeline inference đầy đủ trong điều kiện thực tế.

## 1.6 Tóm Tắt Chương

Chương này trình bày bối cảnh và động lực của đề tài: khoảng cách giữa nhu cầu phân tích dữ liệu trong bóng đá hiện đại và chi phí hạ tầng cao của các giải pháp chuyên dụng, cùng cơ hội khai thác học sâu để thu hẹp khoảng cách đó từ video broadcast sẵn có. Tổng quan nghiên cứu liên quan trình bày tiến triển của từng lĩnh vực kỹ thuật cấu thành hệ thống: phát hiện đối tượng từ HOG-SVM đến các kiến trúc học sâu hiện đại, tracking đa đối tượng từ SORT đến các phương pháp kết hợp đặc trưng ngoại hình, Re-ID từ benchmark người đi bộ đến dữ liệu đặc thù bóng đá, và hiệu chỉnh camera từ phương pháp truyền thống đến keypoint detection bằng học sâu. Các thách thức kỹ thuật đặc thù của bóng đá, bao gồm đồng phục đồng nhất, che khuất dày đặc và camera chuyển động, được xác định rõ và định hướng các quyết định thiết kế sẽ được trình bày chi tiết trong các chương tiếp theo.

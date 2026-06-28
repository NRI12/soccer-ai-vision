# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

---

## 2.1 Tổng Quan về Học Sâu (Deep Learning)

Học sâu (Deep Learning) là một nhánh của học máy (Machine Learning) dựa trên mạng nơ-ron nhân tạo có nhiều tầng ẩn. Thay vì dựa vào đặc trưng thiết kế thủ công như các phương pháp truyền thống, học sâu cho phép mô hình tự động học biểu diễn đặc trưng phân cấp trực tiếp từ dữ liệu thô. Điều này đã tạo ra bước đột phá trong nhiều bài toán thị giác máy tính, bao gồm nhận dạng ảnh, phát hiện đối tượng, phân đoạn ngữ nghĩa và theo dõi đa đối tượng — tất cả đều là các thành phần cốt lõi trong hệ thống phân tích video bóng đá được trình bày trong luận văn này.

### 2.1.1 Mạng Nơ-ron Tích Chập (CNN)

Mạng nơ-ron tích chập (Convolutional Neural Network — CNN) là kiến trúc học sâu được thiết kế đặc biệt cho dữ liệu có cấu trúc không gian, chẳng hạn như ảnh và video. Nguyên lý hoạt động của CNN dựa trên ba cơ chế chính: **chia sẻ trọng số** (weight sharing), **kết nối cục bộ** (local connectivity) và **bất biến dịch chuyển** (translation invariance).

**Phép tích chập (Convolution)**

Tầng tích chập là đơn vị cơ bản của CNN. Với ảnh đầu vào $I \in \mathbb{R}^{H \times W \times C}$ (chiều cao $H$, chiều rộng $W$, số kênh $C$) và bộ lọc (kernel) $K \in \mathbb{R}^{k \times k \times C \times C'}$, đầu ra của phép tích chập tại vị trí $(i, j)$ trên kênh đầu ra thứ $c'$ được tính như sau:

$$O[i, j, c'] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \sum_{c=0}^{C-1} I[i \cdot s + m,\ j \cdot s + n,\ c] \cdot K[m, n, c, c'] + b[c']$$

trong đó $s$ là bước trượt (stride), $k$ là kích thước kernel, và $b[c']$ là hệ số bias tương ứng. Khác với lớp kết nối đầy đủ (fully connected layer), CNN áp dụng cùng một bộ lọc tại mọi vị trí không gian trên ảnh — đây chính là cơ chế chia sẻ trọng số giúp giảm đáng kể số tham số và tạo tính bất biến dịch chuyển.

Kích thước đầu ra của tầng tích chập được xác định bởi:

$$O_H = \left\lfloor \frac{H - k + 2p}{s} \right\rfloor + 1, \quad O_W = \left\lfloor \frac{W - k + 2p}{s} \right\rfloor + 1$$

với $p$ là số padding thêm vào biên ảnh để kiểm soát kích thước đầu ra.

**Hàm kích hoạt (Activation Function)**

Sau mỗi tầng tích chập, một hàm kích hoạt phi tuyến được áp dụng để tăng khả năng biểu diễn của mạng. Hàm ReLU (Rectified Linear Unit) là lựa chọn phổ biến nhất:

$$\text{ReLU}(x) = \max(0, x)$$

ReLU giải quyết vấn đề gradient biến mất (vanishing gradient) vì đạo hàm của nó bằng 1 với $x > 0$ và bằng 0 với $x \leq 0$. Trong các kiến trúc hiện đại như YOLO11, hàm SiLU (Sigmoid Linear Unit) cũng được sử dụng:

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

SiLU cho phép gradient trơn hơn và không bị chặn hoàn toàn ở vùng âm, thường mang lại hiệu suất tốt hơn ReLU trên các bài toán phát hiện đối tượng.

**Chuẩn hóa theo lô (Batch Normalization)**

Batch Normalization (BN) là kỹ thuật chuẩn hóa được áp dụng sau lớp tích chập để ổn định quá trình huấn luyện. Với mini-batch $\mathcal{B} = \{x_1, x_2, \ldots, x_m\}$, BN chuẩn hóa đầu vào về phân phối zero-mean, unit-variance, sau đó áp dụng biến đổi affine học được:

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

trong đó $\mu_\mathcal{B}$ và $\sigma_\mathcal{B}^2$ là giá trị trung bình và phương sai của batch, $\epsilon$ là hằng số nhỏ tránh chia cho 0, còn $\gamma$ và $\beta$ là các tham số học được (scale và shift). BN cho phép sử dụng learning rate lớn hơn, giảm sự phụ thuộc vào khởi tạo trọng số và có tác dụng regularization nhẹ.

**Kết nối dư (Residual Connection)**

Kiến trúc ResNet giới thiệu kết nối dư (skip connection) để giải quyết vấn đề suy giảm gradient trong mạng rất sâu. Thay vì học ánh xạ $\mathcal{H}(x)$ trực tiếp, khối residual học ánh xạ dư $\mathcal{F}(x) = \mathcal{H}(x) - x$, sau đó cộng với đầu vào:

$$y = \mathcal{F}(x, \{W_i\}) + x$$

Kết nối dư đảm bảo gradient có thể truyền ngược trực tiếp qua shortcut path mà không bị suy giảm, đồng thời cho phép xây dựng mạng với hàng trăm tầng. Gần như tất cả các kiến trúc hiện đại — YOLO11, HRNet, OSNet — đều tích hợp kết nối dư làm nền tảng.

**Tầng pooling và downsampling**

Tầng pooling (max pooling hoặc average pooling) thực hiện downsampling không gian, giảm kích thước đặc trưng và tăng receptive field hiệu quả. Max pooling tại vùng $k \times k$ lấy giá trị lớn nhất:

$$O[i, j] = \max_{0 \leq m, n < k} I[i \cdot s + m,\ j \cdot s + n]$$

Trong các kiến trúc phát hiện đối tượng hiện đại, downsampling thường được thực hiện bằng tích chập có stride lớn hơn 1 thay vì tầng pooling riêng biệt, để giữ lại nhiều thông tin hơn.

### 2.1.2 Transfer Learning và Fine-tuning

Transfer learning (học chuyển giao) là kỹ thuật sử dụng kiến thức học được từ một bài toán nguồn (source task) — thường là phân loại ảnh trên ImageNet — để cải thiện hiệu suất trên bài toán đích (target task) với ít dữ liệu hơn. Trong lĩnh vực computer vision, các backbone như ResNet, HRNet hay CSP (Cross Stage Partial) thường được pretrain trên ImageNet với hàng triệu ảnh và hàng nghìn lớp, giúp mô hình học được các đặc trưng thị giác tổng quát từ thấp đến cao: cạnh, kết cấu, hình dạng, và cuối cùng là ngữ nghĩa đối tượng.

**Hai chiến lược fine-tuning chính** được sử dụng trong thực tế:

Chiến lược thứ nhất là **đóng băng backbone** (frozen backbone): giữ nguyên toàn bộ trọng số của backbone pretrained và chỉ huấn luyện các tầng đầu ra (detection head). Chiến lược này phù hợp khi dữ liệu mục tiêu rất ít hoặc rất tương đồng với dữ liệu pretrain, vì risk của overfitting thấp và tốc độ hội tụ nhanh.

Chiến lược thứ hai là **fine-tuning toàn bộ mạng** (full fine-tuning): cập nhật tất cả các trọng số với learning rate nhỏ hơn nhiều so với quá trình huấn luyện ban đầu, thường theo dạng học rate khác nhau cho các tầng khác nhau (discriminative fine-tuning) — các tầng sớm hơn (học đặc trưng thấp) dùng learning rate nhỏ hơn các tầng sau (học đặc trưng cao cấp, đặc thù bài toán). Đây là chiến lược được áp dụng cho YOLO11 trong đề tài này khi fine-tune trên SoccerNet Tracking Dataset.

Lợi ích cơ bản của transfer learning trong bài toán phân tích video bóng đá là: dữ liệu annotate cho bóng đá chuyên nghiệp tốn kém và khó thu thập, trong khi mô hình cần nhận diện các đối tượng nhỏ (cầu thủ ở xa camera), bị che khuất và di chuyển nhanh. Khởi tạo từ pretrained weights cho phép mô hình hội tụ nhanh hơn và đạt hiệu suất cao hơn đáng kể so với huấn luyện từ đầu (training from scratch).

### 2.1.3 Các Hàm Mất Mát Trong Detection và Tracking

Hàm mất mát (loss function) là thước đo định lượng sự sai lệch giữa dự đoán của mô hình và nhãn thực tế, là cơ sở để tính gradient và cập nhật trọng số qua thuật toán gradient descent.

**Binary Cross-Entropy Loss (BCE)**

BCE được sử dụng cho bài toán phân loại nhị phân, trong đó mỗi class được dự đoán độc lập với xác suất $\hat{y}_i \in [0, 1]$:

$$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

Trong YOLO11, BCE được dùng cho bài toán phân loại đối tượng theo từng lớp (multi-label classification), nơi một detection box có thể thuộc nhiều lớp đồng thời (dù trong thực tế bóng đá là multi-class exclusive).

**Focal Loss**

Trong các bài toán phát hiện đối tượng, sự mất cân bằng nghiêm trọng giữa nền (background) và đối tượng (foreground) khiến mô hình bị chi phối bởi các mẫu dễ phân loại (easy negatives). Focal Loss giải quyết vấn đề này bằng cách giảm trọng số đóng góp của các mẫu dễ:

$$\mathcal{L}_{FL} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

trong đó $p_t$ là xác suất dự đoán đúng, $\alpha_t$ là hệ số cân bằng lớp, và $\gamma \geq 0$ là tham số tập trung (focusing parameter). Khi $\gamma = 0$, Focal Loss tương đương với cross-entropy tiêu chuẩn. Khi $\gamma > 0$, các mẫu dễ ($p_t \to 1$) bị giảm trọng số đáng kể, buộc mô hình tập trung học từ các mẫu khó.

**IoU Loss và các biến thể**

Hàm mất mát hồi quy bounding box truyền thống dựa trên MSE (Mean Squared Error) trên tọa độ $(x, y, w, h)$ không trực tiếp tối ưu hóa metric IoU mà các phương pháp đánh giá detection thực sự sử dụng. Các biến thể IoU Loss khắc phục điều này:

*IoU Loss* đơn giản:
$$\mathcal{L}_{IoU} = 1 - \frac{|B \cap \hat{B}|}{|B \cup \hat{B}|}$$

*CIoU Loss* (Complete IoU) bổ sung thêm phạt về khoảng cách tâm và tỉ lệ khung hình:
$$\mathcal{L}_{CIoU} = 1 - IoU + \frac{\rho^2(\mathbf{b}, \hat{\mathbf{b}})}{c^2} + \alpha_v v$$

trong đó $\rho^2(\mathbf{b}, \hat{\mathbf{b}})$ là bình phương khoảng cách Euclidean giữa tâm của bounding box thực và dự đoán, $c$ là đường chéo của hình chữ nhật bao ngoài nhỏ nhất chứa cả hai box, $v = \frac{4}{\pi^2}(\arctan\frac{w}{h} - \arctan\frac{\hat{w}}{\hat{h}})^2$ đo sự chênh lệch tỉ lệ khung hình, và $\alpha_v = \frac{v}{(1 - IoU) + v}$ là hệ số cân bằng.

*DFL Loss (Distribution Focal Loss)* được giới thiệu trong GFL và áp dụng trong YOLO11 để dự đoán phân phối xác suất liên tục của khoảng cách từ tâm đến biên box thay vì dự đoán trực tiếp một giá trị scalar, chi tiết sẽ trình bày trong mục 2.2.4.

**L2 Heatmap Regression Loss**

Đối với bài toán dự đoán heatmap — được sử dụng trong HRNet cho NBJW — hàm mất mát L2 (Mean Squared Error) tính sai lệch giữa heatmap dự đoán $\hat{H}$ và heatmap ground-truth $H$ (thường là phân phối Gaussian 2D tâm tại vị trí keypoint):

$$\mathcal{L}_{heatmap} = \frac{1}{N_k} \sum_{k=1}^{N_k} \left\| H_k - \hat{H}_k \right\|_F^2$$

trong đó $N_k$ là tổng số keypoints và $\|\cdot\|_F$ là chuẩn Frobenius. Heatmap ground-truth tại keypoint $(x^*, y^*)$ được định nghĩa là:

$$H_k(x, y) = \exp\left(-\frac{(x - x^*)^2 + (y - y^*)^2}{2\sigma^2}\right)$$

với $\sigma$ là độ lệch chuẩn của Gaussian (trong NBJW, $\sigma = 2$ pixel).

---

## 2.2 Phát Hiện Đối Tượng — YOLO11

### 2.2.1 Lịch Sử Phát Triển Họ YOLO

Phát hiện đối tượng (Object Detection) là bài toán xác định đồng thời vị trí (bounding box) và nhãn lớp (class label) của tất cả các đối tượng trong ảnh. Trước khi có YOLO, các phương pháp hai giai đoạn (two-stage detectors) như Faster R-CNN đạt độ chính xác cao nhưng tốc độ chậm do phải qua hai bước: sinh vùng đề xuất (Region Proposal Network) rồi mới phân loại và tinh chỉnh. YOLO (You Only Look Once) ra đời năm 2015 với triết lý cơ bản: đưa toàn bộ bài toán detection vào một mạng duy nhất, xử lý ảnh chỉ một lần.

**YOLOv1 (2015)** chia ảnh thành lưới $S \times S$ ô (cell). Mỗi ô dự đoán $B$ bounding boxes cùng confidence score và xác suất lớp cho $C$ lớp. Đây là kiến trúc one-stage đầu tiên, đạt 45 FPS nhưng độ chính xác thấp do thiếu cơ chế xử lý đa tỉ lệ.

**YOLOv2/v3 (2016–2018)** bổ sung anchor boxes — các box tham chiếu với kích thước được xác định bằng k-means clustering trên training data — và multi-scale prediction trên các feature map khác nhau. YOLOv3 dùng darknet-53 làm backbone với 53 tầng convolution.

**YOLOv4/v5 (2020)** tích hợp nhiều kỹ thuật tăng cường: CSP (Cross Stage Partial Network) giảm phép tính, PANet (Path Aggregation Network) tổng hợp đặc trưng đa tỉ lệ, Mosaic augmentation. YOLOv5 là phiên bản đầu tiên triển khai bằng PyTorch + Ultralytics, trở thành chuẩn mực trong công nghiệp.

**YOLOv8 (2023)** và các phiên bản kế tiếp chuyển sang kiến trúc anchor-free, sử dụng decoupled head tách biệt hai nhánh dự đoán vị trí (regression) và nhãn lớp (classification), cùng với thuật toán gán nhãn Task-Aligned Assignment (TAL) thay thế cho gán nhãn dựa trên overlap đơn giản.

**YOLO11 (2024)** — phiên bản được sử dụng trong đề tài — là bản phát triển tiếp theo của Ultralytics với backbone C3k2 được tối ưu hóa (Cross Stage Partial with kernel size 2), mô-đun SPPF (Spatial Pyramid Pooling-Fast) tổng hợp đặc trưng đa tỉ lệ, và module C2PSA (Cross-Stage Partial with Parallel Spatial Attention) tích hợp attention mechanism. YOLO11 giảm số tham số so với YOLOv8 trong khi duy trì hoặc cải thiện độ chính xác, phù hợp với các ứng dụng yêu cầu inference nhanh trên thiết bị hạn chế tài nguyên lẫn HPC cluster.

### 2.2.2 Kiến Trúc YOLO11

Kiến trúc YOLO11 được tổ chức thành ba phần chính: backbone, neck và detection head.

**Backbone** có trách nhiệm trích xuất đặc trưng từ ảnh đầu vào ở nhiều mức độ trừu tượng. YOLO11 sử dụng chuỗi các tầng Conv-BN-SiLU xen kẽ với các khối C3k2 (một biến thể của CSP bottleneck). Khối C3k2 chia luồng đặc trưng thành hai nhánh: một nhánh đi qua bottleneck module (học đặc trưng sâu), một nhánh đi thẳng qua (preserve identity). Hai nhánh sau đó được nối (concatenate) lại theo chiều kênh, giúp duy trì gradient flow và giảm số phép tính. Sau backbone, mô-đun SPPF áp dụng max pooling với nhiều kích thước kernel $(5 \times 5)$ liên tiếp rồi concatenate kết quả, tạo ra biểu diễn đa tỉ lệ không gian tại lớp cuối cùng của backbone.

**Neck** (cổ) sử dụng kiến trúc FPN+PAN (Feature Pyramid Network + Path Aggregation Network) để kết hợp đặc trưng từ các độ phân giải khác nhau của backbone. FPN truyền thông tin từ tầng sâu (ngữ nghĩa cao, độ phân giải thấp) lên các tầng nông (chi tiết tốt, độ phân giải cao) qua upsampling và concatenation. PAN sau đó truyền ngược thông tin từ tầng nông xuống tầng sâu. Kết quả là ba feature map đầu ra ở ba tỉ lệ khác nhau — thường là $80 \times 80$, $40 \times 40$ và $20 \times 20$ với ảnh đầu vào $640 \times 640$ — cho phép phát hiện đối tượng nhỏ, trung bình và lớn tương ứng. Module C2PSA được tích hợp trong neck của YOLO11 để bổ sung cơ chế chú ý không gian (spatial attention), tập trung học đặc trưng tại các vùng có khả năng chứa đối tượng cao.

**Detection Head** nhận ba feature map từ neck và với mỗi vị trí trên feature map, dự đoán bounding box và nhãn lớp. Trong YOLO11, head được thiết kế theo kiểu decoupled — hai nhánh tách biệt cho regression và classification. Điều này quan trọng vì đặc trưng tối ưu cho hồi quy vị trí và phân loại nhãn lớp là khác nhau về bản chất: classification cần đặc trưng ngữ nghĩa cao, trong khi regression cần đặc trưng hình học cục bộ.

### 2.2.3 Cơ Chế Anchor-Free và Task-Aligned Assignment

**Anchor-free detection** loại bỏ khái niệm anchor box — những hộp tham chiếu cố định được đặt tại mỗi vị trí trên feature map. Thay vào đó, mô hình dự đoán trực tiếp khoảng cách từ tâm của ô feature map đến bốn cạnh của bounding box $(l, t, r, b)$ — khoảng cách đến cạnh trái, trên, phải, dưới tương ứng. Tọa độ bounding box được tái tạo như sau:

$$x_1 = x_{center} - l, \quad y_1 = y_{center} - t$$
$$x_2 = x_{center} + r, \quad y_2 = y_{center} + b$$

Cách tiếp cận này loại bỏ sự cần thiết phải thiết kế anchor thủ công (k-means clustering, tỉ lệ aspect ratio) và cho phép mô hình linh hoạt hơn trong việc dự đoán đối tượng với tỉ lệ kích thước đa dạng.

**Task-Aligned Assignment (TAL)** là thuật toán gán nhãn động (dynamic label assignment) trong quá trình training, thay thế cho các phương pháp gán nhãn tĩnh dựa trên IoU đơn thuần. TAL gán nhãn cho mỗi anchor point dựa trên một hàm điểm kết hợp cả đánh giá phân loại và hồi quy:

$$t = s^\alpha \cdot u^\beta$$

trong đó $s$ là confidence score phân loại được dự đoán, $u$ là IoU giữa bounding box dự đoán và ground-truth, còn $\alpha$ và $\beta$ là các siêu tham số cân bằng tầm quan trọng của hai thành phần. Điểm $t$ phản ánh chất lượng tổng hợp của cả classification lẫn localization, giúp chọn ra top-$k$ anchor points chất lượng cao nhất cho mỗi ground-truth box.

### 2.2.4 Hàm Mất Mát của YOLO11

Hàm mất mát tổng của YOLO11 gồm ba thành phần:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{box} + \lambda_2 \mathcal{L}_{cls} + \lambda_3 \mathcal{L}_{dfl}$$

**Box loss** ($\mathcal{L}_{box}$) sử dụng CIoU Loss để đo sự sai lệch về vị trí và kích thước bounding box, như đã trình bày trong mục 2.1.3.

**Classification loss** ($\mathcal{L}_{cls}$) sử dụng Binary Cross-Entropy với logits, được tính trung bình trên tất cả positive anchor points và tất cả các lớp:

$$\mathcal{L}_{cls} = -\frac{1}{N_{pos}} \sum_{i \in \text{pos}} \sum_{c=1}^{C} \left[ y_{i,c} \log(\hat{p}_{i,c}) + (1 - y_{i,c}) \log(1 - \hat{p}_{i,c}) \right]$$

**Distribution Focal Loss** ($\mathcal{L}_{dfl}$) là đóng góp quan trọng của Generalized Focal Loss (GFL). Thay vì hồi quy trực tiếp khoảng cách $d \in \mathbb{R}$ (ví dụ khoảng cách từ tâm đến cạnh trái), mô hình dự đoán một phân phối rời rạc trên tập $\{0, 1, \ldots, \text{reg\_max}\}$, biểu diễn xác suất tại mỗi vị trí nguyên. Giá trị $d$ được lấy bằng kỳ vọng:

$$\hat{d} = \sum_{j=0}^{\text{reg\_max}} j \cdot p_j, \quad \text{với } p_j = \text{softmax}(o_j)$$

DFL Loss là cross-entropy giữa phân phối dự đoán và phân phối "sharp" tập trung tại hai số nguyên $\lfloor d^* \rfloor$ và $\lceil d^* \rceil$ (trong đó $d^*$ là giá trị thực của khoảng cách):

$$\mathcal{L}_{dfl} = -\left[ (d^* - \lfloor d^* \rfloor) \log p_{\lceil d^* \rceil} + (\lceil d^* \rceil - d^*) \log p_{\lfloor d^* \rfloor} \right]$$

DFL cho phép mô hình học được sự không chắc chắn (uncertainty) về vị trí biên hộp, đặc biệt có lợi trong trường hợp đối tượng bị che khuất một phần hoặc biên không rõ ràng.

### 2.2.5 Các Độ Đo Đánh Giá Detection

**Precision** và **Recall** là hai độ đo cơ bản trong bài toán phát hiện đối tượng:

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

trong đó TP (True Positive) là số detection đúng (IoU với ground-truth vượt ngưỡng), FP (False Positive) là detection sai (không khớp với ground-truth nào), và FN (False Negative) là ground-truth không được phát hiện.

**Average Precision (AP)** là diện tích dưới đường cong Precision-Recall khi ngưỡng confidence thay đổi từ 0 đến 1:

$$AP = \int_0^1 p(r)\, dr \approx \sum_{k=1}^{K} p(r_k) \cdot \Delta r_k$$

**mAP50** là giá trị AP trung bình trên tất cả các lớp, tính với ngưỡng IoU cố định là 0.5:

$$\text{mAP50} = \frac{1}{C} \sum_{c=1}^{C} AP_c^{IoU=0.5}$$

**mAP50-95** (còn gọi là mAP trong COCO benchmark) là giá trị AP trung bình trên 10 ngưỡng IoU từ 0.5 đến 0.95 với bước 0.05, cho đánh giá nghiêm ngặt hơn về độ chính xác localization:

$$\text{mAP50-95} = \frac{1}{10} \sum_{t \in \{0.50, 0.55, \ldots, 0.95\}} \text{mAP}^{IoU=t}$$

Trong đề tài, YOLO11 finetuned trên SoccerNet đạt mAP50-95 = 0.582 với Precision = 0.949 và Recall = 0.798, cho thấy mô hình có độ chính xác cao nhưng vẫn còn bỏ sót một số trường hợp phức tạp (recall chưa đạt 1.0), điều này là hoàn toàn chấp nhận được trong bối cảnh video broadcast bóng đá với mật độ đối tượng cao và occlusion thường xuyên xảy ra.

---

## 2.3 Theo Dõi Đa Đối Tượng — ByteTrack

### 2.3.1 Bài Toán Multi-Object Tracking và Độ Đo MOTA/IDF1

Multi-Object Tracking (MOT) là bài toán ước lượng bounding box và danh tính (identity) của nhiều đối tượng đồng thời trong chuỗi video. Khác với phát hiện đối tượng — vốn chỉ xử lý từng frame độc lập — MOT yêu cầu gán ID nhất quán xuyên suốt các frame, tức là cùng một cầu thủ phải được gán cùng một ID từ frame đầu đến frame cuối.

Hướng tiếp cận phổ biến nhất hiện nay là **tracking-by-detection**: ở mỗi frame, đầu tiên chạy detector để lấy tập detection $\mathcal{D}_t$, sau đó thực hiện bài toán gán (assignment) để khớp từng detection với tracklet đang tồn tại. Thách thức chính của bài toán này bao gồm: (1) đối tượng bị che khuất một phần hoặc hoàn toàn; (2) đối tượng ra khỏi và quay lại khung hình; (3) đối tượng có ngoại hình tương đồng (ví dụ cầu thủ cùng đội); (4) detector bỏ sót đối tượng (false negative); (5) detector phát hiện sai (false positive).

**MOTA (Multiple Object Tracking Accuracy)** là độ đo tổng hợp phổ biến nhất, phạt cả False Negative (bỏ sót), False Positive (phát hiện nhầm) và Identity Switch (chuyển ID):

$$\text{MOTA} = 1 - \frac{\sum_t (FN_t + FP_t + IDSW_t)}{\sum_t GT_t}$$

trong đó $GT_t$ là tổng số ground-truth object trong frame $t$. MOTA có thể âm khi số lỗi vượt quá số đối tượng thực sự.

**IDF1 (ID F1 Score)** đo tỉ lệ phần trăm detections được gán đúng ID:

$$\text{IDF1} = \frac{2 \cdot IDTP}{2 \cdot IDTP + IDFP + IDFN}$$

trong đó IDTP là số detection được gán đúng ID, IDFP là detection sai ID, và IDFN là ground-truth không được gán đúng. IDF1 phản ánh khả năng duy trì danh tính nhất quán — đây là độ đo quan trọng hơn MOTA trong các ứng dụng phân tích thể thao vì việc nhầm lẫn cầu thủ này với cầu thủ khác sẽ dẫn đến thống kê sai hoàn toàn.

**HOTA (Higher Order Tracking Accuracy)** là độ đo thế hệ mới kết hợp cả DetA (Detection Accuracy) và AssA (Association Accuracy):

$$\text{HOTA} = \sqrt{\text{DetA} \cdot \text{AssA}}$$

ByteTrack đạt MOTA = 80.3, IDF1 = 77.3 và HOTA = 63.1 trên MOT17 — kết quả tốt nhất tại thời điểm công bố (ECCV 2022).

### 2.3.2 Bộ Lọc Kalman (Kalman Filter)

Bộ lọc Kalman là thuật toán lọc tuyến tính tối ưu cho các hệ thống động tuyến tính với nhiễu Gaussian, được sử dụng trong ByteTrack để dự đoán vị trí của tracklet ở frame kế tiếp khi không có detection phù hợp.

**Vector trạng thái** của mỗi tracklet được định nghĩa là:

$$\mathbf{x} = [x,\ y,\ a,\ h,\ \dot{x},\ \dot{y},\ \dot{a},\ \dot{h}]^T$$

trong đó $(x, y)$ là tọa độ tâm bounding box, $a = w/h$ là tỉ lệ khung hình (aspect ratio), $h$ là chiều cao, và $(\dot{x}, \dot{y}, \dot{a}, \dot{h})$ là các vận tốc tương ứng. Mô hình chuyển động giả định vận tốc không đổi giữa các frame liên tiếp.

**Ma trận chuyển trạng thái** $\mathbf{F}$ mô hình hóa quá trình chuyển động:

$$\mathbf{F} = \begin{bmatrix} \mathbf{I}_4 & \Delta t \cdot \mathbf{I}_4 \\ \mathbf{0}_4 & \mathbf{I}_4 \end{bmatrix}$$

với $\Delta t = 1$ (một frame).

**Bước dự đoán (Predict Step)**: Tại frame $t$, Kalman Filter dự đoán trạng thái mới từ trạng thái tại frame $t-1$:

$$\hat{\mathbf{x}}_{t|t-1} = \mathbf{F} \hat{\mathbf{x}}_{t-1|t-1}$$
$$\mathbf{P}_{t|t-1} = \mathbf{F} \mathbf{P}_{t-1|t-1} \mathbf{F}^T + \mathbf{Q}$$

trong đó $\mathbf{P}$ là ma trận hiệp phương sai sai số ước lượng và $\mathbf{Q}$ là ma trận hiệp phương sai nhiễu quá trình (process noise covariance), phản ánh mức độ không chắc chắn của mô hình chuyển động.

**Bước cập nhật (Update Step)**: Khi có observation mới (detection) $\mathbf{z}_t$ (bounding box từ detector), Kalman Filter cập nhật ước lượng trạng thái:

$$\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}^T \left(\mathbf{H} \mathbf{P}_{t|t-1} \mathbf{H}^T + \mathbf{R}\right)^{-1}$$
$$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t \left(\mathbf{z}_t - \mathbf{H} \hat{\mathbf{x}}_{t|t-1}\right)$$
$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}) \mathbf{P}_{t|t-1}$$

trong đó $\mathbf{K}_t$ là ma trận Kalman Gain, $\mathbf{H}$ là ma trận quan sát (ánh xạ từ không gian trạng thái sang không gian quan sát — chỉ quan sát $(x, y, a, h)$, không quan sát vận tốc trực tiếp), và $\mathbf{R}$ là ma trận hiệp phương sai nhiễu đo lường (measurement noise covariance).

Kalman Gain $\mathbf{K}_t$ quyết định mức độ tin tưởng vào observation so với dự đoán: khi $\mathbf{R}$ nhỏ (detector đáng tin), $\mathbf{K}_t$ lớn và trạng thái được cập nhật nhiều về phía observation; khi $\mathbf{R}$ lớn (detector không chắc chắn), $\mathbf{K}_t$ nhỏ và mô hình giữ nguyên dự đoán từ bước trước nhiều hơn.

### 2.3.3 Hungarian Algorithm — Gán ID Tối Ưu

Sau khi Kalman Filter dự đoán vị trí của tất cả tracklets, bài toán gán (assignment) yêu cầu tìm khớp tối ưu giữa tập dự đoán $\mathcal{T} = \{t_1, \ldots, t_M\}$ và tập detection $\mathcal{D} = \{d_1, \ldots, d_N\}$. Đây là bài toán gán tuyến tính (Linear Assignment Problem), được giải bởi thuật toán Hungarian (còn gọi là Kuhn-Munkres algorithm) với độ phức tạp $O(\min(M, N)^3)$.

Ma trận chi phí $\mathbf{C} \in \mathbb{R}^{M \times N}$ được xây dựng dựa trên IoU giữa predicted box của tracklet $t_i$ và bounding box của detection $d_j$:

$$C_{ij} = 1 - \text{IoU}(\text{pred}(t_i),\ d_j)$$

Mục tiêu của Hungarian Algorithm là tìm phép gán song ánh (bijection) $\pi: \mathcal{T} \to \mathcal{D}$ sao cho tổng chi phí là nhỏ nhất:

$$\pi^* = \arg\min_\pi \sum_{i} C_{i,\pi(i)}$$

Sau khi gán, các cặp $(t_i, d_{\pi(i)})$ với $C_{i,\pi(i)} < \theta_{IoU}$ (ngưỡng IoU tối thiểu) được coi là khớp thành công; các tracklet và detection còn lại chuyển sang bước xử lý tiếp theo.

### 2.3.4 Ý Tưởng Cốt Lõi của ByteTrack

Phương pháp BYTE (viết tắt từ "Byte" — đơn vị cơ bản, nhấn mạnh rằng mọi detection đều có giá trị) do Zhang et al. (ECCV 2022) đề xuất giải quyết một vấn đề nền tảng trong MOT: hầu hết các phương pháp trước đây chỉ sử dụng detection có confidence cao (thường $> 0.5$) và loại bỏ hoàn toàn detection có confidence thấp. Tuy nhiên, đối tượng bị che khuất một phần hoặc ở xa camera vẫn có thể được detector phát hiện với confidence thấp, và loại bỏ chúng dẫn đến mất track không thể phục hồi.

**Cơ chế hai giai đoạn (Two-Stage Matching):**

Tại mỗi frame $f_k$, toàn bộ detection $\mathcal{D}_k$ được phân thành hai tập theo ngưỡng confidence $\tau$ (thường $\tau = 0.5$):

$$\mathcal{D}_{high} = \{d \in \mathcal{D}_k : d.score \geq \tau\}, \quad \mathcal{D}_{low} = \{d \in \mathcal{D}_k : d.score < \tau\}$$

**Giai đoạn 1:** Áp dụng Kalman Filter dự đoán vị trí tất cả tracklets trong $\mathcal{T}$ (bao gồm cả các tracklet đang ở trạng thái "lost"). Thực hiện Hungarian Algorithm để khớp $\mathcal{D}_{high}$ với $\mathcal{T}$ dựa trên IoU similarity (hoặc kết hợp IoU với Re-ID feature distance). Kết quả:

- Các cặp khớp thành công: tracklet được cập nhật bằng Kalman Update với observation mới.
- $\mathcal{T}_{remain}$: tracklets không khớp với bất kỳ detection nào trong $\mathcal{D}_{high}$.
- $\mathcal{D}_{remain}$: detections trong $\mathcal{D}_{high}$ không được khớp với tracklet nào.

**Giai đoạn 2:** Thực hiện lần khớp thứ hai giữa $\mathcal{T}_{remain}$ (tracklets chưa khớp) và $\mathcal{D}_{low}$ (detection confidence thấp), sử dụng **chỉ IoU** (không dùng Re-ID feature, vì detection confidence thấp thường bị nhiễu và đặc trưng ngoại hình không đáng tin):

- Tracklets khớp thành công với $\mathcal{D}_{low}$: đối tượng bị che khuất một phần được giữ lại, track được duy trì liên tục.
- $\mathcal{D}_{low}$ không khớp: bị loại bỏ hoàn toàn (coi là nhiễu nền).
- $\mathcal{T}_{re-remain}$: tracklets không khớp sau cả hai giai đoạn → chuyển sang trạng thái "Lost".

**Khởi tạo track mới:** Các detection trong $\mathcal{D}_{remain}$ (high-score nhưng không khớp với tracklet nào) được khởi tạo thành tracklet mới với ID mới.

Ý nghĩa của thiết kế hai giai đoạn: detection confidence thấp trong giai đoạn 2 chỉ được dùng để **duy trì** tracklet hiện có (không tạo track mới), trong khi detection mới (foreground thực sự) phải có confidence cao mới được khởi tạo. Điều này cân bằng giữa khả năng khôi phục track và tránh tạo track giả từ nhiễu nền.

### 2.3.5 Vòng Đời của Một Track

Mỗi tracklet trong ByteTrack trải qua các trạng thái sau trong vòng đời của nó:

**New (Tentative):** Khi một detection mới trong $\mathcal{D}_{remain}$ (high-score, không khớp với tracklet nào) được khởi tạo thành tracklet. Track ở trạng thái tentative trong tham số `min_frames` frame (mặc định là 3 trong đề tài). Trong giai đoạn này, track chưa được xuất ra output để tránh false positive ngắn hạn.

**Confirmed (Activated):** Sau khi track được xác nhận liên tục qua ít nhất `min_frames` frame, track chuyển sang trạng thái Confirmed và bắt đầu được xuất ra output với ID xác định.

**Lost:** Khi một Confirmed track không tìm được detection khớp trong cả hai giai đoạn của một frame, nó chuyển sang trạng thái Lost. Track Lost vẫn được duy trì trong buffer `lost_buffer` frame (mặc định 90 frame ≈ 3 giây @ 30fps trong đề tài). Trong thời gian này, Kalman Filter tiếp tục dự đoán vị trí và track vẫn tham gia vào quá trình gán ở frame tiếp theo.

**Reactivated (Refind):** Nếu trong thời gian buffer, track Lost tìm được detection khớp, nó quay trở lại trạng thái Confirmed với cùng ID — đây là cơ chế "track rebirth" quan trọng để xử lý trường hợp đối tượng ra ngoài frame rồi quay lại, hoặc bị che khuất lâu.

**Removed:** Sau khi track ở trạng thái Lost quá `lost_buffer` frame mà vẫn không tìm được detection khớp, track bị xóa hoàn toàn khỏi hệ thống.

Trong đề tài, các siêu tham số được chỉnh: `activation_threshold = 0.35`, `lost_buffer = 90` frame, `min_frames = 3` — được tối ưu hóa thực nghiệm cho đặc thù video bóng đá với 30fps và cầu thủ thường ra khỏi frame ngắn (bị che khuất bởi cầu thủ khác hoặc đứng ngoài biên camera).

---

## 2.4 Hiệu Chỉnh Camera — NBJW và HRNet-W48

Phần này trình bày lý thuyết và phương pháp hiệu chỉnh camera sân bóng đá dựa trên công trình "No Bells, Just Whistles: Sports Field Registration by Leveraging Geometric Properties" của Gutiérrez-Pérez và Agudo (CVPR Workshop CVsports, 2024). Toàn bộ nội dung được phân tích trực tiếp từ bài báo gốc.

### 2.4.1 Mô Hình Hình Học Camera (Pinhole Camera Model)

**Mô hình camera lỗ kim (Pinhole Camera Model)** là mô hình toán học cơ bản mô tả quá trình chiếu không gian 3D vào mặt phẳng ảnh 2D. Ma trận chiếu camera $\mathbf{P} \in \mathbb{R}^{3 \times 4}$ được biểu diễn như sau:

$$\mathbf{P} = \mathbf{K} \mathbf{R} \begin{bmatrix} \mathbf{I} & -\mathbf{c} \end{bmatrix}$$

trong đó:

- $\mathbf{K} \in \mathbb{R}^{3 \times 3}$ là **ma trận nội tại** (intrinsic matrix) chứa các thông số nội của camera:

$$\mathbf{K} = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

với $f_x, f_y$ là tiêu cự tính bằng pixel theo chiều ngang và dọc, $(c_x, c_y)$ là tọa độ điểm chính (principal point, thường gần tâm ảnh), và $s$ là hệ số nghiêng (skew, thường bằng 0).

- $\mathbf{R} \in \mathbb{R}^{3 \times 3}$ là **ma trận quay** (rotation matrix) mô tả hướng của camera trong không gian thế giới.

- $\mathbf{c} \in \mathbb{R}^{3}$ là **vị trí tâm camera** trong không gian thế giới.

Quá trình chiếu một điểm 3D $\mathbf{X} = [X, Y, Z, 1]^T$ (tọa độ thuần nhất — homogeneous) sang tọa độ ảnh $\mathbf{x} = [u, v, 1]^T$ được thực hiện qua:

$$\lambda \mathbf{x} = \mathbf{P} \mathbf{X}$$

với $\lambda$ là nhân tử vô hướng (projective depth). Chia tọa độ thuần nhất để lấy tọa độ pixel: $u = (\mathbf{P}\mathbf{X})_1 / (\mathbf{P}\mathbf{X})_3$, $v = (\mathbf{P}\mathbf{X})_2 / (\mathbf{P}\mathbf{X})_3$.

Trong bài toán hiệu chỉnh camera sân bóng đá, NBJW sử dụng mô hình camera với giả thiết: skew $s = 0$, tỉ lệ khung hình đã biết ($f_x / f_y$ cố định từ metadata video), và điểm chính tại tâm ảnh ($c_x = W/2$, $c_y = H/2$). Điều này giảm số bậc tự do cần ước lượng, giúp bài toán hiệu chỉnh ổn định hơn với ít điểm tham chiếu hơn.

### 2.4.2 Homography và Phép Biến Đổi Mặt Phẳng

**Homography** (hay phép biến đổi phối cảnh) là ánh xạ tuyến tính thuần nhất giữa hai mặt phẳng trong không gian chiếu. Trong bối cảnh sân bóng đá, homography mô tả ánh xạ từ mặt phẳng sân (ground plane, $Z = 0$) sang mặt phẳng ảnh.

Khi $Z = 0$, ma trận chiếu $\mathbf{P} = \mathbf{K}\mathbf{R}[\mathbf{I}|-\mathbf{c}]$ được rút gọn thành ma trận homography $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ bằng cách lấy cột 1, 2, 4 của $\mathbf{P}$ (bỏ cột 3 tương ứng với $Z$):

$$\mathbf{H} = \mathbf{K} \begin{bmatrix} \mathbf{r}_1 & \mathbf{r}_2 & \mathbf{t} \end{bmatrix}$$

trong đó $\mathbf{r}_1, \mathbf{r}_2$ là cột đầu và cột hai của ma trận quay $\mathbf{R}$, và $\mathbf{t} = -\mathbf{R}\mathbf{c}$ là vector tịnh tiến.

Điểm sân $\mathbf{X}_{2D} = [X, Y, 1]^T$ (tọa độ thực tế trên sân, đơn vị mét) được chiếu sang tọa độ ảnh $\mathbf{x}$ qua:

$$\lambda \mathbf{x} = \mathbf{H} \mathbf{X}_{2D}$$

Ngược lại, để tính tọa độ thực tế từ tọa độ ảnh (dùng cho bài toán đo quãng đường cầu thủ), ta dùng ma trận nghịch đảo $\mathbf{H}^{-1}$:

$$\mathbf{X}_{2D} \sim \mathbf{H}^{-1} \mathbf{x}$$

**Thuật toán DLT (Direct Linear Transform)** là phương pháp ước lượng ma trận homography từ ít nhất $N \geq 4$ cặp điểm tương ứng $\{(\mathbf{x}_i, \mathbf{X}_{2D,i})\}_{i=1}^N$. Với mỗi cặp điểm, điều kiện $\lambda \mathbf{x}_i = \mathbf{H} \mathbf{X}_{2D,i}$ dẫn đến hệ phương trình tuyến tính thuần nhất:

$$\mathbf{A}_i \mathbf{h} = \mathbf{0}$$

trong đó $\mathbf{h} = \text{vec}(\mathbf{H}) \in \mathbb{R}^9$ là vector dạng cột của ma trận $\mathbf{H}$, và $\mathbf{A}_i \in \mathbb{R}^{2 \times 9}$ được xây dựng từ tọa độ của cặp điểm thứ $i$. Tập hợp tất cả $N$ cặp điểm tạo thành hệ:

$$\mathbf{A} \mathbf{h} = \mathbf{0}, \quad \mathbf{A} \in \mathbb{R}^{2N \times 9}$$

Nghiệm của bài toán là vector riêng (eigenvector) tương ứng với giá trị kỳ dị (singular value) nhỏ nhất của $\mathbf{A}$, tính qua phân rã giá trị kỳ dị (SVD): $\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$, và $\mathbf{h} = \mathbf{V}_{:,9}$ (cột cuối của $\mathbf{V}$).

Để tăng tính ổn định số học, DLT được áp dụng trên dữ liệu đã chuẩn hóa (normalized DLT): tọa độ điểm ảnh được dịch về tâm ảnh và scale để giá trị trung bình khoảng cách đến gốc tọa độ bằng $\sqrt{2}$.

NBJW sử dụng DLT kết hợp **RANSAC** (Random Sample Consensus) để loại bỏ các điểm tương ứng sai (outliers) do lỗi trong bước phát hiện keypoint. Cụ thể, RANSAC lặp lại quá trình: (1) chọn ngẫu nhiên 4 cặp điểm để tính $\mathbf{H}$; (2) đếm số điểm nằm trong ngưỡng reprojection error $\epsilon$; (3) giữ mô hình có số inliers cao nhất. Sau đó DLT được tính lại trên toàn bộ inliers, và refinement tiếp tục bằng tối ưu Levenberg-Marquardt để tối thiểu hóa tổng reprojection error:

$$\mathbf{H}^* = \arg\min_{\mathbf{H}} \sum_{i \in \text{inliers}} \left\| \mathbf{x}_i - \frac{\mathbf{H} \mathbf{X}_{2D,i}}{(\mathbf{H} \mathbf{X}_{2D,i})_3} \right\|^2$$

### 2.4.3 Mạng HRNet-W48 (High-Resolution Network)

**HRNet (High-Resolution Network)** là kiến trúc mạng nơ-ron được Wang et al. (TPAMI 2020) đề xuất cho các bài toán đòi hỏi biểu diễn không gian chi tiết như ước lượng pose, phân đoạn ngữ nghĩa và phát hiện keypoint.

Ý tưởng chính của HRNet là **duy trì đặc trưng độ phân giải cao xuyên suốt toàn bộ quá trình xử lý**, thay vì chiến lược encoder-decoder thông thường (downsampling → bottleneck → upsampling). Cụ thể, HRNet bắt đầu từ một luồng độ phân giải cao, sau đó từng bước bổ sung thêm các luồng độ phân giải thấp hơn song song (high-resolution subnetwork). Giữa các giai đoạn, **multi-resolution fusion module** thực hiện trao đổi thông tin theo cả hai hướng: từ độ phân giải thấp lên cao (bổ sung ngữ nghĩa) và từ cao xuống thấp (bổ sung chi tiết không gian).

Trong HRNet-W48, "W48" biểu thị số kênh (width) của luồng độ phân giải cao nhất là 48, trong khi các luồng thấp hơn có số kênh là 96, 192, 384 tương ứng (tăng gấp đôi mỗi bước downsampling). Kích thước đặc trưng đầu ra của các luồng là $\frac{H}{4} \times \frac{W}{4}$, $\frac{H}{8} \times \frac{W}{8}$, $\frac{H}{16} \times \frac{W}{16}$, $\frac{H}{32} \times \frac{W}{32}$.

Trong cấu hình HRNetV2 (được NBJW sử dụng), đầu ra của tất cả các luồng được upsampling về cùng kích thước $\frac{H}{4} \times \frac{W}{4}$ rồi concatenate, tạo ra biểu diễn đa tỉ lệ phong phú với tổng số kênh là $48 + 96 + 192 + 384 = 720$. Đầu ra này sau đó được truyền qua hai tầng tích chập $1 \times 1$ để giảm chiều và dự đoán heatmap cho từng keypoint/line extremity.

Ưu điểm của HRNet so với các kiến trúc encoder-decoder như U-Net là: vì độ phân giải cao được duy trì trong suốt quá trình, biểu diễn tại đầu ra giữ lại nhiều thông tin vị trí chính xác hơn. Điều này đặc biệt quan trọng cho bài toán phát hiện keypoint sân bóng đá, nơi sự chênh lệch vài pixel trong vị trí keypoint có thể gây sai lệch đáng kể về tọa độ sân thực tế sau biến đổi homography.

### 2.4.4 Pipeline NBJW — Gutiérrez-Pérez & Agudo

NBJW (No Bells, Just Whistles) đề xuất một pipeline hiệu chỉnh camera sân bóng đá **tối giản nhưng hiệu quả**, không sử dụng refinement phức tạp hay database tìm kiếm, dựa hoàn toàn vào hình học sân và thuật toán DLT cổ điển.

**Mô hình sân bóng đá và các tập keypoints**

NBJW xây dựng mô hình 3D của sân bóng đá từ các đặc điểm hình học đã biết (kích thước sân chuẩn FIFA). Các keypoints được định nghĩa theo cấu trúc phân cấp thành năm tập:

Tập $\mathcal{K}_p$ gồm tối đa 30 điểm là **giao điểm trực tiếp** giữa các đường kẻ sân (line-line intersections): góc sân, góc vùng cấm địa, điểm giao giữa đường biên ngang và đường vòng cấm, v.v.

Tập $\mathcal{K}_e$ gồm các **giao điểm mở rộng** (extended intersections): tức là giao điểm của các đường kẻ sân khi được kéo dài ra ngoài phạm vi thực tế của chúng. Ví dụ, đường biên ngang và đường biên ngang đối diện nếu kéo dài sẽ cắt nhau tại điểm vô cực của không gian phối cảnh. Tập này cho phép sử dụng các điểm hội tụ (vanishing points) như điểm tham chiếu.

Tập $\mathcal{K}_{p_1}$ gồm các **giao điểm giữa đường kẻ sân và vòng tròn trung tâm** (ellipse intersections): phần vòng tròn trung tâm được nhìn từ camera broadcast trở thành một ellipse do hiệu ứng phối cảnh. Bằng cách fit ellipse (tối thiểu hóa khoảng cách điểm-đến-ellipse), NBJW xác định được các điểm giao giữa đường kẻ sân và ellipse này.

Tập $\mathcal{K}_{p_2}$ gồm các **điểm tiếp tuyến** (ellipse tangent points): từ một điểm ngoài ellipse, có thể vẽ hai đường tiếp tuyến đến ellipse. Các điểm tiếp tuyến này được tính giải tích từ điểm ngoài và phương trình ellipse.

Tập $\mathcal{K}_{p_3}$ gồm **9 điểm dọc theo trục trung tâm** và **4 điểm chia tư vòng tròn** (quarter-turn points), bổ sung thêm điểm tham chiếu tại khu vực giữa sân — vùng thường được camera broadcast quan sát rõ nhất.

Tổng cộng, NBJW có thể sử dụng tối đa hàng chục điểm tham chiếu từ một frame, mỗi điểm có tọa độ 3D đã biết từ mô hình sân, cho phép DLT tính homography/calibration với độ tin cậy cao.

**Phát hiện Keypoints và Line Extremities**

Hai mạng HRNetV2-W48 riêng biệt được huấn luyện để phát hiện keypoints và line extremities:

*Mạng Keypoints* ($f_{kp}$): Đầu ra là $N_{kp}$ heatmaps (một heatmap cho mỗi loại keypoint), trong đó mỗi heatmap là một phân phối xác suất 2D với đỉnh Gaussian ($\sigma = 2$ pixel) tại vị trí keypoint tương ứng trong ảnh. Vị trí keypoint được trích xuất bằng argmax trên heatmap:

$$\hat{\mathbf{x}}_k = \arg\max_{(i,j)} H_k(i, j)$$

Ngoài ra, một kênh bổ sung $H_{inv}$ được tính là nghịch đảo của giá trị cực đại trên tất cả các kênh khác, tạo thành phân phối xác suất hoàn chỉnh: $\sum_k H_k(i,j) + H_{inv}(i,j) = 1$.

*Mạng Lines* ($f_{line}$): Đầu ra là $N_{line}$ cặp heatmaps (hai đỉnh Gaussian cho hai đầu của mỗi đường kẻ sân) cùng một kênh biên (boundary channel) phụ trợ để cung cấp thông tin cấu trúc toàn cục của sân. Vị trí hai đầu đường kẻ được trích xuất bằng max pooling và top-2 detection.

Trong quá trình huấn luyện, khi homography ground-truth không khả dụng (ví dụ frame bị cắt hoặc camera góc lệch), heatmaps của các tập $\mathcal{K}_{p_1}$, $\mathcal{K}_{p_2}$, $\mathcal{K}_{p_3}$ (những tập đòi hỏi homography để tính toán) được **mask** khỏi hàm mất mát L2, tránh truyền gradient sai về mô hình.

**Tính toán Homography và Hiệu Chỉnh Camera**

Từ các keypoints phát hiện được, NBJW thực hiện hiệu chỉnh theo quy trình đa bước:

Đầu tiên, RANSAC + DLT được áp dụng trên tập $\mathcal{K}_p \cup \mathcal{K}_e$ (không bao gồm các điểm trên vòng tròn/ellipse) để ước lượng homography sơ bộ. Homography này được dùng để tính toán các tập $\mathcal{K}_{p_1}$, $\mathcal{K}_{p_2}$, $\mathcal{K}_{p_3}$ trong không gian ảnh qua chiếu ngược.

Tiếp theo, bài toán hiệu chỉnh camera 3D được giải bằng cách sử dụng các điểm 3D trên sân (bao gồm cả các điểm không nằm trên mặt phẳng sân như cột khung thành, xà ngang) thông qua thuật toán giải form đóng (closed-form solution) kết hợp với non-linear refinement qua maximum likelihood estimation.

Để tăng tính bền vững (robustness), NBJW thử nghiệm **nhiều tập con điểm** (calibration subsets): full-keypoints (tất cả 5 tập), main-keypoints (chỉ $\mathcal{K}_p$), ground-plane-keypoints (loại trừ điểm không nằm trên mặt phẳng sân). Kết hợp với lưới các ngưỡng reprojection error trong RANSAC, hệ thống thực hiện **chiến lược bỏ phiếu** (voting): chọn kết quả hiệu chỉnh từ tập con và ngưỡng tạo ra reprojection error nhỏ nhất trên inliers, ưu tiên tập full-keypoints.

**Kết quả đánh giá từ paper gốc**

NBJW đạt **73.7 Final Score (FS)** trên SoccerNet-Calibration (SN22-test-center), vượt xa TVCalib (53.9 FS) — phương pháp SOTA trước đó. Trên WorldCup 2014, phương pháp đạt **96.2% IoU_part** và **97.8% IoU_whole**, tương đương với phương pháp tốt nhất lúc bấy giờ dù sử dụng pipeline đơn giản hơn đáng kể.

**Cấu hình huấn luyện** trong paper gốc: 200 epoch, optimizer Adam ($\beta_1=0.9$, $\beta_2=0.999$), learning rate $10^{-5}$ (single-view model), batch size 22, augmentation bao gồm random horizontal flip, color jitter và Gaussian noise. Hardware: NVIDIA RTX 2080 Ti (12GB VRAM).

Trong đề tài này, weights pretrained của NBJW (SV_kp và SV_lines) được sử dụng trực tiếp mà không fine-tune thêm, với các ngưỡng phát hiện được chỉnh: `keypoint_threshold = 0.1486`, `line_threshold = 0.3880` — được xác định thực nghiệm để cân bằng giữa độ phủ (completeness) và độ chính xác khi áp dụng trên các video SoccerNet trong đề tài.

### 2.4.5 Exponential Moving Average (EMA) Làm Mượt Tọa Độ

Do hiệu chỉnh camera được thực hiện với tần suất một lần mỗi 15 frames (để giảm tải tính toán — mỗi lần chạy hai mạng HRNet-W48 mất vài giây), tọa độ sân của cầu thủ giữa các lần hiệu chỉnh có thể bị giật cục (jittering) do slight variation trong homography giữa các lần hiệu chỉnh.

Để xử lý vấn đề này, **Exponential Moving Average (EMA)** được áp dụng để làm mượt tọa độ sân theo thời gian. Với vị trí sân thực tế của cầu thủ $\mathbf{p}_t = (X_t, Y_t)$ tại frame $t$, tọa độ được làm mượt $\tilde{\mathbf{p}}_t$ được tính như sau:

$$\tilde{\mathbf{p}}_t = \alpha \cdot \mathbf{p}_t + (1 - \alpha) \cdot \tilde{\mathbf{p}}_{t-1}$$

trong đó $\alpha \in (0, 1)$ là hệ số làm mượt (smoothing factor). Trong đề tài, $\alpha = 0.4$, có nghĩa là giá trị hiện tại đóng góp 40% và lịch sử đóng góp 60% vào ước lượng mới.

EMA có trọng số giảm theo hàm mũ đối với các quan sát trong quá khứ: quan sát $\mathbf{p}_{t-k}$ đóng góp trọng số $\alpha (1-\alpha)^k$ vào $\tilde{\mathbf{p}}_t$. Điều này có nghĩa là EMA tự động "quên" các quan sát cũ theo tốc độ được kiểm soát bởi $\alpha$: $\alpha$ lớn → phản ứng nhanh hơn với thay đổi nhưng kém mượt; $\alpha$ nhỏ → mượt hơn nhưng lag nhiều hơn.

Giá trị $\alpha = 0.4$ được chọn để cân bằng giữa tốc độ theo dõi chuyển động cầu thủ (cần phản ứng nhanh với di chuyển tốc độ cao) và loại bỏ nhiễu calibration (cần làm mượt biến thiên ngẫu nhiên giữa các frame hiệu chỉnh).

---

## 2.5 Nhận Dạng Lại Cầu Thủ — OSNet

### 2.5.1 Bài Toán Person Re-Identification

**Person Re-Identification (Re-ID)** là bài toán nhận dạng lại cùng một cá thể (trong đề tài này là cầu thủ) qua nhiều góc nhìn camera khác nhau hoặc sau một khoảng thời gian gián đoạn. Đây là bài toán nhận dạng ở cấp độ instance (instance-level recognition), khác về bản chất với nhận dạng khuôn mặt (face recognition) ở chỗ không dựa vào đặc điểm sinh trắc học rõ ràng mà phải dựa vào đặc trưng tổng thể như màu sắc, hình dạng trang phục, kiểu di chuyển.

Trong bối cảnh phân tích video bóng đá, bài toán Re-ID phát sinh khi một cầu thủ (với tracker ID đã được gán) bị che khuất hoàn toàn hoặc ra khỏi khung hình trong thời gian đủ lâu để ByteTrack xóa tracklet. Khi cầu thủ đó xuất hiện lại, detector sẽ tạo ra một detection mới và ByteTrack sẽ gán cho nó một ID mới — dẫn đến sai sót nghiêm trọng trong thống kê (cùng một cầu thủ nhưng có hai tracker ID khác nhau, làm đứt quãng quỹ đạo và thống kê). Re-ID giải quyết vấn đề này bằng cách so sánh đặc trưng ngoại hình của detection mới với thư viện ảnh (gallery) của các cầu thủ đã được theo dõi.

**Thách thức đặc thù trong bóng đá:** Tất cả cầu thủ cùng đội mặc đồng phục giống nhau → rất ít đặc trưng màu sắc phân biệt trong cùng đội; khoảng cách từ camera đến cầu thủ rất lớn → ảnh cầu thủ nhỏ và mờ; cầu thủ di chuyển nhanh → pose thay đổi liên tục. Đây là lý do SoccerNet Re-ID Dataset được thiết kế đặc biệt cho môi trường này, với 340.993 thumbnail từ 400 trận đấu thực tế.

**Các độ đo đánh giá Re-ID:**

*Rank-1 Accuracy* là xác suất để ảnh cùng identity đứng đầu danh sách kết quả tìm kiếm:

$$\text{Rank-1} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \mathbb{1}[\text{rank}_1(q) \text{ là same identity}]$$

*Mean Average Precision (mAP)* được tính trung bình AP trên toàn bộ queries, trong đó AP cho mỗi query là diện tích dưới đường cong Precision-Recall khi threshold thay đổi trên danh sách kết quả sắp xếp:

$$\text{mAP} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} AP(q)$$

*CMC (Cumulative Matching Characteristic) curve* cho thấy xác suất để ảnh cùng identity xuất hiện trong top-$k$ kết quả tìm kiếm, theo $k$.

### 2.5.2 Kiến Trúc OSNet (Omni-Scale Network)

OSNet (Omni-Scale Network) do Zhou et al. (ICCV 2019) đề xuất là kiến trúc CNN được thiết kế đặc biệt cho bài toán Re-ID, với hai đóng góp chính: **Omni-Scale Feature Learning** và **Unified Aggregation Gate (UAG)**.

**Tích Chập Khả Ly Theo Chiều Sâu (Depthwise Separable Convolution)**

Để xây dựng mạng nhẹ, OSNet phân tích tích chập tiêu chuẩn $k \times k$ thành hai bước kế tiếp: depthwise convolution (áp dụng một bộ lọc $k \times k$ riêng biệt cho từng kênh đầu vào) và pointwise convolution (tích chập $1 \times 1$ trộn thông tin các kênh). Chi phí tính toán giảm từ $k^2 \cdot C \cdot C'$ xuống còn $(k^2 + C) \cdot C'$, tương đương giảm $k^2 / (1 + k^2/C)$ lần.

OSNet sử dụng thứ tự **pointwise → depthwise** (thay vì depthwise → pointwise thông thường như trong MobileNet), được đặt tên là **Lite $3 \times 3$ layer**. Thứ tự này — tăng chiều kênh trước bằng pointwise rồi mới tổng hợp không gian bằng depthwise — thực nghiệm cho thấy hiệu quả hơn trong bài toán omni-scale feature learning.

**Khối Omni-Scale Residual (Omni-Scale Residual Block)**

Khối này mở rộng residual bottleneck cổ điển bằng cách tích hợp nhiều luồng tích chập song song ở các tỉ lệ không gian khác nhau. Tham số **exponent** $t$ kiểm soát tỉ lệ của mỗi luồng: luồng thứ $t$ chồng $t$ tầng Lite $3 \times 3$ liên tiếp, tạo ra trường nhận thức (receptive field) kích thước $(2t+1) \times (2t+1)$. Với $T = 4$ luồng (tỉ lệ $3 \times 3$, $5 \times 5$, $7 \times 7$, $9 \times 9$), khối học các đặc trưng đồng thời ở bốn tỉ lệ không gian khác nhau.

Tổng thể, residual được tính là tổng các đặc trưng đa tỉ lệ với trọng số động từ Aggregation Gate:

$$\tilde{x} = \sum_{t=1}^{T} G(x_t) \odot x_t, \quad x_t = F_t(x)$$

trong đó $F_t(x)$ là luồng tích chập thứ $t$ áp dụng lên input $x$, $G(x_t)$ là vector trọng số channel-wise được tính từ Unified Aggregation Gate, và $\odot$ là tích Hadamard (nhân từng phần tử). Đầu ra của khối:

$$y = x + \tilde{x}$$

**Unified Aggregation Gate (UAG)**

UAG là mô-đun nhỏ chia sẻ tham số (shared) cho tất cả $T$ luồng trong cùng khối. Với đặc trưng $x_t \in \mathbb{R}^{H' \times W' \times C}$ của luồng $t$, UAG tính trọng số channel-wise $G(x_t) \in \mathbb{R}^C$ theo các bước:

Đầu tiên, Global Average Pooling (GAP) thu gọn không gian không gian và trích xuất vector đặc trưng toàn cục:

$$\mathbf{z}_t = \frac{1}{H' \cdot W'} \sum_{i=1}^{H'} \sum_{j=1}^{W'} x_t[i, j, :]$$

Sau đó, một MLP một tầng ẩn với tỉ lệ giảm chiều 16 tính trọng số kênh:

$$G(x_t) = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}\left(\mathbf{W}_1 \mathbf{z}_t\right)\right)$$

trong đó $\mathbf{W}_1 \in \mathbb{R}^{(C/16) \times C}$, $\mathbf{W}_2 \in \mathbb{R}^{C \times (C/16)}$, và $\sigma$ là hàm sigmoid tạo ra trọng số trong khoảng $(0, 1)$.

Giá trị quan trọng là UAG **chia sẻ tham số** $\mathbf{W}_1, \mathbf{W}_2$ cho tất cả $T$ luồng. Điều này có hai hệ quả: (1) số tham số của AG không phụ thuộc vào $T$, giữ mạng nhẹ; (2) gradient từ tất cả luồng đều cùng cập nhật tham số AG: $\frac{\partial \mathcal{L}}{\partial G} = \frac{\partial \mathcal{L}}{\partial \tilde{x}} \cdot \frac{\partial \tilde{x}}{\partial G} = \frac{\partial \mathcal{L}}{\partial \tilde{x}} \cdot \sum_t x_t$, giúp học hiệu quả hơn so với gate riêng biệt cho từng luồng.

**Kiến Trúc Tổng Thể OSNet**

OSNet được xây dựng bằng cách chồng các khối Omni-Scale Residual lên nhau qua nhiều stage, mỗi stage giảm độ phân giải không gian và tăng số kênh. Tổng thể mạng có cấu trúc: Conv($64$) → MaxPool → Stage1[$64$] → Conv($256$) → Stage2[$256$] → Conv($384$) → Stage3[$384$] → Conv($512$) → GAP → FC($512$) → Softmax/Identity.

Trong phiên bản OSNet_x1.0 (được dùng trong đề tài), mạng có khoảng **2.2 triệu tham số** — nhỏ hơn khoảng **10 lần** so với ResNet50 (25M tham số) trong khi đạt Rank-1 accuracy trên Market-1501 là 84.9%, vượt trội nhiều kiến trúc lớn hơn.

### 2.5.3 Cosine Similarity và Xây Dựng Feature Gallery

**Trích xuất embedding:** Với ảnh cầu thủ được cắt từ bounding box, OSNet trả về vector đặc trưng (embedding) $\mathbf{f} \in \mathbb{R}^{512}$ sau tầng GAP và FC cuối. Vector này được chuẩn hóa L2 để tất cả embeddings nằm trên mặt cầu đơn vị $S^{511}$:

$$\hat{\mathbf{f}} = \frac{\mathbf{f}}{\|\mathbf{f}\|_2}$$

**Cosine Similarity** giữa hai embedding $\hat{\mathbf{f}}_1$ và $\hat{\mathbf{f}}_2$ được tính:

$$\text{sim}(\hat{\mathbf{f}}_1, \hat{\mathbf{f}}_2) = \hat{\mathbf{f}}_1 \cdot \hat{\mathbf{f}}_2 = \frac{\mathbf{f}_1 \cdot \mathbf{f}_2}{\|\mathbf{f}_1\|_2 \cdot \|\mathbf{f}_2\|_2}$$

Sau chuẩn hóa L2, cosine similarity tương đương khoảng cách Euclidean âm: $\text{sim}(\hat{\mathbf{f}}_1, \hat{\mathbf{f}}_2) = 1 - \frac{1}{2}\|\hat{\mathbf{f}}_1 - \hat{\mathbf{f}}_2\|_2^2$. Giá trị similarity nằm trong $[-1, 1]$, với 1 là giống nhau hoàn toàn.

**Quản lý Gallery:** Mỗi tracklet đang hoạt động duy trì một gallery gồm nhiều embeddings trích xuất từ các frame gần đây nhất. Khi ByteTrack phát hiện một track mất kết nối (Lost), hệ thống Re-ID thực hiện:

1. Với mỗi detection mới chưa được gán ID, trích xuất embedding $\hat{\mathbf{f}}_{new}$.
2. Tính cosine similarity giữa $\hat{\mathbf{f}}_{new}$ và tất cả embeddings trong gallery của tất cả Lost tracklets.
3. Nếu similarity cao nhất vượt ngưỡng $\theta = 0.65$ (được cấu hình trong đề tài), gán detection mới vào tracklet có similarity cao nhất → khôi phục tracker ID cũ.
4. Nếu không có gallery nào vượt ngưỡng → khởi tạo tracklet mới.

Gallery được cập nhật rolling: chỉ giữ lại embeddings của $n$ frame gần nhất (kiểm soát bởi `max_age = 120` frames). Điều này đảm bảo gallery phản ánh ngoại hình hiện tại của cầu thủ (tránh ảnh hưởng bởi ảnh quá cũ khi cầu thủ đã thay áo hoặc ánh sáng thay đổi đáng kể).

### 2.5.4 Lý Do Lựa Chọn OSNet Cho Bài Toán Này

Quyết định chọn OSNet_x1.0 thay vì các kiến trúc Re-ID khác (như TransReID, AGW, hay các mô hình dựa trên ViT) dựa trên ba tiêu chí kỹ thuật chính:

**Thứ nhất — Hiệu suất trên dữ liệu thể thao:** Weights `osnet_x1_0_sportsreid.pth.tar` được pretrain đặc biệt trên SoccerNet Re-ID Dataset — bộ dữ liệu với 340.993 ảnh cầu thủ bóng đá thực tế, được thu thập từ 400 trận đấu ở 6 giải lớn châu Âu. Pretrain domain-specific này cho phép OSNet học được đặc trưng phù hợp với môi trường bóng đá (đồng phục, ánh sáng stadium, góc máy broadcast) mà không cần fine-tune thêm.

**Thứ hai — Hiệu quả tính toán:** Với chỉ 2.2M tham số và kiến trúc nhẹ (Lite $3 \times 3$), OSNet có thể chạy inference đủ nhanh để xử lý tất cả detections trong một frame mà không tạo ra bottleneck đáng kể trong pipeline 6 giai đoạn.

**Thứ ba — Omni-scale learning phù hợp với Re-ID bóng đá:** Cầu thủ ở các khoảng cách khác nhau từ camera có kích thước bounding box rất khác nhau (từ $30 \times 60$ pixel ở xa đến $100 \times 200$ pixel ở gần). Khả năng học đặc trưng ở nhiều tỉ lệ không gian đồng thời của OSNet giúp mô hình nhận dạng cùng một cầu thủ dù kích thước ảnh thay đổi đáng kể — đây là ưu điểm rõ ràng so với các kiến trúc single-scale truyền thống.

---

## 2.6 Tóm Tắt Chương

Chương này đã trình bày nền tảng lý thuyết của năm thành phần kỹ thuật cốt lõi cấu thành hệ thống phân tích video bóng đá trong luận văn. Bảng 2.1 tổng hợp thông tin chính của từng mô hình/phương pháp.

**Bảng 2.1: Tổng hợp các mô hình và phương pháp được sử dụng**

| Mô hình / Phương pháp | Bài toán | Tác giả / Venue | Đặc điểm kiến trúc chính | Vai trò trong pipeline |
|----------------------|----------|-----------------|--------------------------|----------------------|
| YOLO11 | Object Detection (8 lớp) | Ultralytics, 2024 | Anchor-free, C3k2, SPPF, C2PSA, decoupled head | Phát hiện cầu thủ, bóng, trọng tài mỗi frame |
| ByteTrack | Multi-Object Tracking | Zhang et al., ECCV 2022 | Two-stage matching (BYTE), Kalman Filter, Hungarian | Gán và duy trì tracker ID liên tục xuyên video |
| HRNet-W48 | Keypoint/Line Detection | Wang et al., TPAMI 2020 | Multi-resolution subnetworks, fusion modules, HRNetV2 | Backbone phát hiện keypoints và đường kẻ sân |
| NBJW | Camera Calibration | Gutiérrez-Pérez & Agudo, CVPRW 2024 | 5 keypoint sets, DLT+RANSAC, multi-subset voting | Tính homography, chiếu pixel → tọa độ sân (m) |
| OSNet_x1.0 | Person Re-ID | Zhou et al., ICCV 2019 | Lite 3×3, Omni-scale residual block, UAG | Khôi phục tracker ID khi cầu thủ mất khỏi frame |

Các lý thuyết và phương pháp trong chương này có mối quan hệ phụ thuộc chặt chẽ trong pipeline: YOLO11 cung cấp detection cho ByteTrack; ByteTrack quản lý ID để OSNet có gallery so sánh; NBJW chuyển đổi tọa độ tracking sang không gian thực tế để tính thống kê; EMA làm mượt tọa độ giữa các lần hiệu chỉnh. Chương tiếp theo sẽ trình bày cách các thành phần này được tích hợp, thiết kế và triển khai thành hệ thống hoàn chỉnh.

# HGNN-Attention-Recommender — Project Roadmap

## 1. Tổng quan dự án

**Tên đề tài:** Nghiên cứu và phát triển hệ thống gợi ý đa miền sử dụng Graph Neural Network kết hợp cơ chế Attention.

**Tên tiếng Anh:** Multi-Domain Recommendation Using Heterogeneous Graph Neural Networks with Attention Mechanisms.

### 1.1. Mục tiêu

Xây dựng và đánh giá hệ thống gợi ý trên hai miền dữ liệu:

- **MovieLens Latest Small:** Gợi ý phim cho người dùng.
- **JTH (Job Tracking History):** Gợi ý công việc phù hợp cho ứng viên.

Nghiên cứu tập trung vào việc sử dụng Heterogeneous Graph Neural Network (HGNN) kết hợp Attention để khai thác các loại thực thể và quan hệ khác nhau trong dữ liệu.

Hệ thống cần cho phép so sánh mô hình đề xuất với các phương pháp truyền thống, Machine Learning và Graph Neural Network.

### 1.2. Công nghệ sử dụng

| Thành phần          | Công nghệ           |
| --------------------- | --------------------- |
| Ngôn ngữ            | Python                |
| Deep Learning         | PyTorch               |
| Graph Neural Network  | PyTorch Geometric     |
| Machine Learning      | Scikit-learn, XGBoost |
| Xử lý dữ liệu     | Pandas, NumPy         |
| Trực quan hóa       | Matplotlib, Seaborn   |
| Trực quan hóa graph | NetworkX, PyVis       |
| Huấn luyện          | Google Colab GPU      |
| Lưu trữ             | Google Drive          |
| Backend               | FastAPI               |
| Frontend              | Streamlit             |
| Quản lý mã nguồn  | GitHub                |
| Kiểm thử            | Pytest                |

### 1.3. Các mô hình thực nghiệm

**Nhóm mô hình chính:**

1. Popularity-Based Recommendation.
2. Matrix Factorization với BPR.
3. XGBoost.
4. LightGCN.
5. HGNN không Attention.
6. HGNN + Attention (mô hình đề xuất).

**Baseline bổ sung (nếu đủ thời gian):**

- Random Recommendation.
- Content-Based Recommendation.

### 1.4. Chỉ số đánh giá

**Chất lượng gợi ý:**

- Precision@10.
- Recall@10.
- NDCG@10.

**Hiệu năng hệ thống:**

- Training Time.
- Inference Latency.
- GPU Memory Usage.
- Model Size.
- Trainable Parameters (đối với mô hình neural).

**Phân tích bổ sung:**

- Cold-start Performance.
- Sparsity Analysis.
- Ablation Study.
- Hyperparameter Sensitivity.
- Kết quả theo nhiều random seeds.

---

# 2. Lộ trình triển khai

## Giai đoạn 1 — Khởi tạo dự án và thiết lập môi trường

**Trạng thái:** Hoàn thành.

### Mục tiêu

Thiết lập môi trường phát triển thống nhất giữa máy local, GitHub, Google Colab và Google Drive.

### Công việc

- [X] Khởi tạo repository `HGNN-Attention-Recommender`.
- [X] Thiết lập Git và GitHub Private Repository.
- [X] Thiết lập môi trường Python và virtual environment.
- [X] Cài đặt PyTorch và các thư viện cần thiết.
- [X] Thiết lập PyTorch Geometric.
- [X] Cài đặt Pandas, NumPy, Scikit-learn.
- [X] Bổ sung XGBoost, NetworkX và PyVis.
- [X] Thiết lập Google Colab.
- [X] Kiểm tra khả năng sử dụng GPU trên Colab.
- [X] Kết nối Google Drive.
- [X] Tổ chức cấu trúc thư mục dự án.
- [X] Tạo các file cấu hình MovieLens và JTH.
- [X] Cấu hình đường dẫn dữ liệu, checkpoint và results.
- [X] Thiết lập `.gitignore`.
- [X] Kiểm tra môi trường local và Colab.

### Kết quả đầu ra

- Repository GitHub hoạt động.
- Môi trường Python sẵn sàng.
- Google Colab kết nối GPU và Drive.
- Cấu trúc dự án thống nhất.
- Các thư viện cần thiết đã được cài đặt.

### Điều kiện hoàn thành

- [X] Chạy thành công script kiểm tra môi trường.
- [X] Import được các thư viện chính.
- [X] Đọc được cấu hình MovieLens và JTH.
- [X] Đồng bộ mã nguồn lên GitHub.

---

## Giai đoạn 2 — Exploratory Data Analysis (EDA)

**Trạng thái:** Hoàn thành.

### Mục tiêu

Khảo sát hai dataset, xác định đặc điểm dữ liệu và những vấn đề cần giải quyết trước khi xây dựng mô hình.

### Công việc

#### 2.1. Thu thập dữ liệu

- [X] Tải MovieLens Latest Small.
- [X] Tải JTH (Job Tracking History).
- [X] Kiểm tra cấu trúc file.
- [X] Xác minh schema và kiểu dữ liệu.
- [X] Tổ chức dữ liệu gốc trong `datasets/`.

#### 2.2. Phân tích MovieLens

- [X] Thống kê số lượng users.
- [X] Thống kê số lượng movies.
- [X] Thống kê số lượng ratings.
- [X] Kiểm tra missing values.
- [X] Kiểm tra duplicate records.
- [X] Phân tích rating distribution.
- [X] Phân tích interactions per user.
- [X] Phân tích interactions per movie.
- [X] Phân tích movie genres.
- [X] Tính sparsity.
- [X] Phân tích cold-start.
- [X] Xác định positive interactions.

#### 2.3. Phân tích JTH

- [X] Thống kê số lượng candidates.
- [X] Thống kê số lượng jobs.
- [X] Thống kê số lượng application processes.
- [X] Kiểm tra missing values.
- [X] Kiểm tra duplicate candidate-job pairs.
- [X] Phân tích applications per candidate.
- [X] Phân tích applications per job.
- [X] Phân tích candidate skills.
- [X] Phân tích job skills.
- [X] Phân tích recruitment stages.
- [X] Tính sparsity.
- [X] Phân tích cold-start.
- [X] Khảo sát các mốc thời gian.
- [X] Đề xuất interaction timestamp.
- [X] Xác định nguy cơ temporal leakage.
- [X] Đánh giá khả năng sử dụng các đặc trưng `llm_*`.

#### 2.4. Thiết kế nghiên cứu sơ bộ

- [X] Xác định nhiệm vụ Top-K Recommendation.
- [X] Đề xuất các node types.
- [X] Đề xuất các edge types.
- [X] Đề xuất cấu trúc Heterogeneous Graph.
- [X] So sánh đặc điểm hai dataset.
- [X] Xác định các vấn đề cần xử lý ở giai đoạn tiền xử lý.

#### 2.5. Báo cáo

- [X] Sinh các thống kê EDA.
- [X] Tạo biểu đồ phân tích.
- [X] Viết `EDA_REPORT.md`.
- [X] Đối chiếu số liệu với dữ liệu nguồn.

### Kết quả đầu ra

- `results/movielens/eda/`
- `results/jth/eda/`
- `reports/EDA_REPORT.md`
- Các biểu đồ và thống kê EDA.
- Đề xuất graph schema sơ bộ.

### Điều kiện hoàn thành

- [X] Hai dataset được phân tích.
- [X] Có báo cáo EDA.
- [X] Có thống kê sparsity và cold-start.
- [X] Có định hướng xây dựng Heterogeneous Graph.

---

## Giai đoạn 3 — Data Preprocessing & Dataset Splitting

**Trạng thái:** Hoàn thành.

### Mục tiêu

Chuẩn hóa dữ liệu và tạo các tập Train, Validation, Test phù hợp với bài toán Top-K Recommendation.

### Công việc

#### 3.1. Tiền xử lý MovieLens

- [ ] Đọc `ratings.csv`, `movies.csv`, `tags.csv`.
- [ ] Chuẩn hóa User ID và Movie ID.
- [ ] Chuyển timestamp sang datetime.
- [ ] Giữ lại rating gốc.
- [ ] Tạo positive interaction với `rating >= 4.0`.
- [ ] Phân biệt positive, observed non-positive và unobserved interactions.
- [ ] Chuẩn hóa movie genres.
- [ ] Xử lý metadata thiếu.
- [ ] Tạo User ID mapping.
- [ ] Tạo Movie ID mapping.
- [ ] Tạo Genre ID mapping.
- [ ] Lưu dữ liệu đã chuẩn hóa.

#### 3.2. Tiền xử lý JTH

- [ ] Đọc `candidates.csv`, `jobs.csv`, `history.csv`.
- [ ] Chuẩn hóa Candidate ID và Job ID.
- [ ] Kiểm tra tính hợp lệ của `application_id`.
- [ ] Chuẩn hóa các trường ngày.
- [ ] Tạo proxy `interaction_time`.
- [ ] Kiểm tra thứ tự các mốc tuyển dụng.
- [ ] Xử lý các timestamp bất thường.
- [ ] Phân tích 980 bản ghi lặp theo cặp Candidate–Job.
- [ ] Giữ lại lịch sử quá trình tuyển dụng gốc.
- [ ] Tạo bảng tương tác Candidate–Job duy nhất.
- [ ] Chuẩn hóa candidate skills.
- [ ] Chuẩn hóa job skills.
- [ ] Xử lý `_rare_skill_`.
- [ ] Chuẩn hóa job categories và expertise areas.
- [ ] Xử lý missing values.
- [ ] Kiểm tra khả năng sử dụng các thuộc tính `llm_*`.
- [ ] Tạo Candidate ID mapping.
- [ ] Tạo Job ID mapping.
- [ ] Tạo Skill ID mapping.
- [ ] Lưu dữ liệu đã chuẩn hóa.

#### 3.3. Temporal Splitting

- [ ] Xây dựng module temporal splitting dùng chung.
- [ ] Xác định tỷ lệ Train/Validation/Test.
- [ ] Chọn cutoff thời gian.
- [ ] Chia dữ liệu MovieLens.
- [ ] Chia dữ liệu JTH.
- [ ] Kiểm tra thứ tự thời gian giữa các tập.
- [ ] Kiểm tra trùng lặp tương tác giữa các tập.
- [ ] Kiểm tra trường hợp cùng timestamp tại ranh giới.
- [ ] Thống kê số positive interactions mỗi tập.
- [ ] Lưu split metadata.
- [ ] Bảo đảm kết quả tái lập được bằng seed/config.

#### 3.4. Candidate Pool

- [ ] Xác định candidate pool cho MovieLens.
- [ ] Xác định candidate pool cho JTH.
- [ ] Loại tương tác đã quan sát trước cutoff.
- [ ] Kiểm tra thời điểm tồn tại của item/job.
- [ ] Không sử dụng item xuất hiện sau cutoff.
- [ ] Xử lý các trường hợp cold-start.
- [ ] Thống nhất candidate pool cho tất cả mô hình.

#### 3.5. Negative Sampling

- [ ] Xây dựng negative sampling module.
- [ ] Triển khai uniform negative sampling.
- [ ] Loại các positive interactions khỏi tập negative.
- [ ] Không coi observed non-positive rating là unobserved.
- [ ] Chỉ lấy mẫu từ candidate pool hợp lệ.
- [ ] Thiết lập negative sampling ratio.
- [ ] Thiết lập random seed.
- [ ] Tạo sampling protocol cho training.
- [ ] Tạo evaluation candidate sets cố định.
- [ ] Kiểm tra negative sampling leakage.

#### 3.6. Validation

- [ ] Kiểm tra tính hợp lệ của ID mapping.
- [ ] Kiểm tra missing IDs.
- [ ] Kiểm tra duplicate interactions.
- [ ] Kiểm tra temporal leakage.
- [ ] Kiểm tra overlap Train/Validation/Test.
- [ ] Kiểm tra candidate pool.
- [ ] Kiểm tra negative sampling.
- [ ] Viết unit tests cho preprocessing.
- [ ] Viết unit tests cho temporal split.
- [ ] Viết unit tests cho negative sampling.

#### 3.7. Báo cáo

- [ ] Thống kê dữ liệu trước và sau xử lý.
- [ ] Thống kê số interactions mỗi split.
- [ ] Thống kê cold-start sau splitting.
- [ ] Ghi nhận các bản ghi bị loại.
- [ ] Ghi nhận các quy tắc preprocessing.
- [X] Viết `PREPROCESSING_REPORT.md`.

### Kết quả đầu ra

- `src/data/common/`
- `src/data/movielens/preprocess.py`
- `src/data/jth/preprocess.py`
- `datasets/movielens/processed/`
- `datasets/jth/processed/`
- `tests/test_preprocessing.py`
- `reports/PREPROCESSING_REPORT.md`

### Điều kiện hoàn thành

- [ ] Có dữ liệu Train/Validation/Test cho cả hai dataset.
- [ ] Có ID mappings.
- [ ] Có candidate pools hợp lệ.
- [ ] Có negative sampling module.
- [ ] Không phát hiện data leakage trong các kiểm thử đã định nghĩa.
- [ ] Có báo cáo tiền xử lý.

---

## Giai đoạn 4 — Heterogeneous Graph Construction

**Trạng thái:** Chưa thực hiện.

### Mục tiêu

Chuyển đổi dữ liệu đã tiền xử lý thành Heterogeneous Graph để phục vụ huấn luyện GNN.

### Công việc

#### 4.1. Thiết kế MovieLens Graph

- [ ] Xác định node types: User, Movie, Genre.
- [ ] Xác định edge types: User–Movie, Movie–Genre.
- [ ] Thiết kế reverse edges.
- [ ] Xác định node features.
- [ ] Xác định edge attributes.
- [ ] Thiết kế cơ chế xử lý rating và timestamp.
- [ ] Xác định cấu trúc graph cho từng split.

#### 4.2. Thiết kế JTH Graph

- [ ] Xác định node types: Candidate, Job, Skill.
- [ ] Đánh giá bổ sung Category và ExpertiseArea.
- [ ] Xác định edge types.
- [ ] Thiết kế Candidate–Job edges.
- [ ] Thiết kế Candidate–Skill edges.
- [ ] Thiết kế Job–Skill edges.
- [ ] Thiết kế reverse edges.
- [ ] Xác định node features.
- [ ] Xác định edge attributes.
- [ ] Kiểm soát temporal leakage trong graph.

#### 4.3. Graph Construction

- [ ] Xây dựng MovieLens graph builder.
- [ ] Xây dựng JTH graph builder.
- [ ] Sử dụng PyTorch Geometric `HeteroData`.
- [ ] Tạo node feature tensors.
- [ ] Tạo edge index tensors.
- [ ] Tạo edge attribute tensors nếu cần.
- [ ] Tạo graph theo train cutoff.
- [ ] Kiểm tra node/edge mappings.
- [ ] Lưu graph đã xây dựng.

#### 4.4. Graph Analysis

- [ ] Thống kê số node theo type.
- [ ] Thống kê số edge theo relation.
- [ ] Phân tích degree distribution.
- [ ] Phân tích isolated nodes.
- [ ] Phân tích connected components.
- [ ] Kiểm tra sparsity.
- [ ] Kiểm tra tính hợp lệ của edge index.
- [ ] Đánh giá dung lượng graph.

#### 4.5. Graph Visualization

- [ ] Xây dựng graph visualization module.
- [ ] Trực quan hóa MovieLens subgraph.
- [ ] Trực quan hóa JTH subgraph.
- [ ] Phân biệt node types bằng màu sắc.
- [ ] Phân biệt edge types.
- [ ] Hiển thị node labels.
- [ ] Tạo interactive graph bằng PyVis.
- [ ] Xuất hình phục vụ báo cáo.

### Kết quả đầu ra

- `src/graph/builders/`
- `src/graph/visualization/`
- MovieLens `HeteroData`.
- JTH `HeteroData`.
- Graph statistics.
- Graph visualizations.
- `reports/GRAPH_CONSTRUCTION_REPORT.md`

### Điều kiện hoàn thành

- [ ] Xây dựng thành công graph cho hai dataset.
- [ ] Graph tương thích PyTorch Geometric.
- [ ] Không chứa tương tác validation/test trong training graph.
- [ ] Có hình trực quan hóa graph.
- [ ] Có báo cáo cấu trúc graph.

---

## Giai đoạn 5 — Baseline Models

**Trạng thái:** Chưa thực hiện.

### Mục tiêu

Xây dựng các mô hình đối chứng để đánh giá hiệu quả của HGNN + Attention.

### Công việc

#### 5.1. Popularity-Based Recommendation

- [ ] Tính độ phổ biến item từ training data.
- [ ] Xây dựng Top-K recommendation.
- [ ] Loại item đã tương tác.
- [ ] Đánh giá trên MovieLens.
- [ ] Đánh giá trên JTH.

#### 5.2. Matrix Factorization với BPR

- [ ] Xây dựng User Embedding.
- [ ] Xây dựng Item Embedding.
- [ ] Triển khai BPR Loss.
- [ ] Triển khai negative sampling.
- [ ] Xây dựng training loop.
- [ ] Xây dựng Top-K inference.
- [ ] Đánh giá trên hai dataset.

#### 5.3. XGBoost

- [ ] Xác định đặc trưng User–Item.
- [ ] Tạo feature vectors.
- [ ] Tạo positive/negative training samples.
- [ ] Xây dựng XGBoost ranking/classification baseline.
- [ ] Huấn luyện trên MovieLens.
- [ ] Huấn luyện trên JTH.
- [ ] Chuyển prediction scores thành Top-K.
- [ ] Kiểm tra feature leakage.
- [ ] Đánh giá bằng ranking metrics.

#### 5.4. LightGCN

- [ ] Xây dựng bipartite interaction graph.
- [ ] Triển khai LightGCN.
- [ ] Xây dựng embedding propagation.
- [ ] Triển khai BPR Loss.
- [ ] Huấn luyện trên MovieLens.
- [ ] Huấn luyện trên JTH.
- [ ] Xây dựng Top-K inference.
- [ ] Đánh giá ranking metrics.

#### 5.5. Baseline bổ sung

- [ ] Xem xét Random Recommendation.
- [ ] Xem xét Content-Based Recommendation.
- [ ] Xây dựng baseline bổ sung nếu đủ thời gian.

#### 5.6. Chuẩn hóa Benchmark

- [ ] Thiết kế interface chung cho models.
- [ ] Chuẩn hóa input/output.
- [ ] Sử dụng cùng dataset splits.
- [ ] Sử dụng cùng evaluation candidate pools.
- [ ] Chuẩn hóa ranking metrics.
- [ ] Lưu model configurations.
- [ ] Lưu prediction results.
- [ ] Viết unit tests.

### Kết quả đầu ra

- `src/models/popularity.py`
- `src/models/matrix_factorization.py`
- `src/models/xgboost_baseline.py`
- `src/models/lightgcn.py`
- Baseline training scripts.
- Baseline evaluation results.

### Điều kiện hoàn thành

- [ ] Bốn baseline chính chạy được.
- [ ] Có kết quả trên MovieLens.
- [ ] Có kết quả trên JTH.
- [ ] Có bảng so sánh baseline.
- [ ] Có pipeline benchmark thống nhất.

---

## Giai đoạn 6 — HGNN Without Attention

**Trạng thái:** Chưa thực hiện.

### Mục tiêu

Xây dựng mô hình Heterogeneous Graph Neural Network không sử dụng Attention làm đối chứng trực tiếp.

### Công việc

- [ ] Nghiên cứu kiến trúc Heterogeneous GNN.
- [ ] Thiết kế node embeddings.
- [ ] Thiết kế relation-specific message passing.
- [ ] Triển khai HeteroConv.
- [ ] Sử dụng SAGEConv hoặc convolution phù hợp.
- [ ] Thiết kế nhiều GNN layers.
- [ ] Thiết kế hidden dimensions.
- [ ] Thiết kế aggregation giữa các quan hệ.
- [ ] Thiết kế User–Item scoring function.
- [ ] Triển khai recommendation loss.
- [ ] Triển khai forward propagation.
- [ ] Triển khai backward propagation.
- [ ] Xây dựng training loop.
- [ ] Kiểm tra tensor shapes.
- [ ] Kiểm tra gradients.
- [ ] Kiểm tra GPU compatibility.
- [ ] Đánh giá trên MovieLens.
- [ ] Đánh giá trên JTH.
- [ ] Lưu checkpoint.
- [ ] Thống kê trainable parameters.

### Kết quả đầu ra

- `src/models/hgnn.py`
- HGNN training configuration.
- HGNN checkpoints.
- HGNN evaluation results.

### Điều kiện hoàn thành

- [ ] HGNN huấn luyện thành công.
- [ ] Không sử dụng Attention.
- [ ] Có Top-K recommendations.
- [ ] Có kết quả đánh giá.
- [ ] Có thể dùng làm ablation baseline.

---

## Giai đoạn 7 — HGNN with Attention

**Trạng thái:** Chưa thực hiện.

### Mục tiêu

Xây dựng mô hình nghiên cứu chính kết hợp Heterogeneous Graph Neural Network và Attention.

### Công việc

#### 7.1. Thiết kế kiến trúc

- [ ] Nghiên cứu Graph Attention Network.
- [ ] Nghiên cứu Attention trên heterogeneous graph.
- [ ] Xác định vị trí sử dụng Attention.
- [ ] Thiết kế relation-specific attention.
- [ ] Thiết kế multi-head attention.
- [ ] Thiết kế aggregation giữa các quan hệ.
- [ ] Thiết kế node embeddings.
- [ ] Thiết kế scoring function.
- [ ] Thiết kế recommendation loss.

#### 7.2. Triển khai mô hình

- [ ] Triển khai GATConv hoặc attention layer phù hợp.
- [ ] Tích hợp Attention vào HGNN.
- [ ] Xây dựng multi-layer architecture.
- [ ] Triển khai dropout.
- [ ] Triển khai normalization nếu cần.
- [ ] Triển khai forward propagation.
- [ ] Triển khai backward propagation.
- [ ] Xây dựng Top-K inference.
- [ ] Hỗ trợ lưu attention weights nếu phù hợp.

#### 7.3. Kiểm thử

- [ ] Kiểm tra input/output tensors.
- [ ] Kiểm tra gradient flow.
- [ ] Kiểm tra NaN/Inf.
- [ ] Kiểm tra GPU memory.
- [ ] Kiểm tra model parameters.
- [ ] Kiểm tra inference output.
- [ ] So sánh kiến trúc với HGNN không Attention.

#### 7.4. Ablation Preparation

- [ ] Thiết kế HGNN without Attention.
- [ ] Thiết kế HGNN with Attention.
- [ ] Chuẩn hóa embedding dimensions.
- [ ] Chuẩn hóa training configurations.
- [ ] Chuẩn hóa evaluation protocol.
- [ ] Thiết kế so sánh attention heads.
- [ ] Thiết kế so sánh số GNN layers.

### Kết quả đầu ra

- `src/models/hgnn_attention.py`
- Attention model configuration.
- HGNN-Attention checkpoints.
- Model architecture documentation.

### Điều kiện hoàn thành

- [ ] Mô hình HGNN + Attention chạy thành công.
- [ ] Có thể huấn luyện trên hai dataset.
- [ ] Có thể tạo Top-K recommendations.
- [ ] Có cấu hình phục vụ ablation study.
- [ ] Có thể so sánh trực tiếp với HGNN không Attention.

---

## Giai đoạn 8 — Training & Hyperparameter Optimization

**Trạng thái:** Chưa thực hiện.

### Mục tiêu

Huấn luyện các mô hình trên Google Colab và tối ưu hyperparameters.

### Công việc

#### 8.1. Thiết lập Training Pipeline

- [ ] Hoàn thiện training scripts.
- [ ] Thiết lập Google Colab notebooks.
- [ ] Kết nối Google Drive.
- [ ] Tải processed datasets.
- [ ] Kiểm tra GPU availability.
- [ ] Thiết lập random seeds.
- [ ] Thiết lập experiment configurations.
- [ ] Thiết lập logging.
- [ ] Thiết lập checkpoint management.

#### 8.2. Huấn luyện MovieLens

- [ ] Huấn luyện Popularity baseline.
- [ ] Huấn luyện MF-BPR.
- [ ] Huấn luyện XGBoost.
- [ ] Huấn luyện LightGCN.
- [ ] Huấn luyện HGNN.
- [ ] Huấn luyện HGNN + Attention.
- [ ] Lưu validation results.
- [ ] Lưu best checkpoints.

#### 8.3. Huấn luyện JTH

- [ ] Huấn luyện Popularity baseline.
- [ ] Huấn luyện MF-BPR.
- [ ] Huấn luyện XGBoost.
- [ ] Huấn luyện LightGCN.
- [ ] Huấn luyện HGNN.
- [ ] Huấn luyện HGNN + Attention.
- [ ] Lưu validation results.
- [ ] Lưu best checkpoints.

#### 8.4. Hyperparameter Optimization

- [ ] Thử nghiệm learning rate.
- [ ] Thử nghiệm embedding dimension.
- [ ] Thử nghiệm hidden dimension.
- [ ] Thử nghiệm số GNN layers.
- [ ] Thử nghiệm attention heads.
- [ ] Thử nghiệm dropout.
- [ ] Thử nghiệm batch size.
- [ ] Thử nghiệm weight decay.
- [ ] Thử nghiệm negative sampling ratio.
- [ ] Thiết lập early stopping.
- [ ] Lựa chọn cấu hình dựa trên validation set.

#### 8.5. Experiment Tracking

- [ ] Ghi training loss.
- [ ] Ghi validation metrics.
- [ ] Ghi training time.
- [ ] Ghi GPU memory usage.
- [ ] Lưu hyperparameters.
- [ ] Lưu model checkpoints.
- [ ] Lưu experiment logs.
- [ ] Hỗ trợ resume training.
- [ ] Kiểm tra reproducibility.

### Kết quả đầu ra

- Trained model checkpoints.
- Training logs.
- Validation results.
- Best hyperparameter configurations.
- Training notebooks.

### Điều kiện hoàn thành

- [ ] Các mô hình chính được huấn luyện trên cả hai dataset.
- [ ] Có best checkpoints.
- [ ] Có training curves.
- [ ] Có validation metrics.
- [ ] Có thể tái lập thí nghiệm.

---

## Giai đoạn 9 — Model Evaluation & Experimental Analysis

**Trạng thái:** Chưa thực hiện.

### Mục tiêu

Đánh giá chất lượng và hiệu năng của các mô hình, xác định vai trò của Attention.

### Công việc

#### 9.1. Ranking Evaluation

- [ ] Triển khai Precision@10.
- [ ] Triển khai Recall@10.
- [ ] Triển khai NDCG@10.
- [ ] Đánh giá trên test set.
- [ ] Sử dụng cùng evaluation candidate pools.
- [ ] Tính macro-average theo user/candidate.
- [ ] Lưu kết quả từng mô hình.

#### 9.2. Model Comparison

- [ ] So sánh Popularity.
- [ ] So sánh MF-BPR.
- [ ] So sánh XGBoost.
- [ ] So sánh LightGCN.
- [ ] So sánh HGNN.
- [ ] So sánh HGNN + Attention.
- [ ] Tạo bảng kết quả MovieLens.
- [ ] Tạo bảng kết quả JTH.
- [ ] Phân tích mô hình tốt nhất theo từng dataset.

#### 9.3. Ablation Study

- [ ] So sánh HGNN có/không Attention.
- [ ] Đánh giá ảnh hưởng của attention heads.
- [ ] Đánh giá ảnh hưởng của GNN layers.
- [ ] Đánh giá ảnh hưởng của embedding dimension.
- [ ] Đánh giá ảnh hưởng của node features.
- [ ] Đánh giá ảnh hưởng của edge types.
- [ ] Đánh giá ảnh hưởng của `llm_*` features nếu hợp lệ.
- [ ] Phân tích đóng góp của từng thành phần.

#### 9.4. Performance Benchmark

- [ ] Đo training time.
- [ ] Đo inference latency.
- [ ] Đo GPU memory usage.
- [ ] Đo model size.
- [ ] Đếm trainable parameters.
- [ ] Đánh giá thời gian tạo Top-K.
- [ ] So sánh hiệu năng giữa các mô hình.

#### 9.5. Robustness Analysis

- [ ] Đánh giá theo nhiều random seeds.
- [ ] Tính mean và standard deviation.
- [ ] Phân tích cold-start.
- [ ] Phân tích sparse users/items.
- [ ] Phân tích độ ổn định mô hình.
- [ ] Phân tích các trường hợp dự đoán thất bại.
- [ ] Đánh giá giới hạn của dữ liệu JTH.

#### 9.6. Visualization

- [ ] Vẽ training loss curves.
- [ ] Vẽ Precision@10 comparison.
- [ ] Vẽ Recall@10 comparison.
- [ ] Vẽ NDCG@10 comparison.
- [ ] Vẽ training time comparison.
- [ ] Vẽ inference latency comparison.
- [ ] Vẽ parameter count comparison.
- [ ] Vẽ ablation results.
- [ ] Xuất hình phục vụ báo cáo.

### Kết quả đầu ra

- `results/movielens/evaluation/`
- `results/jth/evaluation/`
- Model comparison tables.
- Evaluation charts.
- Ablation results.
- `reports/EVALUATION_REPORT.md`

### Điều kiện hoàn thành

- [ ] Có kết quả test cho sáu mô hình chính.
- [ ] Có so sánh MovieLens và JTH.
- [ ] Có ablation HGNN vs HGNN + Attention.
- [ ] Có đánh giá thời gian và tài nguyên.
- [ ] Có biểu đồ và nhận xét kết quả.

---

## Giai đoạn 10 — Backend API Development

**Trạng thái:** Chưa thực hiện.

### Mục tiêu

Xây dựng Backend API để cung cấp khả năng gợi ý và truy xuất kết quả thực nghiệm.

### Công việc

#### 10.1. Thiết lập Backend

- [ ] Khởi tạo FastAPI application.
- [ ] Thiết lập project configuration.
- [ ] Thiết lập API routers.
- [ ] Thiết lập request/response schemas.
- [ ] Thiết lập error handling.
- [ ] Thiết lập logging.

#### 10.2. Model Serving

- [ ] Xây dựng model loader.
- [ ] Tải trained checkpoints.
- [ ] Tải ID mappings.
- [ ] Tải metadata.
- [ ] Thiết lập inference pipeline.
- [ ] Hỗ trợ lựa chọn dataset.
- [ ] Hỗ trợ lựa chọn model.

#### 10.3. Recommendation API

- [ ] API lấy danh sách datasets.
- [ ] API lấy danh sách models.
- [ ] API lấy danh sách users/candidates.
- [ ] API lấy danh sách movies/jobs.
- [ ] API Top-K Movie Recommendation.
- [ ] API Top-K Job Recommendation.
- [ ] API lấy thông tin item.
- [ ] API lấy model metrics.
- [ ] API lấy graph statistics.

#### 10.4. Testing

- [ ] Kiểm tra API input validation.
- [ ] Kiểm tra model loading.
- [ ] Kiểm tra recommendation outputs.
- [ ] Kiểm tra trường hợp user/item không tồn tại.
- [ ] Kiểm tra response time.
- [ ] Viết API tests.
- [ ] Tạo API documentation.

### Kết quả đầu ra

- `src/api/`
- FastAPI application.
- Recommendation endpoints.
- API documentation.
- API tests.

### Điều kiện hoàn thành

- [ ] API hoạt động.
- [ ] Tải được checkpoints.
- [ ] Trả về Top-K recommendations.
- [ ] Trả về model metrics.
- [ ] Có API documentation.

---

## Giai đoạn 11 — Frontend & Visualization Dashboard

**Trạng thái:** Chưa thực hiện.

### Mục tiêu

Xây dựng giao diện demo cho hệ thống gợi ý và trực quan hóa kết quả nghiên cứu.

### Công việc

#### 11.1. Thiết lập Frontend

- [ ] Khởi tạo Streamlit application.
- [ ] Thiết kế layout.
- [ ] Thiết kế navigation.
- [ ] Kết nối FastAPI.
- [ ] Xử lý loading/error states.

#### 11.2. Movie Recommendation

- [ ] Tạo trang Movie Recommendation.
- [ ] Cho phép chọn User.
- [ ] Cho phép chọn Model.
- [ ] Cho phép chọn Top-K.
- [ ] Hiển thị danh sách phim gợi ý.
- [ ] Hiển thị thông tin phim.
- [ ] Hiển thị recommendation scores nếu phù hợp.

#### 11.3. Job Recommendation

- [ ] Tạo trang Job Recommendation.
- [ ] Cho phép chọn Candidate.
- [ ] Cho phép chọn Model.
- [ ] Cho phép chọn Top-K.
- [ ] Hiển thị danh sách công việc gợi ý.
- [ ] Hiển thị job skills.
- [ ] Hiển thị thông tin công việc.

#### 11.4. Graph Visualization

- [ ] Tạo trang Graph Visualization.
- [ ] Hiển thị MovieLens subgraph.
- [ ] Hiển thị JTH subgraph.
- [ ] Phân biệt node types.
- [ ] Phân biệt edge types.
- [ ] Hỗ trợ interactive visualization.

#### 11.5. Model Comparison Dashboard

- [ ] Hiển thị Precision@10.
- [ ] Hiển thị Recall@10.
- [ ] Hiển thị NDCG@10.
- [ ] Hiển thị training time.
- [ ] Hiển thị inference latency.
- [ ] Hiển thị model parameters.
- [ ] Hiển thị training curves.
- [ ] Hiển thị ablation results.
- [ ] Hỗ trợ lựa chọn dataset để so sánh.

#### 11.6. Integration Testing

- [ ] Kiểm tra kết nối Frontend–Backend.
- [ ] Kiểm tra recommendation workflow.
- [ ] Kiểm tra model switching.
- [ ] Kiểm tra graph visualization.
- [ ] Kiểm tra dashboard.
- [ ] Kiểm tra giao diện trên các kích thước màn hình.

### Kết quả đầu ra

- `app/`
- Streamlit application.
- Movie Recommendation page.
- Job Recommendation page.
- Graph Visualization page.
- Model Comparison Dashboard.

### Điều kiện hoàn thành

- [ ] Giao diện hoạt động.
- [ ] Có thể gợi ý phim.
- [ ] Có thể gợi ý công việc.
- [ ] Có thể so sánh mô hình.
- [ ] Có thể trực quan hóa graph.
- [ ] Frontend kết nối Backend thành công.

---

## Giai đoạn 12 — Testing, Documentation & Final Report

**Trạng thái:** Chưa thực hiện.

### Mục tiêu

Hoàn thiện dự án, kiểm thử toàn bộ hệ thống và chuẩn bị báo cáo nghiên cứu.

### Công việc

#### 12.1. Kiểm thử tổng thể

- [ ] Kiểm thử preprocessing pipeline.
- [ ] Kiểm thử graph construction.
- [ ] Kiểm thử model training.
- [ ] Kiểm thử model evaluation.
- [ ] Kiểm thử inference pipeline.
- [ ] Kiểm thử Backend API.
- [ ] Kiểm thử Frontend.
- [ ] Kiểm thử end-to-end.
- [ ] Kiểm tra data leakage.
- [ ] Kiểm tra reproducibility.

#### 12.2. Hoàn thiện mã nguồn

- [ ] Refactor source code.
- [ ] Loại bỏ code không sử dụng.
- [ ] Chuẩn hóa naming conventions.
- [ ] Hoàn thiện configuration files.
- [ ] Hoàn thiện requirements.
- [ ] Bổ sung comments/docstrings.
- [ ] Kiểm tra `.gitignore`.
- [ ] Kiểm tra repository structure.

#### 12.3. Documentation

- [ ] Hoàn thiện README.
- [ ] Viết hướng dẫn cài đặt.
- [ ] Viết hướng dẫn chuẩn bị datasets.
- [ ] Viết hướng dẫn chạy preprocessing.
- [ ] Viết hướng dẫn xây dựng graph.
- [ ] Viết hướng dẫn training trên Colab.
- [ ] Viết hướng dẫn evaluation.
- [ ] Viết hướng dẫn chạy Backend.
- [ ] Viết hướng dẫn chạy Frontend.
- [ ] Viết hướng dẫn tái lập thí nghiệm.

#### 12.4. Báo cáo nghiên cứu

- [ ] Viết chương giới thiệu đề tài.
- [ ] Viết mục tiêu và phạm vi nghiên cứu.
- [ ] Trình bày cơ sở lý thuyết Recommendation Systems.
- [ ] Trình bày Collaborative Filtering.
- [ ] Trình bày Graph Neural Networks.
- [ ] Trình bày Heterogeneous Graph Neural Networks.
- [ ] Trình bày Attention Mechanisms.
- [ ] Trình bày Graph Attention Networks.
- [ ] Trình bày Matrix Factorization và BPR.
- [ ] Trình bày LightGCN.
- [ ] Trình bày XGBoost.
- [ ] Mô tả hai datasets.
- [ ] Trình bày EDA.
- [ ] Trình bày preprocessing.
- [ ] Trình bày graph construction.
- [ ] Trình bày kiến trúc HGNN.
- [ ] Trình bày kiến trúc HGNN + Attention.
- [ ] Trình bày training methodology.
- [ ] Trình bày evaluation metrics.
- [ ] Trình bày kết quả thực nghiệm.
- [ ] Trình bày ablation study.
- [ ] Phân tích ưu điểm và hạn chế.
- [ ] Viết kết luận và hướng phát triển.

#### 12.5. Slide và Demo

- [ ] Chuẩn bị slide giới thiệu.
- [ ] Chuẩn bị slide cơ sở lý thuyết.
- [ ] Chuẩn bị slide datasets.
- [ ] Chuẩn bị slide kiến trúc mô hình.
- [ ] Chuẩn bị slide quy trình thực nghiệm.
- [ ] Chuẩn bị slide kết quả.
- [ ] Chuẩn bị slide ablation study.
- [ ] Chuẩn bị slide demo.
- [ ] Chuẩn bị kịch bản demo.
- [ ] Kiểm tra hệ thống trước khi báo cáo.
- [ ] Chuẩn bị trả lời câu hỏi phản biện.

### Kết quả đầu ra

- Source code hoàn chỉnh.
- Bộ kiểm thử.
- README và tài liệu hướng dẫn.
- Báo cáo nghiên cứu.
- Slide thuyết trình.
- Hệ thống demo.

### Điều kiện hoàn thành

- [ ] Pipeline chạy từ dữ liệu đến kết quả gợi ý.
- [ ] Có kết quả thực nghiệm đầy đủ.
- [ ] Có báo cáo và slide.
- [ ] Có demo hoạt động.
- [ ] Repository được hoàn thiện.

---

# 3. Tổng hợp tiến độ

| Giai đoạn | Nội dung                              | Trạng thái |
| ----------- | -------------------------------------- | ------------ |
| 1           | Environment Setup                      | Completed    |
| 2           | Exploratory Data Analysis              | Completed    |
| 3           | Data Preprocessing & Splitting         | Completed    |
| 4           | Heterogeneous Graph Construction       | Not Started  |
| 5           | Baseline Models                        | Not Started  |
| 6           | HGNN Without Attention                 | Not Started  |
| 7           | HGNN With Attention                    | Not Started  |
| 8           | Training & Hyperparameter Optimization | Not Started  |
| 9           | Evaluation & Experimental Analysis     | Not Started  |
| 10          | Backend API Development                | Not Started  |
| 11          | Frontend & Visualization Dashboard     | Not Started  |
| 12          | Testing, Documentation & Final Report  | Not Started  |

---

# 4. Các nguyên tắc bắt buộc

## 4.1. Data Leakage Prevention

- Không sử dụng tương tác validation/test để tạo training graph.
- Không sử dụng dữ liệu tuyển dụng tương lai để tạo feature.
- Không sử dụng outcome tương lai làm input.
- Không sử dụng toàn bộ lịch sử để tính feature trước temporal split.
- Kiểm tra timestamp của các feature và edge.
- Với JTH, phải kiểm soát thời điểm quan sát của các thuộc tính `llm_*`.

## 4.2. Fair Model Comparison

- Sử dụng cùng train/validation/test splits.
- Sử dụng cùng evaluation candidate pools.
- Sử dụng cùng định nghĩa positive interaction.
- Sử dụng cùng ranking metrics.
- Thống nhất quy tắc loại item đã tương tác.
- Báo cáo khác biệt về tài nguyên và số tham số.
- Không sử dụng test set để tối ưu hyperparameters.

## 4.3. Reproducibility

- Cố định random seeds.
- Lưu experiment configurations.
- Lưu dataset split metadata.
- Lưu model checkpoints.
- Lưu training logs.
- Lưu library versions.
- Lưu kết quả theo từng experiment.
- Có hướng dẫn chạy lại thí nghiệm.

## 4.4. Research Integrity

- Không mặc định HGNN + Attention luôn tốt hơn baseline.
- Không so sánh trực tiếp metric tuyệt đối giữa hai dataset khi evaluation protocol khác nhau.
- Không xem unobserved interactions là negative feedback được xác nhận.
- Phân biệt dữ liệu gốc và dữ liệu dẫn xuất.
- Báo cáo rõ hạn chế của JTH.
- Báo cáo kết quả thực tế, kể cả khi mô hình chính không đạt kết quả tốt nhất.

---

# 5. Định nghĩa hoàn thành dự án

Dự án được xem là hoàn thành khi đáp ứng các yêu cầu:

- [ ] Hai dataset được tiền xử lý đầy đủ.
- [ ] Hai Heterogeneous Graph được xây dựng.
- [ ] Sáu mô hình chính được triển khai.
- [ ] Có kết quả thực nghiệm trên cả MovieLens và JTH.
- [ ] Có Precision@10, Recall@10, NDCG@10.
- [ ] Có ablation HGNN và HGNN + Attention.
- [ ] Có đánh giá training time, inference latency và tài nguyên.
- [ ] Có giao diện gợi ý phim và công việc.
- [ ] Có dashboard so sánh mô hình.
- [ ] Có báo cáo nghiên cứu hoàn chỉnh.
- [ ] Có slide và demo.
- [ ] Có thể tái lập các kết quả thực nghiệm.

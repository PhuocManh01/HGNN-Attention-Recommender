# BÁO CÁO ĐỒ ÁN

## Hệ thống gợi ý phim sử dụng đồ thị dị thể và Graph Neural Network

**Tên dự án:** HGNN-Attention-Recommender  
**Dataset:** MovieLens 1M  
**Ngày cập nhật kết quả:** 23/11/2026  
**Nền tảng huấn luyện:** NVIDIA GeForce RTX 2050, CUDA 12.8  
**Framework:** PyTorch 2.11.0 + CUDA, PyTorch Geometric 2.8.1

---

## 1. Tóm tắt đồ án

Đồ án xây dựng hệ thống gợi ý phim cá nhân hóa dựa trên lịch sử đánh giá của người dùng. Dữ liệu được biểu diễn dưới dạng **đồ thị dị thể** gồm ba loại node:

- User
- Movie
- Genre

Hệ thống thử nghiệm và so sánh năm mô hình:

1. Popularity
2. MF-BPR
3. LightGCN
4. Hetero-GCN/RGCN không attention
5. HGT Attention

Mục tiêu chính là đánh giá liệu thông tin đồ thị dị thể và cơ chế attention có cải thiện chất lượng gợi ý so với các baseline truyền thống hay không.

---

## 2. Mục tiêu

- Xây dựng pipeline xử lý dữ liệu MovieLens 1M.
- Chia dữ liệu theo thời gian cho từng user.
- Xây dựng heterogeneous graph User–Movie–Genre.
- Huấn luyện các mô hình ranking recommendation.
- So sánh mô hình GNN không attention với mô hình GNN có attention.
- Đánh giá bằng full-ranking trên Validation và Test.
- Đảm bảo Test set không được dùng để tuning hoặc early stopping.

---

## 3. Dữ liệu và tiền xử lý

### 3.1. Dataset

Dataset sử dụng là MovieLens 1M:

| Thành phần | Số lượng |
|---|---:|
| Users | 6.040 |
| Movies | 3.883 |
| Genres | 18 |
| Interactions | 1.000.209 |
| Positive threshold | Rating >= 4.0 |

### 3.2. Temporal split

Dữ liệu được chia theo thứ tự thời gian của từng user:

| Split | Số interactions | Tỷ lệ xấp xỉ |
|---|---:|---:|
| Train | 799.086 | 79,89% |
| Validation | 100.613 | 10,06% |
| Test | 100.510 | 10,05% |

Validation dùng để chọn checkpoint/số epoch. Sau khi khóa cấu hình, model được train lại trên `Train + Validation` và chỉ đánh giá một lần trên `Test`.

---

## 4. Kiến trúc đồ thị

Đồ thị gồm các quan hệ:

| Quan hệ | Ý nghĩa |
|---|---|
| User → Movie | User đánh giá Movie |
| Movie → User | Reverse relation của User → Movie |
| Movie → Genre | Movie thuộc Genre |
| Genre → Movie | Reverse relation của Movie → Genre |

Graph train/final không chứa các interaction thuộc Test. Khi đánh giá Test, lịch sử được sử dụng để loại các phim đã xuất hiện trong `Train + Validation` khỏi danh sách đề xuất.

---

## 5. Các mô hình

### 5.1. Popularity

Xếp hạng phim dựa trên số lượng positive interaction trong `Train + Validation`. Đây là baseline đơn giản, không có tham số học.

### 5.2. MF-BPR

Matrix Factorization học embedding cho User và Movie. Mục tiêu BPR khuyến khích score của phim positive cao hơn phim negative.

### 5.3. LightGCN

LightGCN truyền embedding trên đồ thị User–Movie, không sử dụng feature transformation phức tạp. Đây là baseline graph recommendation phổ biến.

### 5.4. Hetero-GCN/RGCN không attention

Model sử dụng `HeteroConv` kết hợp nhiều `GraphConv` theo từng loại relation. Model này giữ thông tin quan hệ dị thể nhưng **không sử dụng attention**.

File train RTX: [`train_gnn_rtx.py`](../scripts/train_gnn_rtx.py)

### 5.5. HGT Attention

Model sử dụng `HGTConv` với multi-head attention trên heterogeneous graph. Cấu hình chính:

| Tham số | Giá trị |
|---|---:|
| Hidden dimension | 64 |
| Số layer | 2 |
| Số attention heads | 4 |
| Dropout | 0,1 |

---

## 6. Môi trường thực nghiệm

| Thành phần | Giá trị |
|---|---|
| GPU | NVIDIA GeForce RTX 2050 |
| CUDA | 12.8 |
| PyTorch | 2.11.0+cu128 |
| PyTorch Geometric | 2.8.1 |
| Device | `cuda:0` |
| Seed | 42 |
| Evaluation | Full-ranking@10 |
| Eligible Test users | 5.767 |

Script đánh giá toàn bộ model: [`final_test_all_models_rtx.py`](../scripts/final_test_all_models_rtx.py)

---

## 7. Kết quả Validation

Validation được dùng để lựa chọn số epoch trước khi chạy Test.

| Model | Precision@10 | Recall@10 | NDCG@10 | Best epoch |
|---|---:|---:|---:|---:|
| Popularity | 0,045274 | 0,048219 | 0,057826 | - |
| MF-BPR | 0,046970 | 0,053420 | **0,060125** | 2 |
| LightGCN | 0,045014 | 0,047367 | 0,057338 | 24 |
| Hetero-GCN không attention | **0,048096** | **0,055927** | **0,061506** | 33 |
| HGT Attention | 0,045187 | 0,049514 | 0,057986 | 45 |

Trên Validation, Hetero-GCN không attention đạt kết quả cao nhất trong nhóm model học được.

---

## 8. Kết quả Final Test

Final Test được thực hiện sau khi:

1. Khóa split và metric.
2. Khóa số epoch dựa trên Validation.
3. Train lại trên `Train + Validation`.
4. Xây dựng graph từ `Train + Validation`.
5. Đánh giá một lần trên Test.

| Model | Precision@10 | Recall@10 | NDCG@10 | Epochs |
|---|---:|---:|---:|---:|
| Popularity | 0,045708 | **0,051387** | **0,059509** | 0 |
| MF-BPR | **0,045864** | 0,050673 | 0,059243 | 2 |
| LightGCN | 0,045344 | 0,051285 | 0,059088 | 24 |
| Hetero-GCN không attention | 0,039501 | 0,042797 | 0,048222 | 33 |
| HGT Attention | 0,039622 | 0,044900 | 0,049729 | 45 |

Dữ liệu gốc của bảng nằm trong [`final_test_results.csv`](../experiments/final_test_rtx/final_test_results.csv) và [`final_test_report.json`](../experiments/final_test_rtx/final_test_report.json).

---

## 9. Phân tích kết quả

### 9.1. Model tốt nhất trên Test

- **NDCG@10 cao nhất:** Popularity, `0,059509`.
- **Recall@10 cao nhất:** Popularity, `0,051387`.
- **Precision@10 cao nhất:** MF-BPR, `0,045864`.

MF-BPR chỉ nhỉnh hơn Popularity rất ít về Precision, trong khi Popularity tốt hơn về Recall và NDCG.

### 9.2. So sánh GNN không attention và HGT Attention

| Metric | Hetero-GCN | HGT Attention | Thay đổi của HGT |
|---|---:|---:|---:|
| Precision@10 | 0,039501 | 0,039622 | +0,31% |
| Recall@10 | 0,042797 | 0,044900 | +4,91% |
| NDCG@10 | 0,048222 | 0,049729 | +3,13% |

Trên Test, HGT Attention tốt hơn Hetero-GCN không attention ở cả ba metric. Tuy nhiên, cả hai vẫn thấp hơn Popularity, MF-BPR và LightGCN.

### 9.3. Chênh lệch giữa Validation và Test

Hetero-GCN không attention đạt kết quả rất tốt trên Validation nhưng giảm rõ rệt trên Test. Điều này cho thấy khả năng tổng quát hóa của model chưa ổn định. HGT Attention có kết quả Validation thấp hơn Hetero-GCN nhưng giảm ít hơn tương đối khi chuyển sang Test.

Không nên kết luận rằng attention luôn tốt hơn hoặc luôn kém hơn. Trong thí nghiệm này, attention giúp HGT vượt model GNN không attention trên Test, nhưng chưa đủ để vượt các baseline đơn giản.

---

## 10. Ý nghĩa các metric

- **Precision@10 = 0,045864:** trung bình khoảng 0,46 phim trong mỗi nhóm 10 đề xuất là positive.
- **Recall@10 = 0,051387:** model tìm được khoảng 5,14% các positive movie trong tập Test của user.
- **NDCG@10 = 0,059509:** đo chất lượng xếp hạng, trong đó phim đúng ở vị trí cao được tính trọng số lớn hơn.

Các giá trị này cần được so sánh tương đối giữa các model trên cùng protocol; không nên diễn giải như độ chính xác rating.

---

## 11. Kết luận

Kết quả Final Test cho thấy:

1. Baseline Popularity vẫn đạt hiệu quả rất mạnh trên MovieLens 1M.
2. MF-BPR có Precision@10 cao nhất nhưng chỉ nhỉnh hơn Popularity một khoảng nhỏ.
3. LightGCN không cải thiện được so với Popularity trong thí nghiệm này.
4. HGT Attention tốt hơn Hetero-GCN không attention trên Test.
5. Hai model heterogeneous GNN chưa vượt được các baseline về chất lượng ranking.

Vì vậy, nếu mục tiêu là triển khai DSS có chất lượng gợi ý tốt nhất theo kết quả hiện tại, nên ưu tiên:

- **Popularity** cho Recall/NDCG và tính đơn giản.
- **MF-BPR** cho Precision.

HGT Attention vẫn có giá trị nghiên cứu vì thể hiện cách sử dụng attention trên heterogeneous graph, nhưng cần tuning thêm hoặc chạy nhiều seed trước khi khẳng định ưu thế của kiến trúc.

---

## 12. Hạn chế và hướng phát triển

- Kết quả hiện tại sử dụng một seed duy nhất: `42`.
- Chưa có mean ± std trên nhiều seed.
- HGT và Hetero-GCN chưa được tuning độc lập đầy đủ.
- Test set chỉ được đánh giá một lần trong thí nghiệm này; không dùng Test để tuning.
- Cần kiểm tra thêm các nhóm user theo số lượng lịch sử và mức độ hoạt động.
- Có thể thử ablation User–Movie so với User–Movie–Genre.
- Có thể tối ưu negative sampling, hidden dimension, số layer và learning rate.

---

## 13. Artifact chính

| Artifact | Mô tả |
|---|---|
| [`train_gnn_rtx.py`](../scripts/train_gnn_rtx.py) | Train GNN không attention trên RTX |
| [`train_hgnn_rtx.py`](../scripts/train_hgnn_rtx.py) | Train HGT Attention trên RTX |
| [`final_test_all_models_rtx.py`](../scripts/final_test_all_models_rtx.py) | Retrain và đánh giá 5 model trên Test |
| [`final_test_report.json`](../experiments/final_test_rtx/final_test_report.json) | Báo cáo Test dạng JSON |
| [`final_test_results.csv`](../experiments/final_test_rtx/final_test_results.csv) | Bảng metric Test |
| [`gnn_no_attention_final.pt`](../experiments/final_test_rtx/checkpoints/gnn_no_attention_final.pt) | Checkpoint GNN không attention |
| [`hgt_attention_final.pt`](../experiments/final_test_rtx/checkpoints/hgt_attention_final.pt) | Checkpoint HGT Attention |
| [`mf_bpr_final.pt`](../experiments/final_test_rtx/checkpoints/mf_bpr_final.pt) | Checkpoint MF-BPR |
| [`lightgcn_final.pt`](../experiments/final_test_rtx/checkpoints/lightgcn_final.pt) | Checkpoint LightGCN |


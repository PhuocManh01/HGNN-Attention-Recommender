# PROJECT TECHNICAL NOTES

> **Dự án:** Movie DSS — HGNN + Attention Recommender  
> **Dữ liệu:** MovieLens Latest Small  
> **Mục đích:** Danh sách các quyết định, rủi ro, biện pháp phòng tránh và kiểm thử kỹ thuật xuyên suốt dự án.  
> **Vị trí đề xuất trong repository:** `docs/PROJECT_TECHNICAL_NOTES.md`

## 1. Phạm vi và các quyết định đã chốt

- **Bài toán:** Gợi ý Top-10 phim cho người dùng đã có lịch sử tương tác (warm-start), sử dụng đồ thị dị thể gồm `user`, `movie`, `genre`.
- **Mô hình:** Popularity, MF-BPR, LightGCN, HGNN without Attention, HGNN + Attention.
- **Mô hình chính:** `HeteroConv` kết hợp `GATv2Conv`; Attention được sử dụng thực sự trong message passing.
- **Đánh giá:** `Precision@10`, `Recall@10`, `NDCG@10`; cùng tập user, cùng candidate pool và cùng giao thức.
- **Ablation:** (1) Có/không Attention; (2) Có/không thông tin Genre.
- **Huấn luyện:** Google Colab; phát triển bằng Windows 11 và VS Code.
- **Demo:** Streamlit; chọn user có sẵn, xem lịch sử, so sánh Top-10 của 5 mô hình, xem metric và trực quan hóa Attention weights thực tế của HGNN + Attention.
- **Ngoài phạm vi:** Đăng nhập, thêm rating trực tiếp, genre filtering, backend/database riêng, nghiên cứu chuyên sâu cold-start và suy diễn nhân quả từ Attention weights.

## 2. Ma trận rủi ro và ưu tiên

| Mã | Vấn đề | Ưu tiên | Giai đoạn cần kiểm soát |
|---|---|---|---|
| R01 | Temporal split và data leakage | Rất cao | Phase 1–3, 6 |
| R02 | Negative sampling | Rất cao | Phase 2, 4–5 |
| R03 | Node features và ID mapping | Rất cao | Phase 2–5, 7 |
| R04 | Thiết kế đồ thị dị thể và kiến trúc Attention | Rất cao | Phase 3, 5 |
| R05 | Full-ranking evaluation | Rất cao | Phase 4–7 |
| R06 | So sánh công bằng 5 mô hình | Cao | Phase 4–6 |
| R07 | Cold-start và dữ liệu thưa | Cao | Phase 2, 6–7 |
| R08 | Overfitting và lựa chọn checkpoint | Cao | Phase 4–6 |
| R09 | Trực quan hóa Attention đúng ngữ nghĩa | Trung bình | Phase 5, 7 |
| R10 | Tái lập kết quả và quản lý artifact | Cao | Mọi phase |

## 3. R01 — Temporal split và chống data leakage

### Quy tắc

1. Sắp xếp **tương tác User–Movie** theo `timestamp` và chia toàn cục theo tỷ lệ mục tiêu **80% Train / 10% Validation / 10% Test**. Không chia node user/movie thành ba nhóm riêng.
2. Giữ ID mapping thống nhất giữa các tập. User hoặc movie có thể không có cạnh trong Train; đó là vấn đề cold-start, không phải lý do để chuyển cạnh qua cutoff.
3. Phân biệt **tương tác được quan sát** với **positive label**. Rating `>= 4.0` là positive; rating thấp hơn vẫn là tương tác đã quan sát.
4. Chỉ tạo reverse edges từ các cạnh hợp lệ trong snapshot tương ứng.
5. Không tính popularity, lịch sử user hoặc feature tổng hợp bằng dữ liệu tương lai.

| Giai đoạn | Cạnh tương tác cho message passing | Nhãn mục tiêu |
|---|---|---|
| Train | Train | Positive Train |
| Validation | Train | Positive Validation |
| Huấn luyện cuối | Train + Validation | Positive Train + Validation |
| Test | Train + Validation | Positive Test |

**Chú ý:** Trong huấn luyện BPR, cạnh positive Train có thể đồng thời là quan hệ của đồ thị Train; cần xác định rõ cách xây dựng embedding và loss. Không đưa cạnh Validation/Test vào graph của giai đoạn trước. Nếu đánh giá theo từng sự kiện thay vì theo cửa sổ, phải thiết kế snapshot theo thời điểm tương ứng, không sử dụng rolling history một cách mơ hồ.

**Movie–Genre metadata:** Có thể dùng metadata thể loại nếu được xem là thông tin sẵn có tại cutoff. MovieLens Latest Small không cung cấp đầy đủ thời điểm metadata trở nên khả dụng; phải ghi rõ giả định và giới hạn này. Không dùng nhãn rating tương lai để tạo feature.

### Kiểm thử bắt buộc

- [ ] Các tập tương tác không trùng nhau; timestamp có thứ tự đúng tại ranh giới.
- [ ] Không có cạnh Validation/Test trong message-passing graph của Validation.
- [ ] Không có cạnh Test trong graph của Test.
- [ ] Reverse edges đúng bằng phép đảo của cạnh gốc được phép.
- [ ] Mọi feature thống kê chỉ dùng lịch sử hợp lệ.
- [ ] Lưu cutoff timestamp, số cạnh, số user/movie và tỷ lệ split thực tế.

## 4. R02 — Negative sampling

### Vấn đề

Phim chưa được user đánh giá là **unknown**, không được khẳng định là phim user không thích. Rating thấp là **observed non-positive**, khác với unknown.

### Quy tắc đề xuất

- Positive: rating `>= 4.0`.
- Với BPR, lấy mẫu `j` từ tập movie ứng viên **chưa được user quan sát tại cutoff của giai đoạn huấn luyện**; cặp `(u, i_positive, j_sampled)` chỉ là tín hiệu xếp hạng tương đối.
- Không lấy `j` trùng positive của chính cặp huấn luyện hoặc phim nằm ngoài candidate universe.
- Không dùng Test labels để tạo negative cho Train hoặc Validation. Nếu lọc future positives khi sampling sẽ dùng thông tin tương lai và làm thay đổi phân phối huấn luyện.
- Cố định seed; ghi rõ chiến lược sampling, số negative/positive, phạm vi movie đủ điều kiện và cách xử lý user không có candidate.
- Giữ chính sách rating thấp nhất quán: **đã quan sát thì không được xem là unobserved negative**.

### Kiểm thử

- [ ] Negative không nằm trong observed history tính đến cutoff tương ứng.
- [ ] Negative khác positive và nằm trong tập ứng viên hợp lệ.
- [ ] Sampling tái lập được khi dùng cùng seed và dữ liệu.
- [ ] Không truy cập Validation/Test labels để sampling Train.

## 5. R03 — Node features, embeddings và ID mapping

| Node | Đặc trưng khởi đầu | Ghi chú |
|---|---|---|
| User | Trainable ID embedding | User mới chưa có embedding được học đủ |
| Movie | Trainable ID embedding | Phải giữ thứ tự movie ID nhất quán |
| Genre | Trainable genre embedding | Quan hệ movie–genre cung cấp tín hiệu nội dung |

- Tạo mapping `raw_user_id -> contiguous_index`, `raw_movie_id -> contiguous_index`, `genre_name -> contiguous_index` **một lần** và lưu cùng artifact.
- Không dùng `rating` của Validation/Test làm node feature hoặc edge weight ở thời điểm chưa được phép.
- Cân nhắc cold-start nếu node có trong mapping nhưng không có cạnh Train: embedding ngẫu nhiên chưa mang thông tin hành vi.
- Với **Genre ablation**, xác định rõ cấu hình `with_genre` và `without_genre`; không để thông tin genre lọt vào nhánh `without_genre` qua feature khác.

### Kiểm thử

- [ ] Mapping là duy nhất, liên tục và giống nhau giữa preprocessing, checkpoint, inference, demo.
- [ ] Tensor feature có kích thước đúng cho từng node type.
- [ ] Genre ablation thực sự loại bỏ toàn bộ tín hiệu genre được chỉ định.
- [ ] Không có feature suy ra từ tương tác tương lai.

## 6. R04 — Heterogeneous graph và HGNN + Attention

### Cấu trúc đồ thị

- Node types: `user`, `movie`, `genre`.
- Relations dự kiến: `user --rates--> movie`, `movie --rev_rates--> user`, `movie --has_genre--> genre`, `genre --rev_has_genre--> movie`.
- Chỉ thêm cạnh tương tác đúng snapshot; quan hệ genre tuân thủ giả định metadata ở mục 3.

### Kiến trúc khởi đầu để thử nghiệm

- `HeteroConv` + `GATv2Conv` theo từng relation.
- Embedding dimension thử từ **64**, số tầng **1–2**, attention heads thử từ **2**, dropout thử **0.1–0.3**.
- User–Movie score: dot product; objective chính: BPR ranking loss.
- Đây là **giá trị khởi đầu**, không phải kết quả tuning cuối cùng.
- `HeteroConv` tổng hợp output giữa các relation bằng cơ chế cấu hình (ví dụ sum/mean); **không tự động học relation-level attention**. GATv2 attention chủ yếu ở cấp cạnh/hàng xóm của từng relation.
- Kiểm tra kích thước đầu ra khi `concat=True/False`, self-loop cho quan hệ hai phía khác kiểu node, residual connection và node không nhận message.
- Tránh mô hình quá sâu gây over-smoothing hoặc chi phí huấn luyện không cần thiết.

### Kiểm thử

- [ ] Mỗi edge type có `edge_index` đúng chiều nguồn/đích.
- [ ] Forward pass trả embedding đúng shape và giá trị hữu hạn.
- [ ] Backprop cập nhật được tham số dự kiến.
- [ ] Không đưa nhầm Test edge vào graph.
- [ ] GATv2 trả được attention weights cùng mapping cạnh tương ứng.

## 7. R05 — Full-ranking evaluation

### Giao thức bắt buộc

1. Chọn tập **eligible evaluation users** theo tiêu chí cố định trước khi xem kết quả model.
2. Với mỗi user, lấy tập movie ứng viên hợp lệ và loại **toàn bộ phim đã quan sát trước cutoff**, bao gồm cả rating thấp.
3. Chấm điểm **tất cả** candidate còn lại; chọn Top-10. Không đánh giá chính thức bằng một tập negative ngẫu nhiên nhỏ.
4. Ground truth là **positive interactions trong cửa sổ cần dự đoán**, không đưa chúng vào graph inference.
5. Dùng cùng candidate pool, eligible users và ground truth cho cả 5 mô hình.
6. Chỉ báo cáo ba metric: `Precision@10`, `Recall@10`, `NDCG@10`.

### Công thức

Với `R_u` là tập positive ground truth hợp lệ của user `u` và `Top10_u` là 10 gợi ý:

- `Precision@10 = |Top10_u ∩ R_u| / 10`.
- `Recall@10 = |Top10_u ∩ R_u| / |R_u|` (chỉ tính khi `|R_u| > 0`).
- `DCG@10 = Σ_{r=1}^{10} rel_r / log2(r+1)` với `rel_r ∈ {0,1}`.
- `IDCG@10 = Σ_{r=1}^{min(10, |R_u|)} 1 / log2(r+1)`.
- `NDCG@10 = DCG@10 / IDCG@10` (chỉ tính khi `IDCG > 0`).

**Lưu ý:** `R_u` phải nằm trong candidate universe; nếu positive test không đủ điều kiện gợi ý, cần quy định và báo cáo rõ, không âm thầm làm biến mất nhãn khó.

### Kiểm thử

- [ ] Top-10 không chứa movie đã quan sát trước cutoff.
- [ ] Không có candidate hoặc ground truth bị trùng ID.
- [ ] Cùng user/candidate/label cho cả 5 mô hình.
- [ ] Unit tests cho Precision, Recall, DCG/IDCG/NDCG bằng ví dụ tính tay.
- [ ] Metric tổng hợp được tính trên cùng danh sách eligible users.

## 8. R06 — So sánh công bằng 5 mô hình

- Popularity phải tính từ **Train** khi validation, và **Train + Validation** khi test.
- MF-BPR, LightGCN, HGNN without Attention và HGNN + Attention dùng chung split, evaluation protocol và tiêu chí lựa chọn mô hình.
- Các mô hình trainable cần ngân sách tuning/epoch hợp lý; không tối ưu riêng trên Test.
- **Attention ablation:** Giữ gần tương đương embedding dimension, số tầng, input graph, scoring/loss và training budget. Ghi rõ khác biệt kiến trúc và số tham số nếu không thể bằng nhau hoàn toàn.
- **Genre ablation:** Giữ nguyên tập user/movie, nhãn, split và evaluation; chỉ thay đổi nguồn thông tin genre theo thiết kế.
- Chạy nhiều seed (khuyến nghị ít nhất **3**) cho các mô hình có tính ngẫu nhiên; báo cáo mean và độ biến thiên, không chỉ chọn run tốt nhất.

### Kiểm thử

- [ ] Bảng kết quả có cùng số user và cùng protocol.
- [ ] Tất cả lựa chọn hyperparameter dựa trên Validation.
- [ ] Ablation thay đổi đúng biến độc lập cần khảo sát.
- [ ] Ghi rõ seed, epoch, cấu hình và phiên bản checkpoint.

## 9. R07 — Cold-start và sparsity

- Global temporal split có thể tạo user/movie chưa xuất hiện trong Train.
- Trọng tâm dự án là **warm-start**; cần định nghĩa eligibility rõ ràng (ví dụ user có ít nhất một positive trong lịch sử được phép và một positive trong tập đánh giá).
- Không di chuyển tương tác tương lai về Train để đủ số lượng mỗi user.
- Thống kê số user/movie không có tương tác Train, số user bị loại và số lượng eligible users cho Validation/Test.
- Không chọn user demo dựa trên việc mô hình chính đạt kết quả đẹp.

### Kiểm thử

- [ ] Báo cáo cold-start counts và chính sách xử lý.
- [ ] Không có thao tác sửa cutoff để ép mỗi user có Train/Validation/Test.
- [ ] User demo đáp ứng tiêu chí đã công bố.

## 10. R08 — Overfitting, training và model selection

- Theo dõi Train loss và Validation `NDCG@10`.
- Dùng early stopping/checkpoint selection theo Validation, không theo Test.
- Thử dropout, weight decay, số tầng và embedding dimension trong ngân sách hợp lý.
- Kiểm tra NaN/Inf, gradient, phân phối score, train loss không giảm và sự khác biệt giữa train/validation.
- Sau khi chốt cấu hình bằng Validation, **train lại trên Train + Validation** cho checkpoint Test. Không dùng Test labels để điều chỉnh.
- Nếu sử dụng lịch sử train lại hoặc early stopping cho final training, phải mô tả rõ cách chọn số epoch mà không dựa trên Test.

### Kiểm thử

- [ ] Lưu learning curves và best-validation checkpoint.
- [ ] Không có Test metric trong vòng lặp chọn model.
- [ ] Final checkpoint được xác nhận dùng đúng Train + Validation.

## 11. R09 — Attention Weight Visualization trên Streamlit

**Giữ lại như một chức năng độc lập của demo.** Không cần genre filtering hoặc giải thích nhân quả.

### Yêu cầu

1. Chọn một user hợp lệ và một node/phim liên quan để phân tích.
2. Chạy inference trên graph snapshot hợp lệ cho Test.
3. Trích xuất **attention weights thực tế** từ `GATv2Conv`, không sử dụng dữ liệu minh họa trong bản demo cuối.
4. Lưu mapping `edge_index -> raw node IDs` và loại quan hệ; hiển thị layer/head tương ứng.
5. Nếu nhiều attention heads, ghi rõ hiển thị từng head hay giá trị tổng hợp (ví dụ mean) và cách tính.
6. Không diễn giải weight `0.42` là “42% yêu thích”; attention không chứng minh quan hệ nhân quả.
7. Nếu trực quan hóa chỉ một tập con cạnh, phải ghi rõ cách chọn cạnh và rằng giá trị có thể được chuẩn hóa theo neighborhood gốc.

### Kiểm thử

- [ ] Attention weights khớp đúng cạnh, node, relation, layer và head.
- [ ] Trực quan hóa không dùng Test labels làm feature hoặc graph edge.
- [ ] Không hiển thị attention giả lập như kết quả thực nghiệm.
- [ ] Có chú thích giới hạn diễn giải.

## 12. R10 — Reproducibility và artifact trên Google Colab

### Artifact tối thiểu

| Artifact | Công dụng |
|---|---|
| `model_state_dict` / checkpoint | Nạp lại trọng số |
| Model config | Khởi tạo đúng kiến trúc |
| User/movie/genre mappings | Giải mã ID và giữ thứ tự embedding |
| Split indices + timestamps | Tái tạo Train/Validation/Test |
| Graph snapshot hoặc cách tái tạo | Message passing đúng thời điểm |
| Random seeds + library versions | Tái lập kết quả |
| Metrics + training logs | Đối chiếu thí nghiệm |
| Eligibility/candidate policy | Đánh giá và demo nhất quán |

- Không commit dataset/checkpoint lớn nếu vượt giới hạn repository; sử dụng cơ chế lưu artifact phù hợp và ghi hướng dẫn tải/nạp.
- Lưu phiên bản Python, PyTorch, PyG và môi trường Colab thực tế; tránh giả định phiên bản GPU/driver cố định.
- Test checkpoint bằng quy trình **save → reload → inference** trên một user cố định, đối chiếu score/top-10 trong cùng môi trường và chế độ `eval()`.
- Tách notebook thí nghiệm và mã nguồn tái sử dụng trong `src/`.

## 13. Checklist theo Phase

### Phase 0 — Environment
- [ ] Windows 11 + VS Code hoạt động; Colab kết nối được.
- [ ] Notebook `notebooks/01_environment_setup.ipynb` được giữ nguyên.
- [ ] Xác nhận phiên bản thư viện, GPU/CPU, seed và khả năng import.

### Phase 1 — EDA
- [ ] Kiểm tra schema, timestamp, rating, user/movie/genre.
- [ ] Thống kê sparsity, phân bố lịch sử và tỉ lệ positive.

### Phase 2 — Preprocessing / Split
- [ ] Global temporal split 80/10/10; lưu cutoff và indices.
- [ ] Chốt observed/positive policy, eligibility và ID mappings.
- [ ] Kiểm thử leakage, cold-start counts, negative sampling.

### Phase 3 — Graph Construction
- [ ] Xây đúng node/edge types và reverse edges.
- [ ] Tạo snapshot Train, Train + Validation.
- [ ] Unit test graph không chứa nhãn tương lai.

### Phase 4 — Baselines
- [ ] Popularity, MF-BPR, LightGCN hoạt động.
- [ ] Chốt full-ranking evaluator và unit tests trước khi so sánh.

### Phase 5 — HGNN + Attention
- [ ] HGNN without Attention và HGNN + Attention chạy được.
- [ ] BPR training ổn định, có checkpoint và attention extraction.

### Phase 6 — Evaluation / Ablation
- [ ] So sánh 5 mô hình với 3 metric trên cùng protocol.
- [ ] Chạy Attention ablation và Genre ablation.
- [ ] Nhiều seed; ghi nhận kết quả và giới hạn.

### Phase 7 — Streamlit Demo
- [ ] Chọn user hợp lệ, lịch sử, Top-10 của 5 mô hình, metrics.
- [ ] Attention Visualization dùng weight thật và mapping đúng.
- [ ] Demo chỉ inference, không train và không lộ Test labels vào input.

### Phase 8 — Report
- [ ] Mô tả split, graph snapshots, negative sampling, eligibility, evaluator.
- [ ] Trình bày 5-model comparison, 2 ablations, giới hạn và reproducibility.

## 14. Nguyên tắc chốt trước khi chạy thực nghiệm chính thức

> **Không bắt đầu so sánh HGNN + Attention với baseline nếu chưa kiểm thử xong temporal split, negative sampling, ID mapping và full-ranking evaluator.**

Khi thay đổi bất kỳ quy tắc nào về dữ liệu, candidate pool, eligibility hoặc đánh giá, phải cập nhật cấu hình, chạy lại các baseline liên quan và ghi rõ phiên bản thực nghiệm; không so sánh kết quả từ các protocol khác nhau.

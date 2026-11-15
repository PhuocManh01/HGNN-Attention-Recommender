# HGNN-Attention-Recommender

> **Movie Decision Support System (DSS) — nghiên cứu Top-K Recommendation bằng Heterogeneous Graph Neural Network và Attention**

Dự án nghiên cứu hệ thống gợi ý phim cá nhân hóa trên **MovieLens Latest Small**, biểu diễn dữ liệu dưới dạng đồ thị dị thể `user–movie–genre`, sử dụng **HeteroConv + GATv2Conv** để học biểu diễn và xếp hạng phim. Trọng tâm là **kiểm chứng thực nghiệm** hiệu quả của thông tin đồ thị dị thể và Attention; ứng dụng DSS chỉ nhằm minh họa kết quả nghiên cứu. **Không dự đoán chính xác điểm rating (rating regression).**

**Trạng thái:** Đang triển khai theo phase; các kết quả thực nghiệm chưa có sẽ không được coi là đã chứng minh.

## 1. Mục tiêu, câu hỏi nghiên cứu và phạm vi

### 1.1. Mục tiêu

1. Xây dựng pipeline có thể tái lập từ MovieLens đến Top-K recommendation.
2. So sánh công bằng **5 mô hình** theo cùng giao thức đánh giá.
3. Đo lường đóng góp của **Attention** và thông tin **Genre** qua ablation.
4. Xây dựng demo tối giản: chọn một user MovieLens có sẵn, suy luận bằng 5 mô hình đã huấn luyện và so sánh Top-10.

### 1.2. Research Questions

- **RQ1:** HGNN khai thác quan hệ `user–movie–genre` có cải thiện Top-K so với Popularity, MF-BPR và LightGCN không?
- **RQ2:** GATv2 Attention có cải thiện kết quả so với HGNN không dùng Attention khi giữ điều kiện thực nghiệm tương đương không?
- **RQ3:** Việc bổ sung cạnh `movie–genre` tác động thế nào đến chất lượng gợi ý, đặc biệt theo nhóm user có lịch sử tương tác khác nhau?

### 1.3. Phạm vi và ngoài phạm vi

**Trong phạm vi:** MovieLens Latest Small; implicit-feedback ranking; 5 mô hình; temporal evaluation; full-ranking; 2 ablation chính (Attention, Genre); demo Streamlit dùng checkpoint đã huấn luyện để so sánh 5 mô hình trên cùng user.

**Ngoài phạm vi:** XGBoost; dataset thứ hai; đăng ký/đăng nhập và quản lý tài khoản; cơ sở dữ liệu hoặc backend API riêng; Docker/Kubernetes; hệ thống production; thu thập rating mới; train trực tiếp trên giao diện; streaming/online learning; rating regression; suy diễn nhân quả từ Attention weights; sử dụng `tags.csv` hoặc `links.csv` làm đặc trưng. Không nghiên cứu chuyên sâu cold-start, nhưng vẫn kiểm tra và báo cáo các trường hợp phát sinh.

## 2. Dữ liệu MovieLens

Nguồn: [GroupLens — MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/).

| Tệp | Vai trò |
| --- | --- |
| `ratings.csv` | `userId`, `movieId`, `rating`, `timestamp`; nguồn tương tác |
| `movies.csv` | `movieId`, `title`, `genres`; metadata phim |
| `tags.csv` | Không sử dụng trong phiên bản nghiên cứu này |
| `links.csv` | Không sử dụng trong phiên bản nghiên cứu này |

**Quy ước implicit feedback:** `rating >= 4.0` là **positive**. Rating thấp hơn vẫn là **observed interaction**, không tự động là nhãn negative. Cặp user–movie chưa quan sát là **unobserved**, không phải negative đã được xác nhận. `timestamp` là thời điểm đánh giá, **không phải** thời điểm phim được phát hành hoặc bắt đầu khả dụng.

**Metadata:** `genres` tách bằng `|`, loại bỏ giá trị `(no genres listed)` khỏi danh sách thể loại có ý nghĩa; phim không có thể loại vẫn tồn tại trong graph. Metadata phim từ `movies.csv` được xem là thông tin tĩnh theo giả định của bài toán; MovieLens không cung cấp lịch sử thay đổi metadata/availability, vì vậy không thể khẳng định mô phỏng hoàn hảo môi trường online ở mọi cutoff.

## 3. Đồ thị dị thể

### 3.1. Node và edge

| Node | Mô tả |
| --- | --- |
| `user` | Người dùng trong MovieLens |
| `movie` | Phim trong danh mục được lựa chọn |
| `genre` | Thể loại hợp lệ |

| Canonical edge type | Ý nghĩa |
| --- | --- |
| `('user','rates','movie')` | Positive user–movie có sẵn tại snapshot |
| `('movie','rev_rates','user')` | Cạnh ngược của `rates` |
| `('movie','has_genre','genre')` | Metadata phim–thể loại |
| `('genre','rev_has_genre','movie')` | Cạnh ngược của `has_genre` |

Cạnh ngược chỉ hỗ trợ message passing, **không** được tính là một nhãn tương tác mới. Mọi reverse edge phải được tạo từ forward edge hợp lệ của **cùng snapshot**. Không dùng cạnh positive thuộc Validation/Test trong graph Train.

### 3.2. Node features (v1)

- **User:** trainable ID embedding, khởi tạo theo seed; không dùng thống kê từ tương lai.
- **Movie:** trainable ID embedding và vector `multi-hot` thể loại; đưa về cùng `hidden_dim` bằng phép chiếu tuyến tính, kết hợp theo cấu hình cố định. Không dùng tương tác Validation/Test để tạo đặc trưng.
- **Genre:** embedding học được cho từng thể loại; mapping thể loại cố định và lưu lại.
- **ID mappings:** mỗi loại node có ID liên tục riêng; mapping tạo từ catalog/định danh được phép tại thời điểm thực nghiệm, lưu cùng checkpoint. Các ID không có lịch sử Train là **cold-start** và phải được báo cáo riêng; embedding ID chưa được huấn luyện không được coi là mô hình đã giải quyết cold-start.

**Ablation Genre:** bỏ `movie–genre`/reverse edges **và** loại genre-derived movie features để kiểm tra đóng góp của toàn bộ nguồn thông tin thể loại; Đảm bảo so sánh cùng split, candidate pool và seed.

## 4. Temporal split và chính sách dữ liệu

### 4.1. Quy trình chia

1. Đọc **toàn bộ** ratings, chuẩn hóa kiểu dữ liệu, timestamp UTC, loại bản ghi lỗi; kiểm tra trùng `(userId,movieId)` và công bố quy tắc xử lý (mặc định giữ rating có timestamp muộn nhất nếu trùng).
2. Sắp xếp **tất cả observed interactions** theo `(timestamp, userId, movieId)` để xử lý hòa thời gian ổn định.
3. Xác định **hai cutoff thời gian toàn cục** tương ứng khoảng **80% / 10% / 10%** số bản ghi observed; các bản ghi có **cùng timestamp không bị chia qua hai tập**. Tỷ lệ thực tế có thể chênh lệch và phải được ghi lại.
4. Chia thành `Train`, `Validation`, `Test` theo cutoff; **sau đó** gắn nhãn positive với ngưỡng `rating >= 4.0` trong từng phần.
5. Giữ cả bảng `observed` (mọi rating) và `positive` (rating đạt ngưỡng) cho mỗi phần; **chỉ positive edges** dùng cho user–movie message passing trong phiên bản đầu.
6. Không dịch chuyển tương tác tương lai sang Train chỉ để đảm bảo mọi user có positive Train. Báo cáo `warm-user`, `cold-user`, `warm-item`, `cold-item`; đánh giá chính tập trung vào user có positive history tại cutoff, các trường hợp còn lại báo cáo riêng.

**Lưu ý:** Việc chọn cutoff theo toàn bộ ratings trước khi chọn positive giúp các mốc thời gian nhất quán. Cutoff, số dòng, số user/movie, số positive và mật độ dữ liệu từng split phải được lưu trong manifest.

### 4.2. Các tập dùng cho ranking

- **Validation targets:** positive trong khoảng Validation; chỉ đánh giá user có positive target và đủ điều kiện warm-user theo giao thức chính.
- **Test targets:** positive trong khoảng Test; chỉ đánh giá user có positive target và đủ điều kiện warm-user theo giao thức chính.
- Nếu user đã tương tác với phim trước cutoff, phim đó được loại khỏi candidate pool để đo **gợi ý phim chưa từng tương tác**. Positive target trùng phim đã xem trước cutoff không được tính là target hợp lệ; số lượng bị loại phải được báo cáo.
- Mỗi user đóng góp một giá trị metric; kết quả cuối cùng là **macro-average theo user đủ điều kiện**. Không gán Recall@K cho user không có target.

## 5. Rolling History và chống data leakage

| Giai đoạn | User–movie edges dùng cho message passing | Positive labels dùng để tối ưu/đánh giá |
| --- | --- | --- |
| Train | Train positive | Train positive, tách supervision khỏi edges trong mini-batch khi cần |
| Validation | Train positive | Validation positive (chỉ đánh giá/chọn cấu hình) |
| Final training (sau khi chọn hyperparameters) | Train + Validation positive | Train + Validation positive |
| Test | Train + Validation positive | Test positive (chỉ đánh giá một lần) |

- **Train:** Nếu sử dụng graph snapshot gồm toàn bộ Train positive, việc dự đoán chính các cạnh đó có nguy cơ *edge-label leakage* qua message passing. **Bắt buộc** dùng cơ chế tách supervision/message-passing edges (ví dụ edge-disjoint split theo batch hoặc loại positive edge đang được tối ưu cùng reverse edge tương ứng), và kiểm thử sự tách biệt này. Không được chỉ dựa vào việc Test/Validation không nằm trong Train graph.
- **Validation:** checkpoint và hyperparameters được chọn bằng metric Validation, không cập nhật trọng số trên Validation labels.
- **Final training:** sau khi khóa cấu hình, huấn luyện lại trên Train + Validation bằng cùng chiến lược edge-disjoint supervision, không dùng Test; lịch trình epoch/early stopping phải được quyết định từ Validation trước khi chạy Test.
- **Test:** chỉ dùng Train + Validation history, không có Test labels/edges trong graph; không điều chỉnh mô hình sau khi xem Test.
- `movie–genre` edges chỉ dựa trên metadata tĩnh đã khai báo; mọi feature thống kê như popularity/count phải được tính từ lịch sử được phép tại snapshot.

**Tính nhất quán:** Đánh giá Validation và Test đều sử dụng một mô hình đã được huấn luyện tương ứng với snapshot của giai đoạn đó; không lấy checkpoint Train rồi tùy tiện gắn thêm tương tác Validation vào graph Test mà không xác định rõ phương pháp cập nhật embedding.

## 6. Negative sampling và loss

**Mục tiêu:** Pairwise ranking với **Bayesian Personalized Ranking (BPR)**:

\[
\mathcal{L}_{\mathrm{BPR}}=-\frac{1}{|\mathcal{B}|}\sum_{(u,i,j)\in\mathcal{B}}\log\sigma(s(u,i)-s(u,j))+\lambda\lVert\Theta\rVert_2^2
\]

- `i`: positive item trong tập supervision hợp lệ; `j`: item được lấy mẫu từ danh mục ứng viên **chưa quan sát tính đến cutoff tương ứng**.
- Mặc định **uniform negative sampling**, 1 negative/positive cho phiên bản đầu; thử tỷ lệ khác chỉ trong nghiên cứu bổ sung.
- Loại khỏi negative pool mọi item có **observed rating** trước hoặc tại cutoff huấn luyện tương ứng, bao gồm rating thấp. Không lấy negative trùng positive của batch/user.
- Không tra cứu tương tác tương lai để làm feature, edge hoặc lựa chọn negative của một snapshot. Do đó một negative giả định có thể trở thành positive về sau; đây là hạn chế của implicit feedback, phải ghi nhận.
- Cố định random seed, lưu cấu hình sampler; lấy mẫu chỉ trong Train/final training. Validation/Test sử dụng **full-ranking**, không sampled negatives.
- Các mô hình không sử dụng BPR (ví dụ Popularity) vẫn phải đánh giá theo cùng protocol.

## 7. Kiến trúc HGNN + Attention

```text
User ID embedding       Movie ID embedding + Genre multi-hot       Genre embedding
        \                         |                                  /
         +---------------- HeteroData ------------------------------+
                                |
                  HeteroConv (per-edge-type GATv2Conv)
                                |
                 Relation aggregation (sum/mean; fixed)
                                |
                  User / Movie node embeddings
                                |
                score(u, movie) = dot(z_u, z_movie)
                                |
                    BPR loss / Top-K ranking
```

- **Mô hình chính:** `HeteroConv` với `GATv2Conv` theo loại quan hệ, hỗ trợ bipartite input cho các edge types khác loại node; kiểm tra shape, self-loop và xử lý node không có incoming edges. Không dùng self-loop mặc định không phù hợp cho bipartite edges.
- **Attention:** GATv2 tính attention trên **các hàng xóm trong từng quan hệ**. `HeteroConv(aggr='sum'/'mean')` **không** mặc định học attention giữa các loại quan hệ; không được tuyên bố có relation-level attention nếu chưa xây dựng thêm.
- **Scoring:** dot product giữa user và movie embeddings; lọc observed items trước khi lấy Top-K.
- **Hyperparameters khởi đầu (không phải kết quả tối ưu):** `hidden_dim=64`, `num_layers=2`, `heads=2`, `dropout=0.2`, `lr=1e-3`, `weight_decay=1e-5`, `batch_size=1024`, `seed=42`. Điều chỉnh bằng Validation; lưu tất cả vào YAML.
- **HGNN without Attention:** giữ cùng node features, edge types, scoring, loss, số tầng và ngân sách tuning; thay GATv2Conv bằng phép message aggregation không attention (ví dụ SAGEConv theo từng quan hệ). Báo cáo số tham số và thời gian chạy; không khẳng định hai kiến trúc hoàn toàn tương đương chỉ vì dùng cùng `hidden_dim`.

## 8. Năm mô hình nghiên cứu

| Mô hình | Vai trò |
| --- | --- |
| **Popularity** | Baseline xếp hạng theo số positive Train tại snapshot |
| **MF-BPR** | Collaborative filtering bằng trainable user/movie embeddings và BPR |
| **LightGCN** | Graph collaborative filtering trên user–movie positive bipartite graph |
| **HGNN without Attention** | Đối chứng graph dị thể không Attention |
| **HGNN + Attention (GATv2)** | Mô hình nghiên cứu chính |

Không sử dụng **XGBoost**. Baseline chỉ dùng thông tin hợp lệ theo snapshot; cùng user cohort, item catalog/candidate policy, metric và cutoff. Các mô hình có khả năng biểu diễn khác nhau được so sánh theo kết quả thực tế, không giả định mô hình phức tạp luôn tốt hơn.

## 9. Full-ranking evaluation

- Với mỗi user đủ điều kiện, tính điểm cho **toàn bộ phim hợp lệ trong catalog tại snapshot**, không chỉ một tập negative lấy mẫu.
- **Mask** mọi item đã có observed interaction trước cutoff (cả rating thấp và cao), sau đó lấy Top-K từ item còn lại.
- Chỉ dùng **positive targets** chưa bị mask để tính relevance; item chưa quan sát không mặc định là negative thật.
- Catalog mặc định lấy từ `movies.csv` với giả định metadata tĩnh. Không dùng timestamp của rating đầu tiên làm ngày phát hành. Phải công bố giới hạn **item availability** và phân tích cold-item riêng.
- **Primary metrics:** `Precision@10`, `Recall@10`, `NDCG@10` (binary relevance, macro-average trên eligible users). `K=10` cố định cho bảng kết quả chính.
- **Báo cáo bổ sung tối thiểu:** số user được đánh giá, số positive targets và tỷ lệ cold-start bị loại/được xử lý. Thời gian chạy và số tham số chỉ ghi nhận khi cần giải thích chi phí mô hình; không bắt buộc HitRate@10, Coverage@10 hay peak memory.
- **Tie-breaking:** nếu score bằng nhau, sắp theo `movieId` tăng dần; kiểm tra item mask, target set, eligible-user count và candidate counts.
- **Model selection:** chọn checkpoint/hyperparameters bằng **Validation NDCG@10**, không nhìn Test. Dùng **cùng evaluation implementation** cho tất cả mô hình.
- **Reporting:** mean ± std qua ít nhất **3 seeds** cho các mô hình học được, công bố từng seed, cấu hình và số user được đánh giá. Popularity có thể deterministic.

## 10. Thiết kế ablation và thực nghiệm

| Mã | Cấu hình | Câu hỏi |
| --- | --- | --- |
| A0 | HGNN without Attention, có Genre | Attention có mang lại lợi ích? |
| A1 | HGNN + GATv2, có Genre | Mô hình chính |
| A2 | HGNN + GATv2, không dùng Genre edges/features | Thông tin thể loại đóng góp bao nhiêu? |

- Chạy trên cùng splits, candidate policy, seeds, optimizer/loss phù hợp và ngân sách tuning được công bố.
- Bảng chính: 5 mô hình × Precision/Recall/NDCG@10; bảng ablation: A0/A1/A2. Không bắt buộc so sánh số head, peak memory hoặc phân tích chuyên sâu theo nhóm user.
- Trực quan hóa Attention weights của mô hình HGNN + Attention trong ứng dụng demo để minh họa cơ chế tổng hợp thông tin trên đồ thị. Không yêu cầu thực nghiệm riêng về khả năng giải thích của Attention.
- Không thực hiện phân tích chuyên sâu cold-start. Không coi Attention weights là bằng chứng giải thích nhân quả cho kết quả gợi ý.
- Kết luận chỉ dựa trên kết quả đo được, tránh khẳng định cải thiện khi chênh lệch nằm trong độ biến thiên giữa seeds.

## 11. Demo nghiên cứu (Streamlit tối giản)

**Mục tiêu:** minh họa kết quả nghiên cứu, **không** xây dựng website gợi ý phim hoàn chỉnh. Người trình bày chọn **một `userId` đã có trong MovieLens**, sau đó xem **Top-10 từ cả 5 mô hình** và so sánh kết quả trên cùng giao thức.

**Luồng demo:**

1. Chọn `userId` trong danh sách user đủ điều kiện của snapshot Test; mặc định có thể chọn sẵn một user minh họa.
2. Hiển thị lịch sử rating **chỉ đến Test cutoff** (số lượt đánh giá, phim đã tương tác, thể loại từ lịch sử hợp lệ). Không hiển thị Test targets như lịch sử đầu vào.
3. Nạp checkpoint/model artifacts, ID mappings, cấu hình, graph snapshot và candidate mask tương ứng của **5 mô hình đã huấn luyện trước trên Colab**. Không train lại khi người dùng bấm demo.
4. Tính điểm trên **cùng tập phim ứng viên hợp lệ**, loại phim đã tương tác trước cutoff, trả về Top-10 của từng mô hình: thứ hạng, tên phim và thể loại.
5. Hiển thị 5 danh sách cạnh nhau hoặc theo tab; so sánh phim trùng lặp, thứ hạng khác nhau và bảng **Precision@10, Recall@10, NDCG@10** của toàn bộ tập Test. Có thể hiện metric riêng user được chọn **chỉ khi có ground truth Test hợp lệ**, kèm nhãn phân biệt với metric tổng thể.
6. **Attention Weight Visualization:** Hiển thị trọng số Attention được trích xuất trực tiếp từ mô hình HGNN + Attention (GATv2Conv) dưới dạng biểu đồ. Cho phép lựa chọn phim hoặc node liên quan để quan sát trọng số Attention trên các cạnh tương ứng. Thông tin hiển thị gồm loại quan hệ, tầng Attention, attention head và trọng số. Chức năng này chỉ áp dụng cho HGNN + Attention, không yêu cầu các mô hình còn lại hỗ trợ.

**Lưu ý kỹ thuật:** GATv2Conv có thể trả về Attention weights khi cấu hình forward phù hợp. Cần lưu ánh xạ giữa chỉ số cạnh và node gốc để hiển thị đúng phim, user hoặc genre.

**Không cần:** đăng ký/đăng nhập, CRUD user, thêm rating, lọc theo thể loại, API/backend riêng, database riêng, huấn luyện trên UI hay triển khai production. Không xây dựng hệ thống giải thích quyết định gợi ý bằng Attention theo quan hệ nhân quả. Attention Weight Visualization chỉ nhằm minh họa cơ chế hoạt động của mô hình. Điểm ranking giữa các mô hình **không cùng thang đo**; so sánh thứ hạng và metric, không so sánh trực tiếp score thô. Nếu không có checkpoint/metadata phù hợp, demo phải báo lỗi rõ ràng thay vì tạo kết quả giả.

## 12. Công nghệ và môi trường

| Thành phần | Công nghệ |
| --- | --- |
| Phát triển | Windows 11, VS Code, Git/GitHub |
| Training | **Google Colab (GPU nếu khả dụng)** |
| Ngôn ngữ | Python |
| Data | pandas, NumPy, SciPy |
| ML/baselines | PyTorch; scikit-learn chỉ khi thực sự cần tiện ích |
| Graph | PyTorch Geometric (PyG), NetworkX (khảo sát) |
| Visualization | Matplotlib (đủ cho báo cáo cơ bản) |
| Config | YAML (OmegaConf tùy chọn) |
| Demo | Streamlit (giai đoạn DSS) |

Cài dependency theo **runtime Colab thực tế**, không hard-code phiên bản CUDA/PyG khi chưa kiểm tra tương thích. `requirements-colab.txt` là danh sách dependency mục tiêu; ghi phiên bản đã chạy thành công vào artifact môi trường. **Không bắt buộc có GPU trên Windows.**

### 12.1. Notebook Phase 0 — giữ nguyên tên

**Giữ:** [`notebooks/00_environment_setup.ipynb`](./notebooks/00_environment_setup.ipynb). Notebook dùng để kiểm tra runtime, thiết lập thư mục, cài dependency, mount Drive tùy chọn và kiểm tra PyTorch/PyG/GPU. Không đổi tên hoặc thay thế notebook này khi cập nhật README.

Quy trình Colab: mở notebook từ repository/Drive → chọn runtime → chạy lần lượt cell → kiểm tra import và thiết bị → lưu báo cáo phiên bản. Checkpoint/dataset processed có thể lưu Drive để tồn tại qua các phiên; không commit dữ liệu lớn hoặc secrets vào Git.

### 12.2. Kiểm tra local (tùy chọn)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-colab.txt
python scripts/check_environment.py
```

Lưu ý: lệnh trên giả định các file đã tồn tại ở đúng đường dẫn và dependency tương thích Windows. `requirements-colab.txt` có thể chứa package chỉ phù hợp Colab; khi đó dùng environment riêng cho local thay vì ép cài bằng mọi giá.

## 13. Cấu trúc repository tối giản (mục tiêu)

> Cấu trúc dưới đây là **đích thiết kế**, không phải khẳng định tất cả file/module đã được tạo.

```text
HGNN-Attention-Recommender/
├── app/                       # Một ứng dụng Streamlit (Phase 7)
├── checkpoints/               # Model checkpoints (không commit file lớn)
├── configs/                   # data.yaml, model.yaml, training.yaml, colab.yaml
├── data/
│   ├── raw/                   # MovieLens gốc, không chỉnh sửa
│   ├── processed/             # Node features, mappings, graph
│   └── splits/                # Temporal splits, manifests
├── docs/                      # Protocol và tài liệu nghiên cứu cần thiết
├── experiments/               # CSV/JSON metrics, cấu hình và kết quả từng seed
├── notebooks/
│   └── 00_environment_setup.ipynb  # GIỮ NGUYÊN notebook Phase 0
├── scripts/
│   └── check_environment.py
├── src/
│   ├── data/                  # Loading, split, sampling, graph
│   ├── evaluation/            # Full-ranking metrics
│   ├── inference/             # Top-K inference
│   ├── models/                # 5 models
│   ├── training/              # Train/validate/retrain
│   └── utils/                 # Seed, logging, configs
├── tests/                     # Unit/integration/leakage tests
├── requirements-colab.txt
└── README.md
```

## 14. Lộ trình triển khai và tiêu chí hoàn thành

| Phase | Mục tiêu | Deliverables | Tiêu chí hoàn thành (DoD) |
| --- | --- | --- | --- |
| **0 — Environment** | Colab + VS Code | `00_environment_setup.ipynb`, config, environment check | Notebook chạy hết; import torch/PyG thành công; ghi device/versions; CPU fallback được kiểm tra |
| **1 — EDA** | Hiểu MovieLens | EDA notebook, data quality report | Thống kê user/movie/rating/genre; kiểm tra null, duplicate, timestamp, sparsity; lưu báo cáo |
| **2 — Splitting** | Dữ liệu thực nghiệm | Train/Val/Test observed+positive, mappings, manifest | Cutoff toàn cục tái lập; timestamp tie không vượt split; không overlap; cohort/cold-start được báo cáo |
| **3 — Graph** | HeteroData snapshots | Graph builder, features, snapshot tests | Edge types/shape đúng; reverse edges đúng; không có future user–movie edges; train supervision-edge isolation test pass |
| **4 — Baselines** | Popularity, MF-BPR, LightGCN | 3 mô hình + configs + metrics | Cùng candidate mask và evaluator; checkpoint/load chạy được; baseline kết quả tái lập |
| **5 — HGNN** | HGNN no Attention và GATv2 | 2 mô hình, training scripts | Forward/backward pass; BPR giảm trên bài test nhỏ; lưu/load checkpoint; không leakage |
| **6 — Experiments** | Full-ranking và 2 ablation chính | Bảng 5 mô hình, A0/A1/A2, kết quả từng seed | Precision/Recall/NDCG@10; ≥3 seeds cho mô hình học được; chọn bằng Validation; Test chỉ đánh giá sau khi chốt |
| **7 — DSS Demo** | So sánh 5 mô hình và trực quan hóa Attention | Một Streamlit app với User Profile, Top-10 Recommendations, Model Comparison và Attention Visualization | Chọn user có sẵn → 5 Top-10 hợp lệ; không gợi ý item đã observed; hiển thị metric Test; trực quan hóa Attention weights thực tế từ GATv2Conv, đúng node/cạnh/quan hệ; checkpoint và mapping nạp được; không train trên UI |
| **8 — Report** | Tài liệu khoa học | Report, README final, reproduction guide | Mô tả protocol, giới hạn, kết quả, config/seed, cách chạy lại; không có claim thiếu bằng chứng |

**Nguyên tắc thực hiện:** chỉ chuyển phase khi DoD của phase hiện tại đạt; có thể điều chỉnh thiết kế nhưng mọi thay đổi ảnh hưởng protocol phải được version và chạy lại so sánh cần thiết.

## 15. Kiểm thử bắt buộc và tái lập

- **Data:** số lượng dòng trước/sau cleaning; timestamp monotonic; split disjoint; positive là tập con của observed; mapping ổn định.
- **Graph:** edge index trong phạm vi; forward/reverse nhất quán; không có Validation/Test user–movie edges trong Train graph; không có supervision edge và reverse tương ứng trong message-passing graph khi tối ưu chính edge đó.
- **Sampling:** negative không trùng positive/observed tại cutoff; seed tái lập; không dùng future interactions.
- **Ranking:** observed-mask chính xác; no duplicate Top-K; tie-breaking ổn định; metrics so với ví dụ nhỏ tính tay.
- **Experiment:** seed, cutoff, config, package versions, git commit, model checkpoint, mappings, results và thời gian chạy được lưu; không sử dụng Test để tune.
- **CI/local tests:** ưu tiên unit tests chạy CPU; GPU/Colab tests riêng khi runtime sẵn có.

## 16. Giới hạn và rủi ro

1. MovieLens là dữ liệu rating quan sát được, không có negative ground truth đầy đủ; kết quả offline không bảo đảm hiệu quả online.
2. Dataset không cung cấp lịch sử thời điểm phim khả dụng; full-ranking trên catalog tĩnh có thể bao gồm phim chưa khả dụng ở cutoff thực tế.
3. Ngưỡng `rating >= 4.0` là quy ước nghiên cứu, không phải định nghĩa phổ quát về sở thích.
4. Cold-start user/item chưa được giải quyết đầy đủ bằng ID embeddings; phải báo cáo tỷ lệ và cách xử lý.
5. GATv2 Attention weights không tự động là lời giải thích nhân quả cho quyết định gợi ý.
6. Colab có thể thay đổi runtime, giới hạn RAM/GPU và ngắt phiên; cần lưu checkpoints và dependency versions.
7. Phương pháp edge-disjoint supervision trong graph training phải được hiện thực và kiểm thử cụ thể; chỉ temporal split thôi chưa đủ loại bỏ leakage.

## 17. Tài liệu tham khảo

- [MovieLens Datasets — GroupLens](https://grouplens.org/datasets/movielens/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [GATv2: How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491)
- [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)
- [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126)

---

**Quy ước trạng thái:** Các bảng kiến trúc, phase và tham số trong README là **đặc tả mục tiêu**, không đồng nghĩa mã nguồn đã triển khai hay kết quả đã được xác thực. Các quyết định thực nghiệm thay đổi sau Phase 2 phải được ghi vào changelog/config để bảo đảm tái lập.

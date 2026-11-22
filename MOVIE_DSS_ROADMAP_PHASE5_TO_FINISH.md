# MOVIE DSS — Lộ trình triển khai từ Phase 5 đến hoàn thiện

**Phiên bản:** 1.0 — 22/11/2026**Môi trường:** Windows 11 + VS Code; train Google Colab (GPU T4 khi có)**Dataset:** MovieLens 1M (`ml-1m`), user–movie–genre, per-user temporal split, positive threshold 4.0**Hệ thống cuối:** Streamlit DSS**Giao thức Test đã chốt:** khóa cấu hình bằng Validation → huấn luyện lại trên Train+Validation theo quy tắc đã định trước → Test cuối cùng, không tuning theo Test.

> **Phạm vi:** Tài liệu là kế hoạch triển khai, không phải báo cáo rằng các bước tương lai đã hoàn tất. Dựa trên các file `rgcn.py`, `hgnn_trainer.py`, `train_rgcn.py`, `05_hgnn_attention.ipynb` được cung cấp. Các file khác cần được kiểm tra trong repository trước khi chỉnh sửa.

## A. Trạng thái và quyết định kiến trúc

- Phase 0–4: môi trường, EDA, preprocessing, heterogeneous graph, Popularity/MF-BPR/LightGCN.
- Phase 5: notebook hiện gọi **cả mô hình không attention và HGNN/HGT attention**; không tạo thêm Phase 6 riêng chỉ để train mô hình không attention.
- Mô hình trong `rgcn.py` hiện là `HeteroConv` + `GraphConv` riêng theo edge type, **không phải `RGCNConv` chuẩn**. **Quyết định khuyến nghị: giữ code và đổi tên hiển thị thành `Hetero-GCN (No Attention)`**, có thể giữ tên file/class cũ để tránh phá compatibility; trong báo cáo không gọi là R-GCN chuẩn. Nếu sau này cần R-GCN chuẩn, thực hiện thành một biến thể riêng và đánh giá độc lập.
- Hetero-GCN và HGT khác nhau ở nhiều chi tiết kiến trúc; chênh lệch metric **không tự động chứng minh tác động nhân quả riêng của attention**. Ablation gần tương đương kiến trúc sẽ giải quyết ở Phase 7.
- Phase 6: kiểm chứng trainer, dữ liệu, metric và so sánh công bằng. Phase 7: tuning/ablation. Phase 8: robustness/final Test. Phase 9: Streamlit DSS. Phase 10: test/deploy/report.

### Cảnh báo bảo mật phải xử lý trước tiên

Notebook Phase 5 được cung cấp có **GitHub personal access token viết trực tiếp trong cell clone**. **Thu hồi (revoke) token đó ngay trong GitHub, tạo token mới nếu cần**, xóa token khỏi notebook và git history nếu đã commit. Không đưa token vào prompt/chat, notebook output, commit hoặc báo cáo. Dùng Colab Secrets, credential helper hoặc repo public nếu phù hợp. Xóa khỏi file không tự thu hồi token đã lộ.

### Số liệu đối chiếu dữ liệu (không dùng như kết quả train mới)

- MovieLens 1M: **1.000.209 ratings, 6.040 users**; số phim sau mapping có thể là **3.883**, cần đọc metadata thực tế.
- Positive train edges từng quan sát: **470.016**; phải đối chiếu file metadata/split đang chạy.
- HGT Validation NDCG@10 từng quan sát: **~0,07824**; chỉ dùng làm mốc, không coi là kết quả final.
- **Không đánh giá Test trước Phase 8**.

## B. Quy tắc áp dụng cho mọi phase

1. **Đọc mã nguồn thật trước khi sửa**; không tự giả định tên khóa YAML, trường metadata hoặc đường dẫn.
2. Giữ nguyên split và mapping giữa các mô hình; ghi hash của split/graph, seed, git commit, config, phiên bản package, GPU và timestamp trong mỗi run.
3. Không thay đổi `test.csv` để chọn hyperparameters; không dùng Test trong Early Stopping, tuning, ablation, model selection.
4. Lưu kết quả mỗi run vào thư mục riêng; không ghi đè checkpoint hoặc report trước đó.
5. Thực hiện theo chuỗi: **audit → code → unit test → smoke test → official run → báo cáo → commit**. Nếu test thất bại, dừng và sửa đúng lỗi.
6. Mọi so sánh metric phải dùng cùng protocol: user đủ điều kiện, candidate universe, loại trừ lịch sử, k=10, target interactions và cách xử lý tie.
7. Train ở Colab, phát triển ở Windows; script chạy từ repo root, dùng `pathlib`, không hardcode đường dẫn máy hoặc bí mật.
8. Không giả định nhanh = đúng: đo thời gian bằng `perf_counter`, đồng bộ CUDA trước khi chốt thời gian, ghi `optimizer_steps`, số positive pairs và peak GPU memory.
9. Khi có thay đổi giao thức, version hóa protocol và chạy lại những mô hình bị ảnh hưởng.

## C. Phase 5 — Hoàn thiện trước khi bước sang Phase 6

### P5.1 Audit kiến trúc và naming

**File:** `src/models/rgcn.py`, `src/models/hgnn_attention.py`, `scripts/train_rgcn.py`, notebook 05, README.
**Việc làm:** Giữ HeteroConv+GraphConv; đổi tên model trong report/biểu đồ/báo cáo thành `Hetero-GCN (No Attention)`; ghi rõ relation-specific GraphConv, residual, LayerNorm, dropout. Giữ alias class `RGCNRecommender` tạm thời nếu script cũ cần; tránh thay đổi checkpoint schema không tương thích.

### P5.2 Sửa trainer để loss đúng và chạy được ở bộ nhớ thực tế

**File:** `src/training/hgnn_trainer.py`.

**Hiện trạng đã xác nhận:** mỗi epoch chỉ `propagate(graph)` một lần, cộng loss của các mini-batch rồi `backward()` và `optimizer.step()` **một lần/epoch**. Các batch là chia nhỏ phần tính loss, **không phải mini-batch optimizer steps**. `torch.stack(batch_losses).mean()` cân trọng số mỗi batch ngang nhau, kể cả batch cuối nhỏ.

**Yêu cầu sửa:**

- Tính trung bình loss theo số **positive training pairs** (cân bằng batch size), kiểm tra chính xác cách `bpr_loss` reduction hoạt động.
- Không giữ hàng trăm batch loss cùng autograd graph nếu gây OOM. Ưu tiên **một graph propagation, tính BPR loss theo các khối nhỏ, tích lũy gradient đúng**. Có thể dùng `loss_chunk.backward(retain_graph=True)` cho các chunk trừ chunk cuối (cân nhắc PyTorch autograd và bộ nhớ); hoặc thiết kế phương án khác có kiểm thử tương đương gradient. Không `optimizer.step()` giữa các backward cùng một graph cũ.
- Đặt seed **trước khi khởi tạo mô hình** ở script gọi; trainer set seed lại không thay thế bước này.
- Ghi `epoch_seconds`, `optimizer_steps`, `positive_pairs`, `num_batches`, `train_loss`, `val_ndcg`; nếu CUDA đo thời gian với `torch.cuda.synchronize()` trước/sau.
- Cung cấp lựa chọn training strategy rõ ràng: `full_graph_one_step_per_epoch` hiện tại, và chỉ thêm chiến lược khác khi benchmark có cơ sở. **Không tự ý đổi strategy giữa HGT và Hetero-GCN**.
- Đảm bảo `best_state` hợp lệ, early stopping, `model.eval()` và no_grad trong validation.

**Tests bắt buộc:** gradient của weighted chunk loss khớp unchunked loss trên graph toy (dung sai định nghĩa rõ); không có NaN/Inf; optimizer steps = số epoch đã chạy; loss/metrics finite; smoke test CPU; kiểm tra không leak Test.

### P5.3 Checkpoint, metadata, kiểm tra graph

**File:** `scripts/train_rgcn.py`, `scripts/train_hgnn.py`, configs, notebook 05.

- Script Hetero-GCN hiện **chưa lưu checkpoint**; thêm `state_dict`, model config, graph metadata, mapping fingerprint, seed, best epoch, protocol version; checkpoint phải load lại và tạo score khớp trước khi train tiếp.
- Đọc `cfg['rgcn']` đúng với YAML thực tế, không tự thêm khóa mà không cập nhật config.
- Kiểm tra **set và số lượng** cạnh positive Train, chiều đảo ngược, node counts, index range, dataset identity, split seed, positive threshold; chỉ so sánh count là chưa đủ.
- Đặt output Hetero-GCN và HGT riêng, không ghi đè. Dùng config chung cho các điều kiện đối chứng, báo cáo số tham số mỗi mô hình.
- Notebook 05 gọi cả hai, thu thập và so sánh metric, có thể rerun từng mô hình mà không train lại mô hình kia, backup artifact lên Drive.

### P5.4 Bảo đảm đánh giá chính xác

**File:** `src/evaluation/ranking.py` và tests. Kiểm tra `score_fn` tránh `model.propagate()` không có graph; loại trừ phim đã xem trong Train; validation positives định nghĩa rõ; full-ranking cùng movie universe; cùng eligible users; không dùng Test.

### P5.5 Gate hoàn thành Phase 5

- [ ] Token cũ đã revoke, notebook sạch bí mật.
- [ ] Naming đúng: Hetero-GCN vs HGT.
- [ ] Weighted loss và gradient test PASS.
- [ ] Test evaluator, graph, mapping, leakage PASS.
- [ ] Checkpoint **cả hai** model lưu và load được.
- [ ] Smoke test cả hai PASS; official training đã chạy và có log/metrics (không bịa số).
- [ ] Hai mô hình dùng cùng protocol và Validation; `test_evaluated=false`.
- [ ] Commit code và tài liệu, không commit dữ liệu lớn/token.

**Artifact:** `experiments/phase5_heterogcn/<run_id>/...`, `experiments/phase5_hgt/<run_id>/...`, `docs/phase5_audit.md`.

## D. Phase 6 — Model Evaluation & Reproducibility

**Notebook:** `notebooks/06_model_evaluation.ipynb`
**Mục tiêu:** đối chiếu **5 mô hình** Popularity, MF-BPR, LightGCN, Hetero-GCN, HGT trên cùng validation, không tuning sâu ở phase này.

**Các bước:**

1. Nạp reports/CSV per-user từ Phase 4/5; xác thực hash split, positive threshold, eligible user set, K, candidate rules.
2. Nếu protocol không đồng nhất, sửa protocol và **chạy lại các mô hình liên quan** trước khi so sánh.
3. Tạo bảng Precision@10, Recall@10, NDCG@10, users_evaluated, runtime, optimizer steps, params, peak memory.
4. Vẽ loss và NDCG theo epoch của các model trainable; đánh dấu best epoch/early stopping.
5. Phân tích performance theo nhóm user tương tác ít/nhiều (ngưỡng lấy từ train, khai báo rõ).
6. Audit LightGCN nếu loss đứng gần 0,693: kiểm tra gradients, negative sampling, score variance, optimizer, convergence; không kết luận baseline kém nếu chưa train đúng.
7. Ghi limitations: Hetero-GCN vs HGT khác nhiều thành phần; kết quả chưa chứng minh riêng attention.

**Artifact:** `experiments/phase6/comparison_validation.csv`, `phase6_report.json`, `plots/`, `docs/phase6_protocol.md`.

**Gate:** 5 mô hình có báo cáo hợp lệ cùng protocol; không dùng Test; biểu đồ và báo cáo có thể tạo lại từ artifact; lỗi baseline đã xử lý hoặc được ghi nhận rõ.

## E. Phase 7 — Hyperparameter Tuning & Ablation

**Notebook:** `07_hyperparameter_tuning.ipynb`, `07_ablation_study.ipynb`.

**Tuning:** giữ nguyên split; mỗi mô hình có ngân sách tìm kiếm tương đương và seed/candidate protocol nhất quán; dùng Validation NDCG@10 để chọn cấu hình. Không giả định 80 giây/train cho toàn bộ cấu hình; đo thực tế và kiểm soát GPU budget.

**Không gian đề xuất:** hidden_dim {32,64,128}; layers {1,2,3}; lr {3e-4,1e-3,3e-3}; dropout {0,0.1,0.2}; negatives {1,3,5}; HGT heads {2,4,8} khi hidden_dim chia hết heads. Có thể giới hạn 8–15 trials/model tùy tài nguyên; lưu tất cả trials, seed, metric, thời gian.

**Ablation:**

- User–Movie vs User–Movie–Genre cho Hetero-GCN và HGT.
- HGT attention vs biến thể **gần tương đương HGT nhưng bỏ attention**, giữ các yếu tố khác tối đa; không dùng đơn thuần Hetero-GCN vs HGT để tuyên bố attention gain.
- Tùy tài nguyên: layers/genre edges/negative sampling.

**Artifact:** `experiments/phase7/tuning_results.csv`, `best_configs.json`, `ablation_results.csv`, `plots/`.

**Gate:** cấu hình tốt nhất được chọn **chỉ bằng Validation**; mọi trial có metadata; có bảng ablation và diễn giải thận trọng; chưa đụng Test.

## F. Phase 8 — Robustness, Final Retrain, Test

**Notebook:** `08_robustness_evaluation.ipynb`, `08_final_test.ipynb`.

**Trình tự khóa trước Test:**

1. Khóa split, preprocessing, metric, eligible users, hyperparameters, số epoch/retrain schedule, negative sampling, model selection và seeds; commit `docs/final_test_protocol.md`.
2. Chạy 5 seeds (đề xuất 42,123,2024,2025,2026) trên Train/Validation cho các mô hình stochastic; báo cáo mean±std và runtime. Không điều chỉnh hyperparameters theo Test.
3. Định nghĩa **Train+Validation retrain**: hợp nhất interactions đã được phép quan sát; tạo graph mới chỉ từ positive Train+Validation và movie–genre; cập nhật history để loại trừ toàn bộ Train+Val. **Không dùng bất kỳ Test edge nào** để xây graph/feature, kể cả positive edge.
4. Vì không còn validation để early stop trong final retrain, **khóa số epoch theo quy tắc định trước từ Validation runs** (ví dụ median best epoch trên 5 seeds; quyết định trước khi xem Test). Lưu rõ lựa chọn và áp dụng công bằng.
5. Train final model trên Train+Val theo seeds đã khóa; chạy Test **một lần theo protocol đã khóa**, chỉ dùng Test target cho metrics, không dùng để cập nhật model.
6. Báo cáo Precision/Recall/NDCG/HitRate/Coverage/Novelty@10 nếu định nghĩa rõ; mean±std, phân tích phân nhóm, thời gian inference.
7. Sau Test không tuning theo Test; nếu sửa bug làm sai metric, công bố invalidation và rerun có audit trail.

**Artifact:** `experiments/phase8/final_test_results.csv`, `final_model_checkpoints/`, `docs/final_test_protocol.md`, `docs/phase8_report.md`.

**Gate:** Test không leak, final checkpoints load được, mapping khớp, bảng Test và uncertainty được xuất; kết quả có thể tái lập.

## G. Phase 9 — Streamlit Movie DSS

**Mục tiêu:** demo hỗ trợ ra quyết định chọn phim từ checkpoint đã khóa, không chỉ hiển thị metric.

**Thiết kế tối thiểu:**

- User thuộc MovieLens 1M: chọn user ID, hiển thị lịch sử tương tác được phép dùng và Top-K chưa xem.
- User mới: fallback popularity/genre preferences; **không giả vờ HGT đã học embedding cho user mới**.
- Hiển thị tên phim, năm, genres, điểm phù hợp (giải thích thang điểm là score mô hình, không phải xác suất), rating cộng đồng nếu tính từ lịch sử được phép.
- So sánh 2–3 phim theo genres, mức phổ biến, rating thống kê, model score; giải thích bằng **bằng chứng quan sát được**, không gán attention weight thành lý do nhân quả.
- Sidebar chọn model (ít nhất HGT, Hetero-GCN, Popularity) nếu checkpoint tương ứng có sẵn và tương thích; chọn K=5/10/20.

**Kiến trúc:** Streamlit + Python service, cache embeddings sau propagation (`st.cache_resource` có quản lý device), mapping movie IDs, checkpoint provenance; không cần FastAPI riêng ở MVP.

**File đề xuất:** `app/streamlit_app.py`, `app/model_loader.py`, `app/recommender.py`, `app/movie_service.py`, `app/explanations.py`, `tests/test_app_recommendations.py`, `docs/app_user_guide.md`.

**Gate:** user mẫu nhận đúng Top-K, không lặp phim lịch sử; reload checkpoint ổn; fallback hoạt động; app không đòi Colab runtime; không dùng Test làm input features trái protocol.

## H. Phase 10 — QA, Deployment, Documentation, Final Report

1. **QA:** unit + integration + end-to-end, cold-start, mapping, out-of-range, no duplicates, reproducibility, model provenance, performance latency/RAM.
2. **Deployment:** chạy local trước (`streamlit run app/streamlit_app.py`), đóng gói requirements, cache, README setup; nếu cloud thì chỉ deploy khi checkpoint/requirements phù hợp giới hạn CPU/RAM.
3. **Báo cáo:** (1) Tổng quan DSS, (2) lý thuyết RS/GNN/HGNN/HGT, (3) dữ liệu/split/graph, (4) kiến trúc và training, (5) experiments + tuning/ablation + final Test, (6) Streamlit DSS và QA, (7) kết luận/hạn chế/hướng phát triển.
4. **Release:** `git tag v1.0.0`, README demo, screenshots, report, cấu hình tái lập; không commit secrets hoặc raw data có điều kiện phân phối hạn chế.

**Gate:** người khác clone repo, thiết lập môi trường và chạy được demo theo README; báo cáo trùng artifact; tests pass; không lộ credential.

## I. Thứ tự file/notebook chuẩn

```text
notebooks/
  00_environment_setup.ipynb
  01_movielens_eda.ipynb
  02_data_preprocessing_per_user.ipynb
  03_heterogeneous_graph_per_user.ipynb
  04_baseline_models_per_user.ipynb
  05_hgnn_attention.ipynb       # cả Hetero-GCN và HGT
  06_model_evaluation.ipynb
  07_hyperparameter_tuning.ipynb
  07_ablation_study.ipynb
  08_robustness_evaluation.ipynb
  08_final_test.ipynb
app/
tests/
docs/
experiments/
```

## J. Tiến độ dự kiến và điểm dừng

| Chặng               | Thời gian tham khảo | Điều kiện chuyển tiếp                  |
| -------------------- | --------------------: | ------------------------------------------- |
| Hoàn thiện Phase 5 |            1–3 ngày | checkpoint cả 2, loss/protocol/tests PASS  |
| Phase 6              |            1–2 ngày | bảng 5 model validation hợp lệ           |
| Phase 7              |            3–5 ngày | best configs + ablation khóa               |
| Phase 8              |            2–4 ngày | final Test đã khóa, báo cáo hoàn tất |
| Phase 9              |            3–5 ngày | Streamlit MVP chạy được                 |
| Phase 10             |            2–4 ngày | QA + deploy + report + release              |

*Ước tính 12–23 ngày làm việc; thay đổi theo GPU quota, bugs, số trials.*

---

# K. Bộ prompt copy-paste để tiếp tục từng chặng

**Cách dùng:** Gửi từng prompt trong ChatGPT cùng repository/files mới nhất. Chỉ chuyển prompt kế tiếp sau khi gate của chặng trước PASS. Không yêu cầu ChatGPT tự giả định đã chạy code nếu chưa có logs.

## PROMPT 0 — Audit và hoàn thiện Phase 5 (BẮT ĐẦU TẠI ĐÂY)

```text
Tôi đang phát triển Movie DSS HGNN + Attention trên MovieLens 1M, Windows 11 + VS Code, train Colab T4. Phase 5 hiện train cả HeteroConv+GraphConv không Attention (file rgcn.py, class RGCNRecommender) và HGT Attention (05_hgnn_attention.ipynb). Tôi đã chọn giữ kiến trúc không attention hiện tại, đổi tên học thuật thành Hetero-GCN (No Attention), KHÔNG chuyển sang RGCNConv chuẩn.

Hãy AUDIT toàn bộ Phase 5 dựa trên các file repo tôi cung cấp: rgcn.py, hgnn_trainer.py, train_rgcn.py, train_hgnn.py, hgnn_attention.py, ranking.py, hgnn_attention.yaml, notebook 05, preprocessing/graph metadata, tests. Không giả định trường YAML hoặc metadata chưa được kiểm chứng; nếu thiếu file thì yêu cầu đúng file cần thiết.

Ưu tiên: (1) cảnh báo và hướng dẫn revoke token GitHub từng bị hardcode trong notebook, chuyển sang Colab Secrets; không in token; (2) chuẩn hóa naming Hetero-GCN; (3) sửa weighted loss theo positive pair và kiểm thử gradient, tránh OOM autograd; (4) kiểm tra 1 optimizer step/epoch, đo epoch_seconds/optimizer_steps/VRAM; (5) seed trước model init; (6) lưu/reload checkpoint Hetero-GCN và HGT; (7) graph/split/evaluator/leakage validation; (8) smoke tests rồi full training cả hai trên Colab.

Hãy hướng dẫn TỪNG BƯỚC, cung cấp FULL CODE của từng file cần sửa, lệnh PowerShell/Colab, tests, output dự kiến, checklist PASS/FAIL. Chỉ hoàn thiện Phase 5; không làm Phase 6 hoặc Test. Không bịa kết quả training.
```

## PROMPT 1 — Phase 6: Evaluation & Reproducibility

```text
Phase 5 đã hoàn thành, đây là các reports, training_history, checkpoint metadata và source code mới nhất [tôi sẽ đính kèm]. Hãy bắt đầu Phase 6 theo MOVIE_DSS_ROADMAP_PHASE5_TO_FINISH.md.

Tạo notebook 06_model_evaluation.ipynb, code/scripts/tests để so sánh Popularity, MF-BPR, LightGCN, Hetero-GCN (No Attention), HGT Attention trên MovieLens 1M Validation full-ranking @10. Trước tiên audit tính tương thích dataset, split, mapping, positive threshold, eligible users, candidate exclusion, top-K và provenance. Nếu không tương thích phải dừng và chỉ rõ mô hình nào cần chạy lại. Vẽ loss/NDCG, bảng Precision/Recall/NDCG, runtime/steps/params/memory; phân tích user groups; kiểm tra LightGCN nếu không hội tụ.

Cung cấp mã chạy được trên Colab, file đầu ra, unit tests, tiêu chí hoàn thành. KHÔNG đọc/đánh giá Test, không tuning Phase 7.
```

## PROMPT 2 — Phase 7: Tuning & Ablation

```text
Phase 6 đã PASS. Tôi gửi các reports/plots/configs đã khóa. Hãy thực hiện Phase 7 theo MOVIE_DSS_ROADMAP_PHASE5_TO_FINISH.md.

Thiết kế tuning công bằng trên Validation cho MF-BPR, LightGCN, Hetero-GCN và HGT trong giới hạn GPU Colab T4. Tạo 07_hyperparameter_tuning.ipynb và 07_ablation_study.ipynb cùng scripts/configs/tests. Ghi từng trial: seed, git commit, config, split hash, metrics, runtime. Tạo ablation User–Movie vs User–Movie–Genre và nếu khả thi biến thể HGT no-attention giữ kiến trúc gần tương đương để kiểm tra attention, không đánh đồng Hetero-GCN vs HGT với attention-only effect.

Chọn best configs bằng Validation NDCG@10, xuất best_configs.json, tuning_results.csv, ablation_results.csv, plots, báo cáo và checklist. Tuyệt đối KHÔNG dùng Test.
```

## PROMPT 3 — Phase 8: Robustness + Retrain + Final Test

```text
Phase 7 đã PASS; best_configs và ablation đã khóa. Hãy triển khai Phase 8 theo MOVIE_DSS_ROADMAP_PHASE5_TO_FINISH.md. Tôi chọn final retrain trên Train+Validation trước khi Test.

Bước 1: viết final_test_protocol.md và khóa trước Test: data/split/mapping, candidate rules, seeds [42,123,2024,2025,2026], metrics, cấu hình, quy tắc chọn epoch final retrain từ Validation (không early stop theo Test), negative sampling, graph Train+Val không có Test edges. Bước 2: nhiều seeds trên Train/Validation, mean±std. Bước 3: code dựng Train+Val graph, retrain, checkpoints. Bước 4: chỉ sau khi xác nhận gate, thực hiện Test cuối cùng, xuất Precision/Recall/NDCG và các metric bổ sung có định nghĩa rõ. Bước 5: kiểm tra leakage, reproducibility, báo cáo.

Hướng dẫn từng bước, code/tests, checkpoint và điều kiện dừng; không điều chỉnh hyperparameters sau khi nhìn Test.
```

## PROMPT 4 — Phase 9: Streamlit DSS

```text
Phase 8 đã PASS, final checkpoints/mapping/metadata đã khóa [tôi sẽ đính kèm]. Hãy xây dựng Phase 9 Streamlit Movie DSS theo MOVIE_DSS_ROADMAP_PHASE5_TO_FINISH.md.

MVP: chọn user có trong MovieLens 1M; Top-K phim chưa xem; chọn model HGT/Hetero-GCN/Popularity nếu có checkpoint hợp lệ; metadata title/year/genres; so sánh 2–3 phim; thông tin hỗ trợ giải thích dựa trên lịch sử và genre (không khẳng định attention là giải thích nhân quả); cold-start fallback popularity/genre; caching embeddings để giảm latency. Xây app/streamlit_app.py, model_loader.py, recommender.py, movie_service.py, explanations.py, requirements và tests. Hướng dẫn chạy Windows 11 PowerShell và kiểm thử trên CPU, không cần Colab để chạy app.

Cung cấp đầy đủ mã, cấu trúc, smoke test, demo checklist; chưa triển khai Phase 10.
```

## PROMPT 5 — Phase 10: QA, Deployment, Final Report

```text
Phase 9 đã PASS, đây là repo/app/checkpoints/results mới nhất [tôi sẽ đính kèm]. Hãy triển khai Phase 10 theo MOVIE_DSS_ROADMAP_PHASE5_TO_FINISH.md.

Thực hiện unit/integration/end-to-end tests, kiểm tra mapping/leakage/reproducibility/cold-start, benchmark latency/RAM, xử lý lỗi UI, hướng dẫn chạy local và tùy chọn deploy Streamlit. Chuẩn hóa README, requirements, model card, experiment protocol, screenshots và release checklist. Xây dàn ý báo cáo 7 chương: tổng quan; lý thuyết; MovieLens 1M và graph; kiến trúc/training; thực nghiệm/tuning/ablation/Test; DSS/QA; kết luận/hạn chế. Chỉ sử dụng số liệu thật từ artifacts, không tự tạo kết quả. Hướng dẫn từng bước với code, tests và tiêu chí PASS/FAIL để đóng dự án.
```

## PROMPT 6 — Kiểm toán cuối cùng (sau Phase 10)

```text
Toàn bộ Movie DSS Phase 0–10 đã triển khai. Hãy audit cuối cùng theo MOVIE_DSS_ROADMAP_PHASE5_TO_FINISH.md và các files/reports tôi đính kèm. Kiểm tra nhất quán notebook naming, dataset MovieLens 1M, split/graph leakage, 5 mô hình, naming Hetero-GCN vs HGT, reproducibility, final Test protocol, checkpoint provenance, Streamlit recommendations, security secrets, dependencies và số liệu trong báo cáo. Xuất bảng PASS/FAIL với bằng chứng đường dẫn/file, mức nghiêm trọng, sửa lỗi còn tồn đọng, danh sách artifact cần bàn giao. Không tự đánh dấu PASS nếu chưa có chứng cứ.
```

---

**Điểm bắt đầu ngay:** PROMPT 0. Sau khi PASS mới dùng PROMPT 1. Không cần tạo `06_hgnn_rgcn.ipynb` nữa vì Phase 5 đã huấn luyện cả hai mô hình.

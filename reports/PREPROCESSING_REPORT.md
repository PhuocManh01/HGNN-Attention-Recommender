# PREPROCESSING_REPORT

## 1. Giới thiệu

Báo cáo tổng hợp và xác nhận pipeline preprocessing Phase 3 cho MovieLens và JTH. Giai đoạn này chỉ chuẩn bị dữ liệu, split, candidate pools và negative sampling; chưa xây dựng graph hoặc huấn luyện model.

## 2. Mục tiêu preprocessing

Chuẩn hóa dữ liệu gốc mà không sửa raw CSV, giữ ID gốc, tạo interaction có timestamp, feature/mapping leakage-safe, temporal split và sampler tái lập được.

## 3. Nguồn dữ liệu

- MovieLens: `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`.
- JTH: `candidates.csv`, `jobs.csv`, `history.csv`.
- Raw integrity: tất cả 7 file tồn tại và đã được ghi nhận SHA-256 trong `reports/preprocessing_validation.json`; validation không ghi vào raw directory.

## 4. MovieLens preprocessing

| Chỉ số | Giá trị |
|---|---:|
| Raw ratings | 100,836 |
| Positive ratings | 48,580 |
| Observed non-positive | 52,256 |
| Catalog movies | 9,742 |
| Users | 610 |
| Processed interactions | 100,836 |
| Duplicate User–Movie | 0 |
| Sparsity | 99.182515% |

Rating gốc được giữ lại; positive được định nghĩa bằng `rating >= 4.0`. Observed non-positive không bị biến thành unobserved. `links.csv` không được dùng để suy luận interaction.

## 5. JTH preprocessing

| Chỉ số | Giá trị |
|---|---:|
| Candidates | 37,554 |
| Jobs | 6,011 |
| Raw application processes | 42,288 |
| Unique Candidate–Job pairs | 41,308 |
| Duplicate pair records | 980 |
| Valid interaction timestamps | 41,308 |
| Temporal anomaly rows | 0 |
| Sparsity | 99.981701% |

Giữ nguyên application events; bảng interactions dùng first interaction của mỗi Candidate–Job. `interaction_time` là proxy bằng stage timestamp hợp lệ sớm nhất, không dùng `end_of_process_date` làm mốc khởi đầu.

## 6. ID mappings và features

Mapping deterministic, contiguous từ 0, có reverse mapping và giữ original IDs. MovieLens có user/movie/genre mappings. JTH có candidate/job/skill/category/expertise-area mappings. Learned scaler, imputer và vocabulary phải fit chỉ trên train; `llm_*`, outcomes và stage timestamps không nằm trong temporal-safe Top-K input mặc định.

## 7. Temporal splitting

Split global chronological, mục tiêu 80/10/10, không random shuffle và không tách cùng timestamp. MovieLens có {'train': 38864, 'val': 4858, 'test': 4858}; JTH có {'train': 33041, 'val': 4140, 'test': 4127}. Pair overlap của cả hai dataset đều bằng 0; timestamp overlap đều bằng 0. Test protocol dùng train + validation history nhất quán; train graph không chứa future edges.

## 8. Candidate pool

Pool được sinh lazy theo user/candidate, không xuất full matrix. MovieLens dùng static catalog assumption vì không có availability timestamp đáng tin cậy; loại rating đã quan sát trước cutoff. JTH yêu cầu `job.create_date <= cutoff`, loại pair đã tương tác và không giả định job còn mở. Ground-truth bị loại do job tạo sau cutoff được ghi riêng trong candidate pool statistics.

## 9. Negative sampling

Uniform sampler hỗ trợ negative ratio, fixed seed, unique negatives, epoch resampling và BPR triplets. Sampler chỉ nhận item từ candidate pool. Các unobserved pairs được đánh dấu `assumed_negative`, không phải negative feedback xác nhận; false-negative risk được ghi trong sampling statistics. Sampled evaluation phải dùng candidate set cố định và không được diễn giải như full-ranking metrics.

## 10. Leakage prevention

- Không dùng validation/test edges trong train graph.
- Không dùng future interactions để block pool tại cutoff.
- Không dùng future recruitment outcomes hoặc `last_stage_reached` làm feature.
- Không dùng `llm_*` khi chưa xác minh snapshot time.
- Observed non-positive MovieLens được giữ ngoài unobserved pool.
- Mapping catalog được phân biệt với train-only feature fitting.

## 11. Validation results

Validation end-to-end đã kiểm tra raw integrity, row counts, IDs, duplicates, split overlap/order, future train edges, mappings, candidate pools, negative sampling và reproducibility. Kết quả validator: **PASSED**. Full test suite: **22 passed**.

## 12. Hạn chế và giả định

- MovieLens static catalog không chứng minh movie đã khả dụng tại từng cutoff.
- JTH thiếu closing date đáng tin cậy nên pool không lọc theo trạng thái job đang mở.
- JTH `interaction_time` là proxy, không phải chắc chắn là ngày ứng viên chủ động ứng tuyển.
- Unobserved không đồng nghĩa negative; false negatives vẫn có thể xảy ra.
- Feature snapshot theo thời gian của một số metadata chưa có, đặc biệt `llm_*`.

## 13. Kết luận

Hai dataset đã có output preprocessing, deterministic mappings, leakage-safe temporal splits, candidate pools và training negative sampler. Các kiểm tra Phase 3 đều đạt.

## 14. Chuẩn bị cho Giai đoạn 4

Sẵn sàng định nghĩa graph schema và graph builders bằng cách sử dụng train graph tại cutoff, các mappings đã lưu, feature schemas và candidate/negative protocols. Chưa triển khai bất kỳ graph construction nào trong báo cáo này.

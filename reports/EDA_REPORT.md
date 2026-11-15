# Báo cáo phân tích khám phá dữ liệu (EDA)

## 1. Giới thiệu

Báo cáo này mô tả hai bộ dữ liệu dùng để phát triển mô hình gợi ý dựa trên đồ thị dị thể và Attention:

- **MovieLens**: miền gợi ý phim, với tương tác người dùng–phim được biểu diễn bằng rating.
- **JTH**: miền tuyển dụng, với các thực thể ứng viên, việc làm và lịch sử các giai đoạn của quá trình tuyển dụng.

Mục tiêu EDA là kiểm tra quy mô, kiểu dữ liệu, giá trị thiếu, bản ghi lặp, phân bố tương tác và các vấn đề ảnh hưởng đến việc xây dựng đồ thị, tạo nhãn và đánh giá mô hình.

Các thống kê và biểu đồ được sinh từ `results/movielens/eda/summary.json` và `results/jth/eda/summary.json`.

## 2. MovieLens

### 2.1. Nguồn và cấu trúc dữ liệu

MovieLens gồm bốn bảng CSV: `ratings.csv` (tương tác user–movie), `movies.csv` (danh mục và thể loại), `tags.csv` (nhãn văn bản) và `links.csv` (liên kết IMDb/TMDB). Các loại nút phù hợp là `user`, `movie` và `genre`; cạnh `movie–genre` được tách từ cột `genres` theo ký tự `|`.

### 2.2. Thống kê tổng quan

| Chỉ số | Giá trị |
|---|---:|
| Người dùng có rating | 610 |
| Phim có ít nhất một rating | 9.724 |
| Phim trong danh mục | 9.742 |
| Rating | 100.836 |
| Rating tích cực (`rating >= 4.0`) | 48.580 |
| Tỷ lệ rating tích cực | 48,18% |
| Độ thưa trên phim có rating | 98,30% |
| Cặp `userId–movieId` bị lặp | 0 |
| Rating tham chiếu phim không tồn tại | 0 |

Ngưỡng `rating >= 4.0` được khai báo trong `configs/movielens.yaml`. Rating gốc vẫn cần được giữ lại cho bài toán dự đoán điểm; ngưỡng chỉ nên dùng khi chuyển sang implicit feedback hoặc ranking.

### 2.3. Missing values và duplicates

`ratings.csv` không có cặp người dùng–phim trùng lặp và không có tham chiếu đến phim ngoài danh mục. Trong `links.csv`, có đúng **8 giá trị `tmdbId` thiếu**; đây là missing metadata liên kết ngoài, không phải missing interaction. Các dòng này vẫn giữ lại vì `movieId` và thông tin phim còn hợp lệ; không dùng `tmdbId` làm feature bắt buộc và không tự suy đoán giá trị thiếu. Các trường IMDb/TMDB không phải là tín hiệu tương tác.

### 2.4. Phân bố rating

Rating nằm trên thang 0.5–5.0. Với ngưỡng hiện tại, 48,18% rating được xem là tích cực. Phân bố này phù hợp cho việc so sánh hai thiết lập: dự đoán rating và ranking các phim chưa từng được người dùng đánh giá.

### 2.5. Phân bố tương tác User–Movie

Số tương tác trên mỗi người dùng và mỗi phim không đồng đều. Độ thưa khoảng 98,30% cho thấy đồ thị user–movie rất thưa; cần chú ý đến người dùng/phim ít lịch sử, negative sampling và việc báo cáo kết quả theo nhóm mức độ hoạt động.

| Đơn vị | Min | Max | Mean | Median |
|---|---:|---:|---:|---:|
| Rating trên mỗi user | 20 | 2.698 | 165,30 | 70,5 |
| Rating trên mỗi movie có rating | 1 | 329 | 10,37 | 3 |

### 2.6. Top 10 thể loại

| Hạng | Thể loại | Số phim |
|---:|---|---:|
| 1 | Drama | 4.361 |
| 2 | Comedy | 3.756 |
| 3 | Thriller | 1.894 |
| 4 | Action | 1.828 |
| 5 | Romance | 1.596 |
| 6 | Adventure | 1.263 |
| 7 | Crime | 1.199 |
| 8 | Sci-Fi | 980 |
| 9 | Horror | 978 |
| 10 | Fantasy | 779 |

### 2.7. Sparsity và vấn đề cold-start

Một phim có thể thuộc nhiều thể loại, vì vậy thể loại nên được biểu diễn bằng nút riêng hoặc multi-hot feature. Có 18 phim trong danh mục chưa xuất hiện trong `ratings.csv`; không có user nào trong tập rating có 0 tương tác. Đây là cold-start theo item, không phải cold-start theo user.

### 2.8. Nhận xét cuối chương MovieLens

MovieLens có tín hiệu rating rõ ràng, không có cặp user–movie trùng và có thể chốt positive interaction bằng `rating >= 4.0`. Tuy nhiên, đồ thị vẫn rất thưa và phân bố tương tác lệch mạnh giữa user/movie. Vì vậy, thực nghiệm cần dùng Top-K metrics, negative sampling nhất quán và báo cáo riêng cold-start item. Metadata genre/tag nên được dùng để hỗ trợ các movie ít tương tác nhưng không được làm thay đổi định nghĩa nhãn.

## 3. JTH

### 3.1. Nguồn và cấu trúc dữ liệu

JTH gồm `candidates.csv` (37.554 dòng, 54 cột), `jobs.csv` (6.011 dòng, 34 cột) và `history.csv` (42.288 dòng, 16 cột). Khóa chính dự kiến là `candidate_id`, `job_id` và `application_id`. Các đặc trưng `llm_*` cần được kiểm tra nguồn gốc và thời điểm tạo trước khi đưa vào mô hình.

### 3.2. Thống kê Candidate–Job

| Chỉ số | Giá trị |
|---|---:|
| Ứng viên trong danh mục | 37.554 |
| Việc làm trong danh mục | 6.011 |
| Quá trình tuyển dụng | 42.288 |
| Ứng viên có tương tác | 26.101 |
| Việc làm có tương tác | 4.116 |
| Cặp ứng viên–việc làm lặp | 980 |
| Cặp ứng viên–việc làm duy nhất | 41.308 |
| Tỷ lệ dòng lặp theo cặp | 2,32% |
| Sparsity trên toàn ma trận Candidate–Job | 99,9817% |
| Ứng viên không khớp danh mục | 0 |
| Việc làm không khớp danh mục | 0 |

| Đơn vị phân bố | Min | Max | Mean | Median |
|---|---:|---:|---:|---:|
| Tương tác trên mỗi candidate có tương tác | 1 | 40 | 1,62 | 1 |
| Tương tác trên mỗi job có tương tác | 1 | 213 | 10,27 | 5 |

980 cặp lặp không nên bị xóa ngay: chúng có thể đại diện cho nhiều lần ứng tuyển hoặc nhiều quy trình khác nhau. Trong báo cáo này, `42.288` là số dòng quá trình và `41.308` là số cặp candidate–job duy nhất. Tính sparsity dùng cặp duy nhất trên toàn không gian `37.554 × 6.011`, không dùng số dòng có lặp.

### 3.3. Cold-start và mức độ bao phủ

| Đối tượng | Tổng số | Có tương tác | Không có tương tác | Tỷ lệ cold-start |
|---|---:|---:|---:|---:|
| Candidate | 37.554 | 26.101 | 11.453 | 30,50% |
| Job | 6.011 | 4.116 | 1.895 | 31,52% |

Cold-start của JTH lớn hơn đáng kể so với MovieLens. Cần đánh giá riêng ba nhóm: candidate mới, job mới và candidate–job chưa từng quan sát. Các feature hồ sơ/kỹ năng có thể hỗ trợ cold-start, nhưng chỉ được dùng nếu chúng đã tồn tại tại cutoff time.

### 3.4. Missing values và duplicates

Một số tỷ lệ thiếu đáng chú ý là: `candidates.csv` có `actual_daily_salary` thiếu 97,73%, `actual_salary` 93,88% và `years_experience` 83,50%; `jobs.csv` có `llm_certifications` thiếu 98,20%, `source` 89,49% và `years_experience` 88,87%. Trong `history.csv`, `4th_interview_date` thiếu 99,84%, `3rd_interview_date` 99,46%, `2nd_interview_date` 98,05% và `job_offer_accepted_date` 97,97%.

Phân loại và cách xử lý đề xuất:

| Loại missing | Ví dụ | Diễn giải | Xử lý |
|---|---|---|---|
| Chưa xảy ra theo quy trình | `2nd_interview_date`, `job_offer_accepted_date` | Stage chưa đạt tại thời điểm ghi nhận | Giữ mask `is_observed`; không điền ngày giả |
| Thiếu nghiệp vụ | `actual_salary`, `source`, `zipcode` | Không được ghi nhận hoặc không áp dụng | Impute theo train split hoặc thêm missing indicator |
| Thiếu đặc trưng dẫn xuất | `llm_certifications`, `llm_programming_languages` | Không có dữ liệu enrich hoặc parser không tạo được | Kiểm tra pipeline nguồn; không mặc định là không có kỹ năng |
| Thiếu định danh/thời gian cốt lõi | `candidate_id`, `job_id`, mốc interaction | Không thể xác định mẫu hoặc thứ tự thời gian | Loại khỏi tập huấn luyện sau khi thống kê và báo cáo |

Không nên điền các mốc tuyển dụng thiếu bằng giá trị suy đoán. Giá trị thiếu có thể nghĩa là giai đoạn chưa xảy ra hoặc chưa được ghi nhận; hai trường hợp này cần được phân biệt nếu dữ liệu nghiệp vụ cho phép.

### 3.5. Phân bố quá trình ứng tuyển

Các giá trị `last_stage_reached` phổ biến là: Qualification (17.107), Shortlist (14.697), Resume Sent (4.140), Application Made (3.210), 1st Interview (1.700), Offer Accepted (857), 2nd Interview (416), 3rd Interview (98), Offer Received (43) và 4th Interview (20).

### 3.6. Phân bố kỹ năng

EDA phát hiện 286 kỹ năng ở ứng viên và 75 kỹ năng ở việc làm. Các kỹ năng ứng viên xuất hiện nhiều gồm JavaScript (8.858), PHP (8.331), Python (6.555), Linux (6.199) và Java (5.296). Ở phía việc làm, các kỹ năng phổ biến gồm PHP (1.102), React.js (869), JavaScript (717), Node.js (645) và Python (355). Cần chuẩn hóa chữ hoa/chữ thường, tên đồng nghĩa và dấu phân cách trước khi tạo nút hoặc cạnh kỹ năng.

#### Top 10 skills

| Hạng | Candidate skill | Số lần | Job skill | Số lần |
|---:|---|---:|---|---:|
| 1 | javascript | 8.858 | php | 1.102 |
| 2 | php | 8.331 | react.js | 869 |
| 3 | python | 6.555 | javascript | 717 |
| 4 | linux | 6.199 | node.js | 645 |
| 5 | java | 5.296 | `_rare_skill_` | 591 |
| 6 | project management | 4.981 | python | 355 |
| 7 | sql | 4.963 | java | 320 |
| 8 | c++ | 4.733 | angular.js | 292 |
| 9 | react.js | 4.212 | automation | 250 |
| 10 | c | 4.032 | vue.js | 240 |

#### Phân tích `_rare_skill_`

`_rare_skill_` không phải là một kỹ năng nghiệp vụ thực tế mà là bucket dùng để thay thế các kỹ năng hiếm/ẩn danh trong dữ liệu JTH. Nó xuất hiện 591 lần trong `jobs.csv` và 579 lần trong `candidates.csv`. Không nên tạo một node kỹ năng có nghĩa ngữ nghĩa từ token này; nên giữ nó như `rare_skill_count`/missing-like indicator hoặc loại khỏi vocabulary khi xây dựng cạnh skill. Cần ghi rõ quy tắc bucket và ngưỡng rare trong pipeline tiền xử lý.

### 3.7. Các giai đoạn tuyển dụng

JTH chứa các mốc shortlist, qualification, gửi CV, phỏng vấn lần 1–4, đề xuất offer, chấp nhận offer và kết thúc quy trình. `last_stage_reached` là thông tin tổng hợp của một quá trình và cần được xem là outcome hoặc label, không mặc định là feature đầu vào.

### 3.8. Chốt bài toán JTH Top-K Recommendation

Thiết kế chính được chốt là **candidate-to-job Top-K recommendation**: tại cutoff time `t`, xếp hạng các job chưa được candidate tương tác và trả về Top 10 job có xác suất xuất hiện tương tác ứng tuyển trong horizon tương lai `H`.

- **Đơn vị truy vấn:** một candidate tại `t`.
- **Candidate pool:** các job đang tồn tại trước hoặc tại `t`, loại các cặp đã quan sát trước `t`.
- **Positive label:** candidate–job phát sinh một quá trình trong `(t, t + H]`.
- **Proxy interaction timestamp:** dùng `min()` của các mốc `spontaneous_application_date`, `shortlist_date`, `qualification_date`, `resume_sent_to_company_date`, `1st_interview_date`, `2nd_interview_date`, `3rd_interview_date`, `4th_interview_date`, `job_offer_proposed_date`, `job_offer_accepted_date`; không dùng `end_of_process_date`. Đây là proxy thuận tiện cho thử nghiệm, chưa chứng minh đó là thời điểm ứng viên bắt đầu tương tác thực sự.
- **Secondary task:** dự đoán candidate đã có quá trình sẽ đạt stage nào tiếp theo, ví dụ `job_offer_accepted`; đây chỉ là hướng mở rộng, không phải nhiệm vụ chính và không được gộp vào positive interaction.
- **Metrics bắt buộc:** `Precision@10`, `Recall@10`, `NDCG@10`, tính macro theo candidate và báo cáo thêm theo nhóm cold-start.

### 3.9. Đặc điểm thời gian, interaction_time và leakage

JTH có dữ liệu thời gian từ `2018-03-09` đến `2025-04-21` nếu xét **proxy interaction timestamp** theo định nghĩa trên. Tất cả 42.288 dòng đều có ít nhất một mốc tạo tương tác proxy hợp lệ. `history.csv` có nhiều timestamp cho cùng một quá trình, nên mỗi mẫu cần có cutoff time rõ ràng.

Kiểm tra tính nhất quán thời gian trên dữ liệu hiện tại cho thấy 31.132 dòng có từ hai mốc trở lên và không có dòng nào vi phạm thứ tự thời gian của các stage sau khi bỏ qua giá trị thiếu. Đây là kiểm tra chất lượng dữ liệu, không phải bằng chứng rằng `min()` là thời điểm bắt đầu thực sự. Khi triển khai chính thức cần thêm: đối chiếu với mốc ứng tuyển nguồn nếu có, kiểm tra các dòng chỉ có stage muộn, và chạy sensitivity analysis giữa `min(stage_dates)`, `spontaneous_application_date` và mốc ứng tuyển nghiệp vụ.

#### Định nghĩa tương tác dương cho nhiệm vụ Top-K — vấn đề nghiên cứu bắt buộc

Một dòng trong `history.csv` chỉ cho biết một quá trình tuyển dụng đã được ghi nhận. Nó **không đồng nghĩa** với việc ứng viên thích công việc, muốn tiếp tục quy trình hoặc đã được tuyển dụng. Nhiệm vụ chính đã được thống nhất là **dự đoán Top-K job có khả năng xuất hiện tương tác ứng tuyển**:

- Đơn vị dự đoán là một cặp `candidate–job` tại thời điểm `t`.
- Nhãn dương là cặp phát sinh một quá trình trong cửa sổ dự đoán sau `t`, theo một mốc khởi đầu được định nghĩa rõ.
- Nhãn âm là cặp đủ điều kiện nhưng không phát sinh quá trình trong cùng cửa sổ; không được tùy tiện xem mọi cặp không quan sát là ứng viên không thích.
- Mốc dùng trong thử nghiệm là **proxy interaction timestamp**; nếu thiếu mốc nghiệp vụ đáng tin cậy, phải ghi rõ đây là dự đoán sự xuất hiện của bản ghi lịch sử, không phải dự đoán sở thích.
- `last_stage_reached` và các ngày của stage sau `t` chỉ được dùng để tạo nhãn, không được dùng làm feature đầu vào.

Stage prediction (`1st_interview`, `job_offer_proposed`, `job_offer_accepted`, ...) là hướng mở rộng sau nhiệm vụ Top-K, không phải lựa chọn thay thế. Mỗi thực nghiệm vẫn phải ghi rõ cutoff, horizon, cách tạo nhãn âm và đơn vị đánh giá.

#### Rò rỉ dữ liệu thời gian — vấn đề nghiên cứu bắt buộc

Không được sử dụng thông tin xảy ra sau thời điểm dự đoán để xây dựng feature đầu vào. Khi dự đoán tại `t`, các trường sau chỉ có thể dùng nếu ngày của chúng không muộn hơn `t`:

- các mốc phỏng vấn;
- `job_offer_proposed_date` và `job_offer_accepted_date`;
- `end_of_process_date`;
- `last_stage_reached` nếu giá trị này được suy ra từ toàn bộ quá trình;
- mọi đặc trưng được tổng hợp từ lịch sử tương lai của cùng ứng viên hoặc việc làm.

Quy trình an toàn:

1. Chọn cutoff time `t` cho từng mẫu.
2. Tạo feature bằng dữ liệu có timestamp `<= t`.
3. Tạo nhãn từ sự kiện trong cửa sổ tương lai `(t, t + horizon]`.
4. Chia train/validation/test theo thời gian, không random split toàn bộ lịch sử.
5. Kiểm tra từng feature: nguồn bảng, timestamp và khả năng quan sát tại `t`.

Không được dùng toàn bộ `history.csv` để tạo thống kê ứng viên/việc làm rồi mới chia train/test; cách này có thể đưa hành vi tương lai vào feature hoặc embedding. Khi xây dựng message passing, cạnh và thuộc tính cũng phải được lọc theo cutoff để không truyền thông tin tương lai qua đồ thị.

### 3.10. Đánh giá thuộc tính gốc và `llm_*`

| Nhóm thuộc tính | Ví dụ | Giá trị nghiên cứu | Rủi ro và quy tắc sử dụng |
|---|---|---|---|
| Gốc của candidate | `skills`, `expertise_area`, `job_category`, `years_experience`, `contract_type`, `zipcode` | Đại diện hồ sơ và khả năng nghề nghiệp được quan sát trực tiếp | Kiểm tra missing; chuẩn hóa text; chỉ dùng bản snapshot có trước cutoff |
| Gốc của job | `job_category`, `skills`, `expertise_area`, `years_experience`, `contract_type`, `salary`, `zipcode` | Đại diện yêu cầu và điều kiện việc làm | Không dùng thông tin được cập nhật sau khi job xuất hiện; salary missing cần có mask |
| Dẫn xuất `llm_*` của candidate | `llm_hard_skills`, `llm_soft_skills`, `llm_programming_languages`, `llm_seniority_level` | Bổ sung ngữ nghĩa, kỹ năng và seniority từ hồ sơ | Cần truy nguyên thời điểm sinh; không dùng nếu được tạo từ dữ liệu tuyển dụng tương lai |
| Dẫn xuất `llm_*` của job | `llm_required_*`, `llm_hard_skills`, `llm_tools_technologies`, `llm_seniority_level` | Làm rõ yêu cầu ẩn trong mô tả job | Có thể gây leakage nếu pipeline enrich dùng trạng thái tuyển dụng hoặc kết quả ứng tuyển |

Thiết kế thực nghiệm nên có ba biến thể: **original-only**, **llm-only** và **original + llm**. So sánh này cho biết mức cải thiện đến từ dữ liệu nghiệp vụ gốc hay từ lớp enrich. Các cột `llm_*` không được mặc định là “chân lý”; cần báo cáo missing rate, nguồn sinh và thời điểm snapshot.

## 4. Đề xuất Heterogeneous Graph

### 4.1. MovieLens Graph

- Nút: `user`, `movie`, `genre`.
- Cạnh: `user–movie` từ rating và `movie–genre` từ danh sách thể loại.
- Thuộc tính cạnh: rating, timestamp và cờ rating tích cực.
- Nhiệm vụ: ranking phim hoặc dự đoán rating; cần tách rõ implicit và explicit feedback.

### 4.2. JTH Graph

- Nút: `candidate`, `job`, `skill`, có thể bổ sung `category` hoặc `expertise_area`.
- Cạnh: `candidate–job` từ quá trình ứng tuyển; `candidate–skill` và `job–skill` từ hồ sơ/yêu cầu.
- Thuộc tính cạnh candidate–job: stage đã đạt **tại cutoff**, số lần tương tác hợp lệ và các đặc trưng thời gian không nhìn thấy tương lai.
- Nhãn: chọn riêng bài toán xuất hiện tương tác hoặc bài toán đạt stage.
- Khi xây dựng message passing theo thời gian, phải lọc cạnh và thuộc tính theo cutoff.

### 4.3. Bảng node/edge đề xuất

Các số liệu trong bảng dưới đây là **thống kê graph toàn bộ trước khi lọc positive, áp dụng cutoff và chia train/validation/test**. Vì vậy MovieLens vẫn bao gồm mọi rating, kể cả rating thấp; JTH vẫn bao gồm các event ở mọi thời điểm, kể cả event tương lai đối với một cutoff cụ thể. Đây **không phải graph huấn luyện**.

| Dataset | Node type | Số node dự kiến | Edge type | Số edge dự kiến | Thuộc tính chính |
|---|---|---:|---|---:|---|
| MovieLens | `user` | 610 | `user–movie` | 100.836 | `rating`, `timestamp`, positive flag |
| MovieLens | `movie` | 9.742 | `movie–genre` | 22.050 | `genre` |
| MovieLens | `genre` | 19 | — | — | tên thể loại |
| JTH | `candidate` | 37.554 | `candidate–job` | 41.308 cặp duy nhất | `interaction_time`, stage tại cutoff |
| JTH | `job` | 6.011 | `candidate–skill` | 171.981 | skill/profile features |
| JTH | `skill` | 287 sau chuẩn hóa hiện tại | `job–skill` | 10.988 | `rare_skill` mask |

Số **287 node Skill** được tính theo hợp của vocabulary kỹ năng ở hai bảng sau khi tách chuỗi và chuẩn hóa khoảng trắng: `286` skill khác nhau ở `candidates.csv` cộng `75` skill khác nhau ở `jobs.csv`, trừ `74` skill xuất hiện ở cả hai bảng: `286 + 75 - 74 = 287`. Con số này hiện vẫn bao gồm node placeholder `_rare_skill_`; nếu loại placeholder khỏi graph thì số node semantic sẽ giảm còn 286.

### 4.4. Thống kê graph dự kiến

Đây cũng là thống kê trên raw/full graph trước xử lý. Graph huấn luyện thực tế phải được tạo lại theo split: MovieLens có thể lọc theo `rating >= 4.0` nếu dùng positive ranking; JTH phải lọc cạnh và thuộc tính theo cutoff, chỉ giữ thông tin quan sát được tại thời điểm dự đoán.

| Graph | Node count | Edge count | Ghi chú |
|---|---:|---:|---|
| MovieLens user–movie | 10.352 node chính | 100.836 cạnh tương tác | Chưa tính cạnh genre nếu chỉ benchmark collaborative filtering |
| MovieLens đầy đủ | 10.371 node | 122.886 cạnh | 100.836 user–movie + 22.050 movie–genre |
| JTH candidate–job | 43.565 node | 41.308 cạnh duy nhất | 980 dòng lặp được giữ ở tầng event nếu cần |
| JTH đầy đủ | 43.852 node | 224.277 cạnh | 41.308 candidate–job + 171.981 candidate–skill + 10.988 job–skill |

Các con số trên là thống kê dự kiến cho graph snapshot toàn bộ. Khi huấn luyện JTH theo thời gian, graph thực tế phải được dựng lại hoặc lọc theo từng cutoff; không được dùng số liệu toàn bộ làm feature cho một snapshot quá khứ.

### 4.5. Baseline và mô hình chính

Sáu baseline được thống nhất cho bài toán Top-K là:

| Mô hình | Loại | Vai trò |
|---|---|---|
| Popularity / Most-interacted | Non-personalized | Upper sanity check cho độ lệch phổ biến |
| Content-based | Feature-based | So sánh bằng genre hoặc skill overlap |
| Matrix Factorization / BPR-MF | Collaborative | Baseline latent-factor |
| LightGCN | Graph collaborative | Baseline graph đồng nhất |
| HGNN | Heterogeneous graph | Baseline graph dị thể không Attention |
| XGBoost | Feature-based ranking/classification | Baseline bắt buộc với feature thủ công và feature matching candidate–job |
| **HGNN-Attention** | **Mô hình chính** | Học quan hệ dị thể với attention theo loại node/edge |

Như vậy có đúng **6 baseline**: Popularity, Content-based, Matrix Factorization/BPR-MF, LightGCN, HGNN và XGBoost; cùng **1 mô hình chính** là HGNN-Attention. Random chỉ là sanity check tùy chọn, không tính vào sáu baseline chính. Các mô hình phải dùng cùng candidate pool, cutoff, negative sampling và `Precision@10`, `Recall@10`, `NDCG@10`. XGBoost cần dùng feature được tạo tại cutoff, không được nhận stage hoặc timestamp tương lai.

## 5. So sánh hai dataset

| Khía cạnh | MovieLens | JTH |
|---|---|---|
| Miền | Gợi ý phim | Tuyển dụng |
| Đối tượng chính | User–Movie | Candidate–Job |
| Tín hiệu phản hồi | Rating trực tiếp | Quá trình và stage tuyển dụng |
| Định nghĩa positive | Có thể cố định bằng `rating >= 4` | Phải định nghĩa theo mục tiêu nghiên cứu |
| Thời gian | Timestamp rating/tag | Nhiều mốc stage có thứ tự |
| Rủi ro chính | Sparsity và cold-start | Label ambiguity và temporal leakage |
| Chiến lược split | Có thể dùng random split có kiểm soát hoặc temporal split | Ưu tiên temporal split và as-of features |

JTH khó hơn MovieLens không chỉ vì đồ thị có nhiều loại nút, mà còn vì một quá trình tuyển dụng có trạng thái thay đổi theo thời gian. Kết quả JTH chỉ có ý nghĩa khi mục tiêu và quy tắc quan sát tại thời điểm dự đoán được mô tả đầy đủ.

### 5.1. Bảng so sánh định lượng

| Chỉ số | MovieLens | JTH |
|---|---:|---:|
| Entity phía user/candidate | 610 users | 37.554 candidates |
| Entity phía item/job | 9.742 movies có trong catalog | 6.011 jobs |
| Tương tác dạng dòng | 100.836 ratings | 42.288 application processes |
| Cặp tương tác duy nhất | 100.836 user–movie | 41.308 candidate–job |
| Dòng/cặp lặp | 0 | 980 (2,32%) |
| Sparsity ma trận | 98,30% | 99,9817% |
| Entity phía user/candidate cold-start | 0 users trong tập rating | 11.453 candidates (30,50%) |
| Entity phía item/job cold-start | 18 movies (0,18%) | 1.895 jobs (31,52%) |
| Min số tương tác/entity hoạt động | User 20; Movie 1 | Candidate 1; Job 1 |
| Mean số tương tác/entity hoạt động | User 165,30; Movie 10,37 | Candidate 1,62; Job 10,27 |
| Median số tương tác/entity hoạt động | User 70,5; Movie 3 | Candidate 1; Job 5 |
| Kiểu tín hiệu | Rating và timestamp | Stage và timestamp nhiều bước |
| Split khuyến nghị | Temporal hoặc random có kiểm soát | Temporal bắt buộc |

MovieLens có nhiều tương tác hơn và ít cold-start item hơn, trong khi JTH có không gian candidate–job lớn hơn, sparsity cao hơn và cần giải quyết thêm vấn đề nhãn theo stage cùng leakage thời gian. Vì vậy không nên so sánh trực tiếp giá trị metric tuyệt đối giữa hai dataset nếu không nêu rõ candidate pool, horizon và quy tắc negative sampling.

## 6. Kết luận và định hướng tiền xử lý

1. Giữ riêng dữ liệu thô và dữ liệu đã biến đổi; ghi lại phiên bản tiền xử lý.
2. Với MovieLens, chuẩn hóa thể loại/tag và tạo positive interaction theo ngưỡng đã cấu hình.
3. Với JTH, xác định grain của mẫu, kiểm tra 980 cặp candidate–job lặp và không gộp trước khi hiểu ý nghĩa nghiệp vụ.
4. Nhiệm vụ JTH chính là candidate-to-job Top-K recommendation; stage prediction chỉ là hướng mở rộng.
5. Tạo feature theo nguyên tắc as-of; mọi sự kiện sau cutoff chỉ được dùng làm label.
6. Dùng temporal split cho JTH và báo cáo riêng các nhóm cold-start.
7. Thống nhất metric chính là `Precision@10`, `Recall@10` và `NDCG@10`; báo cáo macro theo candidate và thêm breakdown cold-start.
8. Kiểm tra rò rỉ tự động trước mỗi lần huấn luyện bằng cách so sánh timestamp của feature với cutoff và timestamp của label.
9. Ghi rõ `interaction_time` hiện là proxy; theo dõi kiểm tra thứ tự stage và sensitivity analysis với các proxy thời gian khác.

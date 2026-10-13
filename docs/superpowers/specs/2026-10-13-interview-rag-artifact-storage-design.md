# Thiết kế RAG phỏng vấn IT và lưu trữ artifact bằng MinIO

Ngày: 2026-10-13 (Asia/Saigon).

Trạng thái: các phần RAG đã được người dùng duyệt trong hội thoại; bổ sung MinIO theo yêu cầu mới. Bản spec này cần được đọc lại trước khi lập kế hoạch triển khai. Đây là thiết kế mục tiêu, không phải mô tả tính năng đã chạy.

## 1. Mục tiêu và phạm vi

Xây dựng buổi luyện phỏng vấn IT từ Intern, Fresher, Junior, Middle đến Senior, cá nhân hóa từ CV, JD, nguồn công ty và câu trả lời. Giữ RAG trong project; core quản lý phiên và diễn biến, RAG cung cấp dữ liệu có bằng chứng. Lưu lâu dài tài liệu và artifact của người dùng bằng MinIO qua interface S3.

Đã chốt:

- Corpus đã duyệt làm nền, kết hợp tìm nguồn mới trước buổi phỏng vấn.
- Cho chọn HR, Technical, Hiring Manager, Tổng hợp. Nếu thiếu, suy ra và hiển thị trước khi bắt đầu; người dùng được chỉnh lại.
- Thời lượng người dùng nhập được ưu tiên. Nếu thiếu, suy ra theo nguồn phù hợp công ty/vị trí/cấp độ/vòng; sau đó dùng khung đã duyệt; cuối cùng dùng mức dự phòng 30–45 phút.
- Phỏng vấn thích ứng theo câu trả lời, có rubric và theo dõi năng lực đã kiểm tra.
- CV, JD, transcript và artifact riêng tư không tự đi vào corpus dùng chung.
- MinIO là nơi lưu object lâu dài; Redis phục vụ điều phối/cache, local workspace phục vụ thực thi.

Không thuộc lần triển khai đầu: ngành ngoài IT, cấp Staff/Principal, phỏng vấn bằng giọng nói/video, quyết định tuyển dụng tự động, tách RAG thành service hoặc thay toàn bộ retrieval bằng knowledge graph.

## 2. Hiện trạng đã kiểm tra từ code

| Điểm hiện tại | Vị trí | Hệ quả thiết kế |
| --- | --- | --- |
| Core sử dụng `RetrievalClient.search(RetrievalQuery)` | `core/retrieval/port.py`, `core/retrieval/factory.py` | Giữ tương thích search; không thay orchestration bằng API standalone |
| RAG mặc định dựa vào framework; nguồn công ty mang tính generic | `rag/adapter.py`, `rag/corpus.py` | Cần bổ sung corpus, metadata và chuẩn bị nguồn công ty |
| Profiler tạo candidate_profile, company_intel, interview_spec, stage_plan | `core/profiler/agent.py` | Mở rộng schema và pipeline hiện có |
| Artifact nằm trên local filesystem, ghi JSON có envelope/checksum | `core/workspace/session.py` | Không thay kiểu Path bằng object URL trên toàn bộ core |
| Raw inputs mặc định 24 giờ; workspace mặc định 30 ngày | `core/workspace/retention.py` | Phân biệt dọn cache local với xóa archive |
| Xóa phiên hiện xử lý Redis và workspace | `api/routes.py`, `core/workspace/cleanup.py` | Cần thêm tombstone và xóa object/metadata |
| Inspector có archive SQLite nhưng đọc nội dung artifact từ workspace | `api/inspect.py`, `fe_inspect/` | Cần đọc artifact lưu lâu dài sau khi workspace hết hạn |
| Compose có app, Redis, Redis Insight; chưa có object store | `compose.yaml` | MinIO chưa được triển khai chỉ bằng việc viết spec |

Các thay đổi chưa commit trong checkout phải được bảo toàn. Kiểm tra lại các điểm tích hợp trước khi thực thi kế hoạch; không coi image đang chạy là bản chứa mọi thay đổi source.

## 3. Kiến trúc và ranh giới

| Thành phần | Trách nhiệm | Đầu vào/đầu ra chính |
| --- | --- | --- |
| Document Intake | Đọc tài liệu, OCR, chuẩn hóa có bằng chứng | DocumentRef → CandidateProfile/JDProfile |
| Knowledge Preparation | Tìm nguồn, kiểm tra, deduplicate, lập snapshot | Company/role/level/round → KnowledgeSnapshot |
| Interview Planner | Đối chiếu hồ sơ và JD, chọn rubric, thời lượng | Profiles + snapshot → InterviewPlan |
| Core orchestration | Hỏi, hỏi sâu, theo dõi thời gian và coverage | Plan + answers → turns/directives |
| Assessor | Đánh giá bằng chứng theo criterion | Answer + rubric → evaluation |
| Report | Tổng hợp nhận định, giới hạn và hướng cải thiện | Evaluations + transcript → report |
| Artifact Archive | Lưu, liệt kê, tải, xác nhận, xóa object | Owner/session/artifact → ArtifactRef |

Giữ `RetrievalClient` và bốn collection hiện có. Các khả năng intake, chuẩn bị nguồn và archive có port riêng, không nhồi tất cả vào `search`. API composition root inject adapter; code điều phối phụ thuộc port, không tự tạo MinIO client.

Luồng chính: upload → lưu original + đăng ký owner → trích xuất → người dùng sửa hồ sơ → chuẩn bị nguồn → xác định vòng/thời lượng → hiển thị kế hoạch → phỏng vấn → đánh giá → báo cáo → archive/đối soát.

## 4. Hợp đồng dữ liệu và bằng chứng

Mọi object có `schema_version`, ID ổn định, thời điểm tạo/cập nhật UTC và liên kết artifact/source. Phân biệt thời điểm xuất bản, thời điểm sự kiện và thời điểm thu thập.

### 4.1 DocumentRef và EvidenceSpan

DocumentRef: `document_id`, `owner_id`, `session_id`, `kind`, tên hiển thị, media type, byte size, SHA-256, artifact reference, trạng thái parse và parser version.

EvidenceSpan: `evidence_id`, `document_id` hoặc `source_id`, page/line/offset khi có, excerpt, extraction method, confidence, assertion status (`explicit`, `inferred`, `user_corrected`, `unverified`). Confidence trích xuất không được dùng thay cho mức thành thạo kỹ năng.

Nhận PDF, DOCX, PNG/JPEG và text; nhận diện MIME thực tế, giới hạn byte/page, reject tài liệu mã hóa hoặc không đọc được với lỗi có thể xử lý. OCR giữ trang và vị trí; không đoán giá trị không đọc được. Mặc định đề xuất cho lần đầu: 20 MiB/file, 50 trang, cấu hình được. Không tải URL/file path tùy ý do người dùng đưa vào để né kiểm tra upload.

### 4.2 CandidateProfile

- Tên, vị trí mong muốn, summary và mục tiêu nghề nghiệp.
- Experience: công ty, chức danh, khoảng thời gian, trách nhiệm, kết quả, loại công việc.
- Projects: bài toán, bối cảnh, công nghệ, vai trò cá nhân, quyết định, khó khăn, kết quả, liên kết.
- Skills: công nghệ/công cụ/chuyên môn/giao tiếp, quan hệ với project hoặc experience.
- Education: trường, ngành, thời gian, môn học và thành tích liên quan.
- Awards, certifications, activities, publications/portfolio khi có.
- Claims cần xác minh, thông tin thiếu/mâu thuẫn và evidence references cho từng trường.

Không cộng các việc trùng thời gian một cách máy móc. Công nghệ trong project là claim, không chứng minh ownership; chứng chỉ không chứng minh proficiency. Các trường không có trong CV giữ null/empty và lý do, không điền để schema trông đầy đủ. Lưu correction revision, phân biệt nguyên bản và thông tin người dùng sửa. Không mặc định hỏi tuổi, gia đình hoặc thông tin cá nhân không liên quan.

### 4.3 JDProfile và FitMatrix

JDProfile: company, title, role family, level, responsibilities, requirements, required/preferred/unspecified, work context, interview information nếu có, evidence refs.

FitMatrix: mỗi requirement có priority/weight, CV evidence, gaps, verification probes và criterion ID. Trọng số do framework hoặc planner suy ra phải được gắn `inferred`, không gán thành yêu cầu chính thức của JD. Thiếu CV/JD vẫn chạy từ form và framework, nhưng báo giới hạn cá nhân hóa.

## 5. Nguồn công ty và retrieval

Nguồn ưu tiên: chính thức → recruiter/interviewer có xuất xứ → trải nghiệm ứng viên có bối cảnh → framework đã duyệt. Nguồn giữ URL, publisher/author nếu có, công ty/entity ID, role, level, round, locale, thời điểm, loại nguồn và evidence spans. Không suy ra danh tính tác giả khi nguồn không cung cấp.

Tìm trước buổi, với giới hạn đề xuất: tối đa 12 tài liệu được fetch trong 60 giây; timeout/retry nằm trong tổng deadline. SearchProvider là port cấu hình được; chọn nhà cung cấp, chi phí và OCR engine trong kế hoạch triển khai dựa trên môi trường thực tế. Pipeline có thể hoạt động corpus-only khi provider thiếu hoặc lỗi.

Fetch chỉ HTTP(S) công khai; chặn private/loopback/link-local IP, kiểm tra lại redirects và giới hạn response size. Không vượt đăng nhập/paywall; chỉ giữ nội dung có quyền sử dụng và trích đoạn cần thiết. Dữ liệu nguồn là untrusted input, không được thực thi hướng dẫn nằm trong tài liệu.

Chuẩn hóa canonical URL, content hash và nhóm nguồn gốc để tránh đếm bài sao chép như các chứng cứ độc lập. Nguồn mâu thuẫn giữ song song với bối cảnh; lời kể không đổi thành official fact. Tránh lấy nguồn của công ty trùng tên hoặc cấp độ/vòng khác.

Nguồn mới qua kiểm tra relevance, xuất xứ, thời điểm và contradictions trước khi dùng trong phiên. Chuyển vào corpus dùng chung cần review record với reviewer và thời điểm duyệt. Không chỉ dùng điểm LLM tự đánh giá để gắn nhãn verified.

Giữ hybrid retrieval hiện có làm baseline; bổ sung metadata filters, source trust, recency và reranking đánh giá được. Index theo đơn vị ý nghĩa như requirement, project claim, câu hỏi/rubric và đoạn kể trải nghiệm; không chỉ cắt theo số ký tự. Mỗi kết quả có source/evidence IDs và lý do áp dụng. Unknown role/level không được âm thầm fallback backend/junior; trả unsupported/needs_confirmation.

KnowledgeSnapshot lưu query, nguồn đã chọn, evidence bundle, corpus/index version, thời điểm chuẩn bị, hạn chế và trạng thái web. Đóng băng snapshot của phiên để audit; nguồn bổ sung trong buổi phải tạo revision mới có lý do.

## 6. Vòng, thời lượng và kế hoạch

Chọn vòng: explicit user choice → evidence phù hợp → framework inference; hiển thị giá trị, nguồn và độ chắc chắn trước khi bắt đầu. Không suy ra HR chỉ từ cấp Intern.

Chọn thời lượng: user minutes → official evidence khớp bối cảnh → các lời kể độc lập có bối cảnh → framework → fallback 30–45 phút. Lưu cả `duration_range_minutes` và `planned_minutes` để runtime có một ngân sách cụ thể; khi hoàn toàn thiếu bằng chứng dùng 40 phút và hiển thị khoảng dự phòng 30–45. Với range có bằng chứng, dùng midpoint làm giá trị mặc định, người dùng chỉnh được. Phạm vi input đề xuất 10–120 phút; ngoài phạm vi trả lỗi form rõ ràng.

Nguồn mâu thuẫn không lấy trung bình tùy tiện giữa các vòng khác nhau. Ghi rõ thời lượng ước lượng của một buổi/vòng, không nhầm tổng thời gian cả quy trình tuyển dụng. Không lấy thời gian làm bài take-home làm thời gian phỏng vấn.

InterviewPlan: round, level, role, duration provenance, ordered stages, criterion priorities/weights, required coverage, question intents, CV probes, company-specific claims, references, reserved closing time và limitations. Questions có thể thay đổi, mục tiêu và rubric có phiên bản ổn định.

## 7. Phỏng vấn thích ứng

Mở đầu: chào tên đã có, giới thiệu vòng/thời lượng, mời giới thiệu. Tiếp theo chọn các phần động lực, project/experience, chuyên môn, hành vi và kết thúc theo vòng. Technical không bắt buộc system design cho mọi vị trí/cấp; HR không phải thi kiến thức thuật toán.

Sau mỗi lượt, core cập nhật bằng chứng, thời gian và criterion status: `not_asked`, `insufficient_evidence`, `sufficient_evidence`, `not_assessed`. Câu trả lời chung chung dẫn tới hỏi làm rõ; mâu thuẫn dẫn tới xác minh; đủ bằng chứng chuyển chủ đề. Không coi im lặng hoặc hết giờ là bằng chứng không có kỹ năng.

Intern/Fresher: nền tảng, học tập, khả năng giải thích và đóng góp thực tế. Junior: thực thi và debugging. Middle: ownership, tradeoffs, chất lượng/vận hành. Senior: phạm vi tác động, quyết định kiến trúc, rủi ro vận hành, mentoring và phối hợp phù hợp JD. Đây là hướng rubric, không phải bắt mọi Senior làm quản lý.

Ngân sách dùng monotonic clock cho phiên đang chạy; khi restart chỉ khôi phục từ checkpoint đã ghi, không dựa vào Redis TTL. Waiting/paused time được tách khỏi active interview time và hiển thị nhất quán. Bản đầu là text interview; latency sinh câu không được tính như thời gian ứng viên trả lời. Dành tối thiểu 10% ngân sách hoặc 2 phút, lấy giá trị lớn hơn, cho candidate questions và kết thúc.

Coverage ưu tiên năng lực trọng tâm trong thời gian cho phép, không cố hỏi mọi dòng CV. Chống hỏi trùng theo question intent + criterion + claim đã khai thác. Giới hạn đề xuất 2 follow-ups liên tiếp cho một intent, sau đó chuyển hoặc đánh dấu chưa đủ. Câu hỏi tiếp chỉ dùng company facts có evidence phù hợp.

Report có verified quotes từ transcript, criterion, nhận định, uncertainty, năng lực chưa kiểm tra và gợi ý luyện tập. Phân biệt claim CV với bằng chứng thể hiện trong buổi. Không quảng bá là kết quả tuyển dụng thật của công ty.

## 8. MinIO và lưu artifact lâu dài

### 8.1 Phân vai dữ liệu

Redis: session coordination, queue, lease, idempotency, cache có TTL. Local workspace: working files và cache để giữ tương thích reader hiện tại. MinIO: originals, extracted/corrected profiles, JD, knowledge snapshots, plans, transcripts, evaluations, reports, trace exports và attachment có quyền lưu. Manifest SQLite bền vững: owner/session/artifact catalog, checksum, object version và archive job state; không phụ thuộc Redis TTL.

Chỉ archive artifact được đăng ký, không dump toàn bộ thư mục tạm hay file credential. Các loại spill/cache nội bộ chỉ lưu khi được manifest báo là dependency của artifact cần giữ. Archive trace bằng snapshot/export nhất quán; không upload trực tiếp live SQLite/WAL như một bản backup hợp lệ.

MinIO được truy cập qua `ArtifactStore` port gồm put/get/head/list/delete. Adapter S3 implement port; workspace không tự gọi network trong mỗi thao tác Path. App composition root inject store và archive coordinator. Lựa chọn SDK S3 là một task triển khai, tránh khóa core vào API riêng của MinIO.

### 8.2 Object và manifest

Buckets private: `user-artifacts` và `knowledge-sources`. Object key của dữ liệu riêng tư:

`owners/{owner_id}/sessions/{session_id}/{artifact_kind}/{artifact_id}/{revision}/{sha256}`

ID nội bộ không chứa tên, email hoặc tên CV. Original filename chỉ là metadata được kiểm soát. Knowledge công khai dùng source ID/content hash; tài liệu do user cung cấp nằm ở bucket private.

Artifact manifest: artifact ID, owner/session, logical path, kind, schema version, content type, size, SHA-256 của bytes lưu thực tế, payload checksum nếu JSON envelope có, bucket/key/version ID, created_at, archived_at, status và dependencies. Không dùng ETag như SHA-256. Mỗi revision immutable; con trỏ revision hiện hành nằm trong catalog.

### 8.3 Ghi và phục hồi

1. Ghi local atomically; tạo hoặc cập nhật durable manifest/outbox bằng transaction catalog.
2. Worker upload với key ổn định, timeout và retry giới hạn; xác minh object size/hash theo metadata và kiểm tra đọc lại trong phép thử.
3. Chuyển `pending` → `uploading` → `archived` chỉ sau xác nhận; lỗi → `retryable_failed` hoặc `failed` với lý do.
4. Chỉ dọn bản local có archive được xác nhận và không cần cho tác vụ đang chạy. Reconciler tìm artifact chưa đăng ký hoặc jobs mất sau crash; local write và catalog không được tuyên bố là transaction chung.
5. Reader ưu tiên local; nếu thiếu, kiểm tra owner rồi tải revision đã archive, xác minh checksum và materialize atomically vào workspace khi core cần Path.

Outbox lưu bền trong SQLite, Redis Streams chỉ là transport/wakeup. Retry idempotent, không tạo nhiều revision từ cùng job. Artifact replacement tạo revision mới, không overwrite revision cũ. Không đánh dấu archive complete chỉ vì message được đưa vào Redis.

UI hiển thị report readiness tách biệt archive status (`pending`, `archived`, `failed`). MinIO mất kết nối không làm mất file local; dừng cleanup liên quan, báo degraded và retry. Khi hết dung lượng local, từ chối nhận thêm upload thay vì xác nhận lưu thành công.

Mở lại report/artifact sau Redis expiry cần catalog và kiểm tra quyền bền vững. Resume buổi phỏng vấn đang dở cần checkpoint runtime riêng; không hứa khôi phục chỉ từ một tập object.

### 8.4 Quyền, retention và xóa

Owner ID lấy từ identity server xác thực, không tin owner_id trong body hay session ID như quyền truy cập. Service token hiện tại không thay thế owner authorization; private archive chỉ được bật cho người dùng khi có mapping ownership bền vững và kiểm tra quyền trên list/get/download/delete. Với local studio một người dùng, có owner development cố định được đánh dấu rõ, không quảng bá thành multi-user isolation.

FE tải qua API proxy có authorization trong bản đầu. Nếu bổ sung presigned URL, TTL tối đa 5 phút và phát hành sau kiểm tra owner; không phát URL nội bộ `minio:9000` cho trình duyệt. Không gửi root credential xuống FE.

Default long-term policy mục tiêu: các artifact người dùng đã đăng ký giữ cho đến khi người dùng chủ động xóa, không áp TTL 24 giờ/30 ngày của workspace lên MinIO. Archive bật versioning; không bật Object Lock/WORM mặc định vì xung đột với quyền xóa. Cache, failed temporary uploads và corpus có chính sách riêng, không tự xóa tài liệu private theo cache TTL.

Luồng xóa: ghi tombstone bền → chặn truy cập/ghi mới → hủy jobs → xóa local, Redis, catalog references và tất cả object revisions/versions liên quan → xác nhận hoàn tất. Nếu object delete thất bại, trả trạng thái deletion pending, retry; không báo deleted thành công sớm. Tombstone chặn worker upload trễ tái tạo dữ liệu. Bản backup cũng phải tuân thủ chính sách xóa đã công bố; không coi tombstone là đã xóa mọi backup ngay lập tức.

### 8.5 Compose, vận hành và backup

Thiết kế triển khai thêm service MinIO và init job idempotent để tạo buckets, versioning và application access policy. API internal endpoint `http://minio:9000`; UI quản trị dự kiến loopback host 9001 nếu bản phân phối chọn hỗ trợ console. Chỉ mở loopback cho môi trường local; production dùng TLS và network policy phù hợp. Giữ Redis/Redis Insight/app và hai FE hiện có.

Data dùng named volume riêng `minio_data`; manifest/outbox dùng volume bền vững và được backup. App sử dụng credential giới hạn theo bucket; root credential chỉ cho quản trị/init và qua secret, không hardcode development default trong spec hay log. Init lỗi phải làm archive readiness thất bại rõ ràng. S3 endpoint, bucket, TLS, credential và archive enable flag được cấu hình độc lập.

Volume trên một máy không phải backup. Cần bản backup sang hệ thống độc lập, gồm object, manifest/catalog và archive exports nhất quán; kiểm thử restore trên thư mục/instance trống với checksum và ownership. Mục tiêu ban đầu đề xuất RPO 24 giờ/RTO 4 giờ cho toàn bộ mất host; không được công bố đạt cho tới khi diễn tập đo được. Archive trong cùng host không đảm bảo chống mất đĩa hoặc xóa nhầm volume.

### 8.6 Tình trạng upstream ảnh hưởng triển khai

Kiểm tra ngày 2026-10-13: repo chính thức `minio/minio` hiển thị archived từ 2026-04-25 và README ghi không còn bảo trì; Community phân phối source-only. Nguồn: [repository chính thức](https://github.com/minio/minio), [Docker hướng dẫn trong repo](https://github.com/minio/minio/blob/master/docs/docker/README.md).

Giữ yêu cầu MinIO trong thiết kế, nhưng không chọn `minio/minio:latest` như một default production còn được hỗ trợ. Trước task deployment phải xác nhận distribution, license/credential requirements, registry availability và pin image digest hoặc source commit có provenance. Local evaluation có thể dùng bản legacy pin được kiểm chứng; production cần distribution có maintenance/support phù hợp. Không tự đổi sang provider khác hoặc tự chấp nhận điều khoản thương mại. Interface S3 cho phép thay adapter/backend sau này mà không sửa core interview.

## 9. API, FE và trạng thái

Các route mới dùng prefix `/api/v1`, có auth và ownership:

- Upload tài liệu và xem parse status; xem/sửa profile theo revision.
- Chuẩn bị interview, xem nguồn, vòng, thời lượng và chỉnh trước khi bắt đầu.
- List/download artifact theo owner/session; xem archive và deletion status.
- Review nguồn dùng chung là quyền riêng, không cấp mặc định cho mọi ứng viên.

Không đổi event/state hiện có tùy tiện. Preparation/archive là job state độc lập liên kết session; REPORT_READY không ngụ ý mọi object đã được archive. FE-test hiển thị tiến độ và giới hạn; FE-inspect hiển thị provenance, snapshot/index/plan revisions và archive events đã sanitize. Không hiển thị credentials hoặc PII không cần thiết trong trace.

## 10. Chất lượng và nghiệm thu

Phân biệt mục tiêu nghiệm thu với kết quả đo hiện tại. Các ngưỡng dưới đây là mục tiêu đề xuất cần thực hiện trên tập benchmark có nhãn do người đánh giá:

| Lớp | Cổng nghiệm thu |
| --- | --- |
| Intake | 0 thông tin CV/JD bịa ở bộ ca kiểm tra; 100% factual fields trả ra có evidence/correction provenance; đo extraction precision/recall riêng cho PDF/DOCX/OCR |
| Routing | 0 lỗi âm thầm đổi unknown role thành backend hoặc unknown level thành junior; test đủ Intern đến Senior và bốn vòng |
| Retrieval | 0 fabricated source/unsupported company assertion trong tập gate; human-reviewed Recall@5 ≥ 0.85 và citation support precision ≥ 0.95 trên truy vấn có đáp án |
| Interview | ≥ 90% weighted required criteria được hỏi hoặc có lý do không đánh giá; không coi merely asked là sufficient evidence; ít nhất 90% buổi được người đánh giá chấm phù hợp cấp độ/flow |
| Timing | Kết thúc trong ngân sách + tối đa 10% hoặc 2 phút, lấy giá trị lớn hơn, khi không có pause/provider incident; giữ phần candidate questions |
| Archive | Tất cả artifact đăng ký được tải lại checksum đúng sau app restart và local cleanup; quyền cross-user bị từ chối; retry không nhân bản revisions |
| Deletion | Không còn truy cập/list/download sau tombstone; verified deletion xóa mọi versions và worker trễ không tái tạo |
| Resilience | MinIO/Redis/web outage, crash sau local write/PUT, đầy đĩa, duplicate message, missing index và restore từ backup có test và trạng thái rõ |

Sufficient evidence và chất lượng phỏng vấn cần người có kinh nghiệm IT review theo rubric, không chỉ pass schema hay tự chấm bởi LLM. Bộ test gồm CV/JD có quyền sử dụng, Senior ít thông tin, ứng viên đổi nghề, gap kinh nghiệm, CV mâu thuẫn, source stale/copied/conflicting và công ty không có nguồn. Tách holdout khỏi dữ liệu dùng để chỉnh hệ thống; ghi corpus/model/prompt/index version khi đo. Synthetic cases kiểm tra kỹ thuật, không thay thế benchmark thực tế.

Source không có bằng chứng thì abstain/fallback được coi là đúng; bộ recall không tính câu hỏi thật sự không có đáp án vào denominator. Báo cáo kết quả theo role/level/round và kiểu tài liệu để tổng điểm không che lỗi ở Senior hoặc OCR.

## 11. Các giai đoạn và điều kiện chuyển tiếp

0. **Storage và ownership:** MinIO distribution được kiểm chứng, catalog/outbox, port/adapter, archive/restore/delete, integration với retention và inspector. Hoàn tất restore/authorization trước khi gọi là lưu lâu dài cho nhiều người dùng.
1. **Intake và routing:** upload, parse/OCR, profile correction, JD/FitMatrix; Intern–Senior và không silent fallback. Original/artifact được archive từ đầu.
2. **Nguồn công ty:** provider port, kiểm tra fetch, evidence/dedup/review, versioned snapshot, suy ra vòng/thời lượng và corpus-only fallback.
3. **Phỏng vấn thích ứng:** plan schema, coverage, follow-up, active-time budget, reports grounded; giữ contract hiện tại trong chuyển đổi.
4. **FE và nghiệm thu:** trải nghiệm chuẩn bị, chỉnh hồ sơ/kế hoạch, nguồn/archives, human-reviewed benchmark và diễn tập backup/restore. FE cần thiết cho từng giai đoạn được làm cùng giai đoạn đó; đây là pass hoàn thiện, không trì hoãn UI intake đến cuối.

Mỗi giai đoạn có kế hoạch riêng và kiểm thử cụ thể. Không triển khai toàn bộ qua một patch lớn. Triển khai theo feature flags; phiên cũ giữ schema/plan đã lưu. Backfill archive chỉ upload/copy và đối soát trước; không xóa workspace cũ trong migration. Rollback dừng writer/worker mới nhưng giữ object/catalog và đường đọc tương thích.

## 12. Rà soát bản spec

- Phạm vi IT Intern–Senior, vòng và thời lượng đúng các lựa chọn đã duyệt.
- RAG và core có trách nhiệm riêng; giữ retrieval boundary hiện tại.
- MinIO được mô tả là storage mục tiêu, không khẳng định Compose đã tích hợp.
- Retention local không được âm thầm xóa archive; quyền owner và xóa versions có thiết kế.
- Thời lượng, coverage, bằng chứng và sufficient evidence là các khái niệm riêng.
- Ngưỡng nghiệm thu, budgets và RPO/RTO được đánh dấu mục tiêu đề xuất, không phải kết quả đã xác minh.
- Các quyết định provider/SDK/distribution có task xác minh trước triển khai; không có bước phụ thuộc vào credential chưa được cung cấp.

Sau khi người dùng review spec, dùng skill writing-plans để tạo các kế hoạch theo giai đoạn, bắt đầu storage/ownership; chỉ triển khai sau khi phần thiết kế bổ sung được chốt.

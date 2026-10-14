# FE-test

Giao diện chạy phỏng vấn thật, dùng trực tiếp API của core. Mở `/test/` trên core để tự kết nối cùng origin. `/inspect/` là ứng dụng riêng chỉ đọc trace.

- Tạo phiên bằng tên, vị trí, công ty, kỹ năng, CV và JD do người dùng nhập.
- Core chuẩn bị hồ sơ, bắt đầu phỏng vấn khi READY, rồi nhận câu trả lời từ người dùng qua WebSocket có xác thực.
- Lịch sử phiên và transcript được tải từ API inspector và artifact trên backend. Không có dữ liệu mẫu, candidate persona hoặc transcript dựng sẵn.
- Nếu core chưa dùng provider thật (`real_provider` khác `true`), các thao tác tạo/chạy/trả lời bị khóa; lịch sử vẫn có thể đọc.
- Có thể mở `/test/?session_id=<id>` để tiếp tục phiên đã lưu. Chỉ FE-test mở WebSocket điều khiển; inspector đọc dữ liệu độc lập.
- Câu trả lời chưa được xác nhận có message ID cố định trong localStorage để người dùng chủ động gửi lại cùng ID. Giao diện không tự gửi lại sau reconnect. Dữ liệu này là thông tin khôi phục thao tác; backend là nguồn dữ liệu của lịch sử và transcript.
- ‘Kết nối backend khác’ là tùy chọn nâng cao. Hai ứng dụng chia sẻ `core-studio-base-url` trong localStorage và `core-studio-token` trong sessionStorage của tab.

Không có bước npm build hay thư viện frontend. Core phục vụ HTML, CSS và JavaScript trực tiếp.

## Ngôn ngữ và đọc báo cáo

- Chọn giao diện Tiếng Việt / English ở thanh trên. Mặc định Tiếng Việt; `core-studio-locale` lưu tùy chọn dùng chung với FE-inspect. Đổi giao diện không đổi nội dung hội thoại đã lưu.
- Ngôn ngữ phỏng vấn là lựa chọn riêng trong form tạo phiên (`interview_language`: `vi` hoặc `en`, mặc định `vi`). Core lưu lựa chọn cùng hồ sơ. Phiên cũ không ghi nhận ngôn ngữ được hiển thị rõ, không suy đoán từ giao diện.
- Desktop giữ chiều cao theo viewport: lịch sử, hội thoại và thông tin phiên cuộn độc lập. Mobile có hội thoại và thông tin phiên giới hạn chiều cao để đọc nội dung dài.
- Báo cáo Markdown có tiêu đề, danh sách, bảng, trích dẫn, code và liên kết HTTP/HTTPS/mailto an toàn; HTML đầu vào được escape và không tự tải ảnh. JSON có cây thu gọn. ‘Nội dung gốc’ và sao chép giữ dữ liệu gốc để kiểm tra, kể cả nội dung vượt giới hạn cây JSON (20 cấp / 5.000 node).

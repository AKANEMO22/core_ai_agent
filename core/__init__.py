"""Lõi nghiệp vụ của nền tảng phỏng vấn AI.

Luồng request bắt đầu ở ``api`` rồi gọi các module trong package này: profiler chuẩn bị
hồ sơ và rubric; interview điều khiển từng lượt; orchestration lập directive tiếp theo;
assessor chấm transcript và tổng hợp báo cáo. ``session`` giữ state/event/idempotency,
còn ``workspace`` lưu artifact bền vững. Có thể đọc theo thứ tự đó để lần theo một phiên.
"""

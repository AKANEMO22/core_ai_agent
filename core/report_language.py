"""Report text in the session's interview language (shared by synthesizer and narrative).

decision.json stays in English (canonical, byte-stable audit artifact); only the
rendered report and display fields of final_assessment are localized here.

Văn bản báo cáo theo ngôn ngữ phỏng vấn đã chọn khi tạo phiên. decision.json giữ
tiếng Anh để kiểm tra lại; chỉ phần hiển thị (markdown, disclaimer, limitations) được dịch.
"""

import re
from typing import Dict, List

from core.language import normalize_interview_language

LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "title": "AI Interview Practice & Coaching Report",
        "disclaimer": "Disclaimer",
        "outcome": "1. Outcome",
        "verdict": "Verdict",
        "PASS": "Ready (pass)",
        "FAIL": "Not yet ready (fail)",
        "NEEDS_REVIEW": "Needs human review",
        "weighted_score": "Weighted score",
        "score_detail": "/ 4 (pass threshold {threshold}, borderline ±{band})",
        "human_confirm": "A human reviewer must confirm this outcome.",
        "summary": "2. Summary",
        "criteria": "3. Criteria",
        "criteria_header": "| Criterion | Weight | Must-have | Level (1-4) | Evidence weight (sum of confidence) | Meets bar |",
        "insufficient": "insufficient evidence",
        "yes": "yes",
        "no": "no",
        "findings": "4. Findings by criterion",
        "not_enough": "Not enough evidence was gathered.",
        "closest_level": "Closest level description",
        "strengths": "5. Strengths",
        "improvements": "6. Areas to improve",
        "study_plan": "7. Study plan",
        "claims": "8. CV claims checked",
        "claims_header": "| Claim | Status |",
        "uncovered": "9. Not covered in this interview",
        "delivery": "10. Delivery observations (not scored)",
        "limitations": "11. Limitations",
        "probes": "12. CV/JD probe coverage",
        "probes_note": "Asked means a probe ID was saved on an interviewer turn; it does not mean the claim was independently verified.",
        "asked_probes": "Asked probes",
        "unasked_probes": "Selected but unasked probes",
        "omitted_evidence": "Evidence omitted from this plan",
        "none": "none",
        "early_title": "Session ended early",
        "early_body": "This partial report uses saved evidence only. Outstanding assessments were stopped; human review is required.",
        "saved_transcript": "Saved candidate transcript (may be unassessed)",
    },
    "vi": {
        "title": "Báo cáo luyện tập & góp ý phỏng vấn AI",
        "disclaimer": "Lưu ý",
        "outcome": "1. Kết quả",
        "verdict": "Kết luận",
        "PASS": "Đạt",
        "FAIL": "Chưa đạt",
        "NEEDS_REVIEW": "Cần người duyệt xác nhận",
        "weighted_score": "Điểm có trọng số",
        "score_detail": "/ 4 (ngưỡng đạt {threshold}, vùng ranh giới ±{band})",
        "human_confirm": "Kết quả này cần người duyệt xác nhận.",
        "summary": "2. Tóm tắt",
        "criteria": "3. Tiêu chí",
        "criteria_header": "| Tiêu chí | Trọng số | Bắt buộc | Mức (1-4) | Độ tin cậy bằng chứng (tổng) | Đạt mức |",
        "insufficient": "chưa đủ bằng chứng",
        "yes": "có",
        "no": "không",
        "findings": "4. Nhận xét theo tiêu chí",
        "not_enough": "Chưa thu thập đủ bằng chứng.",
        "closest_level": "Mô tả mức gần nhất",
        "strengths": "5. Điểm mạnh",
        "improvements": "6. Cần cải thiện",
        "study_plan": "7. Kế hoạch ôn luyện",
        "claims": "8. Kiểm chứng thông tin trong CV",
        "claims_header": "| Thông tin | Trạng thái |",
        "uncovered": "9. Chưa được đánh giá trong buổi phỏng vấn",
        "delivery": "10. Quan sát cách trình bày (không chấm điểm)",
        "limitations": "11. Giới hạn",
        "probes": "12. Độ phủ câu hỏi kiểm chứng CV/JD",
        "probes_note": "\"Đã hỏi\" nghĩa là mã câu hỏi được lưu trên lượt của người phỏng vấn; không có nghĩa thông tin đã được xác minh độc lập.",
        "asked_probes": "Đã hỏi",
        "unasked_probes": "Đã chọn nhưng chưa hỏi",
        "omitted_evidence": "Bằng chứng không đưa vào kế hoạch",
        "none": "không có",
        "early_title": "Phiên kết thúc sớm",
        "early_body": "Báo cáo này chỉ dựa trên bằng chứng đã lưu. Các đánh giá còn dở đã bị dừng; cần người duyệt xác nhận.",
        "saved_transcript": "Câu trả lời đã lưu của ứng viên (có thể chưa được đánh giá)",
    },
}

DISCLAIMERS = {
    "en": {
        "practice": (
            "This report is AI-generated coaching feedback for practice and preparation purposes only. "
            "It does not constitute an employment decision, hiring recommendation, qualification guarantee, "
            "or formal assessment."
        ),
        "hiring_assist": (
            "This report is an AI-assisted assessment intended to support, not replace, a human hiring "
            "decision. A qualified reviewer must confirm the outcome before it is used; it is not an "
            "employment decision on its own."
        ),
    },
    "vi": {
        "practice": (
            "Báo cáo này là góp ý do AI tạo ra, chỉ phục vụ luyện tập và chuẩn bị. Báo cáo không phải "
            "quyết định tuyển dụng, khuyến nghị tuyển dụng, bảo đảm năng lực hay đánh giá chính thức."
        ),
        "hiring_assist": (
            "Báo cáo này là đánh giá có hỗ trợ của AI, dùng để hỗ trợ chứ không thay thế quyết định tuyển "
            "dụng của con người. Người duyệt đủ chuyên môn phải xác nhận kết quả trước khi sử dụng; "
            "bản thân báo cáo không phải quyết định tuyển dụng."
        ),
    },
}

BASE_LIMITATIONS = {
    "en": [
        "Scores reflect evidence in this mock interview's transcript only.",
        "Transcripts come from speech recognition and may contain errors.",
        "Delivery metrics (pace, pauses) are descriptive and never scored.",
    ],
    "vi": [
        "Điểm chỉ phản ánh bằng chứng trong transcript của buổi phỏng vấn thử này.",
        "Transcript được tạo từ nhận dạng giọng nói và có thể có lỗi.",
        "Các chỉ số trình bày (tốc độ, khoảng dừng) chỉ mang tính mô tả và không được chấm điểm.",
    ],
}

EARLY_END_LIMITATION = {
    "en": "Session ended early; outstanding assessments were stopped. Unassessed answers are transcript evidence only.",
    "vi": "Phiên kết thúc sớm; các đánh giá còn dở đã bị dừng. Câu trả lời chưa đánh giá chỉ được xem là bằng chứng transcript.",
}

# Reasons are produced in English by decision.py/synthesizer.py; translate their fixed templates.
_VI_REASONS = (
    (re.compile(r"^must-have below pass level: (.*)$"), r"Tiêu chí bắt buộc dưới mức đạt: \1"),
    (re.compile(r"^insufficient evidence for must-have: (.*)$"), r"Thiếu bằng chứng cho tiêu chí bắt buộc: \1"),
    (re.compile(r"^must-have within ±(\S+) of pass level: (.*)$"), r"Tiêu chí bắt buộc nằm trong ±\1 quanh mức đạt: \2"),
    (re.compile(r"^no criterion has enough evidence$"), "Chưa có tiêu chí nào đủ bằng chứng"),
    (re.compile(r"^weighted score (\S+) within ±(\S+) of threshold (\S+)$"), r"Điểm có trọng số \1 nằm trong ±\2 quanh ngưỡng \3"),
    (
        re.compile(r"^covered criteria carry (\S+) of rubric weight \(< (\S+)\); insufficient coverage to pass$"),
        r"Các tiêu chí đã đánh giá chỉ chiếm \1 trọng số rubric (< \2); chưa đủ độ phủ để đạt",
    ),
    (re.compile(r"^weighted score (\S+) ≥ threshold (\S+)$"), r"Điểm có trọng số \1 ≥ ngưỡng \2"),
    (re.compile(r"^weighted score (\S+) < threshold (\S+)$"), r"Điểm có trọng số \1 < ngưỡng \2"),
    (
        re.compile(r"^Session ended early; incomplete evidence requires human review$"),
        "Phiên kết thúc sớm; bằng chứng chưa đầy đủ nên cần người duyệt xác nhận",
    ),
    (
        re.compile(r"^rubric unavailable \(retrieval degraded\): human review required$"),
        "Không có rubric (truy xuất lỗi): cần người duyệt xác nhận",
    ),
)


def labels(language: str) -> Dict[str, str]:
    return LABELS[normalize_interview_language(language)]


def disclaimer(language: str, product_mode: str) -> str:
    key = "hiring_assist" if product_mode == "hiring_assist" else "practice"
    return DISCLAIMERS[normalize_interview_language(language)][key]


def base_limitations(language: str) -> List[str]:
    return list(BASE_LIMITATIONS[normalize_interview_language(language)])


def early_end_limitation(language: str) -> str:
    return EARLY_END_LIMITATION[normalize_interview_language(language)]


def localize_reason(reason: str, language: str) -> str:
    if normalize_interview_language(language) != "vi":
        return reason
    for pattern, replacement in _VI_REASONS:
        if pattern.match(reason):
            return pattern.sub(replacement, reason)
    return reason

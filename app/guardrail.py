"""
guardrail.py

Lớp lọc Guardrail chạy TRƯỚC retrieval:
1. check_emergency: Nhận diện tình huống khẩn cấp ĐANG XẢY RA (113, 114, 115).
2. check_sensitive_info: Cảnh báo khi người dùng thực sự nhập/gửi OTP, mật khẩu, STK, ảnh giấy tờ.
3. guardrail_check: Hàm tổng hợp điều hướng bypass retrieval khi bị trigger.
"""

from __future__ import annotations
import re
from typing import Dict, Any, List, Optional


# Từ khóa hành chính/thủ tục thuần túy (chỉ các từ ĐẶC TRƯNG cho thủ tục hành chính)
ADMIN_KEYWORDS = [
    "thủ tục", "thu tuc",
    "hồ sơ", "ho so",
    "giấy phép", "giay phep",
    "chứng nhận", "chung nhan",
    "đăng ký", "dang ky",
    "mẫu đơn", "mau don",
    "lệ phí", "le phi",
    "xin cấp", "xin cap"
]

# Các cụm HỎI - ĐỊNH NGHĨA / GIẢ ĐỊNH (câu hỏi thông tin)
INFORMATIONAL_MARKERS = [
    "là gì", "la gi",
    "như thế nào", "nhu the nao",
    "quy trình", "quy trinh",
    "ở đâu", "o dau",
    "khi có", "khi co",
    "nếu có", "neu co"
]

# Từ khóa khẩn cấp theo từng nhóm
EMERGENCY_114_KEYWORDS = ["cháy", "chay", "hỏa hoạn", "hoa hoan", "nổ", "no", "mắc kẹt", "mac ket"]
EMERGENCY_113_KEYWORDS = ["cướp", "cuop", "trộm", "trom", "đánh nhau", "danh nhau", "gây rối", "gay roi", "chém", "chem", "giết", "giet", "tấn công", "tan cong", "tai nạn", "tai nan", "đột nhập", "dot nhap", "uy hiếp", "uy hiep"]
EMERGENCY_115_KEYWORDS = ["cấp cứu", "cap cuu", "chảy máu", "chay mau", "bất tỉnh", "bat tinh", "ngưng thở", "ngung tho", "nguy kịch", "nguy kich", "thương nặng", "thuong nang", "bị thương", "bi thuong"]

# Dấu hiệu đang diễn ra/khẩn cấp
URGENT_SIGNALS = ["đang", "dang", "vừa", "vua", "ngay bây giờ", "ngay bay gio", "gấp", "gap", "khẩn cấp", "khan cap", "vừa bị", "vua bi"]

RESCUE_PLEAS = [
    "cứu tôi", "cuu toi", "cứu với", "cuu voi", "xin cứu", "xin cuu",
    "giúp tôi", "giup toi", "giúp với", "giup voi", "xin giúp", "xin giup",
]


def check_emergency(text: str) -> Dict[str, Any]:
    lowered = text.lower().strip()
    has_exclamation = "!" in text
    # Không coi danh từ pháp lý như "cứu nạn, cứu hộ" là lời kêu cứu.
    has_rescue_plea = any(phrase in lowered for phrase in RESCUE_PLEAS)

    # 1. Kiểm tra INFORMATIONAL_MARKERS (Nếu có marker hỏi thông tin VÀ KHÔNG CÓ dấu ! VÀ KHÔNG CÓ lời kêu cứu "cứu/giúp" -> KHÔNG trigger)
    has_info_marker = any(marker in lowered for marker in INFORMATIONAL_MARKERS)
    if has_info_marker and not has_exclamation and not has_rescue_plea:
        return {"triggered": False, "category": None, "message": None}

    # 2. Kiểm tra từ khóa khẩn cấp
    categories = []
    if any(kw in lowered for kw in EMERGENCY_114_KEYWORDS):
        categories.append("114")
    if any(kw in lowered for kw in EMERGENCY_113_KEYWORDS):
        categories.append("113")
    if any(kw in lowered for kw in EMERGENCY_115_KEYWORDS):
        categories.append("115")

    has_urgent_signal = any(sig in lowered for sig in URGENT_SIGNALS) or has_exclamation or has_rescue_plea
    has_admin_keyword = any(admin_kw in lowered for admin_kw in ADMIN_KEYWORDS)

    # NẾU vừa có từ khẩn cấp VỪA có tín hiệu khẩn cấp -> TRIGGER LUÔN (ưu tiên hơn ADMIN_KEYWORDS)
    if categories and has_urgent_signal:
        pass
    # NẾU không có từ khẩn cấp HOẶC không có tín hiệu khẩn cấp HOẶC có từ hành chính -> KHÔNG trigger
    elif not categories or not has_urgent_signal or has_admin_keyword:
        return {"triggered": False, "category": None, "message": None}

    # Sắp xếp danh mục theo thứ tự tiêu chuẩn (113, 114, 115)
    sorted_cats = sorted(categories, key=lambda x: {"113": 1, "114": 2, "115": 3}[x])
    cat_str = ", ".join(sorted_cats)

    if "113" in sorted_cats and "115" in sorted_cats:
        msg = "Anh/chị vui lòng gọi ngay tổng đài khẩn cấp 113 (Công an - an ninh trật tự) và 115 (Cấp cứu y tế) để được hỗ trợ kịp thời."
    elif "113" in sorted_cats and "114" in sorted_cats:
        msg = "Anh/chị vui lòng gọi ngay tổng đài khẩn cấp 113 (Công an) và 114 (PCCC & CNCH) để được hỗ trợ kịp thời."
    elif "114" in sorted_cats and "115" in sorted_cats:
        msg = "Anh/chị vui lòng gọi ngay tổng đài khẩn cấp 114 (PCCC & CNCH) và 115 (Cấp cứu y tế) để được hỗ trợ kịp thời."
    elif "114" in sorted_cats:
        msg = "Anh/chị vui lòng liên hệ ngay tổng đài khẩn cấp 114 đối với tình huống cháy, nổ hoặc cứu nạn cứu hộ đang diễn ra."
    elif "113" in sorted_cats:
        msg = "Anh/chị vui lòng liên hệ ngay tổng đài khẩn cấp 113 đối với tình huống an ninh trật tự hoặc nguy hiểm đang diễn ra để được lực lượng Công an hỗ trợ kịp thời."
    elif "115" in sorted_cats:
        msg = "Anh/chị vui lòng liên hệ ngay tổng đài khẩn cấp 115 để được hỗ trợ cấp cứu y tế kịp thời."
    else:
        msg = "Anh/chị vui lòng gọi tổng đài khẩn cấp 113/114/115 để được hỗ trợ kịp thời."

    return {
        "triggered": True,
        "category": cat_str,
        "message": msg
    }


def check_sensitive_info(text: str) -> Dict[str, Any]:
    lowered = text.lower().strip()

    # 1. Các cụm hỏi tư vấn / cảnh giác lừa đảo
    INQUIRY_INDICATORS = ["có nên", "co nen", "yêu cầu tôi đọc", "yeu cau toi doc", "gọi tự xưng", "goi tu xung", "lừa đảo", "lua dao", "cảnh giác", "canh giac"]
    has_inquiry = any(ind in lowered for ind in INQUIRY_INDICATORS)

    # 2. Pattern chia sẻ OTP thực sự (hỗ trợ cả chữ + số vd: AB12CD)
    otp_share = bool(re.search(r"(mã\s*otp|otp|mã\s*xác\s*thực)[^,.:]*?(là|=|:|\s+là)\s*([0-9a-zA-Z]{4,8})", lowered))

    # 3. Pattern chia sẻ Mật khẩu thực sự
    pass_matches = re.findall(r"(mật\s*khẩu|mat\s*khau|password|pass)[^,.:]*?(hình\s*như)?\s*(là|=|:|\s+là)\s*([a-zA-Z0-9]{4,16})", lowered)
    pass_share = False
    for match in pass_matches:
        secret_val = match[-1]
        # Chỉ loại trừ nếu secret_val này đứng ngay sau "mã hồ sơ", "mã lỗi", "số hồ sơ"
        if not re.search(rf"(mã\s*hồ\s*sơ|số\s*hồ\s*sơ|mã\s*lỗi)\s*(của\s*tôi)?\s*(là|=|:)\s*{re.escape(secret_val)}", lowered):
            pass_share = True
            break

    # 4. Pattern chia sẻ STK / Tài khoản ngân hàng thực sự
    stk_matches = re.findall(r"(số\s*tài\s*khoản|stk|tài\s*khoản)[^,.:]*?(là|=|:|\s+là)\s*([0-9]{6,})", lowered)
    stk_share = False
    for match in stk_matches:
        stk_val = match[-1]
        if not re.search(rf"(mã\s*hồ\s*sơ|số\s*hồ\s*sơ|mã\s*lỗi)\s*(của\s*tôi)?\s*(là|=|:)\s*{re.escape(stk_val)}", lowered):
            stk_share = True
            break

    # 5. Chia sẻ ảnh giấy tờ
    IMAGE_KEYWORDS = ["ảnh cccd", "anh cccd", "ảnh cmnd", "anh cmnd", "ảnh giấy tờ", "anh giay to", "mặt trước cccd", "mat truoc cccd", "gửi ảnh", "gui anh"]
    img_share = any(kw in lowered for kw in IMAGE_KEYWORDS)

    # Nếu là câu hỏi tư vấn lừa đảo và KHÔNG có pattern chia sẻ bí mật thực sự -> KHÔNG trigger
    if has_inquiry and not (otp_share or pass_share or stk_share or img_share):
        return {"triggered": False, "message": None}

    if otp_share or pass_share or stk_share or img_share:
        msg = "Anh/chị tuyệt đối không cung cấp mật khẩu, mã OTP, mã xác thực, thông tin tài khoản ngân hàng hoặc ảnh giấy tờ tùy thân qua Chatbot để bảo đảm an toàn thông tin cá nhân."
        return {
            "triggered": True,
            "message": msg
        }

    return {"triggered": False, "message": None}


def guardrail_check(text: str) -> Dict[str, Any]:
    emerg_res = check_emergency(text)
    if emerg_res["triggered"]:
        return {
            "triggered": True,
            "category": emerg_res["category"],
            "message": emerg_res["message"],
            "bypass_retrieval": True
        }

    sens_res = check_sensitive_info(text)
    if sens_res["triggered"]:
        return {
            "triggered": True,
            "category": "sensitive_info",
            "message": sens_res["message"],
            "bypass_retrieval": True
        }

    return {
        "triggered": False,
        "bypass_retrieval": False
    }

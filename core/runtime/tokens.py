"""Token estimation for context budgeting without a provider-specific tokenizer dependency."""

import json
import math
import re
from typing import Any, Dict, Iterable, Optional

# Characters of Latin script with Vietnamese diacritics (combining and precomposed ranges).
_VIETNAMESE_MARKS = re.compile(
    r"[ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]"
)
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


class TokenCounter:
    """Estimates tokens per text, adapting the characters-per-token ratio to the script.

    Uses `tiktoken` when installed (exact for OpenAI models); otherwise a heuristic:
    English ~4.0 chars/token, Vietnamese ~3.0, CJK ~1.5. Estimates err on the high side so
    budgets are conservative.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self._encoder = None
        try:  # optional dependency
            import tiktoken  # type: ignore

            try:
                self._encoder = tiktoken.encoding_for_model(model or "gpt-4o")
            except Exception:
                self._encoder = tiktoken.get_encoding("o200k_base")
        except Exception:
            self._encoder = None

    @staticmethod
    def chars_per_token(text: str) -> float:
        """Heuristic characters-per-token ratio for the dominant script of `text`."""
        if not text:
            return 4.0
        sample = text[:4000]
        if len(_CJK.findall(sample)) > len(sample) * 0.2:
            return 1.5
        if len(_VIETNAMESE_MARKS.findall(sample)) > len(sample) * 0.03:
            return 3.0
        return 4.0

    def count(self, text: Optional[str]) -> int:
        """Token estimate for one string."""
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return max(1, math.ceil(len(text) / self.chars_per_token(text)))

    def count_message(self, message: Dict[str, Any]) -> int:
        """Tokens for one chat message including role and tool-call overhead."""
        total = 4  # per-message framing overhead
        content = message.get("content")
        if isinstance(content, str):
            total += self.count(content)
        elif content is not None:
            total += self.count(json.dumps(content, default=str))
        for call in message.get("tool_calls") or []:
            total += self.count(json.dumps(call, default=str))
        return total

    def count_messages(self, messages: Iterable[Dict[str, Any]]) -> int:
        """Tokens for a message list."""
        return sum(self.count_message(m) for m in messages) + 2

"""Verbatim quote verification shared by evaluator, claim checker and narrative."""

import re
import unicodedata


def normalize_for_match(text: str) -> str:
    """Case/accent/punctuation/whitespace folding used for quote verification."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


# Spoken fillers. Evaluators are told to ignore them, so quotes often omit them; they are removed
# from both the quote and the transcript before matching (symmetric, so no content is invented).
_FILLERS = re.compile(r"\b(?:um+|uh+|uhm|erm|er|ah|hmm+|like|you know|i mean|kind of|sort of)\b")


def _strip_fillers(normalized: str) -> str:
    return re.sub(r"\s+", " ", _FILLERS.sub(" ", normalized)).strip()


_ELLIPSIS = re.compile(r"\s*(?:\.{3,}|…|\[\.\.\.\]|\[…\])\s*")


def verify_quote(quote: str, source: str, min_words: int = 3) -> bool:
    """True if the normalized quote (at least `min_words` words) occurs in the normalized source.

    A quote shortened with an ellipsis ("I owned ... the migration") is accepted when every
    segment occurs verbatim, in order, in the source, and each segment is at least two words, so an
    elided quote still cannot stitch together isolated words.
    """
    segments = [_strip_fillers(normalize_for_match(s)) for s in _ELLIPSIS.split(quote or "")]
    segments = [s for s in segments if s]
    if sum(len(s.split()) for s in segments) < min_words:
        return False
    haystack = _strip_fillers(normalize_for_match(source))
    if len(segments) == 1:
        return segments[0] in haystack
    position = 0
    for segment in segments:
        if len(segment.split()) < 2:
            return False
        found = haystack.find(segment, position)
        if found < 0:
            return False
        position = found + len(segment)
    return True

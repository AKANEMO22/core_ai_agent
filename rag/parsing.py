"""Conservative PDF/text intake: extracted facts retain evidence spans."""

from pathlib import Path

from .indexing import EXTRA_CONCEPTS, digest
from .ontology import canonical, mentions


def parse_document(path: str | Path, kind: str):
    if kind not in {"cv", "jd"}:
        raise ValueError("kind must be cv or jd")
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        pages = [page.extract_text() or "" for page in PdfReader(path).pages]
        if not any(text.strip() for text in pages):
            raise ValueError("Scanned PDF has no text layer; OCR is required")
    elif path.suffix.lower() in {".txt", ".md"}:
        pages = [path.read_text(encoding="utf-8")]
    else:
        raise ValueError("Supported input: PDF with text layer, UTF-8 .txt or .md")
    labels = {canonical(s) for s in EXTRA_CONCEPTS}
    evidence = []
    for number, text in enumerate(pages, 1):
        for line_number, line in enumerate(text.splitlines(), 1):
            for label in sorted(labels):
                if mentions(line, label):
                    evidence.append(
                        {
                            "skill": label,
                            "page": number,
                            "line": line_number,
                            "excerpt": line[:500],
                            "status": "claimed"
                            if kind == "cv"
                            else "mentioned_requirement",
                        }
                    )
    # Do not invent project ownership, proficiency, required/preferred weighting or contact fields.
    return {
        "kind": kind,
        "source_locator": path.name,
        "content_hash": digest(pages),
        "raw_text": "\n\n".join(pages),
        "skills": sorted({e["skill"] for e in evidence}),
        "evidence": evidence,
        "review_required": True,
        "limitations": [
            "Rule-based mention extraction; must-have vs preferred and project ownership require review",
            "Scanned PDFs require external OCR",
        ],
    }

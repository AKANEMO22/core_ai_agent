"""Bounded text extraction for candidate-provided CV and job-description files."""

from __future__ import annotations

from io import BytesIO
from pathlib import PurePath
import re
import zipfile
import xml.etree.ElementTree as ET


MAX_DOCUMENT_BYTES = 12 * 1024 * 1024
MAX_EXTRACTED_CHARS = 180_000
MAX_PDF_PAGES = 120
MAX_DOCX_MEMBERS = 300
MAX_DOCX_UNCOMPRESSED_BYTES = 30 * 1024 * 1024


class DocumentExtractionError(ValueError):
    """The uploaded file is unsupported, unsafe, or contains no extractable text."""


def extract_document_text(filename: str, content: bytes) -> tuple[str, str]:
    """Return a safe extension and bounded text for a supported CV/JD document.

    Original filenames are never used as workspace paths. Scanned PDFs need OCR and are
    rejected with an explicit message instead of being decoded as arbitrary binary text.
    """
    if not content:
        raise DocumentExtractionError("The uploaded file is empty")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise DocumentExtractionError("The uploaded file exceeds the 12 MB limit")
    suffix = PurePath(filename or "").suffix.casefold()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise DocumentExtractionError("Upload a PDF, DOCX, or TXT file")

    if suffix == ".pdf":
        text = _extract_pdf(content)
    elif suffix == ".docx":
        text = _extract_docx(content)
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentExtractionError("TXT files must use UTF-8 encoding") from exc
        if "\x00" in text:
            raise DocumentExtractionError("The TXT file contains binary data")

    normalized = _normalize_text(text)
    if not normalized:
        raise DocumentExtractionError(
            "No readable text was found. Scanned PDFs require OCR, which is not configured yet."
        )
    return suffix[1:], normalized[:MAX_EXTRACTED_CHARS]


def _extract_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content), strict=False)
        if reader.is_encrypted:
            raise DocumentExtractionError("Password-protected PDFs are not supported")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise DocumentExtractionError(f"PDFs are limited to {MAX_PDF_PAGES} pages")
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except DocumentExtractionError:
        raise
    except Exception as exc:
        raise DocumentExtractionError("The PDF could not be read") from exc


def _extract_docx(content: bytes) -> str:
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            members = archive.infolist()
            if len(members) > MAX_DOCX_MEMBERS:
                raise DocumentExtractionError("The DOCX file contains too many components")
            if sum(item.file_size for item in members) > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise DocumentExtractionError("The DOCX expands beyond the supported size")
            document = archive.read("word/document.xml")
        root = ET.fromstring(document)
        namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paragraphs = []
        for paragraph in root.iter(f"{namespace}p"):
            value = "".join(node.text or "" for node in paragraph.iter(f"{namespace}t"))
            if value.strip():
                paragraphs.append(value)
        return "\n".join(paragraphs)
    except DocumentExtractionError:
        raise
    except (KeyError, OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        raise DocumentExtractionError("The DOCX file could not be read") from exc


def _normalize_text(text: str) -> str:
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[\t\f\v ]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()

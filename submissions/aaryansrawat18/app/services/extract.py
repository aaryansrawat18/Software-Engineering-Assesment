"""Turn an uploaded file into page text.

A PDF becomes one entry per page. A text or Markdown file is a single page
numbered 1. Failures raise ExtractFailed. Callers store the short code
`extract_failed`, not the exception text, so file contents never land in the database.
"""

from io import BytesIO

from pypdf import PdfReader

PDF_MEDIA_TYPE = "application/pdf"
TEXT_MEDIA_TYPE = "text/plain"
MARKDOWN_MEDIA_TYPE = "text/markdown"
PDF_MAGIC = b"%PDF"


class ExtractFailed(Exception):
    """The file could not be read as pages. Do not put the message in the database."""


def choose_media_type(file_bytes: bytes, filename: str, content_type: str | None) -> str:
    """Decide if this upload is a PDF or UTF-8 text.

    A PDF is accepted when the browser says `application/pdf` or the bytes
    start with `%PDF`. Anything else must be valid UTF-8. Markdown is UTF-8
    whose filename ends in `.md`.
    """
    normalized_type = (content_type or "").split(";")[0].strip().lower()
    if normalized_type == PDF_MEDIA_TYPE or file_bytes.startswith(PDF_MAGIC):
        return PDF_MEDIA_TYPE
    try:
        file_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractFailed("not utf-8") from exc
    if filename.lower().endswith(".md"):
        return MARKDOWN_MEDIA_TYPE
    return TEXT_MEDIA_TYPE


def read_pages(file_bytes: bytes, media_type: str) -> list[dict]:
    """Return `[{page, text}, ...]` for a file we already accepted.

    Empty pages are skipped. If nothing readable is left, the result is an
    empty list and the caller treats that as extract_failed.
    """
    if media_type == PDF_MEDIA_TYPE or file_bytes.startswith(PDF_MAGIC):
        return _read_pdf_pages(file_bytes)
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractFailed("not utf-8") from exc
    if not text.strip():
        return []
    return [{"page": 1, "text": text}]


def _read_pdf_pages(file_bytes: bytes) -> list[dict]:
    """Read each PDF page with pypdf. A broken file raises ExtractFailed."""
    try:
        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": page_number, "text": text})
        return pages
    except Exception as exc:
        raise ExtractFailed("pdf") from exc

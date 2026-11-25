"""Document text extraction for PDF / DOCX / TXT with an alternate (fallback) strategy.

The *primary* strategy uses layout-aware extraction; the *alternate* strategy is used by the
Error Recovery Agent when the primary one fails or yields unusable text.
"""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document

from app.schemas.candidate import DocumentDiagnostics

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS = 80
SUPPORTED_TYPES = {".pdf": "pdf", ".docx": "docx", ".txt": "txt"}
_BULLETS = re.compile(r"^[\s]*[•●▪◦■□➢➤►✓✔\uf0b7\uf0a7*·]+\s*", re.MULTILINE)
_LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\u2013": "–", "\u00a0": " ", "\u200b": ""}


class DocumentParsingError(Exception):
    """Raised with a user-facing message when a document cannot be read."""


class UnsupportedFileTypeError(DocumentParsingError):
    pass


class ScannedDocumentError(DocumentParsingError):
    pass


class EmptyDocumentError(DocumentParsingError):
    pass


@dataclass
class LoadedDocument:
    text: str
    diagnostics: DocumentDiagnostics


def normalise_text(text: str) -> str:
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = _BULLETS.sub("• ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def load_document(path: Path, strategy: str = "primary") -> LoadedDocument:
    suffix = path.suffix.lower()
    file_type = SUPPORTED_TYPES.get(suffix)
    if file_type is None:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Please upload a PDF, DOCX or TXT file."
        )
    loaders = {"pdf": _load_pdf, "docx": _load_docx, "txt": _load_txt}
    doc = loaders[file_type](path, strategy)
    doc.text = normalise_text(doc.text)
    doc.diagnostics.char_count = len(doc.text)
    doc.diagnostics.word_count = len(doc.text.split())
    _validate(doc)
    return doc


def _validate(doc: LoadedDocument) -> None:
    diag = doc.diagnostics
    min_chars = 15 if diag.file_type == "txt" else MIN_TEXT_CHARS
    if diag.char_count < min_chars:
        if diag.image_count > 0:
            diag.is_scanned = True
            raise ScannedDocumentError(
                "This appears to be a scanned/image-based document with no selectable text. "
                "OCR is not enabled - please upload a text-based PDF or DOCX resume."
            )
        raise EmptyDocumentError("The document contains no readable text.")


def _load_pdf(path: Path, strategy: str) -> LoadedDocument:
    try:
        pdf = fitz.open(path)
    except Exception as exc:  # PyMuPDF raises several error types for corrupt files
        raise DocumentParsingError("The PDF file is corrupt or cannot be opened.") from exc
    with pdf:
        if pdf.needs_pass:
            raise DocumentParsingError("The PDF is password protected.")
        pages: list[str] = []
        image_count = table_count = 0
        for page in pdf:
            image_count += len(page.get_images(full=True))
            if strategy == "primary":
                pages.append(page.get_text("text", sort=True))
                table_count += _count_tables(page)
            else:  # alternate: rebuild text from positioned blocks
                blocks = sorted(page.get_text("blocks"), key=lambda b: (round(b[1]), b[0]))
                pages.append("\n".join(str(b[4]) for b in blocks if b[6] == 0))
        diag = DocumentDiagnostics(
            file_type="pdf",
            parser_used=f"PyMuPDF ({'layout text' if strategy == 'primary' else 'block reconstruction'})",
            page_count=pdf.page_count,
            image_count=image_count,
            table_count=table_count,
        )
    return LoadedDocument("\n".join(pages), diag)


def _count_tables(page: fitz.Page) -> int:
    try:
        return len(page.find_tables().tables)
    except Exception:  # table detection is best-effort diagnostics only
        return 0


def _load_docx(path: Path, strategy: str) -> LoadedDocument:
    if strategy == "primary":
        try:
            document = Document(str(path))
        except Exception as exc:
            raise DocumentParsingError("The DOCX file is corrupt or cannot be opened.") from exc
        parts = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        diag = DocumentDiagnostics(
            file_type="docx",
            parser_used="python-docx",
            page_count=1,
            image_count=len(document.inline_shapes),
            table_count=len(document.tables),
        )
        return LoadedDocument("\n".join(parts), diag)
    try:  # alternate: read raw WordprocessingML
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise DocumentParsingError("The DOCX file is corrupt or cannot be opened.") from exc
    xml = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    diag = DocumentDiagnostics(file_type="docx", parser_used="raw OOXML reader", page_count=1)
    return LoadedDocument(text, diag)


def _load_txt(path: Path, strategy: str) -> LoadedDocument:
    raw = path.read_bytes()
    encoding = "utf-8" if strategy == "primary" else "latin-1"
    text = raw.decode(encoding, errors="replace")
    diag = DocumentDiagnostics(file_type="txt", parser_used=f"plain text ({encoding})", page_count=1)
    return LoadedDocument(text, diag)

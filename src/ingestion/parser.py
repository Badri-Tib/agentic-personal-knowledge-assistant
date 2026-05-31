"""
parser.py — Document parsing for the Personal Knowledge Assistant.

Handles:
- Native PDFs via pymupdf (fitz): preserves layout, extracts text per page
- Scanned PDFs via pytesseract (OCR): detects when a page has no selectable text
- Images (.png, .jpg, .jpeg): direct OCR

Each parsed document returns a list of ParsedPage objects with:
  - page_num, text, source_file, doc_type, is_ocr
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional heavy imports — guarded so the module can be imported even if a dep
# is missing (useful during development / partial installs).
try:
    import fitz  # pymupdf
except ImportError:
    fitz = None  # type: ignore

try:
    import pytesseract
    from PIL import Image
    import io
except ImportError:
    pytesseract = None  # type: ignore
    Image = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedPage:
    """Represents one page of extracted text from a document."""
    source_file: str          # Relative or absolute path of the source file
    page_num: int             # 1-indexed page number (or 1 for single-page images)
    text: str                 # Extracted/OCR'd text
    doc_type: str             # e.g. "cv", "transcript", "contract", "unknown"
    is_ocr: bool = False      # True when text was obtained via OCR
    metadata: dict = field(default_factory=dict)  # Extra metadata (future use)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

# Minimum character count to consider a page as "having native text".
# Below this threshold we fall back to OCR.
_MIN_NATIVE_TEXT_LEN = 30

# DPI used when rendering a PDF page to a raster image for OCR.
_OCR_DPI = 200


def _ocr_pil_image(image: "Image.Image") -> str:
    """Run Tesseract OCR on a PIL image and return the extracted text."""
    if pytesseract is None:
        raise ImportError(
            "pytesseract is not installed. "
            "Run: pip install pytesseract pillow"
        )
    return pytesseract.image_to_string(image, lang="fra+eng")


def _pdf_page_to_pil(page: "fitz.Page", dpi: int = _OCR_DPI) -> "Image.Image":
    """Render a pymupdf page to a PIL Image for OCR."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))


def _infer_doc_type(path: Path) -> str:
    """
    Heuristic: guess document type from the filename.
    Extend this mapping as needed.
    """
    name = path.stem.lower()
    mapping = {
        "cv": "cv",
        "resume": "cv",
        "transcript": "transcript",
        "releve": "transcript",
        "notes": "transcript",
        "contract": "contract",
        "contrat": "contract",
        "attestation": "attestation",
        "titre": "titre_sejour",
        "planning": "planning",
        "schedule": "planning",
    }
    for keyword, doc_type in mapping.items():
        if keyword in name:
            return doc_type
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def parse_pdf(path: Path) -> list[ParsedPage]:
    """
    Parse a PDF file. For each page:
      1. Try native text extraction (pymupdf).
      2. Fall back to OCR if native text is too sparse.

    Returns a list of ParsedPage (one per page).
    """
    if fitz is None:
        raise ImportError("pymupdf is not installed. Run: pip install pymupdf")

    path = Path(path)
    doc_type = _infer_doc_type(path)
    pages: list[ParsedPage] = []

    doc = fitz.open(str(path))
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            native_text = page.get_text("text").strip()

            if len(native_text) >= _MIN_NATIVE_TEXT_LEN:
                # Native text is good — use it directly.
                pages.append(ParsedPage(
                    source_file=str(path),
                    page_num=page_index + 1,
                    text=native_text,
                    doc_type=doc_type,
                    is_ocr=False,
                ))
                logger.debug("Page %d: native text (%d chars)", page_index + 1, len(native_text))
            else:
                # Scanned / image-only page — fall back to OCR.
                logger.debug("Page %d: sparse native text, switching to OCR", page_index + 1)
                pil_img = _pdf_page_to_pil(page)
                ocr_text = _ocr_pil_image(pil_img).strip()
                pages.append(ParsedPage(
                    source_file=str(path),
                    page_num=page_index + 1,
                    text=ocr_text,
                    doc_type=doc_type,
                    is_ocr=True,
                ))
                logger.debug("Page %d: OCR result (%d chars)", page_index + 1, len(ocr_text))
    finally:
        doc.close()

    return pages


def parse_image(path: Path) -> list[ParsedPage]:
    """
    Parse a single image file (.png, .jpg, .jpeg) with OCR.
    Returns a list with a single ParsedPage.
    """
    if pytesseract is None or Image is None:
        raise ImportError(
            "pytesseract / pillow not installed. "
            "Run: pip install pytesseract pillow"
        )

    path = Path(path)
    doc_type = _infer_doc_type(path)

    img = Image.open(str(path))
    text = _ocr_pil_image(img).strip()

    return [ParsedPage(
        source_file=str(path),
        page_num=1,
        text=text,
        doc_type=doc_type,
        is_ocr=True,
    )]


SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


def parse_document(path: Path | str) -> list[ParsedPage]:
    """
    Entry point: dispatch to the right parser based on file extension.

    Args:
        path: Path to the document.

    Returns:
        List of ParsedPage objects.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )

    logger.info("Parsing %s (type: %s)", path.name, ext)

    if ext == ".pdf":
        return parse_pdf(path)
    else:
        return parse_image(path)


def parse_directory(directory: Path | str) -> list[ParsedPage]:
    """
    Parse all supported documents in a directory (non-recursive).

    Returns all ParsedPage objects from all documents, in file-alphabetical order.
    """
    directory = Path(directory)
    all_pages: list[ParsedPage] = []

    files = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        logger.warning("No supported documents found in %s", directory)
        return []

    for file in files:
        try:
            pages = parse_document(file)
            all_pages.extend(pages)
            logger.info("  ✓ %s — %d page(s)", file.name, len(pages))
        except Exception as e:
            logger.error("  ✗ %s — %s", file.name, e)

    return all_pages

"""
PDF and document text extraction pipeline.
Supports both text-layer extraction (PyPDF) and multimodal OCR fallback.
"""

import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Hybrid PDF processor:
    1. Attempts text-layer extraction with PyPDF
    2. Falls back to multimodal OCR if text is insufficient
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key

    def extract_text(self, file_path: str) -> str:
        """
        Primary extraction: attempt PyPDF text-layer extraction.
        Returns extracted text or empty string if failed.
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf_text(str(file_path))
        elif suffix in {".html", ".htm"}:
            return self._extract_html_text(str(file_path))
        elif suffix == ".txt":
            return file_path.read_text(encoding="utf-8", errors="ignore")
        else:
            logger.warning(f"Unknown file type: {suffix}")
            return ""

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF using PyPDF."""
        try:
            import pypdf
            text_parts = []
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                logger.info(f"PDF has {len(reader.pages)} pages")
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    text_parts.append(f"--- Page {i+1} ---\n{page_text}")
            full_text = "\n".join(text_parts)
            logger.info(f"PyPDF extracted {len(full_text)} characters")
            return full_text
        except Exception as e:
            logger.error(f"PyPDF extraction failed: {e}")
            return ""

    def _extract_html_text(self, file_path: str) -> str:
        """Extract text from HTML using BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
            soup = BeautifulSoup(html, "html.parser")
            # Remove script and style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            logger.info(f"BeautifulSoup extracted {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"HTML extraction failed: {e}")
            return ""

    def ocr_extract(self, file_path: str) -> str:
        """
        OCR fallback: Convert PDF pages to images and use multimodal LLM.
        """
        file_path = Path(file_path)
        if file_path.suffix.lower() != ".pdf":
            logger.warning(f"OCR only supports PDFs, got: {file_path.suffix}")
            return self.extract_text(str(file_path))

        self._log_ocr_start(file_path)

        try:
            images = self._convert_pdf_to_images(str(file_path))
            if not images:
                logger.warning("No images produced from PDF")
                return ""

            all_text = []
            for i, image_data in enumerate(images):
                logger.info(f"OCR processing page {i+1}/{len(images)}")
                page_text = self._ocr_image(image_data, page_num=i + 1)
                all_text.append(f"--- Page {i+1} ---\n{page_text}")

            combined = "\n".join(all_text)
            logger.info(f"OCR extracted {len(combined)} characters total")
            return combined

        except Exception as e:
            logger.error(f"OCR pipeline failed: {e}")
            return ""

    def _log_ocr_start(self, file_path: Path):
        logger.info(f"Starting OCR pipeline for: {file_path.name}")

    def _convert_pdf_to_images(self, pdf_path: str) -> list[bytes]:
        """Convert PDF pages to PNG images using pdf2image."""
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(pdf_path, dpi=150, fmt="png")
            images = []
            for page in pages:
                buf = io.BytesIO()
                page.save(buf, format="PNG")
                images.append(buf.getvalue())
            logger.info(f"Converted {len(images)} PDF pages to images")
            return images
        except ImportError:
            logger.warning("pdf2image not available, trying PyMuPDF fallback")
            return self._convert_pdf_via_pymupdf(pdf_path)
        except Exception as e:
            logger.error(f"PDF to image conversion failed: {e}")
            return []

    def _convert_pdf_via_pymupdf(self, pdf_path: str) -> list[bytes]:
        """Fallback: convert PDF to images using PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            images = []
            for page in doc:
                mat = fitz.Matrix(1.5, 1.5)  # 1.5x zoom
                pix = page.get_pixmap(matrix=mat)
                images.append(pix.tobytes("png"))
            logger.info(f"PyMuPDF converted {len(images)} pages")
            return images
        except Exception as e:
            logger.error(f"PyMuPDF conversion failed: {e}")
            return []

    def _ocr_image(self, image_bytes: bytes, page_num: int) -> str:
        """Send a page image to OpenAI's vision API for OCR."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            b64 = base64.b64encode(image_bytes).decode("utf-8")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "You are an OCR system. Extract ALL text from this document page image. "
                                    "Do not summarize or paraphrase. Preserve all headings, section numbers, "
                                    "bullet points, and table content exactly as they appear. "
                                    "Output only the extracted text with no commentary."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"}
                            }
                        ]
                    }
                ]
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Vision OCR failed for page {page_num}: {e}")
            return f"[OCR failed for page {page_num}: {e}]"

    def process_uploaded_file(self, file_bytes: bytes, filename: str, save_dir: str = "data/raw_documents") -> str:
        """
        Process an uploaded file from the Streamlit UI.
        Saves the file locally and returns the local path.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        local_path = save_path / filename
        local_path.write_bytes(file_bytes)
        logger.info(f"Saved uploaded file: {local_path}")
        return str(local_path)

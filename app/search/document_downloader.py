"""
Document downloader: fetches PDFs and HTML pages from the web.
Saves files locally in data/raw_documents/.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

DOWNLOAD_DIR = Path("data/raw_documents")
TIMEOUT = 30
MAX_SIZE_BYTES = 50 * 1024 * 1024  # 50MB max


class DocumentDownloader:
    """
    Downloads regulatory documents from the web.
    Supports PDFs and HTML pages.
    """

    def __init__(self, download_dir: Path = DOWNLOAD_DIR):
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; RegulatoryConsultant/1.0)",
            "Accept": "text/html,application/pdf,*/*"
        })

    def download(self, url: str, framework_name: str) -> Optional[str]:
        """
        Download a document from a URL and save it locally.

        Returns:
            Local file path if successful, None otherwise
        """
        if not self._is_valid_url(url):
            logger.warning(f"Invalid URL: {url}")
            return None

        # Check if already downloaded
        existing = self._find_existing(url)
        if existing:
            logger.info(f"File already downloaded: {existing}")
            return existing

        try:
            logger.info(f"Downloading: {url}")
            response = self.session.get(url, timeout=TIMEOUT, stream=True, allow_redirects=True)
            response.raise_for_status()

            # Check size
            content_length = int(response.headers.get("Content-Length", 0))
            if content_length > MAX_SIZE_BYTES:
                logger.warning(f"File too large ({content_length} bytes), skipping: {url}")
                return None

            # Determine file type
            content_type = response.headers.get("Content-Type", "")
            extension = self._get_extension(url, content_type)

            # Generate filename
            safe_name = self._safe_filename(framework_name)
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"{safe_name}_{url_hash}{extension}"
            local_path = self.download_dir / filename

            # Save file
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = local_path.stat().st_size
            logger.info(f"Downloaded {file_size} bytes -> {local_path}")
            return str(local_path)

        except requests.RequestException as e:
            logger.error(f"Download failed for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return None

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            parsed = urlparse(url)
            return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
        except Exception:
            return False

    def _find_existing(self, url: str) -> Optional[str]:
        """Check if file was already downloaded."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        for file in self.download_dir.iterdir():
            if url_hash in file.name:
                return str(file)
        return None

    def _get_extension(self, url: str, content_type: str) -> str:
        """Determine file extension from URL and content type."""
        url_lower = url.lower()
        if ".pdf" in url_lower or "pdf" in content_type:
            return ".pdf"
        elif ".htm" in url_lower or "html" in content_type:
            return ".html"
        elif ".txt" in url_lower or "text/plain" in content_type:
            return ".txt"
        else:
            return ".html"

    def _safe_filename(self, name: str) -> str:
        """Convert a framework name to a safe filename."""
        safe = "".join(c if c.isalnum() else "_" for c in name)
        return safe.lower()[:30]

    def fetch_html_content(self, url: str) -> Optional[str]:
        """Fetch and return HTML content as text."""
        try:
            response = self.session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch HTML: {e}")
            return None

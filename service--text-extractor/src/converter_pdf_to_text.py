from io import BytesIO

from loguru import logger
from pypdf import PdfReader


class PdfToTextConverter:
    def __call__(self, doc_bytes: bytes) -> str:
        reader = PdfReader(BytesIO(doc_bytes))
        logger.info(f"Start pypdf, {len(reader.pages)} pages")
        pages = [page.extract_text() for page in reader.pages]
        txt = "\n".join(pages)
        logger.info(f"PDF len text = {len(txt)}")
        return txt

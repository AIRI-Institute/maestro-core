from loguru import logger
from mmar_mapi import FileStorage
from mmar_mapi.api import ResourceId, TextExtractorAPI

from src.config import Config
from src.converter_image_to_text import ImageToTextConverter
from src.converter_pdf_to_text import PdfToTextConverter


class TextExtractor(TextExtractorAPI):
    def __init__(self, config: Config):
        self.file_storage = FileStorage(config.files_dir)
        self.pdf_to_text = PdfToTextConverter()
        self.image_to_text = ImageToTextConverter()

    def extract(self, *, resource_id: ResourceId) -> ResourceId:
        logger.info(f"Received request with resource_id={resource_id}")

        doc_bytes = self.file_storage.download(resource_id)
        doc_type = resource_id.split(".")[-1].lower()

        text = self._get_text(doc_bytes, doc_type)
        ext_resource_id = self.file_storage.upload(content=text, fname="text.txt")
        return ext_resource_id

    def _get_text(self, doc_bytes: bytes, doc_type: str) -> str:
        match doc_type:
            case "pdf":
                return self.pdf_to_text(doc_bytes)
            case "jpg" | "png" | "jpeg":
                return self.image_to_text(doc_bytes)
            case _:
                logger.warning(f"Unprocessable doc type is {doc_type=}")
                return ""

from loguru import logger
from mmar_mapi import FileStorage
from mmar_mapi.services import ResourceId, TextExtractorAPI

from text_extractor.converter_image_to_text import ImageToTextConverter
from text_extractor.converter_pdf_to_text import PdfToTextConverter


class TextExtractor(TextExtractorAPI):
    def __init__(
        self,
        file_storage: FileStorage,
        pdf_to_text: PdfToTextConverter,
        image_to_text: ImageToTextConverter,
    ):
        self.file_storage = file_storage
        self.pdf_to_text = pdf_to_text
        self.image_to_text = image_to_text

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

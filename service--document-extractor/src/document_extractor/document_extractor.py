import json

from loguru import logger
from mmar_mapi import FileStorage, maybe_lru_cache
from mmar_mapi.services import DocExtractionOutput, DocExtractionSpec, DocumentExtractorAPI, ResourceId, DOC_SPEC_DEFAULT

from document_extractor.config import Config
from document_extractor.docling_document_extractor import DoclingDocumentExtractor
from document_extractor.legacy import trace_duration


class DocumentExtractor(DocumentExtractorAPI):
    def __init__(self, config: Config):
        self.file_storage = FileStorage(config.files_dir)
        self.docling_document_extractor = DoclingDocumentExtractor(config.pdf, file_storage=self.file_storage)
        self._extract = maybe_lru_cache(maxsize=config.cache_maxsize, func=self._extract)

    @trace_duration(logger, label="extract", show_args=True)
    def extract(self, *, resource_id: ResourceId, spec: DocExtractionSpec = DOC_SPEC_DEFAULT) -> ResourceId | None:
        return self._extract(resource_id, spec)

    # todo fix workaround: right now mmar-ptag don't respect methods, decorated in __init__
    def _extract(self, resource_id: ResourceId, spec: DocExtractionSpec) -> ResourceId | None:
        # todo validate resource_id and return None if bad
        doc_bytes = self.file_storage.download(resource_id)
        doc_type = resource_id.split(".")[-1].lower()
        if doc_type != "pdf":
            logger.warning(f"Expected only doc_type=pdf, but passed: {doc_type}")

        output: DocExtractionOutput = self.docling_document_extractor.extract(doc_bytes=doc_bytes, spec=spec)
        output_json = json.dumps(output.model_dump(), ensure_ascii=False, indent=2)
        output_resource_id = self.file_storage.upload(output_json, fname="extraction.json")
        return output_resource_id

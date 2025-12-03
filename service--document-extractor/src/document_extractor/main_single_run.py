import os
from pathlib import Path

from mmar_ptag import init_logger
from loguru import logger

from document_extractor.config import Config, load_config
from document_extractor.models import OCROutput
from document_extractor.document_extractor import DocumentExtractor
from document_extractor.utils_models import parse_ocr_config


def main():
    config: Config = load_config()
    init_logger(config.logger.level)
    logger.debug(f"Config: {config}")

    # todo support many queries
    query = os.environ["QUERY"]
    logger.info(f"Query: {query}")
    parse_ocr_config(query, failfast=True)
    input_dir = Path(os.environ["INPUT_DIR"])
    output_dir = Path(os.environ["OUTPUT_DIR"])

    logger.info(f"query: {query}")
    if not input_dir.is_dir():
        raise ValueError(f"Expected exist: {input_dir}")
    if not output_dir.is_dir():
        raise ValueError(f"Expected exist: {output_dir}")

    ocr = DocumentExtractor(config)

    # support images?
    docs = list(input_dir.glob("*.pdf"))
    if not docs:
        logger.warning("No documents found...")
        return
    logger.info(f"Going to process {len(docs)} documents")
    for ii, pdf_path in enumerate(docs):
        logger.info(f"[{ii}] Document {pdf_path}: started processing")
        doc_bytes = pdf_path.read_bytes()
        doc_type = "pdf"
        query = query
        doc_name = pdf_path.stem
        oo: OCROutput = ocr.interpret_raw(doc_name, doc_bytes, doc_type, query)

        if not oo:
            logger.error(f"[{ii}] Document {pdf_path}: failed")
            continue

        if oo.text:
            out_path_text = output_dir / (doc_name + ".txt")
            out_path_text.write_text(oo.text)

        if oo.config:
            json_output = oo.model_dump_json()
            out_path_json = output_dir / (doc_name + ".json")
            out_path_json.write_text(json_output)

        logger.info(f"[{ii}] Document {pdf_path}: done processing")


if __name__ == "__main__":
    main()

import io
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import (
    BaseAnnotation,
    DescriptionAnnotation,
    DocItem,
    DoclingDocument,
    PageItem,
    PictureItem,
    TableItem,
)
from loguru import logger
from mmar_mapi import FileStorage, ResourceId
from mmar_mapi.services import (
    DocExtractionOutput,
    DocExtractionSpec,
    ExtractedPageImage,
    ExtractedPicture,
    ExtractedTable,
    ExtractionEngineSpec,
    ForceOCR,
    OutputType,
)
from mmar_utils import clean_and_fix_text, parallel_map
from more_itertools import flatten
from PIL import Image as PILImage
from pypdf import PdfReader

from document_extractor.legacy import merge_outputs, split_range

ENG_RUS = ["eng", "rus"]
Device = Literal["CPU", "CUDA"]
# MD = TextType.MARKDOWN, todo
FilePath = str


class PdfConfig:
    num_threads: int
    device: Device
    workers: int
    empty_page_chars_threshold: int
    output_dir: Path | None = None


def parse_device(device: Device) -> AcceleratorDevice:
    if device == "CUDA":
        return AcceleratorDevice.CUDA
    if device == "CPU":
        return AcceleratorDevice.CPU
    raise ValueError(f"Bad device: {device}")


def is_force_ocr(pdf_cfg: PdfConfig, engine_spec: ExtractionEngineSpec, text_basic: str) -> bool:
    if engine_spec.force_ocr == ForceOCR.ENABLED:
        return True
    if engine_spec.force_ocr == ForceOCR.DISABLED:
        return False
    return len(text_basic) < pdf_cfg.empty_page_chars_threshold


class DoclingDocumentExtractor:
    def __init__(self, pdf_cfg: PdfConfig, file_storage: FileStorage):
        device = pdf_cfg.device
        self.accelerator_options = AcceleratorOptions(
            num_threads=pdf_cfg.num_threads,
            device=parse_device(device),
        )
        self.pdf_cfg = pdf_cfg
        self.file_storage = file_storage
        logger.info(f"Docling settings: {settings}")
        # self.chunks = pdf_cfg.chunks

    def _setup_converter(
        self, spec: ExtractionEngineSpec, force_ocr: bool, lang: list[str] = ENG_RUS
    ) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = self.accelerator_options
        pipeline_options.do_ocr = spec.do_ocr
        pipeline_options.generate_page_images = spec.generate_page_images
        pipeline_options.do_table_structure = spec.do_table_structure
        pipeline_options.table_structure_options.do_cell_matching = spec.do_cell_matching
        if spec.do_image_extraction:
            pipeline_options.images_scale = spec.images_scale
            pipeline_options.generate_picture_images = True
            pipeline_options.generate_table_images = True
        if spec.do_annotations:
            pipeline_options.do_picture_description = True
        pipeline_options.ocr_options = TesseractOcrOptions(force_full_page_ocr=force_ocr)
        pipeline_options.ocr_options.lang = lang

        fo = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        converter = DocumentConverter(format_options=fo)
        return converter

    def _fix_extracted_table(self, table: ExtractedTable) -> ExtractedTable:
        return ExtractedTable(
            page=table.page,
            formatted_str=clean_and_fix_text(table.formatted_str),
            annotation=clean_and_fix_text(table.annotation),
            caption=clean_and_fix_text(table.caption),
            image_resource_id=table.image_resource_id,
            width=table.width,
            height=table.height,
        )

    def _fix_extracted_picture(self, picture: ExtractedPicture) -> ExtractedPicture:
        return ExtractedPicture(
            page=picture.page,
            annotation=clean_and_fix_text(picture.annotation),
            caption=clean_and_fix_text(picture.caption),
            image_resource_id=picture.image_resource_id,
            width=picture.width,
            height=picture.height,
        )

    def _calculate_pages(self, file_path: str | Path) -> int:
        reader = PdfReader(Path(file_path))
        return len(reader.pages)

    def _save_image(self, image_obj: PILImage.Image | None, image_format: str = "png") -> ResourceId | None:
        if not image_obj:
            return None
        byte_stream = io.BytesIO()
        image_obj.save(byte_stream, format=image_format)
        image_bytes = byte_stream.getvalue()
        fname = f"example.{image_format}"
        return self.file_storage.upload(image_bytes, fname=fname)

    def _get_doc_item_image_contents(
        self, item: DocItem, doc: DoclingDocument
    ) -> tuple[ResourceId, int, int] | tuple[None, None, None]:
        image_resource_id = None
        image_obj = item.get_image(doc=doc)
        if image_obj is None:
            return None, None, None
        image_resource_id = self._save_image(image_obj)
        image_width = image_obj.width
        image_height = image_obj.height
        return image_resource_id, image_width, image_height

    def _get_item_annotation(self, item: DocItem) -> str:
        annotations: list[BaseAnnotation] = list(item.get_annotations())
        annotations_str = [ann.text for ann in annotations if isinstance(ann, DescriptionAnnotation)]
        return "\n".join(annotations_str)

    def _extract_picture(self, doc: DoclingDocument, picture: PictureItem):
        page_num = -1
        for prov in picture.prov:
            page_num = prov.page_no
            break
        annotation = self._get_item_annotation(picture)
        image_resource_id, width, height = self._get_doc_item_image_contents(item=picture, doc=doc)
        picture_obj = ExtractedPicture(
            page=page_num,
            annotation=annotation,
            caption=picture.caption_text(doc=doc),
            image_resource_id=image_resource_id,
            width=width,
            height=height,
        )
        fixed_picture = self._fix_extracted_picture(picture_obj)
        return fixed_picture

    def _extract_table(self, doc: DoclingDocument, table: TableItem):
        page_num = -1
        for prov in table.prov:
            page_num = prov.page_no
            break
        image_resource_id, width, height = self._get_doc_item_image_contents(item=table, doc=doc)
        annotation = self._get_item_annotation(item=table)
        table_obj = ExtractedTable(
            page=page_num,
            formatted_str=table.export_to_markdown(doc=doc),
            image_resource_id=image_resource_id,
            caption=table.caption_text(doc=doc),
            annotation=annotation,
            width=width,
            height=height,
        )
        fixed_table = self._fix_extracted_table(table_obj)
        # extracted_tables.append(fixed_table)
        return fixed_table

    def _extract_page_image(self, page_num: int, page: PageItem) -> ExtractedPageImage | None:
        if not page.image:
            logger.warning("Trying to extract empty image from page!")
            return None
        image_resource_id = self._save_image(page.image.pil_image)
        return ExtractedPageImage(page=page_num, image_resource_id=image_resource_id)

    def _extract_pictures(self, spec: DocExtractionSpec, doc: DoclingDocument) -> list:
        if not spec.engine.do_annotations and not spec.engine.do_image_extraction:
            return []
        try:
            logger.trace(f"Found pictures: {len(doc.pictures)}")
            res = [self._extract_picture(doc, pic) for pic in doc.pictures]
            return res
        except Exception:
            logger.exception("Failed to extract pictures")
            return []

    def _extract_tables(self, spec: DocExtractionSpec, doc: DoclingDocument) -> list[ExtractedTable]:
        if not spec.engine.do_table_structure:
            return []
        try:
            logger.trace(f"Found tables: {len(doc.tables)}")
            res = [self._extract_table(doc, tb) for tb in doc.tables]
            return res
        except Exception:
            logger.exception("Failed to extract tables")
            return []

    def _extract_text(self, spec: DocExtractionSpec, doc: DoclingDocument) -> str:
        try:
            if spec.engine.output_type == OutputType.MARKDOWN:
                extracted_text = doc.export_to_markdown()
            else:
                extracted_text = doc.export_to_text()
            extracted_text = clean_and_fix_text(extracted_text)
            return extracted_text
        except Exception:
            logger.exception("Failed to extract text")
            return ""

    def _extract_page_images(self, spec: DocExtractionSpec, doc: DoclingDocument) -> list[ExtractedPageImage]:
        if not spec.engine.generate_page_images:
            return []
        try:
            res = [self._extract_page_image(page_num, page) for page_num, page in doc.pages.items()]
            filtered_res = [i for i in res if i]
            return filtered_res
        except Exception:
            logger.exception("Failed to extract tables")
            return []

    def extract(self, doc_bytes: bytes, spec: DocExtractionSpec) -> DocExtractionOutput:
        # todo fix perf: this is weird that we are reading and then writing pdfs
        # this is problem when PDF's is big, e.g. > 100 MB
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
            temp_pdf.write(doc_bytes)
            return self._extract(temp_pdf.name, spec)

    def _extract(self, pdf_path: FilePath, spec: DocExtractionSpec) -> DocExtractionOutput:
        pages_count = self._calculate_pages(file_path=pdf_path)
        logger.info(f"Started processing PDF document with {pages_count} pages")

        if self.pdf_cfg.workers == 1:
            page_range_all = spec.page_range or (1, pages_count)
            args = (pdf_path, spec.with_page_range(page_range_all))
            outputs_list = [self._extract_page_range_safe(args)]
        else:
            page_range_all = spec.page_range or (1, pages_count)
            page_ranges = split_range(page_range_all, chunks=self.pdf_cfg.workers)
            args_list = [(pdf_path, spec.with_page_range(pr)) for pr in page_ranges]

            outputs_list = parallel_map(
                func=self._extract_page_range_safe,
                items=args_list,
                process=True,
                max_workers=self.pdf_cfg.workers,
                desc="OCR in parallel",
            )
        outputs = list(flatten(outputs_list))
        res = merge_outputs(outputs)
        return res

    def _extract_page_range_safe(self, args: tuple[FilePath, DocExtractionSpec]) -> list[DocExtractionOutput | None]:
        pdf_path, spec = args
        page_range = spec.page_range
        start = time.time()
        converter = self._setup_converter(spec.engine, force_ocr=False)
        converter_force_ocr = self._setup_converter(spec.engine, force_ocr=True)

        converters = SimpleNamespace(
            converter=converter,
            converter_force_ocr=converter_force_ocr,
        )

        p_a, p_b = page_range
        outputs = []
        for pi in range(p_a, p_b + 1):
            spec_page = spec.with_page_range((pi, pi))
            inner_args = (pdf_path, spec_page)
            output = self._extract_page_safe(converters, inner_args)
            outputs.append(output)

        elapsed = time.time() - start
        logger.debug(f"Processed page_range {page_range} in {elapsed:.2f} seconds")
        return outputs

    def _extract_page_safe(self, converters, args: tuple[FilePath, DocExtractionSpec]) -> DocExtractionOutput | None:
        pdf_path, spec = args
        page_num = spec.page_range[0]
        assert spec.page_range[0] == spec.page_range[1]

        text_basic = self.__extract_basic(args)
        if text_basic is None:
            # error is already logged
            return DocExtractionOutput(spec=spec)

        force_ocr = is_force_ocr(self.pdf_cfg, spec.engine, text_basic)
        if force_ocr:
            converter = converters.converter_force_ocr
        else:
            converter = converters.converter

        try:
            # here assumed that `extract_text()` is fast
            # also it's possible to reuse created reader, but unlikely it'll gain significant amount of time

            # todo fallback to `ignore tesseract detection if fails`
            # todo check CUDA exception and retry if fails
            return self.__extract_page(converter, args)
        except Exception as ex:
            if isinstance(ex, TypeError) and ex.args[0] == "'NoneType' object is not subscriptable":
                # File "/app/ocr/.venv/lib/python3.13/site-packages/docling/models/tesseract_ocr_model.py", line 161, in __call__
                #   doc_orientation = parse_tesseract_orientation(osd["orient_deg"])
                # todo fix: extra check that lines above present in traceback
                logger.warning(f"Probably orientation checking failed for page_num={page_num}, fallback to basic")
            else:
                logger.exception(f"Failed to extract for page_num={page_num}, fallback to basic")
            return DocExtractionOutput(spec=spec, text=text_basic)

    def __extract_basic(self, args: tuple[FilePath, DocExtractionSpec]) -> str | None:
        pdf_path, spec = args
        assert spec.page_range[0] == spec.page_range[1]

        # spec, pdf_path, page_num = args
        page_i = spec.page_range[0] - 1
        try:
            reader = PdfReader(pdf_path)
            text_basic = reader.pages[page_i].extract_text().strip()
            return text_basic
        except Exception:
            logger.exception(f"Basic text extraction failed for page_i={page_i}")
            return None

    def __extract_page(self, converter, args: tuple[FilePath, DocExtractionSpec]) -> DocExtractionOutput:
        pdf_path, spec = args
        assert spec.page_range[0] == spec.page_range[1]
        assert os.path.exists(pdf_path)

        conversion_result = converter.convert(pdf_path, page_range=spec.page_range)
        doc: DoclingDocument = conversion_result.document

        extracted_text = self._extract_text(spec, doc)
        extracted_tables = self._extract_tables(spec, doc)
        extracted_pictures = self._extract_pictures(spec, doc)
        extracted_page_images = self._extract_page_images(spec, doc)

        res = DocExtractionOutput(
            spec=spec,
            text=extracted_text,
            tables=extracted_tables,
            pictures=extracted_pictures,
            page_images=extracted_page_images,
        )
        return res

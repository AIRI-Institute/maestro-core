from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, BaseModel


def _validate_page_range(v: tuple[int, int]) -> tuple[int, int]:
    if v[0] < 1 or v[1] < v[0]:
        raise ValueError("Invalid page range: start must be ≥ 1 and end must be ≥ start.")
    return v


PageRange = Annotated[tuple[int, int], AfterValidator(_validate_page_range)]
ForceOCR = StrEnum("ForceOCR", ["ENABLED", "DISABLED", "AUTO"])
OutputType = StrEnum("OutputType", ["RAW", "PLAIN", "MARKDOWN"])
ResourceId = str


class ExtractionEngineSpec(BaseModel, frozen=True):
    output_type: OutputType = OutputType.MARKDOWN
    force_ocr: ForceOCR = ForceOCR.AUTO
    do_ocr: bool = False
    do_table_structure: bool = False
    do_cell_matching: bool = False
    do_annotations: bool = False
    do_image_extraction: bool = False
    generate_page_images: bool = False
    images_scale: float = 2.0


class DocExtractionSpec(BaseModel, frozen=True):
    page_range: PageRange | None = None
    engine: ExtractionEngineSpec = ExtractionEngineSpec()

    def _update(self, **update):
        return self.model_copy(update=update)

    def _update_engine(self, **engine_update):
        return self._update(engine=self.engine.model_copy(update=engine_update))

    # fmt: off
    def with_output_type_raw(self): return self._update_engine(output_type=OutputType.RAW)
    def with_output_type_plain(self): return self._update_engine(output_type=OutputType.PLAIN)
    def with_ocr(self): return self._update_engine(do_ocr=True)
    def with_tables(self): return self._update_engine(do_table_structure=True, do_cell_matching=True)
    def with_images(self): return self._update_engine(do_image_extraction=True)
    def with_annotations(self): return self._update_engine(do_annotations=True)
    def with_force_ocr_enabled(self): return self._update_engine(force_ocr=ForceOCR.ENABLED)
    def with_force_ocr_disabled(self): return self._update_engine(force_ocr=ForceOCR.DISABLED)
    def with_page_images(self): return self._update_engine(generate_page_images=True)

    def with_page_range(self, page_range: PageRange): return self._update(page_range=page_range)
    # fmt: on

DOC_SPEC_DEFAULT = DocExtractionSpec()


class ExtractedImage(BaseModel):
    page: int
    image_resource_id: ResourceId | None = None


class ExtractedImageMetadata(BaseModel):
    annotation: str = ""
    caption: str = ""
    width: int | None = None
    height: int | None = None


class ExtractedPicture(ExtractedImage, ExtractedImageMetadata):
    "Image of part of page"

    ...


class ExtractedTable(ExtractedImage, ExtractedImageMetadata):
    formatted_str: str


class ExtractedPageImage(ExtractedImage):
    "Image of all page"

    pass


class DocExtractionOutput(BaseModel):
    spec: DocExtractionSpec
    text: str = ""
    tables: list[ExtractedTable] = []
    pictures: list[ExtractedPicture] = []
    page_images: list[ExtractedPageImage] = []


class DocumentExtractorAPI:
    def extract(self, *, resource_id: ResourceId, spec: DocExtractionSpec = DOC_SPEC_DEFAULT) -> ResourceId | None:
        """returns file with DocExtractionOutput"""
        raise NotImplementedError

from dishka import Provider, Scope, provide
from mmar_mapi import FileStorage

from text_extractor.config import Config, load_config
from text_extractor.converter_image_to_text import ImageToTextConverter
from text_extractor.converter_pdf_to_text import PdfToTextConverter
from text_extractor.text_extractor import TextExtractor


class IOC(Provider):
    scope = Scope.APP

    @provide
    def file_storage(self, config: Config) -> FileStorage:
        return FileStorage(config.files_dir)



class IOCLocal(Provider):
    scope = Scope.APP

    @provide
    def config(self) -> Config:
        return load_config()

    pdf_to_text = provide(PdfToTextConverter)
    image_to_text = provide(ImageToTextConverter)
    text_extractor = provide(TextExtractor)


IOCS = [IOC, IOCLocal]
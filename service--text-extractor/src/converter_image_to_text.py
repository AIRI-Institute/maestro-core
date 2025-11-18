from io import BytesIO

import pytesseract
from loguru import logger
from PIL import Image

from mmar_utils import postprocess_text


class ImageToTextConverter:
    def __call__(self, img_bytes: bytes) -> str:
        image = Image.open(BytesIO(img_bytes))
        logger.info(f"Start tesseract, image size = {image.size}")
        txt: str = pytesseract.image_to_string(image=image, lang="rus+eng")
        txt = postprocess_text(txt)
        logger.info(f"Len text from image: {len(txt)}")
        return txt

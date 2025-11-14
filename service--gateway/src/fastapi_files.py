import io

import magic
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger

from src.config import Config
from src.fastapi_errors import MaxFileSizeExceededException, WrongContentTypeException, WrongMimeTypeException

FileName = str
FileData = tuple[FileName, bytes]


def make_streaming_response(f_data: FileData) -> StreamingResponse:
    f_name, f_bytes = f_data
    file_like = io.BytesIO(f_bytes)
    headers = {
        "Content-Disposition": f"attachment; filename={f_name}",
        "Content-Length": str(len(f_bytes)),
    }
    return StreamingResponse(
        file_like,
        media_type="application/octet-stream",
        headers=headers,
    )


# Игнорируем PEP 484, но иначе некорректно формируется doc swagger. Да, режет глаза и делать так нельзя
async def load_file(config: Config, file: UploadFile | None = None) -> FileData | None:
    if not file:
        return None
    # todo fix: are we really needing validation if resource name passed?
    file_name = file.filename
    if not file_name:
        extension = await validate_allowed_type(config, file)
        file_name = f"file.{extension}"
    elif "." not in file_name:
        extension = await validate_allowed_type(config, file)
        file_name = f"{file_name}.{extension}"
    content: bytes = await read_file(config, file)
    return file_name, content


async def read_file(config: Config, file: UploadFile) -> bytes:
    file_length: int = 0
    file_contents = bytes(0)
    while content := await file.read(config.fastapi.files.read_chunk_size):
        file_length += len(content)
        if file_length > config.fastapi.files.max_file_size:
            raise MaxFileSizeExceededException()

        file_contents += content
    return file_contents


async def validate_allowed_type(config: Config, file: UploadFile) -> str:
    first_chunk: bytes = await file.read(1024)  # Чтение первых 1 KB данных
    mime = magic.Magic(mime=True)
    mime_type: str = mime.from_buffer(first_chunk)
    await file.seek(0)  # Возвращаем указатель файла в начало
    # msg = "len(first_chunk)
    size_pretty = "EMPTY" if not first_chunk else "non-empty"
    logger.debug(f"Checking {size_pretty} file, content_type: {file.content_type}, derived mime_type: {mime_type}")

    if file.content_type not in config.fastapi.files.allowed_content_types:
        content_type: str = file.content_type or "unreachable"
        raise WrongContentTypeException(content_type, config.fastapi.files.allowed_content_types)
    if mime_type != file.content_type and mime_type not in config.fastapi.files.allowed_mime_types:
        raise WrongMimeTypeException(mime_type, config.fastapi.files.allowed_content_types)
    return mime_type.split("/", maxsplit=1)[-1]

from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

class BaseErrorSchema(BaseModel):
    detail: list


class ErrorItemSchema(BaseModel):
    error: str


class ErrorSchema(BaseErrorSchema):
    detail: list[ErrorItemSchema]


class UnprocessableEntityItem(BaseModel):
    type: str
    loc: list[str]
    msg: str
    input: str
    url: str | None = None


class UnprocessableErrorSchema(BaseErrorSchema):
    detail: list[UnprocessableEntityItem]


ERR_STATUSES: dict[int | str, dict[str, Any]] | None = {
    status.HTTP_400_BAD_REQUEST: {"model": ErrorSchema},
    status.HTTP_401_UNAUTHORIZED: {"model": ErrorSchema},
    status.HTTP_404_NOT_FOUND: {"model": ErrorSchema},
    status.HTTP_409_CONFLICT: {"model": ErrorSchema},
    status.HTTP_410_GONE: {"model": ErrorSchema},
    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"model": ErrorSchema},
    status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": UnprocessableErrorSchema},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorSchema},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorSchema},
}


async def loguru_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, RequestValidationError):
        logger.error(f"Validation error for {request.url}: {exc.errors()}")
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": exc.errors()})
    else:
        logger.opt(exception=True).error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "Internal server error"}
        )


class DetailedHTTPException(HTTPException):
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)
        logger.error(f"{detail} by {self.__class__.__name__}")

    def _get_content(self) -> BaseModel:
        return ErrorSchema(detail=[ErrorItemSchema(error=self.detail)])

    def to_response(self) -> JSONResponse:
        return JSONResponse(
            content=self._get_content().model_dump(by_alias=True), status_code=self.status_code, headers=self.headers
        )


def _AE(code: int, detail: str) -> Callable[[], Exception]:
    def instantiate_ex() -> Exception:
        return DetailedHTTPException(code, detail=detail)

    return instantiate_ex


MaxFileSizeExceededException = _AE(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Max upload file size exceeded.")
FileNotFoundException = _AE(status.HTTP_400_BAD_REQUEST, "File not found")
FailedToUploadException = _AE(status.HTTP_400_BAD_REQUEST, "Failed to upload")
MethodDeprecatedException = _AE(status.HTTP_405_METHOD_NOT_ALLOWED, "This handle is deprecated!")
WrongClientException = _AE(status.HTTP_401_UNAUTHORIZED, "Wrong client_id.")


class MalformedException(DetailedHTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class WrongContentTypeException(DetailedHTTPException):
    def __init__(self, input_content_type: str, allowed_content_types: set[str]):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Provided Content-Type '{input_content_type}' is not in allowed set {allowed_content_types}",
        )


class WrongMimeTypeException(DetailedHTTPException):
    def __init__(self, input_mime_type: str, allowed_mime_types: set[str]):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Provided MIME type '{input_mime_type}' is not in allowed set {allowed_mime_types}",
        )

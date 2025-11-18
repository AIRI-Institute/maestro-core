import logging
import re
import warnings
from base64 import b64decode, b64encode
from functools import partial
from time import sleep
from typing import Literal

from gigachat import GigaChat
from gigachat._types import FileTypes
from gigachat.models import Embedding, UploadedFile

from .base_chat import AbstractEntryPoint
from .io_sugar import OnError, make_on_exception
from .models import EntrypointPayload

logger = logging.getLogger(__name__)
PATTERN_CID =  "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
CPATTERN_CID = re.compile(PATTERN_CID)


def _validate_cid(field_value, field: str):
    if CPATTERN_CID.fullmatch(field_value):
        return
    warnings.warn(f"Maybe '{field}' is invalid: not matched by '{PATTERN_CID}'")


_validate_client_id = partial(_validate_cid, field="client_id")
_validate_client_secret = partial(_validate_cid, field="client_secret")


def _make_kwargs_auth(
    user: str | None,
    password: str | None,
    client_id: str | None,
    client_secret: str | None,
    authorization_key: str | None,
):
    if user and password:
        return dict(user=user, password=password)
    if client_id and client_secret:
        _validate_client_id(client_id)
        _validate_client_secret(client_secret)
        creds = b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
        return dict(credentials=creds)
    if authorization_key:
        authorization_key_decode = b64decode(authorization_key).decode()
        if ":" not in authorization_key_decode:
            warnings.warn("Maybe bad value for 'authorization_key': expected `:`")
        else:
            client_id, client_secret = authorization_key_decode.split(":", 1)
            _validate_client_id(client_id)
            _validate_client_secret(client_secret)
        return dict(credentials=authorization_key)
    raise ValueError(f"Unexpected combination of auth args: {user=}, {password=}, {client_id=}, {client_secret=}")


BASE_URL_DEFAULT = "https://gigachat.devices.sberbank.ru/api/v1"


class GigaChatEntryPoint(AbstractEntryPoint):
    def __init__(
        self,
        user: str | None = None,
        password: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        authorization_key: str | None = None,
        model_id: str = "GigaChat-Pro",
        warmup: bool = False,
        temperature: float = 0.0,
        base_url: str = BASE_URL_DEFAULT,
        profanity_check: bool = False,
        on_error: OnError = "warn",
        timeout: float | None = None,
    ) -> None:
        kwargs_pre = dict(
            base_url=base_url,
            scope="GIGACHAT_API_CORP",
            model=model_id,
            verify_ssl_certs=False,
            profanity_check=False,
            timeout=timeout,
        )
        kwargs_auth = _make_kwargs_auth(user, password, client_id, client_secret, authorization_key)
        kwargs = kwargs_pre | kwargs_auth
        self._base_url = base_url
        self._model_id = model_id
        self._model = GigaChat(**kwargs)
        self._DIM: int = 1024
        self._ZEROS: list[float] = [0.0 for _ in range(self._DIM)]
        self._ERROR_MESSAGE: str = ""
        self.temperature = temperature
        if warmup:
            self.warmup()
        self.on_error = on_error
        self.on_exception = make_on_exception(on_error, logger, err_msg=f"No response from {model_id}")

    def __call__(self) -> GigaChat:
        return self._model

    def get_response(self, sentence: str) -> str:
        with self.on_exception():
            return self._model.chat(sentence).choices[0].message.content
        return self._ERROR_MESSAGE

    def get_response_by_payload(self, payload: EntrypointPayload) -> str:
        """
        payload:
        - [{"role": "system", "content": system}, {"role": "user", "content": replica}]
        - {"messages": [{"role": "system", "content": system}, {"role": "user", "content": replica}]}
        - {"messages": [...], "attachments": ...}
        """
        if isinstance(payload, list):
            payload_giga = {"messages": payload}
        else:
            payload_giga = payload
        payload_giga["temperature"] = self.temperature
        with self.on_exception():
            return self._model.chat(payload_giga).choices[0].message.content
        return self._ERROR_MESSAGE

    def get_embedding(self, sentence: str) -> list[float]:
        with self.on_exception():
            return self._model.embeddings([sentence]).data[0].embedding
        return self._ZEROS

    def get_embeddings(self, sentences: list[str], request_limit: int = 50) -> list[list[float]]:
        embeddings: list[list[float]] | None = None
        counter: int = 0
        while embeddings is None and counter < request_limit:
            with self.on_exception():
                items: list[Embedding] = self._model.embeddings(sentences).data
                embeddings = [item.embedding for item in items]
                break
            sleep(0.1)
            counter += 1
        if embeddings is not None:
            return embeddings
        return [self._ZEROS for _ in sentences]

    def upload_file(
        self,
        file: FileTypes,
        purpose: Literal["general", "assistant"] = "general",
    ) -> UploadedFile:
        """
        Usage example:
        with Path(fpath).open('rb') as file:
            uploaded_file = entrypoint_giga.upload_file(file=file)
        """
        return self._model.upload_file(file, purpose)

    def __repr__(self):
        class_name = type(self).__name__
        url = self._base_url
        url_info = "" if BASE_URL_DEFAULT == url else f", url={url}"
        model_info = f"model_id={self._model_id}"
        return f"{class_name}({model_info}{url_info})"

    def __str__(self):
        return repr(self)


def _G(**kwargs_base):
    def instantiate_gigachat(**kwargs):
        return GigaChatEntryPoint(**kwargs_base, **kwargs)

    return instantiate_gigachat


GigaChatCensoredEntryPoint = _G(model_id="GigaChat-Pro", profanity_check=True)
GigaMaxEntryPoint = _G(model_id="GigaChat-Max")
GigaMax2EntryPoint = _G(model_id="GigaChat-2-Max")
GigaPlusEntryPoint = _G(model_id="GigaChat-Plus")
GigaMax2SberdevicesEntryPoint = _G(model_id="GigaChat-2-Max", base_url="https://gigachat.sberdevices.ru/v1")

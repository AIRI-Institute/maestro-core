import re
import warnings
from base64 import b64decode, b64encode
from functools import partial
from typing import Literal

from gigachat import GigaChat
from gigachat._types import FileTypes
from gigachat.models import UploadedFile
from mmar_mapi.services import LLMPayload, LLMRequest, LLMResponseExt

from mmar_llm.llm_endpoint import LLMEndpoint
from mmar_llm.utils import dump_messages

PATTERN_CID = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
CPATTERN_CID = re.compile(PATTERN_CID)
SCOPES = {"GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP"}
SCOPE_DEFAULT = "GIGACHAT_API_CORP"
BASE_URL_DEFAULT = "https://gigachat.devices.sberbank.ru/api/v1"
MODEL_DEFAULT = "GigaChat-2-Max"


def _validate_cid(field_value, field: str):
    if CPATTERN_CID.fullmatch(field_value):
        return
    warnings.warn(f"Maybe '{field}' is invalid: not matched by '{PATTERN_CID}'")


_validate_client_id = partial(_validate_cid, field="client_id")
_validate_client_secret = partial(_validate_cid, field="client_secret")


def _validate_scope(scope):
    if scope in SCOPES:
        return
    warnings.warn(f"Maybe '{scope}' is invalid: not in {SCOPES}")


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


class GigaChatEndpoint(LLMEndpoint):
    ALIAS = "gigachat"

    def __init__(
        self,
        user: str | None = None,
        password: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        authorization_key: str | None = None,
        scope: str = SCOPE_DEFAULT,
        model_id: str = MODEL_DEFAULT,
        base_url: str = BASE_URL_DEFAULT,
        temperature: float = 0.0,
        timeout: float | None = None,
    ) -> None:
        _validate_scope(scope)
        kwargs_pre = dict(
            base_url=base_url,
            scope=scope,
            model=model_id,
            verify_ssl_certs=False,
            profanity_check=False,
            timeout=timeout,
        )
        kwargs_auth = _make_kwargs_auth(user, password, client_id, client_secret, authorization_key)
        kwargs = kwargs_pre | kwargs_auth

        self._model = GigaChat(**kwargs)
        self._model_id = model_id
        self._base_url = base_url
        self.temperature = temperature
        # todo fix: eliminate, move to LLMHub
        self._dim = 1024

    def get_response_ext(self, *, request: LLMRequest) -> LLMResponseExt:
        payload = LLMPayload.parse(request)
        messages_json = dump_messages(payload)
        payload_json = {"messages": messages_json, "temperature": self.temperature}

        # todo ensure that content is ok
        text = self._model.chat(payload_json).choices[0].message.content
        return LLMResponseExt(text=text)

    def get_embedding(self, *, prompt: str) -> list[float]:
        return self._model.embeddings([prompt]).data[0].embedding

    def upload_file(
        self,
        file: FileTypes,
        purpose: Literal["general", "assistant"] = "general",
    ) -> UploadedFile:
        return self._model.upload_file(file, purpose)

    def __repr__(self):
        class_name = type(self).__name__
        url_info = "" if "https://gigachat.devices.sberbank.ru/api/v1" == self._base_url else f", url={self._base_url}"
        model_info = f"model_id={self._model_id}"
        return f"{class_name}({model_info}{url_info})"

    def __str__(self):
        return repr(self)


class GigaChatSberdevicesEndpoint(GigaChatEndpoint):
    ALIAS = "gigachat-sberdevices"

    def __init__(self, **kwargs):
        if "base_url" in kwargs:
            base_url = kwargs["base_url"]
            raise ValueError(f"For key 'base_url' found value {base_url}, expected nothing")
        kwargs_fix = {**kwargs, "base_url": "https://gigachat.sberdevices.ru/v1"}
        super().__init__(**kwargs_fix)

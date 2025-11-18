import base64
import logging
from time import sleep

from loguru import logger
from openai import DefaultHttpxClient, OpenAI

from mmar_llm.base_chat import AbstractEntryPoint
from mmar_llm.io_sugar import OnError, make_on_exception
from mmar_llm.models import EntrypointPayload

logger = logging.getLogger(__name__)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterEntryPoint(AbstractEntryPoint):
    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url=OPENROUTER_BASE_URL,
        emb_dim: int = 1024,
        providers: list[str] = [],
        warmup: bool = False,
        on_error: OnError = "warn",
        verify: bool = True,
    ) -> None:
        self.base_url = base_url
        self._model = OpenAI(base_url=base_url, api_key=api_key, http_client=DefaultHttpxClient(verify=verify))
        self.model_id: str = model_id
        self.extra_body: dict[str, dict[str, list[str]]] = {"provider": {"order": providers}}
        self._DIM: int = emb_dim
        self._ZEROS: list[float] = [0 for _ in range(self._DIM)]
        self._ERROR_MESSAGE: str = ""
        if warmup:
            self.warmup()
        self.on_exception = make_on_exception(on_error, logger, err_msg=f"No response from {model_id}")

    def __call__(self) -> OpenAI:
        return self._model

    def get_response(self, sentence: str, **kwargs) -> str:
        with self.on_exception():
            payload = self.create_payload(system_prompt="", user_prompt=sentence)
            return (
                self._model.chat.completions.create(
                    model=self.model_id, messages=payload, extra_body=self.extra_body, **kwargs
                )
                .choices[0]
                .message.content
            ) or ""
        return self._ERROR_MESSAGE

    def get_instruct_response(self, sentence: str, **kwargs) -> str:
        with self.on_exception():
            return self._model.completions.create(model=self.model_id, prompt=sentence, **kwargs).choices[0].text
        return self._ERROR_MESSAGE

    def get_image_response(self, bytesimage: bytes, sentences: str, mimetype: str = "image/jpeg") -> str:
        with self.on_exception():
            encoded_image = base64.b64encode(bytesimage).decode("utf-8")
            payload = self.create_image_payload(
                system_prompt="", user_prompt=sentences, image_encoded=encoded_image, mimetype=mimetype
            )
            return (
                self._model.chat.completions.create(model=self.model_id, messages=payload, extra_body=self.extra_body)
                .choices[0]
                .message.content
            ) or ""
        return self._ERROR_MESSAGE

    def get_response_by_payload(self, payload: EntrypointPayload, **kwargs) -> str:
        """payload: [{"role": "system", "content": system}, {"role": "user", "content": replica}]"""
        if isinstance(payload, dict):
            if "messages" not in payload:
                logger.warning(f"Bad payload: {payload}")
                return self._ERROR_MESSAGE
            if len(payload) > 1:
                bad_keys = payload.keys() - {"messages"}
                logger.warning(f"Unexpected keys: {bad_keys}")
            messages = payload.get("messages")
        elif isinstance(payload, list):
            messages = payload
        else:
            logger.warning(f"Bad payload: {payload}")
            return self._ERROR_MESSAGE

        with self.on_exception():
            return (
                self._model.chat.completions.create(
                    model=self.model_id, messages=messages, extra_body=self.extra_body, **kwargs
                )
                .choices[0]
                .message.content
            ) or ""
        return self._ERROR_MESSAGE

    def get_embedding(self, sentence: str) -> list[float]:
        with self.on_exception():
            return self._model.embeddings.create(model=self.model_id, input=[sentence]).data[0].embedding
        return self._ZEROS

    def get_embeddings(self, sentences: list[str], request_limit=50) -> list[list[float]]:
        embeddings: list[list[float]] | None = None
        counter: int = 0
        while embeddings is None and counter < request_limit:
            with self.on_exception():
                embeddings = [
                    item.embedding for item in self._model.embeddings.create(model=self.model_id, input=sentences).data
                ]
                break
            sleep(0.1)
            counter += 1
        if embeddings is not None:
            return embeddings
        return [self._ZEROS for _ in sentences]

    def __repr__(self):
        class_name = type(self).__name__
        url_info = f", url={self.base_url}" if self.base_url != OPENROUTER_BASE_URL else ""
        return f"{class_name}(model_id={self.model_id}{url_info})"

    def __str__(self):
        return repr(self)

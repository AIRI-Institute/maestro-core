import base64

from openai import DefaultHttpxClient, OpenAI

from mmar_llm.llm_endpoint import LLMEndpoint
from mmar_llm.utils import dump_messages
from mmar_mapi.api import LLMPayload, LLMRequest, LLMResponseExt

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterEndpoint(LLMEndpoint):
    ALIAS = 'openrouter'

    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url=OPENROUTER_BASE_URL,
        emb_dim: int = 1024,
        providers: list[str] = [],
        verify: bool = True,
        extra_create_args: dict | None = None
    ) -> None:
        self.base_url = base_url
        self._model = OpenAI(base_url=base_url, api_key=api_key, http_client=DefaultHttpxClient(verify=verify))
        self.model_id: str = model_id
        self.extra_body: dict[str, dict[str, list[str]]] = {"provider": {"order": providers}}
        self.extra_create_args = extra_create_args or {}
        self._dim: int = emb_dim

    def __call__(self) -> OpenAI:
        return self._model

    def get_response_ext(self, *, request: LLMRequest) -> LLMResponseExt:
        payload = LLMPayload.parse(request)
        messages_json = dump_messages(payload)

        completions = self._model.chat.completions
        response_openai = completions.create(model=self.model_id, messages=messages_json, extra_body=self.extra_body, **self.extra_create_args)  # type: ignore[arg-type]
        text = response_openai.choices[0].message.content or ""
        return LLMResponseExt(text=text)

    def _create_image_payload(
        self, system_prompt: str, user_prompt: str, image_encoded: str, mimetype: str = "image/jpeg"
    ):
        image_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mimetype};base64,{image_encoded}"}},
        ]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": image_content},
        ]

    # todo fix, call via #get_response_ext
    def get_image_response(self, bytesimage: bytes, sentences: str, mimetype: str = "image/jpeg") -> str:
        encoded_image = base64.b64encode(bytesimage).decode("utf-8")
        payload = self._create_image_payload(
            system_prompt="", user_prompt=sentences, image_encoded=encoded_image, mimetype=mimetype
        )
        completions = self._model.chat.completions
        response_openai = completions.create(model=self.model_id, messages=payload, extra_body=self.extra_body)
        text = response_openai.choices[0].message.content or ""
        return text

    def get_embedding(self, *, prompt: str) -> list[float]:
        return self._model.embeddings.create(model=self.model_id, input=[prompt]).data[0].embedding

    def __repr__(self):
        class_name = type(self).__name__
        url_info = f", url={self.base_url}" if self.base_url != OPENROUTER_BASE_URL else ""
        return f"{class_name}(model_id={self.model_id}{url_info})"

    def __str__(self):
        return repr(self)

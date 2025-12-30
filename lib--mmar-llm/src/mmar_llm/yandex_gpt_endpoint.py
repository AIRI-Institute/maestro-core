import contextlib
import json

import requests

from mmar_mapi.api import LLMPayload, LLMRequest

from mmar_llm.llm_endpoint import LLMEndpoint
from mmar_llm.utils import dump_messages


class YandexGPTEndpoint(LLMEndpoint):
    def __init__(
        self,
        token: str,
        folder_id: str,
        iam_url: str = "https://iam.api.cloud.yandex.net/iam/v1/tokens",
        text_url: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        emb_url: str = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding",
        warmup: bool = False,
    ):
        iam_token = requests.post(
            url=iam_url,
            data=json.dumps({"yandexPassportOauthToken": token}),
        ).json()["iamToken"]
        self.folder_id = folder_id
        self.text_url = text_url
        self.emb_url = emb_url
        self.doc_uri = f"emb://{self.folder_id}/text-search-doc/latest"
        self.query_uri = f"emb://{self.folder_id}/text-search-query/latest"
        self.headers = {
            "Authorization": f"Bearer {iam_token}",
            "Content-Type": "application/json",
        }
        self._DIM: int = 256
        self._ZEROS: list[float] = [0 for _ in range(self._DIM)]
        self._ERROR_MESSAGE: str = ""
        self._input_is_long: bool = False
        if warmup:
            self.warmup()

    def __call__(self) -> None:
        return None

    def get_response(self, *, request: LLMRequest) -> str:
        payload = LLMPayload.parse(request)
        messages_json = dump_messages(payload)
        data = {
            "model_uri": f"gpt://{self.folder_id}/yandexgpt/latest",
            "messages": messages_json,
        }
        response = requests.post(url=self.text_url, headers=self.headers, json=data).json()
        return response["result"]["alternatives"][0]["message"]["text"]

    def get_embedding_custom(self, sentence: str, input_is_long: bool = False) -> list[float]:
        data = {
            "modelUri": self.doc_uri if input_is_long else self.query_uri,
            "text": sentence,
        }
        with contextlib.suppress(Exception):
            response = requests.post(url=self.emb_url, headers=self.headers, json=data).json()
            return response["embedding"]
        return self._ZEROS

    def get_embeddings(self, sentences: list[str], input_is_long: bool = False) -> list[list[float]]:
        return [self.get_embedding_custom(sentence, input_is_long) for sentence in sentences]

    def warmup(self) -> None:
        pass

    def get_embedding(self, *, prompt: str) -> list[float]:
        return self.get_embedding_custom(prompt, self._input_is_long)

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from typing import Any

import tiktoken
from httpx import NetworkError
from openai.types.chat import ChatCompletionMessageParam
from mmar_llm.models import EntrypointPayload

logger = logging.getLogger(__name__)


class AbstractEntryPoint(ABC):
    @abstractmethod
    def __call__(self) -> Any:
        pass

    @abstractmethod
    def get_response(self, sentence: str) -> str:
        pass

    @abstractmethod
    def get_response_by_payload(self, payload: EntrypointPayload) -> str:
        pass

    @abstractmethod
    def get_embedding(self, sentence: str) -> list[float]:
        pass

    @abstractmethod
    def get_embeddings(self, sentences: list[str], request_limit: int = 50) -> list[list[float]]:
        pass

    def decorate_with(self, decorator) -> "AbstractEntryPoint":
        return EntryPointFacade(self, decorator)

    def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        response = self.get_response(prompt)
        retry_count = 0
        if response == "":
            while retry_count < retries:
                time.sleep(5)
                logger.warning(f"Response is empty, attempt {retry_count}")
                retry_count += 1
                response = self.get_response(prompt)
                if response != "":
                    break
        if response == "":
            logger.error("Response is empty!")
        if retry_count > 0:
            logger.info(f"Non-empty answer after retry (count={retry_count})")
        return response

    def get_responses(self, sentences: list[str], max_workers: int = 2) -> list[str]:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.get_response, sentence) for sentence in sentences]
            responses = [future.result() for future in futures]
        return responses

    def get_responses_by_payload(
        self, payloads: list[list[ChatCompletionMessageParam]], max_workers: int = 2
    ) -> list[str]:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.get_response_by_payload, payload) for payload in payloads]
            responses = [future.result() for future in futures]
        return responses

    def get_more_embeddings(self, sentences: list[str], batch_size: int = 2, max_workers: int = 4) -> list[list[float]]:
        batches: list[list[str]] = self.make_batches(sentences, size=batch_size)
        if max_workers == 1:
            emb_batches = [self.get_embeddings(batch) for batch in batches]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.get_embeddings, batch) for batch in batches]
                emb_batches = [future.result() for future in futures]
        return list(chain.from_iterable(emb_batches))

    @staticmethod
    def count_tokens(sentences: list[str]) -> list[int]:
        encoding = tiktoken.get_encoding("cl100k_base")
        return [len(encoding.encode(sentence)) for sentence in sentences]

    @staticmethod
    def make_batches(items: list, size: int = 500) -> list[list[str]]:
        slices = [(i * size, (i + 1) * size) for i in range(len(items) // size + 1)]
        return [items[st:ed] for st, ed in slices]

    def warmup(self) -> None:
        response = self.get_response("Прогрев")
        if not response or not sum(self.get_embedding("Прогрев")):
            raise NetworkError("Нет доступа к ллм!")

    @staticmethod
    def create_payload(system_prompt: str, user_prompt: str) -> list[ChatCompletionMessageParam]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def create_image_payload(system_prompt: str, user_prompt: str, image_encoded: str, mimetype="image/jpeg"):
        image_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mimetype};base64,{image_encoded}"}},
        ]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": image_content},
        ]


class EntryPointFacade(AbstractEntryPoint):
    def __init__(self, base: AbstractEntryPoint, decorator):
        self._base = base

        # with basic class (not ABC) it's possible to override method directly
        # like this: `self.get_response = decorator(self._base.get_response)`
        self._get_response = decorator(self._base.get_response)
        self._get_response_by_payload = decorator(self._base.get_response_by_payload)
        self._get_embedding = decorator(self._base.get_embedding)
        self._get_embeddings = decorator(self._base.get_embeddings)

    def __call__(self) -> Any:
        return self._base.__call__()

    def get_response(self, sentence: str) -> str:
        return self._get_response(sentence)

    def get_response_by_payload(self, payload: list[ChatCompletionMessageParam]) -> str:
        return self._get_response_by_payload(payload)

    def get_embedding(self, sentence: str) -> list[float]:
        return self._get_embedding(sentence)

    def get_embeddings(self, sentences: list[str], request_limit: int = 50) -> list[list[float]]:
        return self._get_embeddings(sentences, request_limit=request_limit)

import json
import time
from enum import StrEnum, auto

import requests
from mmar_mapi.api import LLMRequest
from openai.types.chat import ChatCompletionMessageParam

from mmar_llm.llm_endpoint import LLMEndpoint


class ResponseStatus(StrEnum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    INITIAL = auto()
    DONE = auto()
    PROCESSING = auto()
    FAIL = auto()


class FusionBrainEndpoint(LLMEndpoint):
    def __init__(
        self, api_key: str, secret_key: str, warmup: bool = True, max_gen_attempts: int = 6, gen_ping_timeout: int = 5
    ):
        self.auth_headers = {
            "X-Key": f"Key {api_key}",
            "X-Secret": f"Secret {secret_key}",
        }
        self.base_url = "https://api-key.fusionbrain.ai"
        self.pipeline_id: str = self._get_pipeline()

        self.max_gen_attempts: int = max_gen_attempts
        self.gen_ping_timeout: int = gen_ping_timeout

        if warmup:
            self.warmup()

    def get_response(self, *, request: LLMRequest) -> str:
        raise NotImplementedError

    def get_response_custom(
        self, sentence: str, width: int = 1024, height: int = 1024, negative_prompt: str = "", style: str = ""
    ) -> str:
        uuid = self._start_generation(
            sentence, width=width, height=height, negative_prompt=negative_prompt, style=style
        )
        response = self._end_generation(uuid)
        if result := response.get("result"):
            return result["files"][0]

        if response["status"] in [ResponseStatus.INITIAL, ResponseStatus.PROCESSING]:
            raise RuntimeError(
                f"Max generation time of {self.max_gen_attempts * self.gen_ping_timeout} seconds exceeded!"
            )

        raise RuntimeError("Error getting generation result " + response["status"])

    def _get_pipeline(self) -> str:
        response = requests.get(f"{self.base_url}/key/api/v1/pipelines", headers=self.auth_headers)
        if response.status_code >= 400:
            raise RuntimeError(f"Error getting pipeline id {response.status_code} {response.reason}")

        data = response.json()
        return data[0]["id"]

    def _start_generation(
        self, sentence: str, width: int = 1024, height: int = 1024, negative_prompt: str = "", style: str = ""
    ) -> str:
        params = {
            "type": "GENERATE",
            "numImages": 1,
            "width": width,
            "height": height,
            "style": style,
            "negativePromptDecoder": negative_prompt,
            "generateParams": {"query": f"{sentence}"},
        }

        data = {"pipeline_id": (None, self.pipeline_id), "params": (None, json.dumps(params), "application/json")}
        response = requests.post(
            f"{self.base_url}/key/api/v1/pipeline/run",
            headers=self.auth_headers,
            files=data,  # type: ignore[arg-type]
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Error sending generation request to FusionBrain {response.status_code} {response.reason}"
            )

        return response.json()["uuid"]

    def _check_generation(self, uuid: str) -> dict:
        response = requests.get(f"{self.base_url}/key/api/v1/pipeline/status/{uuid}", headers=self.auth_headers)
        return response.json()

    def _end_generation(self, uuid: str) -> dict:
        attempts = 0
        while attempts < self.max_gen_attempts:
            data = self._check_generation(uuid)
            if data["status"] == ResponseStatus.DONE:
                return data

            attempts += 1
            time.sleep(self.gen_ping_timeout)

        return data

    def warmup(self) -> None:
        pass

    def get_response_by_payload(self, payload: list[ChatCompletionMessageParam]) -> str:
        raise NotImplementedError()

    def get_embedding(self, *, prompt: str) -> list[float]:
        raise NotImplementedError()

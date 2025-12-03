import numpy as np
from mmar_mapi.services import LLMHubAPI, LLMCallProps

from question_detector.config import Config


class GigaEmbedding:
    def __init__(self, config: Config, llm_hub: LLMHubAPI) -> None:
        self.llm = llm_hub
        self.config = config
        self.props = LLMCallProps(
            endpoint_key=self.config.llm.endpoint_key,
            attempts=self.config.llm.max_retries,
        )

    def __call__(self, text):
        return self.get_embedding(text)

    def get_embedding(self, text: str) -> np.ndarray:
        model_response: list[float] = self.llm.get_embedding(prompt=text, props=self.props)
        vector = np.array(model_response)
        return vector / np.linalg.norm(vector)

import numpy as np
from mmar_mapi.api import LLMAccessorAPI, LLMCallProps

from src.config import Config


# todo rename to entrypoint_embedding
class GigaEmbedding:
    def __init__(self, config: Config, llm_accessor: LLMAccessorAPI) -> None:
        self.llm = llm_accessor
        self.config = config
        self.props = LLMCallProps(
            entrypoint_key=self.config.llm.entrypoint_key,
            attempts=self.config.llm.max_retries,
        )

    def __call__(self, text):
        return self.get_embedding(text)

    def get_embedding(self, text: str) -> np.ndarray:
        model_response: list[float] = self.llm.get_embedding(prompt=text, props=self.props)
        vector = np.array(model_response)
        return vector / np.linalg.norm(vector)

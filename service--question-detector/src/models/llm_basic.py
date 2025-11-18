import numpy as np
from mmar_mapi.api import LLMAccessorAPI, LLMCallProps

from src.config import Config


class GigaBasicCheck:
    def __init__(self, config: Config, llm_accessor: LLMAccessorAPI) -> None:
        self.llm = llm_accessor
        self.config = config
        self.props = LLMCallProps(
            entrypoint_key=self.config.llm.entrypoint_key,
            attempts=self.config.llm.max_retries,
        )

    def __call__(self, text: str) -> np.ndarray:
        sentence = self.config.models.basic_check_prompt.format(text=text)
        model_response: str = self.llm.get_response(request=sentence, props=self.props).lower()
        if "нет" in model_response:
            return np.array([0])
        if "да" in model_response:
            return np.array([1])
        return np.array([0])

import numpy as np
from openai import OpenAI

from question_detector.config import Config


class GigaBasicCheck:
    def __init__(self, config: Config, client: OpenAI) -> None:
        self.client = client
        self.config = config
        self.model = self.config.llm.question_detector_model

    def __call__(self, text: str) -> np.ndarray:
        sentence = self.config.models.basic_check_prompt.format(text=text)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": sentence}],
        )
        model_response = response.choices[0].message.content.lower()
        if "нет" in model_response:
            return np.array([0])
        if "да" in model_response:
            return np.array([1])
        return np.array([0])

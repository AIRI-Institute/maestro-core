import numpy as np
import pymorphy3

from question_detector.config import ModelsConfig


class FirstWordCheck:
    def __init__(self, model_config: ModelsConfig) -> None:
        self.first_words: list[str] = self._load_data(model_config.first_word_list_path)
        self.morph = pymorphy3.MorphAnalyzer()

    def __call__(self, text: str) -> np.ndarray:
        return self.encode(text)

    def _get_first_word_normal(self, text: str) -> str:
        first_word: str = text.lower().split(" ")[0]
        return self.morph.parse(first_word)[0].normal_form

    def encode(self, text: str) -> np.ndarray:
        normal_first_word: str = self._get_first_word_normal(text)
        is_in_list: bool = normal_first_word in self.first_words
        return np.array([int(is_in_list)])

    @staticmethod
    def _load_data(path: str) -> list[str]:
        with open(path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file]

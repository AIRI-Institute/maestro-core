import numpy as np


class QuestionMarkCheck:
    def __call__(self, text: str) -> np.ndarray:
        return self.encode(text)

    def encode(self, text: str) -> np.ndarray:
        has_question_mark: bool = bool(text) and text[-1] == "?"
        return np.array([int(has_question_mark)])

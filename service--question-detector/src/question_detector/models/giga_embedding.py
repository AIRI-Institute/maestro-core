import numpy as np
from openai import OpenAI
from question_detector.config import Config


class GigaEmbedding:
    def __init__(self, config: Config, client: OpenAI) -> None:
        self.client = client
        self.config = config
        self.model = self.config.llm.question_detector_model

    def __call__(self, text):
        return self.get_embedding(text)

    def get_embedding(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=text)
        vector = np.array(response.data[0].embedding)

        # Validate embedding dimension
        expected_dim = self.config.models.expected_embedding_dim
        if vector.shape[0] != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim} dimensions "
                f"but got {vector.shape[0]}. "
                f"Make sure the embedding model is configured correctly. "
                f"Check the 'question_detector_model' in your config."
            )

        return vector / np.linalg.norm(vector)

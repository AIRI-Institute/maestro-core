from loguru import logger

from mmar_llm.models import UnsupportedModelException
from mmar_llm.openrouter_endpoint import OpenRouterEndpoint


class AiriChatEndpoint(OpenRouterEndpoint):
    ALIAS = 'airi'

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str = "",
        emb_dim: int = 1024,
        verify: bool = True,
        extra_create_args: dict | None = None
    ) -> None:
        super().__init__(model_id=model_id, base_url=base_url, api_key=api_key, emb_dim=emb_dim, verify=False, extra_create_args=extra_create_args)
        self.api_key = api_key
        self.verify = verify
        # todo fix: move in warmup?
        # self.check_current(model_id=model_id)

    def check_current(self, model_id: str) -> None:
        model_list = self.get_available_models()

        if model_id not in model_list:
            logger.error("Model '%s' not found. Available models: %s", model_id, model_list)
            raise UnsupportedModelException(model_id, model_list)
        else:
            logger.info("Model '%s' is found.", model_id)

    def get_available_models(self) -> list[str]:
        models_data = self._model.models.list().data
        models_ids = [model.id for model in models_data]
        logger.info(f"Available models: {models_ids}")
        return models_ids

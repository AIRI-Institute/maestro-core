from openai.types.chat import ChatCompletionMessageParam


class ServiceUnavailableException(Exception):
    def __init__(self, message="Сервис недоступен. Возможно, требуется включить VPN или сервис не запущен."):
        super().__init__(message)


class UnsupportedModelException(Exception):
    def __init__(
        self, model_id, model_names, message="Модель {model_id} недоступна. Список доступных моделей - {model_names}"
    ):
        super().__init__(message.format(model_id=model_id, model_names=model_names))


EndpointLLMPayload = list[ChatCompletionMessageParam] | dict

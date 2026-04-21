from mmar_mapi.models.chat import Chat


class TextProcessorAPI:
    def process(self, *, text: str, chat: Chat | None = None) -> str:
        raise NotImplementedError

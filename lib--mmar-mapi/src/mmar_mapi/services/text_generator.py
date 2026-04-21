from mmar_mapi.models.chat import Chat


class TextGeneratorAPI:
    def process(self, *, chat: Chat) -> str:
        raise NotImplementedError

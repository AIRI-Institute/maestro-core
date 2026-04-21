class TranslatorAPI:
    def get_lang_codes(self) -> list[str]:
        raise NotImplementedError

    def translate(self, *, text: str, lang_code_from: str | None = None, lang_code_to: str) -> str:
        raise NotImplementedError

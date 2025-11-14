from enum import StrEnum

from mmar_mapi import AIMessage, Chat, FileStorage, HumanMessage
from mmar_mapi.api import LLMAccessorAPI, Message, Payload
from mmar_mapi.tracks import SimpleTrack, TrackResponse
from mmar_ptag import ptag_client

from src.config import DOMAINS, Config

S = StrEnum("State", ["EMPTY", "START"])
RECIPE_EXAMPLE_1 = """
Рецепт борща
📍 *Бульон*
- [ ] ~800 г мяса и несколько костей
- [ ] луковица
- [ ] соль
- [ ] лавровый лист
- варить 3..5 часов, пенку снимать

📍 *Обжарка*
- [ ] 2 луковицы
- [ ] 2.5 морковки
- [ ] 2 средние свеклы
- [ ] чеснок

📍 *Сборка*
- обжарка, бульон (мясо порвать)
- [ ] капуста по вкусу
- [ ] картошка по вкусу
- [ ] чёрный перец
- [ ] хмели-сунели
- [ ] уксус (2..3 столовые ложки) или выжать пол-лимона
- [ ] томатная паста
""".strip()
SYSTEM_PROMPT = f"""Тебе пользователь на вход пришлёт рецепт. Твоя задача - преобразовать этот рецепт в формат
- заголовок
- этап готовки
- в каждом этапе перечислены шаги

Дополнительные правила:
- если шаг совпадает с нужным ингридиентом, то в шаге нужно добавить checkbox `[ ]`
- если ингридиент повторяется, то чекбокс не нужен
- этапы отформатируй в формате `📍 *Название этапа*`
- сохраняй все количества, температуры и временные параметры
===
{RECIPE_EXAMPLE_1}
===
"""


class RecipesSummarizer(SimpleTrack):
    DOMAIN = DOMAINS.examples
    CAPTION = "👩‍🍳 Recipes Summarizer"

    def __init__(self, config: Config):
        self.file_storage = FileStorage(config.files_dir)
        self.llm_accessor = ptag_client(LLMAccessorAPI, config.addresses.llm_accessor)

    def generate_response(self, chat: Chat, user_message: HumanMessage) -> TrackResponse:
        text = user_message.text
        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=text),
        ]
        payload = Payload(messages=messages)
        response = self.llm_accessor.get_response(request=payload)
        return AIMessage(content=response)

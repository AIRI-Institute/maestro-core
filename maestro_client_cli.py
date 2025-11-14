import os
from pathlib import Path
from types import SimpleNamespace

import fire
from mmar_llm import EntrypointsConfig
from mmar_mapi import AIMessage, Context, HumanMessage
from mmar_mcli import MaestroClient
from mmar_utils import take_exactly_one

FORCE = bool(int(os.environ.get("force", "0")))
NO_INPUT = bool(int(os.environ.get("no_input", "0")))


class NoRecords(Exception):
    pass


def ask_is_yes(question) -> bool:
    if FORCE:
        return True
    assert question.endswith("?"), f"This is not question: '{question}'"
    # todo mark colors
    response = input(question.strip() + " (y (yes: default) / n (no) )")
    answer = response.strip().lower()
    return len(answer) == 0 or answer.startswith("y")


def user_func(ai_msg: AIMessage, records) -> HumanMessage:
    if ai_msg.state == "final":
        print("Exit!")
    print(f"\nBot (state={ai_msg.state})> {ai_msg.text}")
    buttons = ai_msg.widget and ai_msg.widget.buttons
    if buttons:
        print("Choose:\n" + "\n".join(f"{ii + 1}) {btn[0]}" for ii, btn in enumerate(buttons)))

    if records:
        expected_state, text = records[0]
        if expected_state != ai_msg.state:
            raise ValueError(f"Expected {expected_state}, found: {ai_msg.state}")
        del records[0]
        print(f"User (AUTO)> {text}")
    else:
        if NO_INPUT:
            raise NoRecords()
        text = input("User> ")
    if text.isnumeric():
        ii = int(text)
        if buttons and ii in range(1, len(buttons) + 1):
            text = buttons[ii - 1][0]
            print(f"User (decrypted)> {text}")
    return HumanMessage(content=text)


async def repl(mc, context, records):
    ai_msg = take_exactly_one(await mc.send_message(context, HumanMessage(content="/start")))
    while ai_msg.state != "final":
        human_msg = user_func(ai_msg, records)
        ai_msg = take_exactly_one(await mc.send_message(context, human_msg))
    print(f'\nBot ({ai_msg.state})> {ai_msg.content}')
    return ai_msg.text


def parse_records(records: list[str] | None) -> list[tuple[str, str]]:
    if not records:
        return []
    parsed = []
    for record in records:
        parts = record.split("=")
        if len(parts) != 2:
            raise ValueError(f"Bad record: {record}")
        parsed.append(tuple(parts))
    return parsed


ENTRYPOINTS_WIZARD = "EntrypointsWizard"
TRACKS = {"Dummy", ENTRYPOINTS_WIZARD, "Chatbot"}


def process_entrypoints_wizard_result(result):
    entrypoints_json = result
    print("Result:")
    print(entrypoints_json)
    EntrypointsConfig.model_validate_json(entrypoints_json)
    if not ask_is_yes("Update entrypoints.json?"):
        print("Exit")
        return
    Path("entrypoints.json").write_text(entrypoints_json)
    print("Updated entrypoints.json")


async def main(track_id=None, *records):
    if not track_id:
        print(f"Allowed tracks: {TRACKS}")
        return
    records = parse_records(records)
    addr = "localhost:7732"
    config = SimpleNamespace(addresses__maestro=addr)
    mc = MaestroClient(config)
    context = Context(track_id=track_id)

    try:
        result = await repl(mc, context, records)
        if track_id == ENTRYPOINTS_WIZARD:
            process_entrypoints_wizard_result(result)
    except KeyboardInterrupt:
        print("\nInterrupted! Exit!")
    except NoRecords:
        print("No more records! Exit!")


if __name__ == "__main__":
    fire.Fire(main)

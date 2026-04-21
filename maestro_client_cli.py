import os
from pathlib import Path
from types import SimpleNamespace

import fire
from mmar_mapi import AIMessage, Context, HumanMessage, make_content
from mmar_mcli import FileData, MaestroClient
from mmar_utils import take_exactly_one

FORCE = bool(int(os.environ.get("force", "0")))
NO_INPUT = bool(int(os.environ.get("no_input", "0")))
ADDR = os.environ.get("addresses__maestro", "localhost:7731")
HEADERS_EXTRA = os.environ.get("headers_extra", None)
CLIENT_ID = "TEST"


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


def parse_file_path(text) -> Path | None:
    if not text.startswith("@"):
        return None
    maybe_path = Path(text[1:])
    return maybe_path if maybe_path.is_file() else None


async def _upload(mc, fpath) -> dict:
    resource_name = fpath.name
    fd: FileData = (resource_name, fpath.read_bytes())
    resource_id = await mc.upload_resource(fd, CLIENT_ID)
    resource = {"resource_id": resource_id, "resource_name": resource_name}
    return resource


async def user_func(mc, ai_msg: AIMessage, records) -> HumanMessage:
    if ai_msg.state == "final":
        print("Exit!")
    print(f"\nBot (state={ai_msg.state})> {ai_msg.text}")
    buttons = ai_msg.widget and ai_msg.widget.buttons
    if buttons:
        print("Choose:\n" + "\n".join(f"{ii + 1}) {btn[0]}" for ii, btn in enumerate(buttons)))

    if records:
        expected_state, text = records[0]
        if expected_state and expected_state != ai_msg.state:
            raise ValueError(f"Expected {expected_state}, found: {ai_msg.state}")
        del records[0]
        print(f"User (AUTO)> {text}")
    else:
        if NO_INPUT:
            raise NoRecords()
        text = input("User> ")

    user_fpath = parse_file_path(text)

    if user_fpath:
        resource = await _upload(mc, user_fpath)
        content = make_content(resource=resource)
    elif text.isnumeric():
        ii = int(text)
        if buttons and ii in range(1, len(buttons) + 1):
            text = buttons[ii - 1][0]
            print(f"User (decrypted)> {text}")
        content = text
    else:
        content = text

    return HumanMessage(content=content)


async def repl(mc, context, records):
    msg = HumanMessage(content="/start")
    response = await mc.send(context, msg)
    if not response:
        raise ValueError(f"No response for context={repr(context)} and msg={repr(msg)}")
    ai_msg = take_exactly_one(response)
    while ai_msg.state != "final":
        human_msg = await user_func(mc, ai_msg, records)
        ai_msg = take_exactly_one(await mc.send(context, human_msg))
    print(f"\nBot ({ai_msg.state})> {ai_msg.content}")
    return ai_msg.text


def parse_records(records: list[str] | None) -> list[tuple[str, str]]:
    if not records:
        return []
    parsed = []
    for record in records:
        parts = record.split("=", 1)
        if len(parts) != 2:
            raise ValueError(f"Bad record: {record}")
        parsed.append(tuple(parts))
    return parsed


LLM_CONFIG_WIZARD = "LLMConfigWizard"
TRACKS = {"Dummy", LLM_CONFIG_WIZARD, "Chatbot"}


def process_llm_config_wizard_result(result):
    llm_config_toml = result
    print("\n" + "=" * 60)
    print("Generated LLM Hub Configuration (TOML):")
    print("=" * 60)
    print(llm_config_toml)
    print("=" * 60)
    print("\nTo use this configuration:")
    print("1. Copy the content above to service--llm-hub/llm-config.toml")
    print("2. Restart llm-hub: make restart-llm-hub")
    print()
    if not ask_is_yes("Save to llm-config.toml?"):
        print("Exit")
        return
    Path("llm-config.toml").write_text(llm_config_toml)
    print(f"\n✓ Saved to llm-config.toml")
    print("Next steps:")
    print("  cp llm-config.toml service--llm-hub/llm-config.toml")
    print("  make restart-llm-hub")


def _make_maestro_client():
    config = SimpleNamespace(addresses__maestro=ADDR, headers_extra=HEADERS_EXTRA)
    mc = MaestroClient(config)
    return mc


async def track(track_id=None, *records):
    if not track_id:
        print(f"Allowed tracks: {TRACKS}")
        return
    records = parse_records(records)
    mc = _make_maestro_client()
    context = Context(track_id=track_id)

    try:
        result = await repl(mc, context, records)
        if track_id == LLM_CONFIG_WIZARD:
            process_llm_config_wizard_result(result)
    except KeyboardInterrupt:
        print("\nInterrupted! Exit!")
    except NoRecords:
        print("No more records! Exit!")


async def upload(fpath: str):
    if not os.path.exists(fpath):
        return f"Expected path to existing file, found: {fpath}"
    fpath = Path(fpath)
    assert fpath.is_file()

    mc = _make_maestro_client()
    resource = await _upload(mc, fpath)
    return resource["resource_id"]


if __name__ == "__main__":
    fire.Fire({"track": track, "upload": upload})

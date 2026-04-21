"""Test script framework for validating AI responses in maestro conversations.

This module provides a fluent DSL for writing test scripts that simulate
conversations and validate AI responses.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, TypeAlias

from mmar_mapi import AIMessage, Context, HumanMessage, make_content, make_session_id

MsgInit: TypeAlias = str | HumanMessage
StatesTransitions: TypeAlias = dict[str, set[str]]
StateToAction: TypeAlias = dict[str, set[str]]


@dataclass
class TrackEvent(ABC):
    """Base class for all events in a test script."""

    @abstractmethod
    async def execute(self, client, context: Context) -> list[AIMessage]:
        """Execute the event and return any AI messages produced."""


@dataclass
class HumanEvent(TrackEvent):
    """Event that sends a human message to the maestro service."""

    message: MsgInit

    async def execute(self, client, context: Context) -> list[AIMessage]:
        # Ensure we always have a HumanMessage
        if isinstance(self.message, str):
            # For commands starting with "/", use plain string
            # For other strings, use make_content to create structured content
            if self.message.startswith("/"):
                msg = HumanMessage(content=self.message)
            else:
                msg = HumanMessage(content=make_content(self.message))
        elif isinstance(self.message, dict):
            # Dict from make_content() - wrap in HumanMessage
            msg = HumanMessage(content=self.message)
        elif isinstance(self.message, HumanMessage):
            msg = self.message
        else:
            raise ValueError(f"Unsupported message type: {type(self.message)}")

        # Log human message
        print(f"Human> {msg.content}")
        result = await client.send(context, msg)
        return result or []


@dataclass
class HumanFileEvent(TrackEvent):
    """Event that uploads a file and sends a message with the resource_id to the maestro service."""

    path: str | Path

    async def execute(self, client, context: Context) -> list[AIMessage]:
        path = Path(self.path)
        if not path.exists():
            raise ValueError(f"File not found: {path}")

        # Log file upload
        print(f"Human> file: {path.name}")

        file_bytes = path.read_bytes()
        resource_id = await client.upload_resource(
            file_data=(path.name, file_bytes),
            client_id=context.client_id,
        )

        if not resource_id:
            raise ValueError(f"Failed to upload file: {path}")

        # Send message with resource_id
        msg = HumanMessage(content=make_content(resource_id=resource_id))
        result = await client.send(context, msg)
        return result or []


# Predicates for checking AI messages
class Predicate(ABC):
    """Base class for predicates used to validate AI messages."""

    @abstractmethod
    def __call__(self, msg: AIMessage) -> bool:
        """Check if the predicate matches the given AI message."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable description of this predicate."""
        ...

    def describe_fail(self, msg: AIMessage) -> str:
        """Return a description of why this predicate failed for the given message."""
        return f"expected {self.describe()}, but got: state={msg.state}, action={msg.action}, text={msg.text[:50] if msg.text else None}..."

    def __and__(self, other: "Predicate") -> "Predicate":
        return AndPredicate(self, other)

    def __or__(self, other: "Predicate") -> "Predicate":
        return OrPredicate(self, other)


class AndPredicate(Predicate):
    """Combines two predicates with logical AND."""

    def __init__(self, left: Predicate, right: Predicate) -> None:
        self.left = left
        self.right = right

    def __call__(self, msg: AIMessage) -> bool:
        return self.left(msg) and self.right(msg)

    def describe(self) -> str:
        return f"({self.left.describe()} and {self.right.describe()})"


class OrPredicate(Predicate):
    """Combines two predicates with logical OR."""

    def __init__(self, left: Predicate, right: Predicate) -> None:
        self.left = left
        self.right = right

    def __call__(self, msg: AIMessage) -> bool:
        return self.left(msg) or self.right(msg)

    def describe(self) -> str:
        return f"({self.left.describe()} or {self.right.describe()})"


class CallbackPredicate(Predicate):
    """A predicate that uses a callback function to check the message."""

    def __init__(
        self,
        callback: Callable[[AIMessage], bool],
        description: str = "callback",
        failure_detail: Callable[[AIMessage], str] | None = None,
    ):
        self.callback = callback
        self._description = description
        self._failure_detail = failure_detail

    def __call__(self, msg: AIMessage) -> bool:
        return self.callback(msg)

    def describe(self) -> str:
        return self._description

    def describe_fail(self, msg: AIMessage) -> str:
        if self._failure_detail:
            return self._failure_detail(msg)
        return f"expected {self._description}, but got: state={msg.state}, action={msg.action}"


class HasTextPredicate(Predicate):
    """Predicate that checks if message text contains expected string."""

    def __init__(self, expected: str, exact: bool = False):
        self.expected = expected
        self.exact = exact

    def __call__(self, msg: AIMessage) -> bool:
        if self.exact:
            return msg.text == self.expected
        return self.expected in msg.text

    def describe(self) -> str:
        return f"text{'==' if self.exact else ' contains'} '{self.expected}'"

    def describe_fail(self, msg: AIMessage) -> str:
        return f"text {'==' if self.exact else 'contains'} '{self.expected}'; actual text: '{msg.text[:100] if msg.text else ''}...'"


class HasActionPredicate(Predicate):
    """Predicate that checks if message action equals expected value."""

    def __init__(self, expected: str):
        self.expected = expected

    def __call__(self, msg: AIMessage) -> bool:
        return msg.action == self.expected

    def describe(self) -> str:
        return f"action=='{self.expected}'"

    def describe_fail(self, msg: AIMessage) -> str:
        return f"action=='{self.expected}'; actual action: '{msg.action}'"


class HasStatePredicate(Predicate):
    """Predicate that checks if message state equals expected value."""

    def __init__(self, expected: str):
        self.expected = expected

    def __call__(self, msg: AIMessage) -> bool:
        return msg.state == self.expected

    def describe(self) -> str:
        return f"state=='{self.expected}'"

    def describe_fail(self, msg: AIMessage) -> str:
        return f"state=='{self.expected}'; actual state: '{msg.state}'"


class HasResourceIdPredicate(Predicate):
    """Predicate that checks if message has a resource_id with optional extension."""

    def __init__(self, *, ext: str | None = None):
        self.ext = ext

    def __call__(self, msg: AIMessage) -> bool:
        if msg.resource_id is None:
            return False
        if self.ext is None:
            return True
        return msg.resource_id.endswith(f".{self.ext}")

    def describe(self) -> str:
        if self.ext:
            return f"resource_id.endswith('.{self.ext}')"
        return "resource_id is not None"

    def describe_fail(self, msg: AIMessage) -> str:
        if self.ext:
            return f"resource_id.endswith('.{self.ext}'); actual: '{msg.resource_id}'"
        return f"resource_id is not None; actual: '{msg.resource_id}'"


class HasButtonsPredicate(Predicate):
    """Predicate that checks if message widget has buttons containing expected text."""

    def __init__(self, *, include: str | None = None):
        self.include = include

    def __call__(self, msg: AIMessage) -> bool:
        if msg.widget is None or msg.widget.buttons is None:
            return False
        if self.include is None:
            return True
        return any(self.include.lower() in item.lower() for row in msg.widget.buttons for item in row)

    def describe(self) -> str:
        if self.include:
            return f"buttons contain '{self.include}'"
        return "buttons are not None"

    def describe_fail(self, msg: AIMessage) -> str:
        if self.include:
            return f"buttons contain '{self.include}'; actual buttons: {msg.widget.buttons if msg.widget else None}"
        return f"buttons are not None; actual: {msg.widget.buttons if msg.widget else None}"


class HasInlineButtonsPredicate(Predicate):
    """Predicate that checks if message widget has inline buttons containing expected text."""

    def __init__(self, *, including: str | None = None):
        self.including = including

    def __call__(self, msg: AIMessage) -> bool:
        if msg.widget is None or msg.widget.ibuttons is None:
            return False
        if self.including is None:
            return True
        return any(self.including.lower() in item.lower() for row in msg.widget.ibuttons for item in row)

    def describe(self) -> str:
        if self.including:
            return f"inline buttons contain '{self.including}'"
        return "inline buttons are not None"

    def describe_fail(self, msg: AIMessage) -> str:
        if self.including:
            return f"inline buttons contain '{self.including}'; actual inline buttons: {msg.widget.ibuttons if msg.widget else None}"
        return f"inline buttons are not None; actual: {msg.widget.ibuttons if msg.widget else None}"


@dataclass
class CheckAIEvent(TrackEvent):
    """Event that validates AI messages against predicates.

    The predicates are combined with logical AND - all must pass.
    If a callback is provided, it will be called with the first matching message.
    """

    predicates: list[Predicate] = field(default_factory=list)
    callback: Callable[[AIMessage], Any] | None = None

    async def execute(self, client, context: Context) -> list[AIMessage]:
        return []

    def check(self, messages: list[AIMessage]) -> AIMessage:
        """Check that messages satisfy all predicates.

        Returns the first message that passes all predicates.

        Raises:
            ValueError: If no message satisfies all predicates.
        """
        for msg in messages:
            failed_preds = []
            for pred in self.predicates:
                if not pred(msg):
                    failed_preds.append(pred)
            if not failed_preds:
                if self.callback is not None:
                    self.callback(msg)
                return msg

        # Build helpful error message
        error_parts = []
        for pred in failed_preds:
            error_parts.append(f"  - {pred.describe_fail(messages[-1]) if messages else pred.describe()}")
        raise ValueError("No message satisfied all predicates:\n" + "\n".join(error_parts))


@dataclass
class CheckerScript:
    """A test script that executes a sequence of events.

    Example:
        script = CheckerScript(
            context=Context(client_id="test"),
            events=[
                human("Hello"),
                check_ai(P.has_text("Hello") & P.has_state("GREETING")),
            ]
        )
        await script.run(client)

    With state transitions validation:
        script = CheckerScript(
            context=Context(client_id="test"),
            states_transitions={
                "empty": {"start"},
                "start": {"final"},
                "final": set(),
            },
            state_to_action={
                "empty": set(),
                "start": {"greet"},
                "final": {"answer", "complete"},
            },
            events=[
                human("Hello"),
                check_ai(P.has_action("greet")),
                human("Goodbye"),
                check_ai(P.has_action("complete")),
            ]
        )
        await script.run(client)
    """

    context: Context
    events: list[TrackEvent] = field(default_factory=list)
    states_transitions: StatesTransitions | None = None
    state_to_action: StateToAction | None = None
    auto_session_id: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_context()
        self._validate_state_config()
        if self.auto_session_id:
            self._auto_set_session_id()

    def _validate_context(self) -> None:
        """Validate that context has required fields."""
        if not self.context.track_id:
            raise ValueError("track_id must not be empty")

    def _auto_set_session_id(self) -> None:
        """Auto-generate and set a unique ISO session_id in the context."""
        new_session_id = make_session_id(with_millis=True)
        self.context = self.context.model_copy(update={"session_id": new_session_id})

    def _validate_state_config(self) -> None:
        """Validate that states_transitions and state_to_action are consistent."""
        if self.states_transitions is None or self.state_to_action is None:
            return

        states = set(self.states_transitions.keys())
        states_by_actions = set(self.state_to_action.keys())

        if states != states_by_actions:
            raise ValueError(
                f"Inconsistent state configuration: states_transitions has {states}, "
                f"but state_to_action has {states_by_actions}"
            )

    async def _run(self, client, context: Context | None = None) -> list[AIMessage]:
        """Internal method that runs the test script with the given client.

        Args:
            client: Maestro client instance (MaestroClientI).
            context: Optional context override. If not provided, uses self.context.

        Returns:
            List of all AI messages produced during execution.
        """
        ctx = self._resolve_context(context)
        all_messages: list[AIMessage] = []

        prev_state: str | None = None
        states_visited: set[str] = set()

        for event in self.events:
            messages = await event.execute(client, ctx)
            all_messages.extend(messages)

            # Log AI messages
            for msg in messages:
                print(f"AI (state={msg.state})> {msg.text}")
                if msg.widget is not None:
                    prefix = "AI .............> "
                    if msg.widget.buttons is not None:
                        print(f"{prefix}buttons={msg.widget.buttons}")
                    if msg.widget.ibuttons is not None:
                        print(f"{prefix}ibuttons={msg.widget.ibuttons}")
                if msg.resource_id is not None:
                    prefix = "AI .............> "
                    print(f"{prefix}resource_id={msg.resource_id}")

            if isinstance(event, CheckAIEvent):
                event.check(all_messages)

            # Validate state transitions and actions if configured
            if messages:
                self._validate_states_and_actions(messages, prev_state, states_visited)
                # Update prev_state to the last message's state
                prev_state = messages[-1].state

        return all_messages

    async def arun(self, client, context: Context | None = None) -> list[AIMessage]:
        """Run the test script asynchronously with the given client.

        Args:
            client: Maestro client instance (MaestroClientI).
            context: Optional context override. If not provided, uses self.context.

        Returns:
            List of all AI messages produced during execution.
        """
        import time

        start_time = time.time()
        try:
            messages = await self._run(client, context)
            elapsed = time.time() - start_time
            print(f"\n✓ SMOKE TEST PASSED in {elapsed:.2f} seconds - {len(messages)} messages exchanged")
            return messages
        except Exception:
            elapsed = time.time() - start_time
            print(f"\n✗ SMOKE TEST FAILED in {elapsed:.2f} seconds")
            raise

    def run(self, client, context: Context | None = None) -> list[AIMessage]:
        """Run the test script synchronously with the given client.

        This is a synchronous wrapper around arun() that handles the asyncio event loop.

        Args:
            client: Maestro client instance (MaestroClientI).
            context: Optional context override. If not provided, uses self.context.

        Returns:
            List of all AI messages produced during execution.
        """
        import asyncio

        return asyncio.run(self.arun(client, context))

    def _validate_states_and_actions(
        self,
        messages: list[AIMessage],
        prev_state: str | None,
        states_visited: set[str],
    ) -> None:
        """Validate state transitions and actions for the given messages.

        Args:
            messages: List of AI messages to validate.
            prev_state: Previous state for transition validation.
            states_visited: Set to track visited states.

        Raises:
            ValueError: If state transition or action validation fails.
        """
        if self.states_transitions is None or self.state_to_action is None:
            return

        valid_states = set(self.states_transitions.keys())

        for msg in messages:
            state = msg.state

            # Validate state is known
            if state not in valid_states:
                raise ValueError(f"Unknown state '{state}'. Valid states are: {valid_states}")

            states_visited.add(state)

            # Validate action is allowed for this state
            if msg.action:
                allowed_actions = self.state_to_action.get(state, set())
                if msg.action not in allowed_actions:
                    raise ValueError(
                        f"Action '{msg.action}' not allowed for state '{state}'. Allowed actions: {allowed_actions}"
                    )

        # Validate state transition from previous state
        # If prev_state is None but "empty" state exists, use "empty" as the starting point
        transition_from = prev_state
        if transition_from is None and "empty" in valid_states:
            transition_from = "empty"

        if transition_from is not None:
            last_state = messages[-1].state
            allowed_targets = self.states_transitions.get(transition_from, set())
            if last_state not in allowed_targets:
                raise ValueError(
                    f"Invalid state transition: '{transition_from}' -> '{last_state}'. "
                    f"Allowed transitions from '{transition_from}': {allowed_targets}"
                )

    def _resolve_context(self, override: Context | None) -> Context:
        if override is not None:
            return override
        return self.context


# Predicate namespace for cleaner API
P = SimpleNamespace(
    has_text=lambda expected, exact=False: HasTextPredicate(expected, exact),
    has_action=lambda expected: HasActionPredicate(expected),
    has_state=lambda expected: HasStatePredicate(expected),
    has_resource_id=HasResourceIdPredicate,
    has_buttons=HasButtonsPredicate,
    has_inline_buttons=HasInlineButtonsPredicate,
    has_ibuttons=HasInlineButtonsPredicate,  # backward compat
    has_content=lambda callback: CallbackPredicate(callback, description="custom content check"),
)


def human(message: MsgInit) -> HumanEvent:
    """Create a human event that sends a message."""
    return HumanEvent(message)


def human_file(path: str | Path) -> HumanFileEvent:
    """Create a human event that uploads a file and sends it with resource_id.

    This encapsulates:
    - Reading the file bytes
    - Uploading via client.upload_resource()
    - Validating non-empty resource_id
    - Sending the message with make_content(resource_id=...)

    Args:
        path: Path to the file to upload.

    Returns:
        A HumanFileEvent that will upload and send the file.
    """
    return HumanFileEvent(path)


def check_ai(
    *predicates: Predicate,
    callback: Callable[[AIMessage], Any] | None = None,
) -> CheckAIEvent:
    """Create a check event that validates AI responses.

    Args:
        *predicates: Predicates to check (combined with AND).
        callback: Optional callback to call with the matching message.

    Returns:
        A CheckAIEvent that will validate against the given predicates.
    """
    return CheckAIEvent(predicates=list(predicates), callback=callback)


# Event namespace for cleaner API
E = SimpleNamespace(
    human=human,
    human_file=human_file,
    check_ai=check_ai,
)

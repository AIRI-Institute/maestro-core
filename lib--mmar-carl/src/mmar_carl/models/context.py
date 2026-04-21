"""
Reasoning context for CARL reasoning system.
"""

import warnings
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, PrivateAttr, field_serializer

from mmar_carl.llm import OpenAIClientConfig, OpenAICompatibleClient
from mmar_carl.models.base import SelfCriticEvaluatorBase
from mmar_carl.models.config import LLMStepConfig
from mmar_carl.models.enums import Language
from mmar_carl.models.llm_client_base import LLMClientBase
from mmar_carl.models.replan import ReplanCheckerBase


class ReasoningContext(BaseModel):
    """
    Context object that maintains state during reasoning execution.

    Contains the input data, API object for LLM calls, execution history, and configuration.
    Supports tool registry and memory storage for extended step types.

    Individual LLM steps can override the model using llm_config:
        ```python
        LLMStepDescription(
            number=1,
            title="Complex Task",
            aim="...",
            llm_config=LLMStepConfig(model="anthropic/claude-3.5-sonnet")
        )
        ```

    Callbacks for monitoring execution:
        ```python
        context = ReasoningContext(
            outer_context=data,
            api=client,
            on_step_start=lambda num, title: print(f"Starting step {num}"),
            on_step_complete=lambda result: print(f"Step {result.step_number} done"),
            on_progress=lambda completed, total: print(f"Progress: {completed}/{total}"),
        )
        ```

    For streaming LLM responses:
        ```python
        def handle_chunk(chunk: str):
            print(chunk, end="", flush=True)

        context = ReasoningContext(
            outer_context=data,
            api=client,
            on_llm_chunk=handle_chunk,
        )
        ```
    """

    outer_context: str = Field(..., description="Input data as string (it can be CSV or other text information)")
    api: Any = Field(..., description="API object for LLM execution (LLMClientBase or compatible)")
    model: str = Field(default="default", description="Specific model to use")
    retry_max: int = Field(default=3, description="Maximum retry attempts")
    history: list[str] = Field(default_factory=list, description="Accumulated reasoning history")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata and state")
    language: Language = Field(default=Language.RUSSIAN, description="Language for reasoning prompts")
    system_prompt: str = Field(default="", description="System prompt to include in each reasoning step")

    # Deprecated: Use per-step llm_config instead
    default_model: Optional[str] = Field(
        default=None,
        description="Deprecated. Use per-step llm_config instead.",
    )

    # Memory storage for MEMORY step types
    memory: dict[str, dict[str, Any]] = Field(default_factory=dict, description="Memory storage organized by namespace")

    # === History Management ===
    max_history_entries: int = Field(
        default=0,
        ge=0,
        description="Maximum history entries to keep (0 = unlimited). Prevents context overflow in long chains.",
    )

    # === Callbacks for monitoring execution ===
    # Note: Callbacks are excluded from JSON serialization via @field_serializer
    on_step_start: Optional[Callable[[int, str], None]] = Field(
        default=None,
        description="Callback called when a step starts: on_step_start(step_number, step_title)",
    )
    on_step_complete: Optional[Callable[[Any], None]] = Field(
        default=None,
        description="Callback called when a step completes: on_step_complete(StepExecutionResult)",
    )
    on_chain_complete: Optional[Callable[[Any], None]] = Field(
        default=None,
        description="Callback called when the entire chain completes: on_chain_complete(ReasoningResult)",
    )
    on_progress: Optional[Callable[[int, int], None]] = Field(
        default=None,
        description="Progress callback: on_progress(completed_steps, total_steps)",
    )
    on_llm_chunk: Optional[Callable[[str], None]] = Field(
        default=None,
        description="Streaming callback for LLM responses: on_llm_chunk(chunk_text)",
    )

    # === Cancellation support ===
    _cancelled: bool = PrivateAttr(default=False)

    # === Private instance attributes (NOT class attributes!) ===
    # FIX: Tool registry must be instance-level, not class-level
    _tool_registry: dict[str, Callable] = PrivateAttr(default_factory=dict)

    # Internal LLM client cache (keyed by model for per-step clients)
    _llm_client: LLMClientBase | None = PrivateAttr(default=None)
    _llm_client_cache: dict[str, LLMClientBase] = PrivateAttr(default_factory=dict)

    # Self-critic evaluator registry
    _self_critic_evaluator_registry: dict[str, SelfCriticEvaluatorBase] = PrivateAttr(default_factory=dict)

    # RE-PLAN checker registry
    _replan_checker_registry: dict[str, ReplanCheckerBase] = PrivateAttr(default_factory=dict)

    # === Serialization: Exclude callbacks from JSON/dict output ===
    @field_serializer(
        "on_step_start",
        "on_step_complete",
        "on_chain_complete",
        "on_progress",
        "on_llm_chunk",
        when_used="json-unless-none",
    )
    def serialize_callback(self, value: Optional[Callable]) -> Optional[str]:
        """Serialize callbacks as '<callback>' to avoid JSON serialization errors."""
        return "<callback>" if value is not None else None

    def model_post_init(self, __context: Any) -> None:
        """Create LLM client after model initialization."""

        # Initialize instance-level tool registry (fix for class attribute bug)
        if not hasattr(self, "_tool_registry") or self._tool_registry is None:
            self._tool_registry = {}

        # Initialize LLM client cache
        if not hasattr(self, "_llm_client_cache") or self._llm_client_cache is None:
            self._llm_client_cache = {}

        # Initialize self-critic evaluator registry
        if not hasattr(self, "_self_critic_evaluator_registry") or self._self_critic_evaluator_registry is None:
            self._self_critic_evaluator_registry = {}

        # Initialize RE-PLAN checker registry
        if not hasattr(self, "_replan_checker_registry") or self._replan_checker_registry is None:
            self._replan_checker_registry = {}

        # Warn about deprecated default_model
        if self.default_model is not None:
            warnings.warn(
                "ReasoningContext.default_model is deprecated. Use per-step llm_config instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Register default self-critic evaluator to avoid race conditions
        self._register_default_self_critic_evaluator()

        # Check if api is already an LLMClientBase (e.g., OpenAICompatibleClient)
        self._llm_client = self.api

        # Initialize default namespace for memory
        if "default" not in self.memory:
            self.memory["default"] = {}

    @property
    def llm_client(self) -> LLMClientBase:
        """Get the default LLM client (creates it if not already created)."""
        return self.api

    def get_llm_client_for_step(self, llm_config: Optional[LLMStepConfig] = None) -> LLMClientBase:
        """
        Get an LLM client for a specific step, potentially with overrides.

        Args:
            llm_config: Optional per-step LLM configuration

        Returns:
            LLM client with appropriate configuration
        """

        # If no override, return default client
        if llm_config is None:
            return self.llm_client

        # Check if we need to create an overridden client
        has_override = (
            llm_config.model is not None or llm_config.temperature is not None or llm_config.max_tokens is not None
        )

        if not has_override:
            return self.llm_client

        # For OpenAI-compatible clients, handle model/temperature/max_tokens overrides
        if isinstance(self._llm_client, OpenAICompatibleClient):
            # Create a cache key based on the override parameters
            cache_key = f"openai:{llm_config.model or ''}:{llm_config.temperature or ''}:{llm_config.max_tokens or ''}"

            if cache_key not in self._llm_client_cache:
                # Create a new client with overridden config
                base_config = self._llm_client.config

                new_config = OpenAIClientConfig(
                    base_url=base_config.base_url,
                    api_key=base_config.api_key,
                    model=llm_config.model or base_config.model,
                    temperature=llm_config.temperature
                    if llm_config.temperature is not None
                    else base_config.temperature,
                    max_tokens=llm_config.max_tokens if llm_config.max_tokens is not None else base_config.max_tokens,
                    timeout=base_config.timeout,
                    verify_ssl=base_config.verify_ssl,
                    extra_headers=base_config.extra_headers,
                    extra_body=base_config.extra_body,
                )
                self._llm_client_cache[cache_key] = OpenAICompatibleClient(new_config)

            return self._llm_client_cache[cache_key]

        return self.llm_client

    def add_to_history(self, entry: str) -> None:
        """
        Add a new entry to the reasoning history.

        If max_history_entries is set (> 0), older entries are trimmed
        to prevent context overflow in long chains.
        """
        self.history.append(entry)

        # Trim history if limit is set
        if self.max_history_entries > 0 and len(self.history) > self.max_history_entries:
            # Keep the most recent entries
            self.history = self.history[-self.max_history_entries :]

    def get_current_history(self) -> str:
        """Get the current reasoning history as a single string."""
        return "\n".join(self.history)

    # === Cancellation Methods ===

    def cancel(self) -> None:
        """Request cancellation of the running chain."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled

    def reset_cancellation(self) -> None:
        """Reset cancellation flag for a new execution."""
        self._cancelled = False

    # === Tool Registry Methods ===

    def register_tool(self, tool_name: str, tool_callable: Callable) -> None:
        """
        Register a tool/function for use in TOOL steps.

        Args:
            tool_name: Name of the tool (must match ToolStepConfig.tool_name)
            tool_callable: The callable to execute

        Note:
            Tools MUST be stateless for safe parallel execution.
            If a tool has instance state, it may cause race conditions
            when multiple steps execute in parallel.
        """
        # Warn if tool appears to have instance state
        if hasattr(tool_callable, "__self__") and hasattr(tool_callable.__self__, "__dict__"):
            if tool_callable.__self__.__dict__:
                warnings.warn(
                    f"Tool '{tool_name}' has instance state and may not be safe for parallel execution. "
                    "Ensure tools are stateless or use appropriate locking.",
                    UserWarning,
                    stacklevel=2,
                )
        self._tool_registry[tool_name] = tool_callable

    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """Get a registered tool by name."""
        return self._tool_registry.get(tool_name)

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tool_registry

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tool_registry.keys())

    # === Self-Critic Evaluator Registry Methods ===

    def register_self_critic_evaluator(self, name: str, evaluator: SelfCriticEvaluatorBase) -> None:
        """
        Register a self-critic evaluator strategy by name.

        Args:
            name: Evaluator name referenced by LLMStepConfig.self_critic_evaluators
            evaluator: Evaluator strategy implementation
        """
        if not name or not name.strip():
            raise ValueError("Self-critic evaluator name cannot be empty")
        self._self_critic_evaluator_registry[name.strip()] = evaluator

    def get_self_critic_evaluator(self, name: str) -> Optional[SelfCriticEvaluatorBase]:
        """Get a registered self-critic evaluator by name."""
        return self._self_critic_evaluator_registry.get(name)

    def list_self_critic_evaluators(self) -> list[str]:
        """List all registered self-critic evaluator names."""
        return list(self._self_critic_evaluator_registry.keys())

    # === RE-PLAN Checker Registry Methods ===

    def register_replan_checker(self, name: str, checker: ReplanCheckerBase) -> None:
        """
        Register a RE-PLAN checker strategy by name.

        Args:
            name: Checker name referenced by RegisteredReplanCheckerConfig.name
            checker: Checker strategy implementation
        """
        if not name or not name.strip():
            raise ValueError("RE-PLAN checker name cannot be empty")
        self._replan_checker_registry[name.strip()] = checker

    def get_replan_checker(self, name: str) -> Optional[ReplanCheckerBase]:
        """Get a registered RE-PLAN checker by name."""
        return self._replan_checker_registry.get(name)

    def list_replan_checkers(self) -> list[str]:
        """List all registered RE-PLAN checker names."""
        return list(self._replan_checker_registry.keys())

    # === Default Self-Critic Evaluator Registration ===

    def _register_default_self_critic_evaluator(self) -> None:
        """Register the built-in 'llm' self-critic evaluator if not already present."""
        # Lazy import to avoid circular dependency
        from mmar_carl.step_executors import LLMSelfCriticEvaluator

        default_name = "llm"
        if self.get_self_critic_evaluator(default_name) is None:
            self.register_self_critic_evaluator(default_name, LLMSelfCriticEvaluator())

    # === Memory Methods ===

    def memory_read(self, key: str, namespace: str = "default", default: Any = None) -> Any:
        """Read a value from memory."""
        ns = self.memory.get(namespace, {})
        return ns.get(key, default)

    def memory_write(self, key: str, value: Any, namespace: str = "default") -> None:
        """Write a value to memory."""
        if namespace not in self.memory:
            self.memory[namespace] = {}
        self.memory[namespace][key] = value

    def memory_append(self, key: str, value: Any, namespace: str = "default") -> None:
        """Append a value to a list in memory (creates list if not exists)."""
        if namespace not in self.memory:
            self.memory[namespace] = {}
        if key not in self.memory[namespace]:
            self.memory[namespace][key] = []
        if isinstance(self.memory[namespace][key], list):
            self.memory[namespace][key].append(value)
        else:
            raise ValueError(f"Memory key '{key}' is not a list")

    def memory_delete(self, key: str, namespace: str = "default") -> bool:
        """Delete a value from memory. Returns True if key existed."""
        if namespace in self.memory and key in self.memory[namespace]:
            del self.memory[namespace][key]
            return True
        return False

    def memory_list(self, namespace: str = "default") -> list[str]:
        """List all keys in a memory namespace."""
        return list(self.memory.get(namespace, {}).keys())

    async def close(self) -> None:
        """
        Close any open LLM clients and release resources.

        Should be called when the context is no longer needed, especially for
        OpenAI-compatible clients that maintain HTTP connections.

        Note: For proper cleanup when using asyncio.run(), the caller should
        wait for all background tasks after calling this method.

        Note: After calling close(), the context can still be used for a new
        execution - LLM clients will be recreated on demand.

        Example:
            ```python
            context = ReasoningContext(...)
            try:
                result = chain.execute(context)
            finally:
                await context.close()
            ```
        """

        # Close the main client
        if isinstance(self._llm_client, OpenAICompatibleClient):
            await self._llm_client.close()
        # FIX: Reset the main client so it can be recreated if context is reused
        self._llm_client = None

        # Close any cached clients
        for client in self._llm_client_cache.values():
            if isinstance(client, OpenAICompatibleClient):
                await client.close()

        self._llm_client_cache.clear()

    model_config = {"arbitrary_types_allowed": True}

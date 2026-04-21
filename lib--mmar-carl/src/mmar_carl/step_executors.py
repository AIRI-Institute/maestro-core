"""
Step executors for different step types in CARL reasoning chains.

Each step type has a dedicated executor that handles its specific execution logic.
"""

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from simpleeval import EvalWithCompoundTypes

from .models import (
    ConditionalStepConfig,
    ExecutionMode,
    LLMStepConfig,
    Language,
    MCPStepConfig,
    MemoryOperation,
    MemoryStepConfig,
    PromptTemplate,
    ReasoningContext,
    SelfCriticDecision,
    SelfCriticEvaluatorBase,
    StepDescription,
    StepExecutionResult,
    StepType,
    ToolStepConfig,
    TransformStepConfig,
    StructuredOutputStepConfig,
)


def _get_nested_value(data: Any, path: str) -> Any:
    """Get nested value from dict/list by dotted path."""
    if not path:
        return data

    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            if part in current:
                current = current[part]
            else:
                return None
        elif isinstance(current, list):
            if part.isdigit() and int(part) < len(current):
                current = current[int(part)]
            else:
                return None
        else:
            return None
    return current


def resolve_context_reference(source: str, context: ReasoningContext) -> Any:
    """Resolve a dynamic value from history, memory, metadata, step results, or outer context."""

    # Check for string literals first (quoted strings)
    if (source.startswith('"') and source.endswith('"')) or \
       (source.startswith("'") and source.endswith("'")):
        return source[1:-1]  # Remove quotes and return as-is

    if source.startswith("$history"):
        match = re.match(r"\$history\[(-?\d+)\]", source)
        if match:
            index = int(match.group(1))
            if context.history:
                return context.history[index]
            return None
        return context.get_current_history()

    if source.startswith("$memory."):
        parts = source[8:].split(".", 1)
        if len(parts) == 2:
            return context.memory_read(parts[1], namespace=parts[0])
        return context.memory_read(parts[0])

    if source.startswith("$metadata."):
        return _get_nested_value(context.metadata, source[10:])

    if source.startswith("$steps."):
        return _get_nested_value(context.metadata.get("step_results", {}), source[7:])

    if source == "$outer_context":
        outer_value = context.outer_context
        # If outer context is a string that looks like JSON, try to parse it
        if isinstance(outer_value, str):
            outer_value = outer_value.strip()
            if outer_value.startswith(("{", "[")) or \
               (outer_value.startswith('"') and outer_value.endswith('"')):
                try:
                    return json.loads(outer_value)
                except json.JSONDecodeError:
                    return outer_value  # Not valid JSON, return as-is
        return outer_value

    return context.metadata.get(source)


def _extract_json_payload(text: str) -> Any:
    """
    Extract and parse JSON payload from model response.

    Handles multiple formats:
    - Raw JSON string
    - JSON wrapped in markdown code blocks (```json...```)
    - JSON embedded in text (extracts first {...} or [...])

    Args:
        text: Raw text response from LLM

    Returns:
        Parsed JSON object (dict or list)

    Raises:
        json.JSONDecodeError: If no valid JSON found
    """
    stripped = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", stripped, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


class StepExecutorBase(ABC):
    """Abstract base class for step executors."""

    @abstractmethod
    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        """
        Execute a step and return the result.

        Args:
            step: The step description to execute
            context: The reasoning context
            prompt_template: Optional prompt template for LLM steps

        Returns:
            StepExecutionResult with execution outcome
        """
        pass


class LLMSelfCriticEvaluator(SelfCriticEvaluatorBase):
    """Default self-critic evaluator that uses the step LLM."""

    @staticmethod
    def _build_prompt(base_prompt: str, candidate: str, custom_instruction: str = "") -> str:
        extra_instruction = custom_instruction.strip()
        extra_block = (
            f"\nAdditional reviewer instruction:\n{extra_instruction}\n"
            if extra_instruction
            else ""
        )
        return (
            "You are a strict reviewer of an LLM answer.\n"
            "Evaluate whether the answer fully satisfies the task.\n"
            "Return only a valid JSON object with this exact schema:\n"
            '{"verdict":"APPROVE|DISAPPROVE","review":"short text review (1-3 lines)"}\n'
            "Do not return markdown, explanations, or any text outside this JSON object."
            f"{extra_block}\n"
            f"Task:\n{base_prompt}\n\n"
            f"Candidate answer:\n{candidate}"
        )

    @staticmethod
    def _parse_decision(text: str) -> SelfCriticDecision:
        try:
            payload = _extract_json_payload(text)
        except Exception:
            return SelfCriticDecision(
                verdict="DISAPPROVE",
                review_text="Evaluator response is not valid JSON.",
                metadata={"llm_calls": 1},
            )

        if not isinstance(payload, dict):
            return SelfCriticDecision(
                verdict="DISAPPROVE",
                review_text="Evaluator JSON response must be an object.",
                metadata={"llm_calls": 1},
            )

        verdict_raw = str(payload.get("verdict", "")).strip().upper()
        verdict = verdict_raw if verdict_raw in {"APPROVE", "DISAPPROVE"} else "DISAPPROVE"

        review_value = payload.get("review", "")
        review_text = str(review_value).strip() if review_value is not None else ""

        if not review_text:
            review_text = "Evaluator JSON response has empty 'review' field."
            verdict = "DISAPPROVE"

        if verdict_raw and verdict_raw not in {"APPROVE", "DISAPPROVE"}:
            review_text = f"Invalid verdict '{verdict_raw}'. {review_text}"
            verdict = "DISAPPROVE"

        return SelfCriticDecision(
            verdict=verdict,
            review_text=review_text,
            metadata={"llm_calls": 1},
        )

    async def evaluate(
        self,
        step: Any,
        candidate: str,
        base_prompt: str,
        context: Any,
        llm_client: Any,
        retries: int,
    ) -> SelfCriticDecision:
        _ = context
        llm_config = getattr(step, "llm_config", None)
        custom_instruction = ""
        if llm_config is not None:
            custom_instruction = getattr(llm_config, "self_critic_instruction", "") or ""

        critique_prompt = self._build_prompt(base_prompt, candidate, custom_instruction=custom_instruction)
        critique = await llm_client.get_response_with_retries(critique_prompt, retries=retries)
        return self._parse_decision(critique)


class LLMStepExecutor(StepExecutorBase):
    """Executor for standard LLM reasoning steps."""

    _DEFAULT_SELF_CRITIC_EVALUATOR = "llm"
    _DEFAULT_SELF_CRITIC_REVISIONS = 1

    async def _execute_with_streaming(
        self,
        llm_client: Any,
        prompt: str,
        retries: int,
        on_chunk: Callable[[str], None],
    ) -> str:
        """
        Execute LLM call with streaming and retry logic.

        Args:
            llm_client: The LLM client with stream_response method
            prompt: The prompt to send
            retries: Maximum retry attempts
            on_chunk: Callback for each chunk

        Returns:
            Complete response as string
        """
        last_error: Exception | None = None

        for attempt in range(retries):
            try:
                full_response = ""
                async for chunk in llm_client.stream_response(prompt):
                    full_response += chunk
                    try:
                        on_chunk(chunk)
                    except Exception:
                        pass  # Don't fail execution if callback fails
                return full_response
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)

        raise last_error or Exception("All streaming retries failed")

    @staticmethod
    def _resolve_execution_mode(step: StepDescription) -> ExecutionMode:
        """Resolve execution mode from per-step LLM config."""
        llm_config = getattr(step, "llm_config", None)
        if llm_config is None:
            return ExecutionMode.FAST

        mode = getattr(llm_config, "execution_mode", ExecutionMode.FAST)
        if isinstance(mode, ExecutionMode):
            return mode
        if isinstance(mode, str):
            try:
                return ExecutionMode(mode)
            except ValueError:
                return ExecutionMode.FAST
        return ExecutionMode.FAST

    @staticmethod
    def _resolve_model_name(llm_client: Any) -> str | None:
        """Resolve model name for tracing."""
        if hasattr(llm_client, "config") and hasattr(llm_client.config, "model"):
            return llm_client.config.model
        return None

    async def _execute_llm_call(
        self,
        llm_client: Any,
        prompt: str,
        retries: int,
        context: ReasoningContext,
        model_name: str | None,
        generation_name: str,
        allow_streaming: bool,
    ) -> str:
        """Execute one LLM call with optional tracing and streaming."""
        parent_span = context.metadata.get("__langfuse_span")
        generation = None
        if parent_span is not None:
            gen_kwargs: dict[str, Any] = {"name": generation_name, "input": prompt}
            if model_name:
                gen_kwargs["model"] = model_name
            generation = parent_span.start_observation(**gen_kwargs, as_type="generation")

        try:
            if allow_streaming and context.on_llm_chunk and hasattr(llm_client, "stream_response"):
                result = await self._execute_with_streaming(llm_client, prompt, retries, context.on_llm_chunk)
            else:
                result = await llm_client.get_response_with_retries(prompt, retries=retries)
        except Exception as exc:
            if generation is not None:
                generation.update(output=f"ERROR: {exc}")
                generation.end()
            raise

        if generation is not None:
            generation.update(output=result)
            generation.end()

        return result

    @staticmethod
    def _build_regeneration_prompt(base_prompt: str, candidate: str, review_text: str) -> str:
        """Build prompt to regenerate the candidate after self-critic disapproval."""
        return (
            "One or more evaluators DISAPPROVED the candidate answer.\n"
            "Regenerate the same task output with higher quality.\n"
            "Use the review notes below, and return only the improved final answer.\n\n"
            f"Original task:\n{base_prompt}\n\n"
            f"Previous answer:\n{candidate}\n\n"
            f"Review notes:\n{review_text}"
        )

    @staticmethod
    def _append_replan_feedback(full_prompt: str, context: ReasoningContext, step_number: int) -> tuple[str, bool]:
        """
        Append RE-PLAN feedback to the prompt when present.

        The executor injects per-step feedback into context metadata under
        '__replan_feedback'. The step executor consumes it as an additional
        instruction block for regenerated runs.
        """
        _ = step_number
        raw_feedback = context.metadata.get("__replan_feedback")
        if isinstance(raw_feedback, str):
            feedback_items = [raw_feedback]
        elif isinstance(raw_feedback, list):
            feedback_items = [str(item).strip() for item in raw_feedback if str(item).strip()]
        else:
            feedback_items = []

        if not feedback_items:
            return full_prompt, False

        feedback_block = "\n".join(f"- {item}" for item in feedback_items)
        updated_prompt = (
            f"{full_prompt}\n\n"
            "RE-PLAN feedback for this retry:\n"
            "Use these notes to improve your answer quality and direction.\n"
            f"{feedback_block}"
        )
        return updated_prompt, True

    @staticmethod
    def _normalize_llm_calls(value: Any) -> int:
        """Normalize optional llm_calls metadata value."""
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _resolve_disapprove_feedback(llm_config: LLMStepConfig, evaluator_name: str) -> str:
        """Resolve static regeneration feedback for disapproved evaluator."""
        feedback_map = llm_config.self_critic_disapprove_feedback
        if not isinstance(feedback_map, dict):
            return ""

        specific = feedback_map.get(evaluator_name)
        wildcard = feedback_map.get("*")
        resolved = specific if specific is not None else wildcard
        return str(resolved).strip() if resolved is not None else ""

    def _ensure_default_self_critic_evaluator(self, context: ReasoningContext) -> None:
        """Ensure built-in 'llm' self-critic evaluator is available.
        
        Note: This method is kept for backward compatibility. The default evaluator
        is now registered during context initialization to avoid race conditions.
        """
        if context.get_self_critic_evaluator(self._DEFAULT_SELF_CRITIC_EVALUATOR) is None:
            context.register_self_critic_evaluator(
                self._DEFAULT_SELF_CRITIC_EVALUATOR, LLMSelfCriticEvaluator()
            )

    async def _execute_fast_mode(
        self,
        llm_client: Any,
        full_prompt: str,
        retries: int,
        context: ReasoningContext,
        model_name: str | None,
    ) -> tuple[str, dict[str, Any]]:
        result = await self._execute_llm_call(
            llm_client=llm_client,
            prompt=full_prompt,
            retries=retries,
            context=context,
            model_name=model_name,
            generation_name="llm_generation",
            allow_streaming=True,
        )
        return result, {
            "execution_mode": ExecutionMode.FAST.value,
            "llm_calls": 1,
            "rounds": 1,
            "evaluator_decisions": [],
        }

    async def _execute_self_critic_mode(
        self,
        step: StepDescription,
        llm_config: LLMStepConfig,
        llm_client: Any,
        full_prompt: str,
        retries: int,
        context: ReasoningContext,
        model_name: str | None,
    ) -> tuple[str, dict[str, Any]]:
        draft = await self._execute_llm_call(
            llm_client=llm_client,
            prompt=full_prompt,
            retries=retries,
            context=context,
            model_name=model_name,
            generation_name="llm_generation_draft",
            allow_streaming=False,
        )
        llm_calls = 1
        self._ensure_default_self_critic_evaluator(context)

        evaluator_names = llm_config.self_critic_evaluators or [self._DEFAULT_SELF_CRITIC_EVALUATOR]
        max_revisions = max(0, llm_config.self_critic_max_revisions)

        candidate = draft
        round_summaries: list[dict[str, Any]] = []
        max_revisions_reached = False

        for revision_round in range(max_revisions + 1):
            round_approved = True
            disapprove_reviews: list[str] = []
            evaluator_decisions: list[dict[str, Any]] = []

            for evaluator_name in evaluator_names:
                evaluator = context.get_self_critic_evaluator(evaluator_name)
                if evaluator is None:
                    raise ValueError(
                        f"Self-critic evaluator '{evaluator_name}' is not registered. "
                        f"Available evaluators: {context.list_self_critic_evaluators()}"
                    )

                decision = await evaluator.evaluate(
                    step=step,
                    candidate=candidate,
                    base_prompt=full_prompt,
                    context=context,
                    llm_client=llm_client,
                    retries=retries,
                )
                if not isinstance(decision, SelfCriticDecision):
                    raise TypeError(
                        f"Self-critic evaluator '{evaluator_name}' returned invalid result type: "
                        f"{type(decision).__name__}. Expected SelfCriticDecision."
                    )

                decision_meta = decision.metadata if isinstance(decision.metadata, dict) else {}
                decision_llm_calls = self._normalize_llm_calls(decision_meta.get("llm_calls", 0))
                llm_calls += decision_llm_calls

                verdict = decision.normalized_verdict()
                review_text = (decision.review_text or "").strip()
                if verdict == "DISAPPROVE":
                    static_feedback = self._resolve_disapprove_feedback(llm_config, evaluator_name)
                    if static_feedback:
                        review_text = f"{review_text}\n{static_feedback}".strip() if review_text else static_feedback
                if not review_text:
                    review_text = f"Evaluator '{evaluator_name}' returned empty review text."
                    verdict = "DISAPPROVE"

                evaluator_decisions.append(
                    {
                        "evaluator": evaluator_name,
                        "verdict": verdict,
                        "has_review": bool(review_text),
                        "llm_calls": decision_llm_calls,
                    }
                )
                if verdict == "DISAPPROVE":
                    round_approved = False
                    disapprove_reviews.append(f"[{evaluator_name}] {review_text}")

            round_summaries.append(
                {
                    "round": revision_round + 1,
                    "approved": round_approved,
                    "evaluators": evaluator_decisions,
                }
            )

            if round_approved:
                break

            if revision_round >= max_revisions:
                max_revisions_reached = True
                break

            # Keep current round review text local; only used for this regeneration pass.
            current_review_text = "\n\n".join(disapprove_reviews).strip()
            candidate = await self._execute_llm_call(
                llm_client=llm_client,
                prompt=self._build_regeneration_prompt(full_prompt, candidate, current_review_text),
                retries=retries,
                context=context,
                model_name=model_name,
                generation_name="llm_generation_regenerate",
                allow_streaming=False,
            )
            llm_calls += 1

        mode_details = {
            "execution_mode": ExecutionMode.SELF_CRITIC.value,
            "llm_calls": llm_calls,
            "rounds": len(round_summaries),
            "max_revisions": max_revisions,
            "evaluator_policy": "all_must_approve",
            "evaluator_decisions": round_summaries,
        }
        if max_revisions_reached:
            mode_details["quality_warning"] = (
                f"Reached self_critic_max_revisions={max_revisions} without full evaluator approval."
            )
        return candidate, mode_details

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        """Execute an LLM reasoning step."""
        start_time = time.time()
        template = prompt_template or PromptTemplate()

        try:
            # Generate prompt for this step with RAG-like context extraction
            step_prompt = template.format_step_prompt(step, context.outer_context, context.language)
            full_prompt = template.format_chain_prompt(
                outer_context=context.outer_context,
                current_task=step_prompt,
                history=context.get_current_history(),
                language=context.language,
                system_prompt=context.system_prompt,
            )
            full_prompt, used_replan_feedback = self._append_replan_feedback(full_prompt, context, step.number)

            # Get LLM client for this step (may have per-step overrides)
            llm_config = getattr(step, "llm_config", None)
            llm_client = context.get_llm_client_for_step(llm_config)

            # Get retry count (per-step override or context default)
            step_retries = getattr(step, "retry_max", None)
            retries = step_retries if step_retries is not None else context.retry_max

            model_name = self._resolve_model_name(llm_client)
            execution_mode = self._resolve_execution_mode(step)

            if execution_mode == ExecutionMode.SELF_CRITIC:
                if llm_config is None:
                    llm_config = LLMStepConfig(
                        execution_mode=ExecutionMode.SELF_CRITIC,
                        self_critic_max_revisions=self._DEFAULT_SELF_CRITIC_REVISIONS,
                    )
                result, mode_details = await self._execute_self_critic_mode(
                    step=step,
                    llm_config=llm_config,
                    llm_client=llm_client,
                    full_prompt=full_prompt,
                    retries=retries,
                    context=context,
                    model_name=model_name,
                )
            else:
                result, mode_details = await self._execute_fast_mode(
                    llm_client=llm_client,
                    full_prompt=full_prompt,
                    retries=retries,
                    context=context,
                    model_name=model_name,
                )

            context.metadata.setdefault("execution_mode_details", {})
            if used_replan_feedback:
                mode_details["replan_feedback_used"] = True
            context.metadata["execution_mode_details"][str(step.number)] = mode_details

            # Update context history
            mode_suffix = "" if execution_mode == ExecutionMode.FAST else f" [{execution_mode.value}]"
            if context.language == Language.ENGLISH:
                step_result = f"Step {step.number}. {step.title}{mode_suffix}\nResult: {result}\n"
            else:  # Russian
                step_result = f"Шаг {step.number}. {step.title}{mode_suffix}\nРезультат: {result}\n"
            updated_history = context.history.copy()
            updated_history.append(step_result)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.LLM,
                result=result,
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.LLM,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )


class ToolStepExecutor(StepExecutorBase):
    """Executor for tool/function call steps."""

    def _resolve_input_value(self, source: str, context: ReasoningContext) -> Any:
        """Resolve an input value from context."""
        return resolve_context_reference(source, context)

    def _coerce_value(self, value: Any, param_name: str, config: ToolStepConfig) -> Any:
        """Coerce value to the expected type based on parameter definition."""
        # Find parameter definition
        param_def = None
        for param in config.parameters:
            if param.name == param_name:
                param_def = param
                break

        if param_def is None:
            return value  # No type info, return as-is

        # Coerce based on parameter type
        if param_def.type == "int":
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        elif param_def.type == "float":
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        elif param_def.type == "bool":
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        elif param_def.type == "list":
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value
        elif param_def.type == "dict":
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value

        return value

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        """Execute a tool call step."""
        start_time = time.time()
        config: ToolStepConfig = step.step_config  # type: ignore

        try:
            # Get the tool callable
            tool_callable = context.get_tool(config.tool_name)
            if tool_callable is None:
                raise ValueError(f"Tool '{config.tool_name}' not registered in context")

            # Build arguments from input mapping
            kwargs = {}
            for param_name, source in config.input_mapping.items():
                raw_value = self._resolve_input_value(source, context)
                kwargs[param_name] = self._coerce_value(raw_value, param_name, config)

            # Execute the tool with timeout
            if asyncio.iscoroutinefunction(tool_callable):
                result = await asyncio.wait_for(tool_callable(**kwargs), timeout=config.timeout)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: tool_callable(**kwargs)), timeout=config.timeout
                )

            # Convert result to string for history
            result_str = json.dumps(result) if not isinstance(result, str) else result

            # Format history entry
            if context.language == Language.ENGLISH:
                step_result = f"Step {step.number}. {step.title} [TOOL: {config.tool_name}]\nResult: {result_str}\n"
            else:
                step_result = f"Шаг {step.number}. {step.title} [ИНСТРУМЕНТ: {config.tool_name}]\nРезультат: {result_str}\n"

            updated_history = context.history.copy()
            updated_history.append(step_result)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.TOOL,
                result=result_str,
                result_data=result,
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except asyncio.TimeoutError:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.TOOL,
                result="",
                success=False,
                error_message=f"Tool execution timed out after {config.timeout}s",
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )
        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.TOOL,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )


class MCPStepExecutor(StepExecutorBase):
    """
    Executor for MCP (Model Context Protocol) steps.

    Note: This is an EXPERIMENTAL feature. Full MCP support requires:
      - mcp-sdk library installed (pip install mcp)
      - Proper server connection handling
      - Stdio transport is partially implemented

    For production use, consider registering MCP tools as regular Python tools
    via context.register_tool() instead.
    """

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        """Execute an MCP protocol step."""
        start_time = time.time()
        config: MCPStepConfig = step.step_config  # type: ignore

        try:
            # Build arguments
            arguments = config.arguments.copy()
            for arg_name, source in config.argument_mapping.items():
                arguments[arg_name] = self._resolve_input_value(source, context)

            # Try to use MCP SDK if available
            try:
                from mcp import ClientSession  # noqa: F401
                from mcp.client.stdio import stdio_client  # noqa: F401

                result = await self._execute_mcp_call(config, arguments)
            except ImportError:
                # MCP SDK not available - return error
                raise ImportError(
                    "MCP SDK not installed. Install with: pip install mcp"
                    "\nAlternatively, register the MCP tool as a regular tool."
                )

            # Convert result to string
            result_str = json.dumps(result) if not isinstance(result, str) else result

            if context.language == Language.ENGLISH:
                step_result = f"Step {step.number}. {step.title} [MCP: {config.server.server_name}/{config.tool_name}]\nResult: {result_str}\n"
            else:
                step_result = f"Шаг {step.number}. {step.title} [MCP: {config.server.server_name}/{config.tool_name}]\nРезультат: {result_str}\n"

            updated_history = context.history.copy()
            updated_history.append(step_result)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.MCP,
                result=result_str,
                result_data=result,
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.MCP,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )

    def _resolve_input_value(self, source: str, context: ReasoningContext) -> Any:
        """Resolve an input value from context."""
        return resolve_context_reference(source, context)

    async def _execute_mcp_call(self, config: MCPStepConfig, arguments: dict) -> Any:
        """Execute the actual MCP call. Requires mcp-sdk."""
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        if config.server.transport == "stdio":
            server_params = StdioServerParameters(
                command=config.server.command or "",
                args=config.server.args,
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.call_tool(config.tool_name, arguments=arguments), timeout=config.timeout
                    )
                    return result.content
        else:
            raise NotImplementedError(f"MCP transport '{config.server.transport}' not yet implemented")


class MemoryStepExecutor(StepExecutorBase):
    """Executor for memory read/write steps."""

    def _resolve_value(self, source: str, context: ReasoningContext) -> Any:
        """Resolve a value from context for write operations."""
        resolved = resolve_context_reference(source, context)
        return source if resolved is None and not source.startswith("$") else resolved

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        """Execute a memory operation step."""
        start_time = time.time()
        config: MemoryStepConfig = step.step_config  # type: ignore

        try:
            result_data: Any = None
            result_str = ""

            if config.operation == MemoryOperation.READ:
                result_data = context.memory_read(config.memory_key, config.namespace, config.default_value)
                result_str = json.dumps(result_data) if not isinstance(result_data, str) else result_data

            elif config.operation == MemoryOperation.WRITE:
                if config.value_source:
                    value = self._resolve_value(config.value_source, context)
                else:
                    value = config.default_value
                context.memory_write(config.memory_key, value, config.namespace)
                result_str = f"Written to {config.namespace}.{config.memory_key}"
                result_data = {"key": config.memory_key, "namespace": config.namespace, "value": value}

            elif config.operation == MemoryOperation.APPEND:
                if config.value_source:
                    value = self._resolve_value(config.value_source, context)
                else:
                    value = config.default_value
                context.memory_append(config.memory_key, value, config.namespace)
                result_str = f"Appended to {config.namespace}.{config.memory_key}"
                result_data = {"key": config.memory_key, "namespace": config.namespace, "appended": value}

            elif config.operation == MemoryOperation.DELETE:
                existed = context.memory_delete(config.memory_key, config.namespace)
                result_str = f"Deleted {config.namespace}.{config.memory_key}" if existed else "Key not found"
                result_data = {"key": config.memory_key, "namespace": config.namespace, "existed": existed}

            elif config.operation == MemoryOperation.LIST:
                keys = context.memory_list(config.namespace)
                result_str = ", ".join(keys) if keys else "(empty)"
                result_data = keys

            # Format history entry
            if context.language == Language.ENGLISH:
                step_result = f"Step {step.number}. {step.title} [MEMORY: {config.operation}]\nResult: {result_str}\n"
            else:
                step_result = f"Шаг {step.number}. {step.title} [ПАМЯТЬ: {config.operation}]\nРезультат: {result_str}\n"

            updated_history = context.history.copy()
            updated_history.append(step_result)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.MEMORY,
                result=result_str,
                result_data=result_data,
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.MEMORY,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )


class TransformStepExecutor(StepExecutorBase):
    """Executor for data transformation steps (no LLM call)."""

    def _get_input(self, input_key: str, context: ReasoningContext) -> Any:
        """Get input data based on input_key."""
        resolved = resolve_context_reference(input_key, context)
        return "" if resolved is None else resolved

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        """Execute a data transformation step."""
        start_time = time.time()
        config: TransformStepConfig = step.step_config  # type: ignore

        try:
            input_data = self._get_input(config.input_key, context)
            result_data: Any = None

            if config.transform_type == "extract":
                # Extract using regex expression
                if config.expression:
                    matches = re.findall(config.expression, str(input_data))
                    result_data = matches
                else:
                    result_data = input_data

            elif config.transform_type == "format":
                # Format using output_format template
                if config.output_format:
                    result_data = config.output_format.format(input=input_data)
                else:
                    result_data = str(input_data)

            elif config.transform_type == "aggregate":
                # Simple aggregation - join if list
                if isinstance(input_data, list):
                    result_data = "\n".join(str(item) for item in input_data)
                else:
                    result_data = str(input_data)

            elif config.transform_type == "filter":
                # Filter using expression as regex
                if isinstance(input_data, list) and config.expression:
                    result_data = [item for item in input_data if re.search(config.expression, str(item))]
                elif isinstance(input_data, str) and config.expression:
                    lines = input_data.split("\n")
                    result_data = [line for line in lines if re.search(config.expression, line)]
                else:
                    result_data = input_data

            elif config.transform_type == "map":
                # Apply template to each item
                if isinstance(input_data, list) and config.map_template:
                    result_data = [config.map_template.format(item=item) for item in input_data]
                else:
                    result_data = input_data

            result_str = json.dumps(result_data) if not isinstance(result_data, str) else result_data

            # Format history entry
            if context.language == Language.ENGLISH:
                step_result = f"Step {step.number}. {step.title} [TRANSFORM: {config.transform_type}]\nResult: {result_str}\n"
            else:
                step_result = f"Шаг {step.number}. {step.title} [ТРАНСФОРМАЦИЯ: {config.transform_type}]\nРезультат: {result_str}\n"

            updated_history = context.history.copy()
            updated_history.append(step_result)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.TRANSFORM,
                result=result_str,
                result_data=result_data,
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.TRANSFORM,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )


class ConditionalStepExecutor(StepExecutorBase):
    """
    Executor for conditional branching steps.

    IMPORTANT LIMITATION:
        Conditional steps currently return the next_step number in result_data,
        but the DAG executor proceeds in topological order and does NOT follow
        the branch. This is a known limitation.

        For actual branching behavior, you have two options:
        1. Design your chain with separate branches using dependencies
        2. Use multiple chains and select which one to execute based on conditions

        Example workaround:
        ```python
        # Instead of conditional branching:
        chain = ReasoningChain([
            step_1,
            conditional_step,  # Returns next_step but doesn't affect execution
            step_2a,  # Will execute regardless
            step_2b,  # Will also execute
        ])

        # Use dependency-based branching:
        chain = ReasoningChain([
            step_1,
            step_2a,  # dependencies=[1]
            step_2b,  # dependencies=[1]
            step_3,   # dependencies=[2a, 2b] - waits for both
        ])
        ```

        Future versions may implement true branching execution.

    The executor evaluates conditions using simpleeval for safety.
    """

    def __init__(self):
        """Initialize with safe expression evaluator."""
        self._evaluator = EvalWithCompoundTypes()
        # Only allow safe functions - no code execution
        self._evaluator.functions = {
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'bool': bool,
        }

    def _evaluate_condition(self, condition: str, value: Any) -> bool:
        """
        Evaluate a condition against a value safely.

        Supports simple conditions like:
        - "contains:keyword" - checks if value contains keyword
        - "equals:value" - checks equality
        - "startswith:prefix" - checks prefix
        - "endswith:suffix" - checks suffix
        - "matches:regex" - regex match
        - "empty" - checks if empty
        - "nonempty" - checks if not empty

        For complex expressions (requires simpleeval):
        - "len(value) > 5"
        - "value != 'skip'"
        - "int(value) >= 10"
        """
        value_str = str(value)

        # Built-in condition patterns (always supported, no eval)
        if condition.startswith("contains:"):
            return condition[9:] in value_str
        elif condition.startswith("equals:"):
            return value_str == condition[7:]
        elif condition.startswith("startswith:"):
            return value_str.startswith(condition[11:])
        elif condition.startswith("endswith:"):
            return value_str.endswith(condition[9:])
        elif condition.startswith("matches:"):
            return bool(re.search(condition[8:], value_str))
        elif condition == "empty":
            return not value_str.strip()
        elif condition == "nonempty":
            return bool(value_str.strip())

        # Complex expressions evaluated safely via simpleeval
        try:
            self._evaluator.names = {"value": value, "v": value}
            result = self._evaluator.eval(condition)
            return bool(result)
        except Exception:
            return False

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        """
        Execute a conditional step.

        Returns the next step number in result_data.
        """
        start_time = time.time()
        config: ConditionalStepConfig = step.step_config  # type: ignore

        try:
            # Get the value to evaluate
            value = self._get_condition_value(config.condition_context_key, context)

            # Evaluate branches
            next_step: Optional[int] = None
            matched_condition = ""

            for branch in config.branches:
                if self._evaluate_condition(branch.condition, value):
                    next_step = branch.next_step
                    matched_condition = branch.condition
                    break

            if next_step is None:
                next_step = config.default_step
                matched_condition = "(default)"

            result_str = f"Condition matched: {matched_condition}, next step: {next_step}"

            # Format history entry
            if context.language == Language.ENGLISH:
                step_result = f"Step {step.number}. {step.title} [CONDITIONAL]\nResult: {result_str}\n"
            else:
                step_result = f"Шаг {step.number}. {step.title} [УСЛОВИЕ]\nРезультат: {result_str}\n"

            updated_history = context.history.copy()
            updated_history.append(step_result)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.CONDITIONAL,
                result=result_str,
                result_data={"next_step": next_step, "matched_condition": matched_condition},
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.CONDITIONAL,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )

    def _get_condition_value(self, key: str, context: ReasoningContext) -> Any:
        """Get the value to evaluate conditions against."""
        resolved = resolve_context_reference(key, context)
        return "" if resolved is None else resolved


class StructuredOutputStepExecutor(StepExecutorBase):
    """Executor for structured output generation with schema validation."""

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        start_time = time.time()
        config: StructuredOutputStepConfig = step.step_config  # type: ignore

        try:
            input_data = resolve_context_reference(config.input_source, context)
            serialized_input = input_data if isinstance(input_data, str) else json.dumps(input_data, ensure_ascii=False)

            schema_json = json.dumps(config.output_schema, indent=2, ensure_ascii=False)
            strict_instruction = "Return ONLY valid JSON with no markdown or additional text." if config.strict_json else "Return valid JSON."
            full_prompt = (
                f"You are a structured output generator.\n"
                f"Task: {config.instruction or step.title}\n"
                f"Schema name: {config.schema_name}\n"
                f"JSON Schema:\n{schema_json}\n\n"
                f"Input:\n{serialized_input}\n\n"
                f"{strict_instruction}"
            )

            llm_config = getattr(step, "llm_config", None)
            llm_client = context.get_llm_client_for_step(llm_config)

            # Get retry count (per-step override or context default)
            step_retries = getattr(step, "retry_max", None)
            retries = step_retries if step_retries is not None else context.retry_max

            # Resolve model name for LangFuse generation tracing
            model_name = None
            if hasattr(llm_client, "config") and hasattr(llm_client.config, "model"):
                model_name = llm_client.config.model

            # Create a LangFuse generation observation inside the step span
            parent_span = context.metadata.get("__langfuse_span")
            generation = None
            if parent_span is not None:
                gen_kwargs: dict[str, Any] = {"name": "llm_generation", "input": full_prompt}
                if model_name:
                    gen_kwargs["model"] = model_name
                generation = parent_span.start_observation(**gen_kwargs, as_type="generation")

            raw_result = await llm_client.get_response_with_retries(full_prompt, retries=retries)

            if generation is not None:
                generation.update(output=raw_result)
                generation.end()

            parsed = _extract_json_payload(raw_result)

            if context.language == Language.ENGLISH:
                step_result = (
                    f"Step {step.number}. {step.title} [STRUCTURED_OUTPUT]\n"
                    f"Result: {json.dumps(parsed, ensure_ascii=False)}\n"
                )
            else:
                step_result = (
                    f"Шаг {step.number}. {step.title} [СТРУКТУРИРОВАННЫЙ ВЫВОД]\n"
                    f"Результат: {json.dumps(parsed, ensure_ascii=False)}\n"
                )

            updated_history = context.history.copy()
            updated_history.append(step_result)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.STRUCTURED_OUTPUT,
                result=json.dumps(parsed, ensure_ascii=False),
                result_data=parsed,
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )
        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.STRUCTURED_OUTPUT,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )


# =============================================================================
# Step Executor Factory
# =============================================================================


_EXECUTORS: dict[StepType, StepExecutorBase] = {
    StepType.LLM: LLMStepExecutor(),
    StepType.TOOL: ToolStepExecutor(),
    StepType.MCP: MCPStepExecutor(),
    StepType.MEMORY: MemoryStepExecutor(),
    StepType.TRANSFORM: TransformStepExecutor(),
    StepType.CONDITIONAL: ConditionalStepExecutor(),
    StepType.STRUCTURED_OUTPUT: StructuredOutputStepExecutor(),
}


def get_executor(step_type: StepType) -> StepExecutorBase:
    """Get the executor for a given step type."""
    executor = _EXECUTORS.get(step_type)
    if executor is None:
        raise ValueError(f"No executor registered for step type: {step_type}")
    return executor


def register_executor(step_type: StepType, executor: StepExecutorBase) -> None:
    """Register a custom executor for a step type."""
    _EXECUTORS[step_type] = executor

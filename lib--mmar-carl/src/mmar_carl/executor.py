"""
DAG execution engine for CARL reasoning chains.

Handles parallel execution of reasoning steps based on their dependencies.
Supports multiple step types through pluggable executors.
"""

import asyncio
import copy
import time
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .logging_utils import (
    log_batch_start,
    log_chain_complete,
    log_chain_start,
    log_step_complete,
    log_step_start,
)
from .models import (
    AnyStepDescription,
    PromptTemplate,
    ReasoningContext,
    ReasoningResult,
    ReplanAction,
    ReplanAggregationOutcome,
    ReplanCheckerInput,
    ReplanCheckerVote,
    ReplanEvent,
    ReplanPolicy,
    ReplanRollbackTarget,
    ReplanTargetType,
    ReplanVerdict,
    StepDescription,
    StepDescriptionBase,
    StepExecutionResult,
    StepType,
)
from .replan import CheckerVote, aggregate_replan_votes, create_checker_from_spec
from .step_executors import get_executor
from .tracing import create_chain_trace
from mmar_utils import gather_with_limit


class ExecutionCancelledError(Exception):
    """Raised when chain execution is cancelled by user."""

    pass


class ExecutionNode(BaseModel):
    """
    Represents a node in the execution DAG.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step: Any
    dependencies: list["ExecutionNode"] = Field(default_factory=list)
    dependents: list["ExecutionNode"] = Field(default_factory=list)
    executed: bool = False
    executing: bool = False
    result: StepExecutionResult | None = None

    def can_execute(self) -> bool:
        """Check if this node can be executed (all dependencies completed)."""
        return all(dep.executed for dep in self.dependencies)

    def is_ready(self) -> bool:
        """Check if this node is ready for execution (not executing or executed)."""
        return not self.executing and not self.executed and self.can_execute()


ExecutionNode.model_rebuild()


class _ExecutionSnapshot(BaseModel):
    """Snapshot of mutable chain execution state for rollback support."""

    executed_nodes: set[int]
    all_results: list[StepExecutionResult]
    history: list[str]
    metadata: dict[str, Any]
    memory: dict[str, dict[str, Any]]


class _CheckpointSnapshot(BaseModel):
    """Named checkpoint snapshot with ordering metadata."""

    name: str
    step_number: int
    sequence: int
    snapshot: _ExecutionSnapshot


class DAGExecutor:
    """
    Executes reasoning chains as a Directed Acyclic Graph (DAG).

    Automatically parallelizes execution where dependencies allow.

    Note on parallel execution:
        - Each parallel step gets an isolated copy of memory (deep copy)
        - Tool registry is shared via shallow copy - tools MUST be stateless
        - Memory writes in parallel steps ARE merged back after batch completion
          but are NOT visible to other parallel steps in the same batch

    Note on conditional steps:
        - CONDITIONAL steps return next_step in result_data but DAG execution
          currently proceeds in topological order
        - For branching behavior, use separate chains or design dependencies accordingly
        - This is a known limitation that may be addressed in future versions

    Note on cancellation:
        - Call context.cancel() to request cancellation
        - Execution will stop after the current batch completes
        - Partial results are returned with success=False

    Performance Notes:
        For chains with large memory payloads, deep copying for parallel
        execution can be expensive. Consider:

        1. Using smaller, focused memory namespaces
        2. Reducing max_workers if memory is huge (default is 1)
        3. Using external state storage (Redis, etc.) for very large state
        4. Setting max_workers=1 for memory-heavy chains to avoid deep copies

    Token Usage Tracking:
        Token usage is tracked per step and aggregated in ReasoningResult.
        Note: Token counts are only available when the LLM client provides them
        (e.g., OpenAI API responses include usage data).
    """

    def __init__(
        self,
        max_workers: int = 1,
        prompt_template: PromptTemplate | None = None,
        enable_progress: bool = False,
        timeout: float | None = None,
        replan_policy: ReplanPolicy | None = None,
    ):
        """
        Initialize the DAG executor.

        Args:
            max_workers: Maximum number of parallel executions
            prompt_template: Template for generating prompts
            enable_progress: Whether to enable progress tracking
            timeout: Maximum total execution time in seconds (None = no limit)
        """
        self.max_workers = max_workers
        self.prompt_template = prompt_template or PromptTemplate()
        self.enable_progress = enable_progress
        self.timeout = timeout
        self.replan_policy = replan_policy
        self._execution_stats = {
            "total_steps": 0,
            "executed_steps": 0,
            "failed_steps": 0,
            "parallel_batches": 0,
            "total_time": 0.0,
        }

    def build_execution_graph(self, steps: Sequence[StepDescription | StepDescriptionBase | AnyStepDescription]) -> list[ExecutionNode]:
        """
        Build an execution graph from step descriptions.

        Args:
            steps: list of step descriptions

        Returns:
            list of execution nodes forming the DAG
        """
        if not steps:
            return []

        # Create nodes
        step_map: dict[int, ExecutionNode] = {}
        for step in steps:
            node = ExecutionNode(step=step, dependencies=[], dependents=[])
            step_map[step.number] = node

        # Build dependencies
        for step in steps:
            node = step_map[step.number]
            for dep_number in step.dependencies:
                if dep_number in step_map:
                    dependency_node = step_map[dep_number]
                    node.dependencies.append(dependency_node)
                    dependency_node.dependents.append(node)
                else:
                    raise ValueError(
                        f"Step {step.number} ('{step.title}') depends on non-existent step {dep_number}. "
                        f"Available steps: {sorted(step_map.keys())}. "
                        "Check your dependencies configuration."
                    )

        # Validate no cycles
        self._validate_no_cycles(step_map)

        return list(step_map.values())

    def _validate_no_cycles(self, nodes: dict[int, ExecutionNode]) -> None:
        """
        Validate that the execution graph has no cycles.

        Args:
            nodes: Dictionary of execution nodes

        Raises:
            ValueError: If cycles are detected
        """
        visited = set()
        rec_stack = set()

        def has_cycle(node_num: int) -> bool:
            visited.add(node_num)
            rec_stack.add(node_num)

            node = nodes[node_num]
            for dep in node.dependencies:
                if dep.step.number not in visited:
                    if has_cycle(dep.step.number):
                        return True
                elif dep.step.number in rec_stack:
                    return True

            rec_stack.remove(node_num)
            return False

        for node_num in nodes:
            if node_num not in visited:
                if has_cycle(node_num):
                    raise ValueError(f"Cycle detected in execution graph involving step {node_num}")

    async def execute_step(self, node: ExecutionNode, context: ReasoningContext) -> StepExecutionResult:
        """
        Execute a single step using the appropriate executor based on step type.

        Args:
            node: Execution node to execute
            context: Reasoning context

        Returns:
            Step execution result
        """
        step = node.step
        start_time = time.time()

        # Log step start
        log_step_start(step.number, step.title, str(step.step_type))

        # Get per-step timeout (step override > chain default > None)
        step_timeout = getattr(step, "timeout", None) or self.timeout

        # Call step_start callback if registered
        if context.on_step_start:
            try:
                context.on_step_start(step.number, step.title)
            except Exception:
                # Don't fail execution if callback fails
                pass

        # Create a LangFuse span for this step if tracing is active
        trace = context.metadata.get("__langfuse_trace")
        span = None
        if trace is not None:
            span_input: dict[str, Any] = {
                "step_type": str(step.step_type),
                "number": step.number,
                "title": step.title,
                "dependencies": getattr(step, "dependencies", []),
            }

            # Add step-type-specific details to span input
            step_config = getattr(step, "step_config", None)
            if step_config is not None:
                from .models import StepType as ST

                if step.step_type == ST.TOOL:
                    span_input["tool_name"] = getattr(step_config, "tool_name", None)
                    span_input["input_mapping"] = getattr(step_config, "input_mapping", {})
                elif step.step_type == ST.MEMORY:
                    span_input["operation"] = str(getattr(step_config, "operation", ""))
                    span_input["memory_key"] = getattr(step_config, "memory_key", "")
                    span_input["namespace"] = getattr(step_config, "namespace", "default")
                elif step.step_type == ST.TRANSFORM:
                    span_input["transform_type"] = getattr(step_config, "transform_type", "")
                    span_input["input_key"] = getattr(step_config, "input_key", "")
                elif step.step_type == ST.MCP:
                    span_input["tool_name"] = getattr(step_config, "tool_name", None)
                    server = getattr(step_config, "server", None)
                    if server:
                        span_input["server_name"] = getattr(server, "server_name", "")
                elif step.step_type == ST.CONDITIONAL:
                    span_input["condition_key"] = getattr(step_config, "condition_context_key", "")

            span = trace.start_span(
                name=f"step_{step.number}: {step.title}",
                input=span_input,
            )
            context.metadata["__langfuse_span"] = span

        # Get the appropriate executor for this step type
        executor = get_executor(step.step_type)

        # Inject step-specific RE-PLAN feedback into executor-visible metadata.
        replan_feedback_map = context.metadata.get("__replan_feedback_by_step")
        step_feedback: list[str] | None = None
        if isinstance(replan_feedback_map, dict):
            raw_feedback = replan_feedback_map.get(str(step.number))
            if isinstance(raw_feedback, str):
                step_feedback = [raw_feedback]
            elif isinstance(raw_feedback, list):
                step_feedback = [str(item) for item in raw_feedback if str(item).strip()]
        if step_feedback:
            context.metadata["__replan_feedback"] = step_feedback

        # Execute the step with optional per-step timeout
        try:
            if step_timeout is not None:
                result = await asyncio.wait_for(
                    executor.execute(step, context, self.prompt_template), timeout=step_timeout
                )
            else:
                result = await executor.execute(step, context, self.prompt_template)
        except asyncio.TimeoutError:
            result = StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=step.step_type,
                result="",
                success=False,
                error_message=f"Step timed out after {step_timeout}s",
            )
        finally:
            context.metadata.pop("__replan_feedback", None)

        # Run step-level metrics on successful results
        step_metrics = getattr(step, "metrics", [])
        if step_metrics and result.success and result.result:
            metric_scores: dict[str, float] = {}
            for metric in step_metrics:
                try:
                    score = await metric.compute_async(result)
                    metric_scores[metric.name] = float(score)
                except Exception as e:
                    from .logging_utils import log_warning
                    log_warning(f"Metric '{getattr(metric, 'name', repr(metric))}' failed on step {step.number}: {e}")
            if metric_scores:
                result = result.model_copy(update={"metrics": metric_scores})

        # End the span with step outcome
        if span is not None:
            end_output: dict[str, Any] = {"success": result.success}
            if result.result:
                end_output["result"] = result.result[:1000]
            if result.result_data is not None:
                try:
                    import json as _json

                    serialized = _json.dumps(result.result_data, default=str)
                    end_output["result_data"] = _json.loads(serialized)
                except (TypeError, ValueError):
                    end_output["result_data"] = str(result.result_data)[:1000]
            if result.error_message:
                end_output["error"] = result.error_message
            if result.execution_time:
                end_output["execution_time"] = result.execution_time
            if result.metrics:
                end_output["metrics"] = result.metrics
            span.update(output=end_output)
            span.end()
            context.metadata.pop("__langfuse_span", None)

        # Call step_complete callback if registered
        if context.on_step_complete:
            try:
                context.on_step_complete(result)
            except Exception:
                # Don't fail execution if callback fails
                pass

        # Log step completion
        log_step_complete(step.number, result.success, time.time() - start_time)

        return result

    async def execute_batch(
        self, ready_nodes: list[ExecutionNode], context: ReasoningContext
    ) -> list[StepExecutionResult]:
        """
        Execute a batch of ready nodes in parallel.

        Args:
            ready_nodes: list of nodes ready for execution
            context: Reasoning context

        Returns:
            list of step execution results
        """
        if not ready_nodes:
            return []

        # Create independent contexts for parallel execution
        context_snapshots = []
        try:
            for _ in ready_nodes:
                snapshot = ReasoningContext(
                    outer_context=context.outer_context,
                    api=context.api,
                    model=context.model,
                    retry_max=context.retry_max,
                    history=context.history.copy(),
                    metadata=context.metadata.copy(),
                    language=context.language,
                    system_prompt=context.system_prompt,
                    memory=copy.deepcopy(context.memory),  # Deep copy memory for isolation
                    max_history_entries=context.max_history_entries,
                    # Preserve callbacks
                    on_step_start=context.on_step_start,
                    on_step_complete=context.on_step_complete,
                    on_progress=context.on_progress,
                    on_llm_chunk=context.on_llm_chunk,
                )
                # Copy tool registry (shallow copy is fine for callables)
                snapshot._tool_registry = context._tool_registry.copy()
                # Copy self-critic evaluator registry
                snapshot._self_critic_evaluator_registry = context._self_critic_evaluator_registry.copy()
                # Copy RE-PLAN checker registry
                snapshot._replan_checker_registry = context._replan_checker_registry.copy()
                # Preserve cancellation state
                snapshot._cancelled = context._cancelled
                context_snapshots.append(snapshot)

            # Execute in parallel
            tasks = [self.execute_step(node, ctx) for node, ctx in zip(ready_nodes, context_snapshots)]

            if self.max_workers == 1:
                results = [(await task) for task in tasks]
            else:
                results = await gather_with_limit(*tasks, return_exceptions=True, max_workers=self.max_workers)

            # Process results and handle exceptions
            processed_results = []
            import traceback as _traceback
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_tb = _traceback.format_exception(type(result), result, result.__traceback__)
                    processed_results.append(
                        StepExecutionResult(
                            step_number=ready_nodes[i].step.number,
                            step_title=ready_nodes[i].step.title,
                            step_type=ready_nodes[i].step.step_type,
                            result="",
                            success=False,
                            error_message=str(result),
                            error_traceback="".join(error_tb),
                        )
                    )
                else:
                    processed_results.append(result)

            # Merge memory changes back from successful parallel steps
            # Memory writes in parallel steps are now visible to subsequent batches
            for i, result in enumerate(processed_results):
                if result.success and context_snapshots[i].memory:
                    for namespace, data in context_snapshots[i].memory.items():
                        if namespace not in context.memory:
                            context.memory[namespace] = {}
                        # Merge the namespace data
                        context.memory[namespace].update(data)

            # Merge execution mode diagnostics from snapshot contexts.
            context.metadata.setdefault("execution_mode_details", {})
            for i, result in enumerate(processed_results):
                if not result.success:
                    continue
                snapshot_mode_details = context_snapshots[i].metadata.get("execution_mode_details")
                if not isinstance(snapshot_mode_details, dict):
                    continue
                step_key = str(result.step_number)
                if step_key in snapshot_mode_details:
                    context.metadata["execution_mode_details"][step_key] = snapshot_mode_details[step_key]

            return processed_results
        finally:
            # Close all snapshot contexts to prevent event loop issues
            # (Each snapshot creates its own LLM clients via model_post_init)
            # This ensures cleanup even if an exception occurs during execution
            for snapshot in context_snapshots:
                try:
                    await snapshot.close()
                except Exception:
                    pass  # Don't fail if cleanup fails

    def _capture_snapshot(
        self,
        *,
        executed_nodes: set[int],
        all_results: list[StepExecutionResult],
        context: ReasoningContext,
    ) -> _ExecutionSnapshot:
        """Capture mutable state for rollback."""
        return _ExecutionSnapshot(
            executed_nodes=executed_nodes.copy(),
            all_results=[result.model_copy(deep=True) for result in all_results],
            history=context.history.copy(),
            metadata=copy.deepcopy(context.metadata),
            memory=copy.deepcopy(context.memory),
        )

    def _restore_snapshot(
        self,
        snapshot: _ExecutionSnapshot,
        *,
        nodes: list[ExecutionNode],
        context: ReasoningContext,
    ) -> tuple[set[int], list[StepExecutionResult]]:
        """Restore mutable state from a snapshot."""
        executed_nodes = snapshot.executed_nodes.copy()
        all_results = [result.model_copy(deep=True) for result in snapshot.all_results]

        context.history = snapshot.history.copy()
        context.metadata = copy.deepcopy(snapshot.metadata)
        context.memory = copy.deepcopy(snapshot.memory)

        for node in nodes:
            node.executed = node.step.number in executed_nodes
            node.executing = False
            node.result = None

        return executed_nodes, all_results

    @staticmethod
    def _is_checkpoint_step(step: StepDescription | StepDescriptionBase | AnyStepDescription) -> bool:
        return bool(getattr(step, "checkpoint", False))

    @staticmethod
    def _checkpoint_name(step: StepDescription | StepDescriptionBase | AnyStepDescription) -> str:
        checkpoint_name = getattr(step, "checkpoint_name", None)
        return checkpoint_name if checkpoint_name else f"step_{step.number}"

    @staticmethod
    def _should_evaluate_replan(
        *,
        policy: ReplanPolicy,
        step: StepDescription | StepDescriptionBase | AnyStepDescription,
        result: StepExecutionResult,
    ) -> bool:
        trigger = policy.trigger

        # Step-level override can force-disable/enable checks.
        step_replan_enabled = getattr(step, "replan_enabled", None)
        if step_replan_enabled is False:
            return False

        if trigger.step_numbers and step.number not in trigger.step_numbers:
            return False
        if trigger.step_types and step.step_type not in trigger.step_types:
            return False
        if trigger.checkpoint_only and not bool(getattr(step, "checkpoint", False)):
            return False

        if result.success:
            return trigger.evaluate_after_step
        return trigger.evaluate_after_failure

    @staticmethod
    def _budget_snapshot(
        *,
        chain_replans: int,
        per_step_replans: dict[int, int],
        target_visits: dict[str, int],
        same_target_streak: int,
    ) -> dict[str, Any]:
        return {
            "chain_replans": chain_replans,
            "per_step_replans": {str(step): count for step, count in per_step_replans.items()},
            "target_visits": dict(target_visits),
            "same_target_streak": same_target_streak,
        }

    def _build_checker_input(
        self,
        *,
        result: StepExecutionResult,
        step: StepDescription | StepDescriptionBase | AnyStepDescription,
        context: ReasoningContext,
        checkpoints: list[_CheckpointSnapshot],
        budget_snapshot: dict[str, Any],
        all_results: list[StepExecutionResult],
    ) -> ReplanCheckerInput:
        recent_errors = [
            failed.error_message or "" for failed in all_results if not failed.success and failed.error_message
        ]
        checkpoint_names = [checkpoint.name for checkpoint in checkpoints]
        checkpoint_steps = [checkpoint.step_number for checkpoint in checkpoints]

        return ReplanCheckerInput(
            step_number=result.step_number,
            step_title=result.step_title,
            step_type=result.step_type,
            step_success=result.success,
            step_result=result.result,
            step_result_data=result.result_data,
            step_error=result.error_message,
            history=context.history.copy(),
            recent_errors=recent_errors,
            checkpoint_names=checkpoint_names,
            checkpoint_steps=checkpoint_steps,
            budget_snapshot=budget_snapshot,
            metadata={"step_results_total": len(all_results)},
        )

    @staticmethod
    def _normalize_feedback(verdict: ReplanVerdict) -> list[str]:
        feedback: list[str] = []
        if verdict.reason:
            feedback.append(verdict.reason)
        feedback.extend(hint for hint in verdict.regeneration_hints if hint)
        # Preserve order and remove duplicates.
        seen: set[str] = set()
        deduped: list[str] = []
        for item in feedback:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped

    def _resolve_rollback_target(
        self,
        *,
        action: ReplanAction,
        verdict: ReplanVerdict,
        policy: ReplanPolicy,
        current_step_number: int,
    ) -> ReplanRollbackTarget | None:
        if action == ReplanAction.CONTINUE or action == ReplanAction.FAIL:
            return None
        if action == ReplanAction.RETRY_CURRENT_STEP:
            return ReplanRollbackTarget(target_type=ReplanTargetType.CURRENT_STEP)
        if action == ReplanAction.RESTART_CHAIN:
            return ReplanRollbackTarget(target_type=ReplanTargetType.CHAIN_START)
        if action == ReplanAction.REPLAN_FROM_CHECKPOINT:
            if verdict.suggested_target is not None:
                return verdict.suggested_target
            return policy.default_checkpoint_target
        # Defensive fallback.
        return ReplanRollbackTarget(target_type=ReplanTargetType.CURRENT_STEP, step_number=current_step_number)

    def _resolve_target_snapshot(
        self,
        *,
        target: ReplanRollbackTarget,
        current_step_number: int,
        chain_start_snapshot: _ExecutionSnapshot,
        pre_step_snapshots: dict[int, _ExecutionSnapshot],
        checkpoints: list[_CheckpointSnapshot],
    ) -> tuple[_ExecutionSnapshot | None, str]:
        if target.target_type == ReplanTargetType.CHAIN_START:
            return chain_start_snapshot, "chain_start"

        if target.target_type == ReplanTargetType.CURRENT_STEP:
            snapshot = pre_step_snapshots.get(current_step_number)
            return snapshot, "current_step"

        if target.target_type == ReplanTargetType.STEP_NUMBER:
            if target.step_number is None:
                return None, "step:missing"
            snapshot = pre_step_snapshots.get(target.step_number)
            return snapshot, f"step:{target.step_number}"

        if target.target_type == ReplanTargetType.NAMED_CHECKPOINT:
            name = target.checkpoint_name or ""
            for checkpoint in reversed(checkpoints):
                if checkpoint.name == name:
                    return checkpoint.snapshot, f"named:{name}"
            return None, f"named:{name}"

        if target.target_type == ReplanTargetType.NEAREST_CHECKPOINT:
            for checkpoint in reversed(checkpoints):
                if checkpoint.step_number < current_step_number:
                    return checkpoint.snapshot, f"named:{checkpoint.name}"
            return chain_start_snapshot, "chain_start"

        return None, "unknown"

    async def execute(self, steps: Sequence[StepDescription | StepDescriptionBase | AnyStepDescription], context: ReasoningContext) -> ReasoningResult:
        """
        Execute a complete reasoning chain.

        Args:
            steps: list of step descriptions
            context: Initial reasoning context

        Returns:
            Complete reasoning result

        Raises:
            TimeoutError: If execution exceeds the configured timeout
            ExecutionCancelledError: If cancellation is requested via context.cancel()
        """
        start_time = time.time()
        self._execution_stats["total_steps"] = len(steps)

        # Reset cancellation state for new execution
        context.reset_cancellation()

        # Create a LangFuse trace for this chain execution (no-op when disabled)
        chain_meta = context.metadata.get("chain_metadata", {})
        chain_name = chain_meta.get("trace_name") or chain_meta.get("name", "*Unnamed CARL Chain*")
        session_id = chain_meta.get("session_id")

        # Log chain start
        log_chain_start(chain_name, len(steps), self.max_workers)
        trace = create_chain_trace(
            chain_name=chain_name,
            context_preview=context.outer_context,
            total_steps=len(steps),
            language=str(context.language),
            session_id=session_id,
        )
        context.metadata["__langfuse_trace"] = trace

        # Build execution graph
        nodes = self.build_execution_graph(steps)
        if not nodes:
            trace.update(output={"success": True, "executed_steps": 0})
            trace.end()
            result = ReasoningResult(
                success=True, history=[], step_results=[], total_execution_time=time.time() - start_time
            )
            if context.on_chain_complete:
                try:
                    context.on_chain_complete(result)
                except Exception:
                    pass
            return result

        step_by_number: dict[int, StepDescription | StepDescriptionBase | AnyStepDescription] = {step.number: step for step in steps}

        # Build RE-PLAN checker runtimes once per chain run.
        replan_policy = self.replan_policy
        replan_checkers: list[tuple[str, Any]] = []
        if replan_policy and replan_policy.enabled and replan_policy.checkers:
            for checker_spec in replan_policy.checkers:
                checker_name = getattr(checker_spec, "name", type(checker_spec).__name__)
                checker_runtime = create_checker_from_spec(checker_spec, context)
                replan_checkers.append((checker_name, checker_runtime))

        replan_enabled = bool(replan_policy and replan_policy.enabled and replan_checkers)
        replan_events: list[ReplanEvent] = []

        # Execute DAG
        executed_nodes: set[int] = set()
        all_results: list[StepExecutionResult] = []
        current_history = context.history.copy()
        batch_count = 0
        cancelled = False
        replan_failed = False
        replan_fail_message = ""

        # RE-PLAN runtime state
        chain_start_snapshot = self._capture_snapshot(
            executed_nodes=executed_nodes,
            all_results=all_results,
            context=context,
        )
        pre_step_snapshots: dict[int, _ExecutionSnapshot] = {}
        checkpoints: list[_CheckpointSnapshot] = []
        chain_replans = 0
        per_step_replans: dict[int, int] = {}
        rollback_target_visits: dict[str, int] = {}
        last_rollback_target_key: str | None = None
        same_target_streak = 0

        # Ensure internal metadata containers exist.
        context.metadata.setdefault("__replan_feedback_by_step", {})

        while len(executed_nodes) < len(nodes):
            # Check for cancellation
            if context.is_cancelled():
                cancelled = True
                break

            # Check for timeout
            if self.timeout is not None and (time.time() - start_time) > self.timeout:
                raise TimeoutError(
                    f"Chain execution timed out after {self.timeout}s. "
                    f"Completed {len(executed_nodes)}/{len(nodes)} steps."
                )

            # Find ready nodes
            ready_nodes = [node for node in nodes if node.step.number not in executed_nodes and node.can_execute()]

            if not ready_nodes:
                # This should not happen in a valid DAG
                remaining = [n.step.number for n in nodes if n.step.number not in executed_nodes]
                raise ValueError(
                    f"Deadlock detected: unable to execute steps {remaining}. "
                    f"Check for missing dependencies or circular references."
                )

            batch_count += 1
            log_batch_start(batch_count, len(ready_nodes))
            if self.enable_progress:
                print(f"Executing batch {batch_count} with {len(ready_nodes)} steps")

            # Call on_progress callback if registered
            if context.on_progress:
                try:
                    context.on_progress(len(executed_nodes), len(nodes))
                except Exception:
                    pass

            # Capture pre-execution snapshot for each step in this batch (retry target).
            if replan_enabled:
                batch_snapshot = self._capture_snapshot(
                    executed_nodes=executed_nodes,
                    all_results=all_results,
                    context=context,
                )
                for node in ready_nodes:
                    pre_step_snapshots[node.step.number] = batch_snapshot

            # Execute batch
            batch_results = await self.execute_batch(ready_nodes, context)
            all_results.extend(batch_results)

            # Process conditional routing decisions
            for result in batch_results:
                if result.success and result.step_type == StepType.CONDITIONAL:
                    next_step = result.result_data.get("next_step")
                    if next_step is not None:
                        self._skip_conditional_branches(
                            nodes, result.step_number, next_step, executed_nodes
                        )

            # Update history from successful results
            # Sort by step number to maintain deterministic order
            batch_results.sort(key=lambda r: r.step_number)
            seen_steps = set()

            for result in batch_results:
                if result.success and result.step_number not in seen_steps:
                    # Add the latest history entry from this step
                    if result.updated_history:
                        new_entry = result.updated_history[-1]
                        # FIX: Use add_to_history() to enforce max_history_entries limit
                        context.add_to_history(new_entry)
                        seen_steps.add(result.step_number)

            # Sync current_history with context (may have been trimmed by max_history_entries)
            current_history = context.history.copy()

            # Mark nodes as executed
            for node in ready_nodes:
                node.executed = True
                executed_nodes.add(node.step.number)

            # Store step results into metadata for downstream references ($steps.<step_number>...)
            context.metadata.setdefault("step_results", {})
            for result in batch_results:
                if result.success:
                    context.metadata["step_results"][str(result.step_number)] = {
                        "step_number": result.step_number,
                        "title": result.step_title,
                        "step_type": str(result.step_type),
                        "result": result.result,
                        "result_data": result.result_data,
                    }
                    context.metadata[f"step_{result.step_number}"] = (
                        result.result_data if result.result_data is not None else result.result
                    )

            # Consume step-level RE-PLAN feedback once a step has executed.
            feedback_map = context.metadata.get("__replan_feedback_by_step")
            if isinstance(feedback_map, dict):
                for result in batch_results:
                    feedback_map.pop(str(result.step_number), None)

            # Capture post-step checkpoint snapshots after state updates are merged.
            if replan_enabled:
                for result in batch_results:
                    if not result.success:
                        continue
                    step = step_by_number[result.step_number]
                    if not self._is_checkpoint_step(step):
                        continue
                    checkpoint_snapshot = self._capture_snapshot(
                        executed_nodes=executed_nodes,
                        all_results=all_results,
                        context=context,
                    )
                    checkpoints.append(
                        _CheckpointSnapshot(
                            name=self._checkpoint_name(step),
                            step_number=step.number,
                            sequence=len(checkpoints) + 1,
                            snapshot=checkpoint_snapshot,
                        )
                    )

            # Run chain-level RE-PLAN policy checks.
            rollback_applied = False
            if replan_enabled and replan_policy is not None:
                for result in batch_results:
                    step = step_by_number[result.step_number]
                    if not self._should_evaluate_replan(policy=replan_policy, step=step, result=result):
                        continue

                    budget_before = self._budget_snapshot(
                        chain_replans=chain_replans,
                        per_step_replans=per_step_replans,
                        target_visits=rollback_target_visits,
                        same_target_streak=same_target_streak,
                    )
                    checker_input = self._build_checker_input(
                        result=result,
                        step=step,
                        context=context,
                        checkpoints=checkpoints,
                        budget_snapshot=budget_before,
                        all_results=all_results,
                    )

                    votes: list[CheckerVote] = []
                    for checker_name, checker_runtime in replan_checkers:
                        try:
                            verdict = await checker_runtime.evaluate(checker_input, context)
                            if not isinstance(verdict, ReplanVerdict):
                                raise TypeError(
                                    f"Checker '{checker_name}' returned {type(verdict).__name__}, expected ReplanVerdict"
                                )
                        except Exception as exc:
                            verdict = ReplanVerdict(
                                action=ReplanAction.FAIL,
                                reason=f"RE-PLAN checker '{checker_name}' failed: {exc}",
                                confidence=0.0,
                                metadata={"checker_exception": str(exc)},
                            )
                        votes.append(CheckerVote(checker_name=checker_name, verdict=verdict))

                    aggregate = aggregate_replan_votes(votes, replan_policy.aggregation)
                    final_action = aggregate.selected_verdict.action
                    rollback_target = self._resolve_rollback_target(
                        action=final_action,
                        verdict=aggregate.selected_verdict,
                        policy=replan_policy,
                        current_step_number=result.step_number,
                    )
                    feedback = self._normalize_feedback(aggregate.selected_verdict)
                    budget_exhausted = False
                    event_note = ""

                    if aggregate.triggered and final_action != ReplanAction.CONTINUE:
                        projected_chain_replans = chain_replans + 1
                        projected_step_replans = per_step_replans.get(result.step_number, 0) + 1

                        rollback_target_key = rollback_target.to_key() if rollback_target else "none"
                        projected_target_visits = rollback_target_visits.get(rollback_target_key, 0) + 1
                        projected_streak = (
                            same_target_streak + 1
                            if rollback_target_key == last_rollback_target_key
                            else 1
                        )

                        budgets = replan_policy.budgets
                        budget_limits: list[tuple[bool, str]] = [
                            (
                                budgets.max_replans_per_chain > 0
                                and projected_chain_replans > budgets.max_replans_per_chain,
                                f"max_replans_per_chain={budgets.max_replans_per_chain} exceeded",
                            ),
                            (
                                budgets.max_replans_per_step > 0
                                and projected_step_replans > budgets.max_replans_per_step,
                                (
                                    f"max_replans_per_step={budgets.max_replans_per_step} exceeded for "
                                    f"step {result.step_number}"
                                ),
                            ),
                            (
                                budgets.max_visits_per_checkpoint > 0
                                and projected_target_visits > budgets.max_visits_per_checkpoint,
                                (
                                    f"max_visits_per_checkpoint={budgets.max_visits_per_checkpoint} exceeded for "
                                    f"target {rollback_target_key}"
                                ),
                            ),
                            (
                                budgets.max_same_rollback_target_repeats > 0
                                and projected_streak > budgets.max_same_rollback_target_repeats,
                                (
                                    "max_same_rollback_target_repeats="
                                    f"{budgets.max_same_rollback_target_repeats} exceeded for "
                                    f"target {rollback_target_key}"
                                ),
                            ),
                        ]
                        exhausted_limits = [reason for is_exhausted, reason in budget_limits if is_exhausted]
                        if exhausted_limits:
                            budget_exhausted = True
                            event_note = "; ".join(exhausted_limits)
                            if budgets.fail_on_budget_exhaustion:
                                final_action = ReplanAction.FAIL
                            else:
                                final_action = ReplanAction.CONTINUE

                        if final_action != ReplanAction.CONTINUE:
                            chain_replans = projected_chain_replans
                            per_step_replans[result.step_number] = projected_step_replans
                            rollback_target_visits[rollback_target_key] = projected_target_visits
                            last_rollback_target_key = rollback_target_key
                            same_target_streak = projected_streak

                        if final_action not in {ReplanAction.CONTINUE, ReplanAction.FAIL}:
                            target_snapshot, resolved_target_key = self._resolve_target_snapshot(
                                target=rollback_target or ReplanRollbackTarget(
                                    target_type=ReplanTargetType.CURRENT_STEP
                                ),
                                current_step_number=result.step_number,
                                chain_start_snapshot=chain_start_snapshot,
                                pre_step_snapshots=pre_step_snapshots,
                                checkpoints=checkpoints,
                            )
                            if target_snapshot is None:
                                final_action = ReplanAction.FAIL
                                event_note = f"Unable to resolve rollback target: {resolved_target_key}"
                            else:
                                executed_nodes, all_results = self._restore_snapshot(
                                    target_snapshot,
                                    nodes=nodes,
                                    context=context,
                                )
                                current_history = context.history.copy()
                                checkpoints = [
                                    checkpoint
                                    for checkpoint in checkpoints
                                    if checkpoint.step_number in executed_nodes
                                ]
                                pre_step_snapshots = {
                                    step_number: snapshot
                                    for step_number, snapshot in pre_step_snapshots.items()
                                    if step_number in executed_nodes
                                }
                                rollback_applied = True
                                if feedback:
                                    feedback_map = context.metadata.setdefault("__replan_feedback_by_step", {})
                                    if isinstance(feedback_map, dict):
                                        existing = feedback_map.get(str(result.step_number), [])
                                        if isinstance(existing, str):
                                            existing_items = [existing]
                                        elif isinstance(existing, list):
                                            existing_items = [str(item) for item in existing if str(item).strip()]
                                        else:
                                            existing_items = []
                                        merged: list[str] = []
                                        for item in existing_items + feedback:
                                            if item and item not in merged:
                                                merged.append(item)
                                        feedback_map[str(result.step_number)] = merged
                                event_note = (
                                    event_note or f"Rollback applied to target '{resolved_target_key}'."
                                )

                    aggregation_outcome = ReplanAggregationOutcome(
                        strategy=replan_policy.aggregation.strategy,
                        triggered=aggregate.triggered,
                        trigger_count=aggregate.trigger_count,
                        total_count=aggregate.total_count,
                        mandatory_satisfied=aggregate.mandatory_satisfied,
                        selected_checker=aggregate.selected_checker,
                        selected_action=final_action,
                    )
                    replan_events.append(
                        ReplanEvent(
                            sequence=len(replan_events) + 1,
                            step_number=result.step_number,
                            step_title=result.step_title,
                            checker_votes=[
                                ReplanCheckerVote(
                                    checker_name=vote.checker_name,
                                    action=vote.verdict.action,
                                    reason=vote.verdict.reason,
                                    confidence=vote.verdict.confidence,
                                    suggested_target=vote.verdict.suggested_target,
                                    regeneration_hints=vote.verdict.regeneration_hints,
                                    metadata=vote.verdict.metadata,
                                )
                                for vote in votes
                            ],
                            aggregation=aggregation_outcome,
                            final_action=final_action,
                            rollback_target=rollback_target if final_action != ReplanAction.CONTINUE else None,
                            feedback_passed=feedback if rollback_applied else [],
                            triggering_checkers=aggregate.triggering_checkers,
                            budget_usage=self._budget_snapshot(
                                chain_replans=chain_replans,
                                per_step_replans=per_step_replans,
                                target_visits=rollback_target_visits,
                                same_target_streak=same_target_streak,
                            ),
                            budget_exhausted=budget_exhausted,
                            note=event_note,
                        )
                    )

                    if final_action == ReplanAction.FAIL:
                        replan_failed = True
                        replan_fail_message = (
                            event_note
                            or aggregate.selected_verdict.reason
                            or f"RE-PLAN policy requested fail at step {result.step_number}"
                        )
                        all_results.append(
                            StepExecutionResult(
                                step_number=result.step_number,
                                step_title=result.step_title,
                                step_type=result.step_type,
                                result="",
                                success=False,
                                error_message=f"RE-PLAN failure: {replan_fail_message}",
                                execution_time=0.0,
                                updated_history=context.history.copy(),
                            )
                        )
                        break

                    if rollback_applied:
                        break

            if replan_failed:
                break
            if rollback_applied:
                continue

        # Calculate final stats
        total_time = time.time() - start_time
        successful_steps = [r for r in all_results if r.success]
        failed_steps = [r for r in all_results if not r.success]

        self._execution_stats.update(
            {
                "executed_steps": len(successful_steps),
                "failed_steps": len(failed_steps),
                "parallel_batches": batch_count,
                "total_time": total_time,
                "replan_events": len(replan_events),
            }
        )

        # Determine success (false if cancelled)
        chain_success = len(failed_steps) == 0 and not cancelled and not replan_failed

        # Aggregate token usage from all steps
        total_tokens = {"prompt": 0, "completion": 0, "total": 0}
        for step_result in all_results:
            if step_result.token_usage:
                total_tokens["prompt"] += step_result.token_usage.get("prompt", 0)
                total_tokens["completion"] += step_result.token_usage.get("completion", 0)
        total_tokens["total"] = total_tokens["prompt"] + total_tokens["completion"]

        # Build final output for trace
        trace_output: dict[str, Any] = {
            "success": chain_success,
            "executed_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "cancelled": cancelled,
            "total_execution_time": total_time,
            "token_usage": total_tokens,
        }
        # Include the final step result in trace output
        if successful_steps:
            last_successful = max(successful_steps, key=lambda r: r.step_number)
            if last_successful.result:
                trace_output["final_output"] = last_successful.result[:2000]
        trace_output["replan_failed"] = replan_failed
        trace_output["replan_events"] = len(replan_events)

        # Update trace with final chain outcome
        trace.update(output=trace_output)

        # End the trace to complete it in Langfuse
        trace.end()

        result = ReasoningResult(
            success=chain_success,
            history=current_history,
            step_results=all_results,
            total_execution_time=total_time,
            token_usage=total_tokens,
            replan_events=replan_events,
            metadata={
                "execution_stats": self._execution_stats.copy(),
                "parallel_batches": batch_count,
                "cancelled": cancelled,
                "replan": {
                    "enabled": replan_enabled,
                    "events": len(replan_events),
                    "chain_replans": chain_replans,
                    "per_step_replans": {str(k): v for k, v in per_step_replans.items()},
                    "rollback_target_visits": rollback_target_visits,
                    "failed": replan_failed,
                    "failure_reason": replan_fail_message if replan_failed else "",
                },
            },
        )

        # Call chain_complete callback if registered
        if context.on_chain_complete:
            try:
                context.on_chain_complete(result)
            except Exception:
                pass

        # Log chain completion
        log_chain_complete(chain_success, total_time, len(successful_steps), len(steps))

        # Raise cancellation error after cleanup
        if cancelled:
            raise ExecutionCancelledError(
                f"Chain execution cancelled after completing {len(executed_nodes)}/{len(nodes)} steps. "
                "Partial results are available in the returned ReasoningResult."
            )

        return result

    def _skip_conditional_branches(
        self,
        nodes: list[ExecutionNode],
        conditional_step: int,
        next_step: int,
        executed_nodes: set[int],
    ) -> None:
        """Mark steps that should be skipped due to conditional routing.

        When a conditional step selects next_step=3, all other branches
        (steps not reachable from step 3) should be marked as skipped.
        """
        step_by_number = {node.step.number: node for node in nodes}

        # Find all steps that should be skipped
        for node in nodes:
            if node.step.number == conditional_step:
                continue  # Don't skip the conditional step itself

            if node.step.number in executed_nodes:
                continue  # Already executed

            # Check if this step is reachable from the target next_step
            if not self._is_reachable_from_target(
                node.step.number, next_step, conditional_step, step_by_number
            ):
                # Mark as executed without actually running
                node.executed = True
                executed_nodes.add(node.step.number)

    def _is_reachable_from_target(
        self,
        step_number: int,
        target_step: int,
        conditional_step: int,
        step_by_number: dict[int, ExecutionNode],
    ) -> bool:
        """Check if a step is reachable from the target next_step.

        A step is reachable if:
        - It IS the target step, OR
        - It depends on the target step (directly or transitively)
        """
        if step_number == target_step:
            return True

        node = step_by_number.get(step_number)
        if not node:
            return False

        # Check if this step depends on the target (directly or transitively)
        for dep_node in node.dependencies:
            dep_step = dep_node.step.number
            if dep_step == target_step:
                return True
            if dep_step != conditional_step:  # Don't trace back through conditional
                if self._is_reachable_from_target(dep_step, target_step, conditional_step, step_by_number):
                    return True

        return False

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics from the last run."""
        return self._execution_stats.copy()

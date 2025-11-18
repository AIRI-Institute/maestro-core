"""
DAG execution engine for CARL reasoning chains.

Handles parallel execution of reasoning steps based on their dependencies.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from .models import Language, PromptTemplate, ReasoningContext, ReasoningResult, StepDescription, StepExecutionResult


@dataclass
class ExecutionNode:
    """
    Represents a node in the execution DAG.
    """

    step: StepDescription
    dependencies: list["ExecutionNode"]
    dependents: list["ExecutionNode"]
    executed: bool = False
    executing: bool = False
    result: StepExecutionResult | None = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.dependents is None:
            self.dependents = []

    def can_execute(self) -> bool:
        """Check if this node can be executed (all dependencies completed)."""
        return all(dep.executed for dep in self.dependencies)

    def is_ready(self) -> bool:
        """Check if this node is ready for execution (not executing or executed)."""
        return not self.executing and not self.executed and self.can_execute()


class DAGExecutor:
    """
    Executes reasoning chains as a Directed Acyclic Graph (DAG).

    Automatically parallelizes execution where dependencies allow.
    """

    def __init__(
        self,
        max_workers: int = 3,
        prompt_template: PromptTemplate | None = None,
        enable_progress: bool = False,
    ):
        """
        Initialize the DAG executor.

        Args:
            max_workers: Maximum number of parallel executions
            prompt_template: Template for generating prompts
            enable_progress: Whether to enable progress tracking
        """
        self.max_workers = max_workers
        self.prompt_template = prompt_template or PromptTemplate()
        self.enable_progress = enable_progress
        self._execution_stats = {
            "total_steps": 0,
            "executed_steps": 0,
            "failed_steps": 0,
            "parallel_batches": 0,
            "total_time": 0.0,
        }

    def build_execution_graph(self, steps: list[StepDescription]) -> list[ExecutionNode]:
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
                    raise ValueError(f"Step {step.number} depends on non-existent step {dep_number}")

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
        Execute a single reasoning step.

        Args:
            node: Execution node to execute
            context: Reasoning context

        Returns:
            Step execution result
        """
        start_time = time.time()
        step = node.step

        try:
            # Generate prompt for this step with RAG-like context extraction
            step_prompt = self.prompt_template.format_step_prompt(step, context.outer_context, context.language)
            full_prompt = self.prompt_template.format_chain_prompt(
                outer_context=context.outer_context,
                current_task=step_prompt,
                history=context.get_current_history(),
                language=context.language,
                system_prompt=context.system_prompt,
            )

            # Execute LLM call with retries using LLMClient
            result = await context.llm_client.get_response_with_retries(full_prompt, retries=context.retry_max)

            # Update context history
            if context.language == Language.ENGLISH:
                step_result = f"Step {step.number}. {step.title}\nResult: {result}\n"
            else:  # Russian
                step_result = f"Шаг {step.number}. {step.title}\nРезультат: {result}\n"
            updated_history = context.history.copy()
            updated_history.append(step_result)

            execution_time = time.time() - start_time

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                result=result,
                success=True,
                execution_time=execution_time,
                updated_history=updated_history,
            )

        except Exception as e:
            execution_time = time.time() - start_time

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                result="",
                success=False,
                error_message=str(e),
                execution_time=execution_time,
                updated_history=context.history.copy(),
            )

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
        for _ in ready_nodes:
            snapshot = ReasoningContext(
                outer_context=context.outer_context,
                api=context.api,
                entrypoint_key=context.entrypoint_key,
                retry_max=context.retry_max,
                history=context.history.copy(),
                metadata=context.metadata.copy(),
                language=context.language,
                system_prompt=context.system_prompt,
            )
            context_snapshots.append(snapshot)

        # Execute in parallel
        tasks = [self.execute_step(node, ctx) for node, ctx in zip(ready_nodes, context_snapshots)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    StepExecutionResult(
                        step_number=ready_nodes[i].step.number,
                        step_title=ready_nodes[i].step.title,
                        result="",
                        success=False,
                        error_message=str(result),
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def execute(self, steps: list[StepDescription], context: ReasoningContext) -> ReasoningResult:
        """
        Execute a complete reasoning chain.

        Args:
            steps: list of step descriptions
            context: Initial reasoning context

        Returns:
            Complete reasoning result
        """
        start_time = time.time()
        self._execution_stats["total_steps"] = len(steps)

        # Build execution graph
        nodes = self.build_execution_graph(steps)
        if not nodes:
            return ReasoningResult(
                success=True, history=[], step_results=[], total_execution_time=time.time() - start_time
            )

        # Execute DAG
        executed_nodes: set[int] = set()
        all_results: list[StepExecutionResult] = []
        current_history = context.history.copy()
        batch_count = 0

        while len(executed_nodes) < len(nodes):
            # Find ready nodes
            ready_nodes = [node for node in nodes if node.step.number not in executed_nodes and node.can_execute()]

            if not ready_nodes:
                # This should not happen in a valid DAG
                remaining = [n.step.number for n in nodes if n.step.number not in executed_nodes]
                raise ValueError(f"Deadlock detected: unable to execute steps {remaining}")

            batch_count += 1
            if self.enable_progress:
                print(f"Executing batch {batch_count} with {len(ready_nodes)} steps")

            # Execute batch
            batch_results = await self.execute_batch(ready_nodes, context)
            all_results.extend(batch_results)

            # Update history from successful results
            # Sort by step number to maintain deterministic order
            batch_results.sort(key=lambda r: r.step_number)
            seen_steps = set()

            for result in batch_results:
                if result.success and result.step_number not in seen_steps:
                    # Add the latest history entry from this step
                    if result.updated_history:
                        new_entry = result.updated_history[-1]
                        current_history.append(new_entry)
                        seen_steps.add(result.step_number)

            # Mark nodes as executed
            for node in ready_nodes:
                node.executed = True
                executed_nodes.add(node.step.number)

            # Update context with current history
            context.history = current_history.copy()

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
            }
        )

        return ReasoningResult(
            success=len(failed_steps) == 0,
            history=current_history,
            step_results=all_results,
            total_execution_time=total_time,
            metadata={"execution_stats": self._execution_stats.copy(), "parallel_batches": batch_count},
        )

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics from the last run."""
        return self._execution_stats.copy()

"""
Main chain definition and execution API for CARL.

Provides the primary interface for defining and executing reasoning chains.
"""

import asyncio
from typing import Any

from .executor import DAGExecutor
from .models import ContextSearchConfig, PromptTemplate, ReasoningContext, ReasoningResult, StepDescription


class ReasoningChain:
    """
    Main interface for defining and executing reasoning chains.

    Provides a high-level API that combines chain definition with DAG execution.
    """

    def __init__(
        self,
        steps: list[StepDescription],
        max_workers: int = 3,
        prompt_template: PromptTemplate | None = None,
        enable_progress: bool = False,
        metadata: dict[str, Any] | None = None,
        search_config: ContextSearchConfig | None = None,
    ):
        self.steps = steps
        self.max_workers = max_workers
        self.enable_progress = enable_progress
        self.metadata = metadata or {}

        # Set up prompt template with search configuration
        if prompt_template:
            self.prompt_template = prompt_template
            if search_config:
                self.prompt_template.search_config = search_config
        else:
            self.prompt_template = PromptTemplate(search_config=search_config or ContextSearchConfig())

        self._validate_steps()
        self.executor = DAGExecutor(
            max_workers=max_workers, prompt_template=self.prompt_template, enable_progress=enable_progress
        )

    def _validate_steps(self) -> None:
        if not self.steps:
            raise ValueError("Reasoning chain must have at least one step")
        step_numbers = [step.number for step in self.steps]

        # Check for duplicate step numbers
        if len(step_numbers) != len(set(step_numbers)):
            duplicates = [num for num in step_numbers if step_numbers.count(num) > 1]
            raise ValueError(f"Duplicate step numbers found: {duplicates}")

        # Check for missing dependencies
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_numbers:
                    raise ValueError(f"Step {step.number} depends on non-existent step {dep}")

        # Check for cycles (basic validation)
        self._check_for_cycles()

    def _check_for_cycles(self) -> None:
        """
        Basic cycle detection using dependency graph.

        Raises:
            ValueError: If cycles are detected
        """
        visited = set()
        rec_stack = set()

        def visit(step_num: int) -> bool:
            if step_num in rec_stack:
                return True  # Cycle detected
            if step_num in visited:
                return False

            visited.add(step_num)
            rec_stack.add(step_num)

            # Visit dependencies
            step = next(s for s in self.steps if s.number == step_num)
            for dep in step.dependencies:
                if visit(dep):
                    return True

            rec_stack.remove(step_num)
            return False

        for step in self.steps:
            if step.number not in visited:
                if visit(step.number):
                    raise ValueError(f"Cycle detected involving step {step.number}")

    async def execute_async(self, context: ReasoningContext) -> ReasoningResult:
        """
        Execute the reasoning chain asynchronously.

        Args:
            context: Reasoning context with input data and LLM client

        Returns:
            Complete reasoning result
        """
        # Add chain metadata to context
        context.metadata.update({"chain_steps": len(self.steps), "chain_metadata": self.metadata})

        return await self.executor.execute(self.steps, context)

    def execute(self, context: ReasoningContext) -> ReasoningResult:
        """
        Execute the reasoning chain synchronously.

        Args:
            context: Reasoning context with input data and LLM client

        Returns:
            Complete reasoning result
        """
        try:
            # Check if we're already in an event loop
            _ = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.execute_async(context))
        else:
            # We're in an event loop, create a task and run it
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.execute_async(context))
                return future.result()

    def get_execution_plan(self) -> dict[str, Any]:
        """
        Get the execution plan showing parallelization opportunities.

        Returns:
            Dictionary describing the execution plan
        """
        # Build dependency levels
        levels = []
        remaining_steps = self.steps.copy()

        while remaining_steps:
            current_level = []
            for step in remaining_steps[:]:
                if all(dep not in [s.number for s in remaining_steps] for dep in step.dependencies):
                    current_level.append(step)
                    remaining_steps.remove(step)

            if current_level:
                levels.append(
                    {
                        "level": len(levels) + 1,
                        "steps": [step.number for step in current_level],
                        "parallelizable": len(current_level) > 1,
                        "step_titles": [step.title for step in current_level],
                    }
                )

        return {
            "total_steps": len(self.steps),
            "max_workers": self.max_workers,
            "execution_levels": levels,
            "estimated_parallel_batches": len(levels),
            "parallelization_ratio": len([s for level in levels for s in level["steps"] if level["parallelizable"]])
            / len(self.steps)
            if self.steps
            else 0,
        }

    def get_step_dependencies(self) -> dict[int, list[int]]:
        """
        Get a mapping of step dependencies.

        Returns:
            Dictionary mapping step numbers to their dependencies
        """
        return {step.number: step.dependencies.copy() for step in self.steps}

    def get_steps_summary(self) -> list[dict[str, Any]]:
        """
        Get a summary of all steps in the chain.

        Returns:
            List of step summaries
        """
        return [
            {
                "number": step.number,
                "title": step.title,
                "aim": step.aim,
                "dependencies": step.dependencies,
                "step_context_queries": step.step_context_queries,
                "has_dependencies": step.has_dependencies(),
            }
            for step in self.steps
        ]


class ChainBuilder:
    """
    Builder pattern for constructing reasoning chains.

    Provides a fluent interface for building complex reasoning chains.
    """

    def __init__(self):
        """Initialize the chain builder."""
        self.steps: list[StepDescription] = []
        self.max_workers: int = 3
        self.prompt_template: PromptTemplate | None = None
        self.search_config: ContextSearchConfig | None = None
        self.enable_progress: bool = False
        self.metadata: dict[str, Any] = {}

    def add_step(
        self,
        number: int,
        title: str,
        aim: str,
        reasoning_questions: str,
        stage_action: str,
        example_reasoning: str,
        dependencies: list[int] | None = None,
        step_context_queries: list[str] | None = None,
    ) -> "ChainBuilder":
        """
        Add a step to the chain.

        Args:
            number: Step number
            title: Step title
            aim: Step objective
            reasoning_questions: Key questions to answer
            stage_action: Action to perform
            example_reasoning: Example of expert reasoning
            dependencies: List of step numbers this depends on
            entities: Entities/concepts this works with

        Returns:
            Self for method chaining
        """
        step = StepDescription(
            number=number,
            title=title,
            aim=aim,
            reasoning_questions=reasoning_questions,
            stage_action=stage_action,
            example_reasoning=example_reasoning,
            dependencies=dependencies or [],
            step_context_queries=step_context_queries or [],
        )
        self.steps.append(step)
        return self

    def with_max_workers(self, max_workers: int) -> "ChainBuilder":
        """
        Set maximum number of parallel workers.

        Args:
            max_workers: Maximum workers

        Returns:
            Self for method chaining
        """
        self.max_workers = max_workers
        return self

    def with_prompt_template(self, template: PromptTemplate) -> "ChainBuilder":
        """
        Set custom prompt template.

        Args:
            template: Prompt template to use

        Returns:
            Self for method chaining
        """
        self.prompt_template = template
        return self

    def with_search_config(self, config: ContextSearchConfig) -> "ChainBuilder":
        """
        Set search configuration for context extraction.

        Args:
            config: Search configuration to use

        Returns:
            Self for method chaining
        """
        self.search_config = config
        return self

    def with_progress(self, enable: bool = True) -> "ChainBuilder":
        """
        Enable or disable progress tracking.

        Args:
            enable: Whether to enable progress

        Returns:
            Self for method chaining
        """
        self.enable_progress = enable
        return self

    def with_metadata(self, **metadata) -> "ChainBuilder":
        """
        Add metadata to the chain.

        Args:
            **metadata: Metadata key-value pairs

        Returns:
            Self for method chaining
        """
        self.metadata.update(metadata)
        return self

    def build(self) -> ReasoningChain:
        """
        Build the reasoning chain.

        Returns:
            Constructed reasoning chain

        Raises:
            ValueError: If chain configuration is invalid
        """
        return ReasoningChain(
            steps=self.steps,
            max_workers=self.max_workers,
            prompt_template=self.prompt_template,
            enable_progress=self.enable_progress,
            metadata=self.metadata,
            search_config=self.search_config,
        )


def create_chain_from_config(config: dict[str, Any]) -> ReasoningChain:
    """
    Create a reasoning chain from a configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Constructed reasoning chain
    """
    steps = []
    for step_config in config.get("steps", []):
        step = StepDescription(**step_config)
        steps.append(step)

    # Create search configuration if provided
    search_config = None
    if "search_config" in config:
        search_config = ContextSearchConfig(**config["search_config"])

    return ReasoningChain(
        steps=steps,
        max_workers=config.get("max_workers", 3),
        enable_progress=config.get("enable_progress", False),
        metadata=config.get("metadata", {}),
        search_config=search_config,
    )

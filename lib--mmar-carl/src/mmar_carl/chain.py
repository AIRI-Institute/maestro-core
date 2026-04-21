"""
Main chain definition and execution API for CARL.

Provides the primary interface for defining and executing reasoning chains.
Supports JSON serialization/deserialization for chain persistence.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from pydantic import BaseModel, Field

from .executor import DAGExecutor
from .models.dataset import DatasetEvaluationReport
from .models import (
    # Enums
    Language,
    # Step Configurations
    ConditionalBranch,
    ConditionalStepConfig,
    ContextQuery,
    ContextSearchConfig,
    ExecutionMode,
    LLMStepConfig,
    MCPServerConfig,
    MCPStepConfig,
    MemoryStepConfig,
    PromptTemplate,
    ReasoningContext,
    ReasoningResult,
    # Step Type Enum
    StepType,
    ToolParameter,
    ToolStepConfig,
    TransformStepConfig,
    StructuredOutputStepConfig,
    # New Typed Step Classes
    StepDescriptionBase,
    LLMStepDescription,
    ToolStepDescription,
    MCPStepDescription,
    MemoryStepDescription,
    TransformStepDescription,
    ConditionalStepDescription,
    StructuredOutputStepDescription,
    AnyStepDescription,
    StepDescription,
    ReplanPolicy,
)


class ReflectionOptions(BaseModel):
    """
    Configuration options for chain reflection.

    Controls what information is included in the reflection prompt
    and how verbose the analysis should be.

    Example:
        ```python
        # Minimal reflection (faster, cheaper)
        options = ReflectionOptions(
            include_chain_structure=False,
            include_dependency_analysis=False,
            max_output_preview_chars=200,
        )
        reflection = chain.reflect_async("Analyze data", options=options)

        # Detailed reflection (more context)
        options = ReflectionOptions(
            include_step_definitions=True,
            include_execution_metrics=True,
            include_dependency_analysis=True,
        )
        ```
    """

    include_chain_structure: bool = Field(
        default=True,
        description="Include chain configuration and step type distribution",
    )
    include_step_definitions: bool = Field(
        default=True,
        description="Include original step definitions (aim, queries, etc.)",
    )
    include_execution_metrics: bool = Field(
        default=True,
        description="Include timing stats and parallel efficiency",
    )
    include_dependency_analysis: bool = Field(
        default=True,
        description="Include dependency graph analysis and parallelization opportunities",
    )
    include_metric_scores: bool = Field(
        default=True,
        description=(
            "Include MetricBase evaluation scores (step-level and chain-level) in the "
            "reflection prompt so the LLM can reference concrete quality signals"
        ),
    )
    extra_feedback: dict[str, Any] | str | None = Field(
        default=None,
        description=(
            "Optional user-provided data or context to append to the reflection prompt. "
            "Pass a dict for labelled entries or a plain string for freeform notes."
        ),
    )
    max_output_preview_chars: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Maximum characters to show in output previews",
    )
    max_result_preview_chars: int = Field(
        default=300,
        ge=50,
        le=2000,
        description="Maximum characters to show in step result previews",
    )
    language: Optional[Language] = Field(
        default=None,
        description="Language for reflection prompt (None = use context language)",
    )
    dataset_report: DatasetEvaluationReport | None = Field(
        default=None,
        description=(
            "Optional dataset evaluation report produced by DatasetEvaluator. "
            "When provided, a dedicated section listing problem cases is added to "
            "the reflection prompt so the LLM can focus improvements on patterns "
            "observed across failing cases."
        ),
    )


class ReasoningChain:
    """
    Main interface for defining and executing reasoning chains.

    Provides a high-level API that combines chain definition with DAG execution.

    Accepts both legacy StepDescription and new typed step classes:
    - StepDescription (legacy, backward compatible)
    - LLMStepDescription, ToolStepDescription, MCPStepDescription, etc. (new API)

    Note on parallel execution:
        - Steps in the same batch execute in parallel with isolated memory
        - Tool registry is shared - tools MUST be stateless for thread safety
        - Memory writes are only visible to subsequent batches, not parallel siblings

    Note on conditional steps:
        - CONDITIONAL steps are currently informational only
        - They return next_step in result_data but execution proceeds topologically
    """

    def __init__(
        self,
        steps: Sequence[StepDescription | StepDescriptionBase | AnyStepDescription],
        max_workers: int = 3,
        prompt_template: PromptTemplate | None = None,
        enable_progress: bool = False,
        metadata: dict[str, Any] | None = None,
        search_config: ContextSearchConfig | None = None,
        timeout: float | None = None,
        trace_name: str | None = None,
        session_id: str | None = None,
        replan_policy: ReplanPolicy | None = None,
        metrics: Optional[list] = None,
    ):
        # Normalize steps to support both legacy and new types
        self.steps: list[StepDescription | StepDescriptionBase | AnyStepDescription] = list(steps)
        self.metrics: list = metrics or []
        self.max_workers = max_workers
        self.enable_progress = enable_progress
        self.metadata = metadata or {}
        self.timeout = timeout
        self.trace_name = trace_name
        self.session_id = session_id
        self.replan_policy = replan_policy

        # Set up prompt template with search configuration
        if prompt_template:
            self.prompt_template = prompt_template
            if search_config:
                self.prompt_template.search_config = search_config
        else:
            self.prompt_template = PromptTemplate(search_config=search_config or ContextSearchConfig())

        self._validate_steps()
        self.executor = DAGExecutor(
            max_workers=max_workers,
            prompt_template=self.prompt_template,
            enable_progress=enable_progress,
            timeout=timeout,
            replan_policy=replan_policy,
        )

        # Store last execution result for reflection
        self._last_result: ReasoningResult | None = None
        self._last_context: ReasoningContext | None = None

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
        # Add chain metadata to context (include trace_name and session_id)
        chain_meta = dict(self.metadata)
        if self.trace_name:
            chain_meta["trace_name"] = self.trace_name
        if self.session_id:
            chain_meta["session_id"] = self.session_id
        context.metadata.update(
            {
                "chain_steps": len(self.steps),
                "chain_metadata": chain_meta,
                "replan_policy_enabled": bool(self.replan_policy and self.replan_policy.enabled),
            }
        )

        result = await self.executor.execute(self.steps, context)

        # Run chain-level metrics on the chain result
        if self.metrics and result.success:
            chain_metric_scores: dict[str, float] = {}
            for metric in self.metrics:
                try:
                    score = await metric.compute_async(result)
                    chain_metric_scores[metric.name] = float(score)
                except Exception as e:
                    from .logging_utils import log_warning

                    log_warning(f"Chain metric '{getattr(metric, 'name', repr(metric))}' failed: {e}")
            if chain_metric_scores:
                result = result.model_copy(update={"metrics": chain_metric_scores})
                # Send to LangFuse as scores (idiomatic way to attach evaluation
                # metrics to a trace; works even after trace.end())
                trace = context.metadata.get("__langfuse_trace")
                if trace is not None:
                    for metric_name, score_value in chain_metric_scores.items():
                        try:
                            trace.score(name=metric_name, value=score_value)
                        except Exception:
                            pass

        # Store for reflection
        self._last_result = result
        self._last_context = context

        return result

    def execute(self, context: ReasoningContext) -> ReasoningResult:
        """
        Execute the reasoning chain synchronously.

        Args:
            context: Reasoning context with input data and LLM client

        Returns:
            Complete reasoning result

        Note:
            This method creates a new event loop and may have limitations when
            called from async contexts (e.g., Jupyter notebooks, FastAPI).

            For async applications, prefer execute_async() instead:

            ```python
            result = await chain.execute_async(context)
            ```

            If you must use execute() in an async context, be aware of potential
            issues with signal handlers and context variables.
        """
        import warnings

        # Warn about sync API limitations
        try:
            asyncio.get_running_loop()
            warnings.warn(
                "execute() is being called from an async context. "
                "This creates a new event loop in a thread pool, which may cause issues. "
                "Prefer execute_async() instead: result = await chain.execute_async(context)",
                UserWarning,
                stacklevel=2,
            )
        except RuntimeError:
            pass  # No event loop running, safe to use asyncio.run

        async def _execute_and_cleanup() -> ReasoningResult:
            """Run the chain and close LLM clients before the event loop shuts down."""
            try:
                return await self.execute_async(context)
            finally:
                await context.close()
                # Wait for background tasks (e.g., httpx connection cleanup) to complete.
                # httpx spawns tasks during aclose() that need time to finish before
                # asyncio.run() closes the event loop.
                current_task = asyncio.current_task()
                # Give tasks time to spawn and complete
                for _ in range(10):
                    await asyncio.sleep(0.01)
                    other_tasks = [t for t in asyncio.all_tasks() if t != current_task]
                    if not other_tasks:
                        break
                    # Wait for any remaining tasks with timeout
                    try:
                        await asyncio.wait_for(asyncio.gather(*other_tasks, return_exceptions=True), timeout=0.1)
                    except asyncio.TimeoutError:
                        # Some tasks are still running, continue waiting
                        pass

        try:
            # Check if we're already in an event loop
            _ = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(_execute_and_cleanup())
        else:
            # We're in an event loop, create a task and run it
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _execute_and_cleanup())
                return future.result()

    async def batch_execute_async(
        self,
        contexts: list[ReasoningContext],
        max_concurrent: int = 1,
    ) -> list[ReasoningResult]:
        """
        Execute the chain on multiple contexts asynchronously.

        Args:
            contexts: List of :class:`ReasoningContext` objects to run.
            max_concurrent: Maximum number of chains running in parallel.
                Defaults to ``1`` (sequential) to respect LLM API rate limits.
                Increase carefully — each concurrent execution issues its own
                LLM requests.

        Returns:
            List of :class:`ReasoningResult` in the same order as ``contexts``.
        """
        if max_concurrent == 1:
            results = []
            for ctx in contexts:
                results.append(await self.execute_async(ctx))
            return results

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _run(ctx: ReasoningContext) -> ReasoningResult:
            async with semaphore:
                return await self.execute_async(ctx)

        return list(await asyncio.gather(*[_run(ctx) for ctx in contexts]))

    def batch_execute(
        self,
        contexts: list[ReasoningContext],
        max_concurrent: int = 1,
    ) -> list[ReasoningResult]:
        """
        Synchronous wrapper around :meth:`batch_execute_async`.

        Args:
            contexts: List of :class:`ReasoningContext` objects to run.
            max_concurrent: Maximum number of chains running in parallel.

        Returns:
            List of :class:`ReasoningResult` in the same order as ``contexts``.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.batch_execute_async(contexts, max_concurrent=max_concurrent))
        else:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.batch_execute_async(contexts, max_concurrent=max_concurrent),
                )
                return future.result()

    # =========================================================================
    # Reflection Methods
    # =========================================================================

    async def reflect_async(
        self,
        task_description: str,
        context: ReasoningContext | None = None,
        language: Language | None = None,
        options: ReflectionOptions | None = None,
    ) -> str:
        """
        Generate a reflection on the chain execution results.

        Analyzes how well the chain accomplished the given task based on
        step results and overall execution.

        Args:
            task_description: Description of the original task/goal
            context: Optional context to use (uses last execution context if not provided)
            language: Language for reflection (uses context language if not provided)
            options: Optional ReflectionOptions to control what's included in the prompt

        Returns:
            Reflection text analyzing the execution results

        Raises:
            RuntimeError: If no execution has been performed yet

        Example:
            ```python
            # Basic reflection
            reflection = await chain.reflect_async("Analyze sentiment")

            # Minimal reflection (faster, cheaper)
            options = ReflectionOptions(
                include_chain_structure=False,
                include_dependency_analysis=False,
                max_output_preview_chars=200,
            )
            reflection = await chain.reflect_async("Analyze sentiment", options=options)
            ```
        """
        if self._last_result is None:
            raise RuntimeError("No execution result available. Call execute() before reflect().")

        ctx = context or self._last_context
        if ctx is None:
            raise RuntimeError("No context available. Provide a context or execute the chain first.")

        # Determine language (options > parameter > context)
        opts = options or ReflectionOptions()
        lang = opts.language or language or ctx.language

        # Build reflection prompt with options
        reflection_prompt = self._build_reflection_prompt(task_description, lang, opts)

        # Get LLM response
        llm_client = ctx.llm_client
        reflection = await llm_client.get_response_with_retries(reflection_prompt, retries=2)

        return reflection

    def reflect(
        self,
        task_description: str,
        context: ReasoningContext | None = None,
        language: Language | None = None,
        options: ReflectionOptions | None = None,
    ) -> str:
        """
        Generate a reflection on the chain execution results (synchronous).

        Analyzes how well the chain accomplished the given task based on
        step results and overall execution.

        Args:
            task_description: Description of the original task/goal
            context: Optional context to use (uses last execution context if not provided)
            language: Language for reflection (uses context language if not provided)
            options: Optional ReflectionOptions to control what's included in the prompt

        Returns:
            Reflection text analyzing the execution results

        Raises:
            RuntimeError: If no execution has been performed yet

        Example:
            ```python
            chain = ReasoningChain(steps=[...])
            context = ReasoningContext(outer_context="data", api=client)
            result = chain.execute(context)

            # Reflect on how well the task was accomplished
            reflection = chain.reflect(
                task_description="Analyze customer sentiment and extract key themes"
            )
            print(reflection)

            # With options for minimal reflection
            from mmar_carl import ReflectionOptions
            options = ReflectionOptions(include_dependency_analysis=False)
            reflection = chain.reflect("Analyze sentiment", options=options)
            ```
        """

        async def _reflect_and_cleanup() -> str:
            """Run reflection and close LLM clients before the event loop shuts down."""
            try:
                return await self.reflect_async(task_description, context, language, options)
            finally:
                ctx = context or self._last_context
                if ctx is not None:
                    await ctx.close()
                    # Wait for background tasks (e.g., httpx connection cleanup) to complete.
                    current_task = asyncio.current_task()
                    for _ in range(10):
                        await asyncio.sleep(0.01)
                        other_tasks = [t for t in asyncio.all_tasks() if t != current_task]
                        if not other_tasks:
                            break
                        try:
                            await asyncio.wait_for(asyncio.gather(*other_tasks, return_exceptions=True), timeout=0.1)
                        except asyncio.TimeoutError:
                            pass

        try:
            # Check if we're already in an event loop
            _ = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(_reflect_and_cleanup())
        else:
            # We're in an event loop, create a task and run it
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _reflect_and_cleanup())
                return future.result()

    def _build_reflection_prompt(self, task_description: str, language: Language, options: ReflectionOptions) -> str:
        """
        Build the reflection prompt based on execution results.

        This method now provides concrete, actionable advice for improving the chain.
        The prompt instructs the LLM to provide specific code/prompt changes that
        a developer can directly implement.

        Args:
            task_description: The original task/goal
            language: Language for the prompt
            options: ReflectionOptions controlling what's included

        Returns:
            Formatted reflection prompt with enhanced context and actionable guidance
        """
        result = self._last_result

        # Gather enhanced context using helper methods (respecting options)
        context_sections = []

        if options.include_chain_structure:
            context_sections.append(self._build_chain_structure_summary())

        if options.include_step_definitions:
            context_sections.append(self._build_step_definitions_summary())

        context_sections.append(self._build_step_details_summary(max_preview_chars=options.max_result_preview_chars))

        if options.include_execution_metrics:
            context_sections.append(self._build_execution_metrics_summary())

        if options.include_dependency_analysis:
            context_sections.append(self._build_dependency_analysis())

        if options.include_metric_scores:
            scores_section = self._build_metric_scores_summary()
            if scores_section:
                context_sections.append(scores_section)

        if options.dataset_report is not None:
            context_sections.append(self._build_dataset_report_section(options.dataset_report, language))

        if options.extra_feedback is not None:
            context_sections.append(self._build_extra_feedback_summary(options.extra_feedback, language))

        full_context = "\n\n".join(context_sections)

        # Final output
        if result is None:
            final_output = "(No execution result available)"
        else:
            final_output = result.get_final_output() or result.get_full_output()
        if len(final_output) > options.max_output_preview_chars:
            final_output = final_output[: options.max_output_preview_chars] + "..."

        if language == Language.ENGLISH:
            return f"""You are an expert chain architecture analyst. Your task is to provide **CONCRETE, ACTIONABLE ADVICE** for rewriting this reasoning chain to better accomplish the given task.

## Original Task
{task_description}

{full_context}

## Final Output
```
{final_output}
```

## Instructions: Provide Actionable Improvements

Analyze the chain execution and provide **specific, implementable changes**. Each recommendation must include exact code/prompt modifications.

### Required Output Format

Provide your analysis in these sections:

#### 1. SPECIFIC CHANGES NEEDED
For each step that needs improvement, provide:
- **Step number and title**
- **Problem**: What specifically is wrong (vague aim, missing context query, wrong dependency, etc.)
- **Fix**: EXACT code to replace the current step

Example format:
```
STEP 2 (Competitive Analysis)
Problem: Queries for "CompetitorData" which doesn't exist in outer_context
Fix: Remove the step OR update step_context_queries to use existing fields:
  step_context_queries=["Comment", "Product", "Rating"]
```

#### 2. STEP-BY-STEP REWRITE INSTRUCTIONS
For steps that need modification, show "before → after":
```
BEFORE:
aim = "Analyze the data"
AFTER:
aim = "Categorize each comment into themes: Product Quality, Shipping, Value for Money, or Customer Service. Count occurrences per product."
```

#### 3. NEW STEPS TO ADD
Identify missing functionality and provide FULL step definitions:
```
ADD NEW STEP 2.5: Count Themes by Product
LLMStepDescription(
    number=2.5,
    title="Aggregate Theme Counts",
    aim="Count theme occurrences per product",
    reasoning_questions="How many comments mention each theme for each product?",
    dependencies=[1, 2]
)
```

#### 4. STEPS TO REMOVE/COMBINE
Identify redundant steps:
```
REMOVE: Step 3 (Redundant Summary)
- This step duplicates what step 4 already does
- Update step 4 dependencies to [1, 2] instead of [1, 2, 3]
```

#### 5. DEPENDENCY FIXES
Identify bottlenecks and provide exact changes:
```
STEP 2: Remove dependency on step 1
CHANGE: step_2.dependencies = []
REASON: Competitive analysis doesn't need theme extraction to run first
BENEFIT: Steps 1 and 2 can run in parallel
```

#### 6. STRUCTURE REORGANIZATION
If chain structure should change:
```
CURRENT BOTTLENECK: Step 3 waits for [1, 2] but only needs step 1
RECOMMENDED: Make step 2 independent, restructure as:
  - Level 1: Steps 1, 2 (parallel)
  - Level 2: Step 3 (depends on 1 only)
  - Level 3: Step 4 (depends on 1, 2, 3)
```

## Critical Rules
1. **Be Specific**: Show exact code changes, not general suggestions
2. **Reference Step Numbers**: Always mention which step you're modifying
3. **Provide Complete Code**: For new steps, include ALL required fields
4. **Explain WHY**: Each change should have a clear rationale
5. **Prioritize Impact**: Focus on changes that will have the biggest effect on output quality
6. **Check Context Queries**: Verify all step_context_queries exist in the provided data
7. **Validate Dependencies**: Ensure dependencies are actually necessary
8. **Use Metric Scores**: If evaluation metric scores are present above, reference them to justify quality issues and prioritize improvements (low scores = higher priority fixes)

Begin your analysis now. Focus on ACTIONABLE advice that can be immediately implemented."""

        else:  # Russian
            return f"""Ты — эксперт по анализу архитектуры цепочек рассуждений. Твоя задача — предоставить **КОНКРЕТНЫЕ, ПРИКЛАДНЫЕ РЕКОМЕНДАЦИИ** по переписыванию этой цепочки для лучшего выполнения поставленной задачи.

## Исходная задача
{task_description}

{full_context}

## Итоговый результат
```
{final_output}
```

## Инструкции: Предоставь конкретные улучшения

Проанализируй выполнение цепочки и предоставь **конкретные, реализуемые изменения**. Каждая рекомендация должна включать точные изменения кода/промптов.

### Требуемый формат вывода

Предоставь анализ в следующих разделах:

#### 1. КОНКРЕТНЫЕ НЕОБХОДИМЫЕ ИЗМЕНЕНИЯ
Для каждого шага, требующего улучшения, укажи:
- **Номер шага и название**
- **Проблема**: Что именно неправильно (неясная цель, отсутствующий контекстный запрос, неправильная зависимость и т.д.)
- **Решение**: ТОЧНЫЙ код для замены текущего шага

Формат примера:
```
ШАГ 2 (Анализ конкурентов)
Проблема: Запрашивает "CompetitorData", которого нет в outer_context
Решение: Удали шаг ИЛИ обнови step_context_queries для использования существующих полей:
  step_context_queries=["Comment", "Product", "Rating"]
```

#### 2. ПОШАГОВЫЕ ИНСТРУКЦИИ ПО ПЕРЕПИСЫВАНИЮ
Для шагов, требующих изменений, покажи "до → после":
```
ДО:
aim = "Проанализировать данные"
ПОСЛЕ:
aim = "Категоризировать каждый комментарий по темам: Качество продукта, Доставка, Соотношение цены и качества или Сервис обслуживания. Подсчитать количество по каждому продукту."
```

#### 3. НОВЫЕ ШАГИ ДЛЯ ДОБАВЛЕНИЯ
Определи отсутствующую функциональность и предоставь ПОЛНЫЕ определения шагов:
```
ДОБАВИТЬ НОВЫЙ ШАГ 2.5: Подсчитать темы по продуктам
LLMStepDescription(
    number=2.5,
    title="Агрегировать количество тем",
    aim="Подсчитать упоминания каждой темы по каждому продукту",
    reasoning_questions="Сколько комментариев упоминают каждую тему для каждого продукта?",
    dependencies=[1, 2]
)
```

#### 4. ШАГИ ДЛЯ УДАЛЕНИЯ/ОБЪЕДИНЕНИЯ
Определи избыточные шаги:
```
УДАЛИТЬ: Шаг 3 (Избыточная сводка)
- Этот шаг дублирует то, что уже делает шаг 4
- Обнови зависимости шага 4 на [1, 2] вместо [1, 2, 3]
```

#### 5. ИСПРАВЛЕНИЕ ЗАВИСИМОСТЕЙ
Определи узкие места и предоставь точные изменения:
```
ШАГ 2: Удали зависимость от шага 1
ИЗМЕНИ: step_2.dependencies = []
ПРИЧИНА: Анализ конкурентов не требует предварительного извлечения тем
ПРЕИМУЩЕСТВО: Шаги 1 и 2 могут выполняться параллельно
```

#### 6. РЕОРГАНИЗАЦИЯ СТРУКТУРЫ
Если структура цепочки должна измениться:
```
ТЕКУЩЕЕ УЗКОЕ МЕСТО: Шаг 3 ожидает [1, 2], но нужен только шаг 1
РЕКОМЕНДАЦИЯ: Сделай шаг 2 независимым, реструктурируй как:
  - Уровень 1: Шаги 1, 2 (параллельно)
  - Уровень 2: Шаг 3 (зависит только от 1)
  - Уровень 3: Шаг 4 (зависит от 1, 2, 3)
```

## Критические правила
1. **Будь конкретным**: Показывай точные изменения кода, а не общие рекомендации
2. **Указывай номера шагов**: Всегда упоминай, какой шаг ты изменяешь
3. **Предоставляй полный код**: Для новых шагов включай ВСЕ необходимые поля
4. **Объясняй ПОЧЕМУ**: Каждое изменение должно иметь чёткое обоснование
5. **Приоритет влияния**: Сосредоточься на изменениях с наибольшим эффектом на качество вывода
6. **Проверяй контекстные запросы**: Убедись, что все step_context_queries существуют в предоставленных данных
7. **Проверяй зависимости**: Убедись, что зависимости действительно необходимы
8. **Используй метрики**: Если выше присутствуют оценочные метрики, ссылайся на них для обоснования проблем качества и расстановки приоритетов улучшений (низкие баллы = более высокий приоритет исправления)

Начни анализ сейчас. Сосредоточься на ПРИКЛАДНЫХ рекомендациях, которые можно немедленно реализовать."""

    # =========================================================================
    # Reflection Helper Methods
    # =========================================================================

    def _build_chain_structure_summary(self) -> str:
        """
        Build a summary of the chain structure and configuration.

        Returns:
            Formatted chain structure summary
        """
        # Count step types
        step_type_counts: dict[str, int] = {}
        for step in self.steps:
            st = step.step_type
            step_type_counts[st] = step_type_counts.get(st, 0) + 1

        # Calculate parallelization potential
        exec_plan = self.get_execution_plan()
        parallelizable = sum(1 for level in exec_plan["execution_levels"] if level["parallelizable"])
        total_levels = len(exec_plan["execution_levels"])

        return f"""## Chain Structure

**Configuration:**
- Total Steps: {len(self.steps)}
- Max Workers: {self.max_workers}
- Parallelization Potential: {parallelizable}/{total_levels} levels can run in parallel
- Parallelization Ratio: {exec_plan.get("parallelization_ratio", 0):.1%}

**Step Type Distribution:**
{chr(10).join(f"- {step_type}: {count}" for step_type, count in sorted(step_type_counts.items()))}

**Execution Plan (Batches):**
{chr(10).join(f"Level {l['level']}: Steps {l['steps']} ({'parallel' if l['parallelizable'] else 'sequential'})" for l in exec_plan["execution_levels"])}"""  # noqa: E741

    def _build_step_definitions_summary(self) -> str:
        """
        Build a summary of the original step definitions for analysis.

        Returns:
            Formatted step definitions summary
        """
        lines = ["## Original Step Definitions", ""]
        for step in self.steps:
            lines.append(f"### Step {step.number}: {step.title}")
            lines.append(f"Type: {step.step_type}")
            lines.append(f"Dependencies: {step.dependencies if step.dependencies else 'None'}")

            # LLM-specific fields
            if step.is_llm_step():
                aim = getattr(step, "aim", "")
                reasoning_questions = getattr(step, "reasoning_questions", "")
                step_context_queries = getattr(step, "step_context_queries", None)
                lines.append(f"Aim: {aim}")
                lines.append(f"Reasoning Questions: {reasoning_questions}")
                lines.append(f"Context Queries: {step_context_queries if step_context_queries else 'None'}")

            # Non-LLM step config
            if step.step_config is not None:
                lines.append(f"Config: {step.step_config}")

            lines.append("")

        return "\n".join(lines)

    def _build_step_details_summary(self, max_preview_chars: int = 500) -> str:
        """
        Build a summary of step execution results with detailed information.

        Args:
            max_preview_chars: Maximum characters to show in result previews

        Returns:
            Formatted step details summary
        """
        result = self._last_result
        if not result:
            return "## Step Execution Results\n\nNo execution results available."

        lines = ["## Step Execution Results", ""]
        for step_result in result.step_results:
            status = "✓ SUCCESS" if step_result.success else "✗ FAILED"
            lines.append(f"### Step {step_result.step_number} ({step_result.step_type}): {step_result.step_title}")
            lines.append(f"Status: {status}")

            if step_result.execution_time:
                lines.append(f"Execution Time: {step_result.execution_time:.3f}s")

            if not step_result.success and step_result.error_message:
                lines.append(f"Error: {step_result.error_message}")

            # Show result preview
            result_preview = step_result.result if step_result.result else "(no result)"
            if len(result_preview) > max_preview_chars:
                result_preview = result_preview[:max_preview_chars] + "..."
            lines.append(f"Result Preview:\n{result_preview}")

            lines.append("")

        return "\n".join(lines)

    def _build_metric_scores_summary(self) -> str:
        """
        Build a summary of MetricBase evaluation scores from step and chain results.

        Step scores and chain scores are both included when available.
        Returns an empty string if no metrics were computed.

        This section is fed to the LLM during reflection so it can reference
        concrete quality signals (e.g. low keyword_coverage on a step) when
        producing improvement recommendations.
        """
        result = self._last_result
        if not result:
            return ""

        lines: list[str] = []

        # Step-level metric scores
        steps_with_metrics = [sr for sr in result.step_results if sr.metrics]
        if steps_with_metrics:
            lines.append("### Step Metric Scores")
            for sr in steps_with_metrics:
                for name, value in sr.metrics.items():
                    lines.append(f"- Step {sr.step_number} ({sr.step_title}): {name} = {value:.4g}")

        # Chain-level metric scores
        if result.metrics:
            lines.append("### Chain Metric Scores (on final output)")
            for name, value in result.metrics.items():
                lines.append(f"- {name} = {value:.4g}")

        if not lines:
            return ""

        return "## Evaluation Metric Scores\n\n" + "\n".join(lines)

    def _build_extra_feedback_summary(self, extra_feedback: dict[str, Any] | str, language: Language) -> str:
        """
        Format optional user-provided feedback/context for the reflection prompt.

        Args:
            extra_feedback: Dict of labelled entries or a plain string.
            language: Language for the section header.

        Returns:
            Formatted section string ready to be appended to the prompt.
        """
        if isinstance(extra_feedback, str):
            content = extra_feedback
        else:
            content = "\n".join(f"- **{k}**: {v}" for k, v in extra_feedback.items())

        if language == Language.ENGLISH:
            header = "## Additional Feedback"
        else:
            header = "## Дополнительный контекст"

        return f"{header}\n\n{content}"

    def _build_dataset_report_section(self, report: DatasetEvaluationReport, language: Language) -> str:
        """
        Build a prompt section from a DatasetEvaluationReport.

        Summarises dataset-level statistics and lists each selected problem case
        with input/output previews and its metric score.

        Args:
            report: The evaluation report produced by DatasetEvaluator.
            language: Language for section headers and labels.

        Returns:
            Formatted section string ready to be appended to the prompt.
        """
        total = len(report.all_results)
        k = len(report.selected_cases)

        strategy = report.strategy
        if strategy.mode == "threshold":
            direction = "below" if strategy.higher_is_better else "above"
            strategy_desc = f"threshold {strategy.threshold} ({direction})"
        else:  # top_k_worst
            strategy_desc = f"top-{strategy.k} worst"

        lines: list[str] = []

        if language == Language.ENGLISH:
            lines += [
                "## Dataset Evaluation",
                "",
                f"**Metric:** {report.metric_name}",
                f"**Cases evaluated:** {total}  |  "
                f"**Mean score:** {report.mean_score:.3f}  "
                f"(min: {report.min_score:.3f}, max: {report.max_score:.3f})",
                f"**Problem cases selected:** {k}/{total}  (strategy: {strategy_desc})",
                "",
                "> **Overfitting warning:** Optimize for patterns visible across "
                "problem cases, not individual quirks. Verify that improvements do "
                "not degrade the mean score.",
            ]
            if not report.selected_cases:
                lines += ["", "_No problem cases found — all cases pass the selection criterion._"]
            else:
                lines += ["", "### Problem Cases", ""]
                for i, r in enumerate(report.selected_cases, 1):
                    label = r.case.label or f"case_{i}"
                    input_preview = r.case.input[:300].replace("\n", " ")
                    output_preview = r.chain_output[:300].replace("\n", " ")
                    lines += [
                        f"**Case {i}** `{label}`  —  score: **{r.score:.3f}**",
                        f"- Input:  `{input_preview}`",
                        f"- Output: `{output_preview}`",
                        "",
                    ]
        else:  # Russian
            lines += [
                "## Оценка датасета",
                "",
                f"**Метрика:** {report.metric_name}",
                f"**Кейсов оценено:** {total}  |  "
                f"**Средний балл:** {report.mean_score:.3f}  "
                f"(мин: {report.min_score:.3f}, макс: {report.max_score:.3f})",
                f"**Проблемных кейсов отобрано:** {k}/{total}  (стратегия: {strategy_desc})",
                "",
                "> **Предупреждение о переоптимизации:** Оптимизируй под паттерны, "
                "видимые по нескольким проблемным кейсам, а не под отдельные случаи. "
                "Убедись, что улучшения не ухудшают средний балл.",
            ]
            if not report.selected_cases:
                lines += ["", "_Проблемных кейсов не найдено — все кейсы проходят критерий отбора._"]
            else:
                lines += ["", "### Проблемные кейсы", ""]
                for i, r in enumerate(report.selected_cases, 1):
                    label = r.case.label or f"case_{i}"
                    input_preview = r.case.input[:300].replace("\n", " ")
                    output_preview = r.chain_output[:300].replace("\n", " ")
                    lines += [
                        f"**Кейс {i}** `{label}`  —  балл: **{r.score:.3f}**",
                        f"- Входные данные:  `{input_preview}`",
                        f"- Результат цепочки: `{output_preview}`",
                        "",
                    ]

        return "\n".join(lines)

    def _build_execution_metrics_summary(self) -> str:
        """
        Build a summary of execution metrics and timing.

        Returns:
            Formatted execution metrics summary
        """
        result = self._last_result
        if not result:
            return "## Execution Metrics\n\nNo execution metrics available."

        successful = result.get_successful_steps()
        failed = result.get_failed_steps()

        # Calculate timing stats
        individual_times = [sr.execution_time for sr in result.step_results if sr.execution_time]
        total_individual_time = sum(individual_times) if individual_times else 0
        avg_time = total_individual_time / len(individual_times) if individual_times else 0
        wall_time = result.total_execution_time or 0

        # Calculate parallel efficiency
        efficiency = 0.0
        if wall_time > 0 and total_individual_time > 0:
            efficiency = min(1.0, total_individual_time / (wall_time * self.max_workers))

        lines = [
            "## Execution Metrics",
            "",
            f"**Overall Status:** {'SUCCESS' if result.success else 'FAILED'}",
            f"**Total Steps:** {len(result.step_results)}",
            f"**Successful:** {len(successful)}",
            f"**Failed:** {len(failed)}",
            "",
            "**Timing:**",
            f"Wall Clock Time: {wall_time:.3f}s",
            f"Total Individual Step Time: {total_individual_time:.3f}s",
            f"Average Step Time: {avg_time:.3f}s",
            f"Parallel Efficiency: {efficiency:.1%}",
            "",
        ]

        # Show slowest steps
        if individual_times:
            sorted_by_time = sorted(
                [(sr.step_number, sr.step_title, sr.execution_time) for sr in result.step_results if sr.execution_time],
                key=lambda x: x[2],
                reverse=True,
            )
            lines.append("**Slowest Steps:**")
            for step_num, title, time_val in sorted_by_time[:3]:
                lines.append(f"  Step {step_num} ({title}): {time_val:.3f}s")

        return "\n".join(lines)

    def _build_dependency_analysis(self) -> str:
        """
        Build an analysis of the dependency structure.

        Returns:
            Formatted dependency analysis
        """
        lines = [
            "## Dependency Analysis",
            "",
            "**Step Dependencies:**",
        ]

        # Find entry points (steps with no dependencies)
        entry_points = [s for s in self.steps if not s.dependencies]
        lines.append(f"Entry Points (no dependencies): {[s.number for s in entry_points]}")

        # Find leaves (steps that nothing depends on)
        depended_on = set()
        for step in self.steps:
            depended_on.update(step.dependencies)
        leaves = [s.number for s in self.steps if s.number not in depended_on]
        lines.append(f"Leaf Steps (nothing depends on them): {leaves}")

        # Calculate max depth
        def get_depth(step_num: int, memo: dict[int, int]) -> int:
            if step_num in memo:
                return memo[step_num]
            step = next(s for s in self.steps if s.number == step_num)
            if not step.dependencies:
                memo[step_num] = 1
                return 1
            max_dep_depth = max(get_depth(dep, memo) for dep in step.dependencies)
            memo[step_num] = max_dep_depth + 1
            return memo[step_num]

        depths = {s.number: get_depth(s.number, {}) for s in self.steps}
        max_depth = max(depths.values()) if depths else 0
        lines.append(f"Maximum Dependency Depth: {max_depth}")

        # Find potential parallelization opportunities
        lines.append("")
        lines.append("**Parallelization Opportunities:**")

        # Group steps by depth
        by_depth: dict[int, list[int]] = {}
        for step_num, depth in depths.items():
            by_depth.setdefault(depth, []).append(step_num)

        for depth in sorted(by_depth.keys()):
            steps_at_depth = by_depth[depth]
            if len(steps_at_depth) > 1:
                lines.append(f"Depth {depth}: Steps {steps_at_depth} can run in parallel")
            else:
                lines.append(f"Depth {depth}: Step {steps_at_depth[0]} (sequential)")

        # Check for potential bottlenecks
        lines.append("")
        lines.append("**Potential Bottlenecks:**")
        bottleneck_steps = []
        for step in self.steps:
            # Count how many steps depend on this one
            dependents = sum(1 for s in self.steps if step.number in s.dependencies)
            if dependents > 2:
                bottleneck_steps.append((step.number, step.title, dependents))

        if bottleneck_steps:
            for step_num, title, count in sorted(bottleneck_steps, key=lambda x: x[2], reverse=True):
                lines.append(f"Step {step_num} ({title}): {count} steps depend on this")
        else:
            lines.append("No significant bottlenecks detected")

        return "\n".join(lines)

    def get_last_result(self) -> ReasoningResult | None:
        """Get the last execution result (if any)."""
        return self._last_result

    def get_last_context(self) -> ReasoningContext | None:
        """Get the last execution context (if any)."""
        return self._last_context

    def get_execution_plan(self) -> dict[str, Any]:
        """
        Get the execution plan showing parallelization opportunities.

        Returns:
            Dictionary describing the execution plan
        """
        # Build dependency levels
        levels = []
        remaining_steps = self.steps.copy()
        # Track which step numbers have been "completed" (added to a level)
        completed_step_numbers = set()

        while remaining_steps:
            current_level = []
            for step in remaining_steps[:]:
                # A step can be added if all its dependencies are in completed_step_numbers
                deps_satisfied = all(dep in completed_step_numbers for dep in step.dependencies)
                if deps_satisfied:
                    current_level.append(step)
                    remaining_steps.remove(step)

            # Mark all steps in this level as completed
            for step in current_level:
                completed_step_numbers.add(step.number)

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
                "step_type": step.step_type,
                "aim": getattr(step, "aim", ""),
                "dependencies": step.dependencies,
                "step_context_queries": getattr(step, "step_context_queries", None),
                "has_dependencies": step.has_dependencies(),
            }
            for step in self.steps
        ]

    # =========================================================================
    # Serialization Methods
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the chain to a dictionary.

        Handles both legacy StepDescription and new typed step classes.

        Returns:
            Dictionary representation of the chain
        """
        serialized_steps = []
        for step in self.steps:
            if isinstance(step, StepDescription):
                # Legacy format - use model_dump directly
                serialized_steps.append(step.model_dump(mode="json"))
            elif isinstance(step, StepDescriptionBase):
                # New typed step classes - need to convert to legacy format for JSON
                step_data = {
                    "number": step.number,
                    "title": step.title,
                    "dependencies": step.dependencies,
                    "step_type": step.step_type,
                    "checkpoint": getattr(step, "checkpoint", False),
                    "checkpoint_name": getattr(step, "checkpoint_name", None),
                    "replan_enabled": getattr(step, "replan_enabled", None),
                }
                # Add step_config for non-LLM steps
                if step.step_config is not None:
                    step_data["step_config"] = step.step_config.model_dump(mode="json")
                # Add LLM-specific fields
                if step.is_llm_step():
                    step_data["aim"] = getattr(step, "aim", "")
                    step_data["reasoning_questions"] = getattr(step, "reasoning_questions", "")
                    step_data["step_context_queries"] = getattr(step, "step_context_queries", [])
                    step_data["stage_action"] = getattr(step, "stage_action", "")
                    step_data["example_reasoning"] = getattr(step, "example_reasoning", "")
                    step_data["llm_config"] = getattr(step, "llm_config", None)
                    if step_data["llm_config"] is not None:
                        step_data["llm_config"] = step_data["llm_config"].model_dump(mode="json")
                    step_data["retry_max"] = getattr(step, "retry_max", None)
                    step_data["timeout"] = getattr(step, "timeout", None)
                serialized_steps.append(step_data)
            else:
                # Fallback - try model_dump
                serialized_steps.append(step.model_dump(mode="json"))

        result = {
            "version": "1.1",  # Updated version for new step class support
            "max_workers": self.max_workers,
            "enable_progress": self.enable_progress,
            "metadata": self.metadata,
            "timeout": self.timeout,
            "replan_policy": self.replan_policy.model_dump(mode="json") if self.replan_policy else None,
            "search_config": self.prompt_template.search_config.model_dump() if self.prompt_template else None,
            "steps": serialized_steps,
        }
        if self.trace_name:
            result["trace_name"] = self.trace_name
        if self.session_id:
            result["session_id"] = self.session_id
        return result

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize the chain to a JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the chain to a JSON file.

        Args:
            path: File path to save to
        """
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any], use_typed_steps: bool = False) -> "ReasoningChain":
        """
        Deserialize a chain from a dictionary.

        Args:
            data: Dictionary representation of the chain
            use_typed_steps: If True, create typed step classes (LLMStepDescription, etc.)
                           If False (default), create legacy StepDescription for backward compatibility

        Returns:
            Reconstructed ReasoningChain
        """
        steps: list[StepDescription | AnyStepDescription] = []
        for step_data in data.get("steps", []):
            # Handle step_config reconstruction based on step_type
            step_type = step_data.get("step_type", "llm")
            step_config_data = step_data.get("step_config")

            if step_config_data is not None:
                step_config = _reconstruct_step_config(step_type, step_config_data)
                step_data["step_config"] = step_config

            llm_config_data = step_data.get("llm_config")
            if llm_config_data is not None and not isinstance(llm_config_data, LLMStepConfig):
                step_data["llm_config"] = LLMStepConfig.model_validate(llm_config_data)

            if use_typed_steps:
                # Create typed step class
                step = _create_typed_step_from_dict(step_data)
            else:
                # Create legacy StepDescription for backward compatibility
                step = StepDescription.model_validate(step_data)
            steps.append(step)

        # Reconstruct search config
        search_config = None
        if data.get("search_config"):
            search_config = ContextSearchConfig.model_validate(data["search_config"])

        # Reconstruct RE-PLAN policy
        replan_policy = None
        if data.get("replan_policy"):
            replan_policy = ReplanPolicy.model_validate(data["replan_policy"])

        return cls(
            steps=steps,
            max_workers=data.get("max_workers", 3),
            enable_progress=data.get("enable_progress", False),
            metadata=data.get("metadata", {}),
            search_config=search_config,
            timeout=data.get("timeout"),
            trace_name=data.get("trace_name"),
            session_id=data.get("session_id"),
            replan_policy=replan_policy,
        )

    @classmethod
    def from_dict_typed(cls, data: dict[str, Any]) -> "ReasoningChain":
        """
        Deserialize a chain from a dictionary using typed step classes.

        This creates LLMStepDescription, ToolStepDescription, etc. instead of
        legacy StepDescription.

        Args:
            data: Dictionary representation of the chain

        Returns:
            Reconstructed ReasoningChain with typed step classes
        """
        return cls.from_dict(data, use_typed_steps=True)

    @classmethod
    def from_json(cls, json_str: str) -> "ReasoningChain":
        """
        Deserialize a chain from a JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            Reconstructed ReasoningChain
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ReasoningChain":
        """
        Load a chain from a JSON file.

        Args:
            path: File path to load from

        Returns:
            Reconstructed ReasoningChain
        """
        path = Path(path)
        json_str = path.read_text(encoding="utf-8")
        return cls.from_json(json_str)


def _reconstruct_step_config(step_type: str, config_data: dict[str, Any]) -> Any:
    """Reconstruct step configuration based on step type."""
    if step_type == StepType.TOOL or step_type == "tool":
        # Reconstruct ToolParameter objects if present
        if "parameters" in config_data:
            config_data["parameters"] = [ToolParameter.model_validate(p) for p in config_data["parameters"]]
        return ToolStepConfig.model_validate(config_data)

    elif step_type == StepType.MCP or step_type == "mcp":
        # Reconstruct MCPServerConfig
        if "server" in config_data:
            config_data["server"] = MCPServerConfig.model_validate(config_data["server"])
        return MCPStepConfig.model_validate(config_data)

    elif step_type == StepType.MEMORY or step_type == "memory":
        return MemoryStepConfig.model_validate(config_data)

    elif step_type == StepType.TRANSFORM or step_type == "transform":
        return TransformStepConfig.model_validate(config_data)

    elif step_type == StepType.CONDITIONAL or step_type == "conditional":
        return ConditionalStepConfig.model_validate(config_data)

    elif step_type == StepType.STRUCTURED_OUTPUT or step_type == "structured_output":
        return StructuredOutputStepConfig.model_validate(config_data)

    return None


def _create_typed_step_from_dict(step_data: dict[str, Any]) -> AnyStepDescription:
    """
    Create a typed step class from dictionary data.

    Args:
        step_data: Dictionary containing step data

    Returns:
        The appropriate typed step description instance
    """
    step_type = step_data.get("step_type", "llm")

    if step_type == StepType.LLM or step_type == "llm":
        return LLMStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            aim=step_data.get("aim", ""),
            reasoning_questions=step_data.get("reasoning_questions", ""),
            step_context_queries=step_data.get("step_context_queries", []),
            stage_action=step_data.get("stage_action", ""),
            example_reasoning=step_data.get("example_reasoning", ""),
            llm_config=step_data.get("llm_config"),
            retry_max=step_data.get("retry_max"),
            timeout=step_data.get("timeout"),
        )
    elif step_type == StepType.TOOL or step_type == "tool":
        return ToolStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    elif step_type == StepType.MCP or step_type == "mcp":
        return MCPStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    elif step_type == StepType.MEMORY or step_type == "memory":
        return MemoryStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    elif step_type == StepType.TRANSFORM or step_type == "transform":
        return TransformStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    elif step_type == StepType.CONDITIONAL or step_type == "conditional":
        return ConditionalStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    elif step_type == StepType.STRUCTURED_OUTPUT or step_type == "structured_output":
        return StructuredOutputStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    else:
        raise ValueError(f"Unknown step type: {step_type}")


class ChainBuilder:
    """
    Builder pattern for constructing reasoning chains.

    Provides a fluent interface for building complex reasoning chains.
    """

    def __init__(self):
        """Initialize the chain builder."""
        self.steps: list[StepDescription | StepDescriptionBase | AnyStepDescription] = []
        self.max_workers: int = 3
        self.prompt_template: PromptTemplate | None = None
        self.search_config: ContextSearchConfig | None = None
        self.enable_progress: bool = False
        self.metadata: dict[str, Any] = {}
        self.timeout: float | None = None
        self.trace_name: str | None = None
        self.session_id: str | None = None
        self.replan_policy: ReplanPolicy | None = None

    def add_step(
        self,
        number: int,
        title: str,
        aim: str,
        reasoning_questions: str,
        stage_action: str,
        example_reasoning: str,
        dependencies: list[int] | None = None,
        step_context_queries: list[ContextQuery | str] | None = None,
        llm_config: LLMStepConfig | None = None,
        execution_mode: ExecutionMode | str | None = None,
        checkpoint: bool = False,
        checkpoint_name: str | None = None,
        replan_enabled: bool | None = None,
    ) -> "ChainBuilder":
        """
        Add an LLM reasoning step to the chain.

        Args:
            number: Step number
            title: Step title
            aim: Step objective
            reasoning_questions: Key questions to answer
            stage_action: Action to perform
            example_reasoning: Example of expert reasoning
            dependencies: List of step numbers this depends on
            step_context_queries: RAG-like context queries
            llm_config: Optional per-step LLM config
            execution_mode: Optional execution mode shortcut ("fast", "self_critic")

        Returns:
            Self for method chaining
        """
        resolved_llm_config = llm_config
        if execution_mode is not None:
            mode = execution_mode if isinstance(execution_mode, ExecutionMode) else ExecutionMode(str(execution_mode))
            if resolved_llm_config is None:
                resolved_llm_config = LLMStepConfig(execution_mode=mode)
            else:
                resolved_llm_config = resolved_llm_config.model_copy(update={"execution_mode": mode})

        step = LLMStepDescription(
            number=number,
            title=title,
            aim=aim,
            reasoning_questions=reasoning_questions,
            stage_action=stage_action,
            example_reasoning=example_reasoning,
            dependencies=dependencies or [],
            step_context_queries=step_context_queries or [],
            llm_config=resolved_llm_config,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
        )
        self.steps.append(step)
        return self

    def add_tool_step(
        self,
        number: int,
        title: str,
        tool_name: str,
        input_mapping: dict[str, str] | None = None,
        dependencies: list[int] | None = None,
        tool_description: str = "",
        timeout: float = 30.0,
        checkpoint: bool = False,
        checkpoint_name: str | None = None,
        replan_enabled: bool | None = None,
    ) -> "ChainBuilder":
        """
        Add a tool execution step to the chain.

        Args:
            number: Step number
            title: Step title
            tool_name: Name of the registered tool to call
            input_mapping: Maps context keys to tool parameters
            dependencies: List of step numbers this depends on
            tool_description: Description of the tool
            timeout: Execution timeout in seconds

        Returns:
            Self for method chaining
        """
        step = ToolStepDescription(
            number=number,
            title=title,
            dependencies=dependencies or [],
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=ToolStepConfig(
                tool_name=tool_name,
                tool_description=tool_description,
                input_mapping=input_mapping or {},
                timeout=timeout,
            ),
        )
        self.steps.append(step)
        return self

    def add_mcp_step(
        self,
        number: int,
        title: str,
        server_name: str,
        tool_name: str,
        command: str | None = None,
        args: list[str] | None = None,
        arguments: dict[str, Any] | None = None,
        argument_mapping: dict[str, str] | None = None,
        dependencies: list[int] | None = None,
        timeout: float = 60.0,
        checkpoint: bool = False,
        checkpoint_name: str | None = None,
        replan_enabled: bool | None = None,
    ) -> "ChainBuilder":
        """
        Add an MCP protocol step to the chain.

        Args:
            number: Step number
            title: Step title
            server_name: MCP server name
            tool_name: MCP tool to call
            command: Command to start stdio server
            args: Arguments for the server command
            arguments: Static arguments for the tool
            argument_mapping: Maps context keys to tool arguments
            dependencies: List of step numbers this depends on
            timeout: Execution timeout in seconds

        Returns:
            Self for method chaining
        """
        step = MCPStepDescription(
            number=number,
            title=title,
            dependencies=dependencies or [],
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=MCPStepConfig(
                server=MCPServerConfig(
                    server_name=server_name,
                    command=command,
                    args=args or [],
                ),
                tool_name=tool_name,
                arguments=arguments or {},
                argument_mapping=argument_mapping or {},
                timeout=timeout,
            ),
        )
        self.steps.append(step)
        return self

    def add_memory_step(
        self,
        number: int,
        title: str,
        operation: str,
        memory_key: str,
        value_source: str | None = None,
        default_value: Any = None,
        namespace: str = "default",
        dependencies: list[int] | None = None,
        checkpoint: bool = False,
        checkpoint_name: str | None = None,
        replan_enabled: bool | None = None,
    ) -> "ChainBuilder":
        """
        Add a memory operation step to the chain.

        Args:
            number: Step number
            title: Step title
            operation: Memory operation (read, write, append, delete, list)
            memory_key: Key in memory store
            value_source: Source of value for write operations
            default_value: Default value if key not found
            namespace: Memory namespace
            dependencies: List of step numbers this depends on

        Returns:
            Self for method chaining
        """
        from .models import MemoryOperation

        step = MemoryStepDescription(
            number=number,
            title=title,
            dependencies=dependencies or [],
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=MemoryStepConfig(
                operation=MemoryOperation(operation),
                memory_key=memory_key,
                value_source=value_source,
                default_value=default_value,
                namespace=namespace,
            ),
        )
        self.steps.append(step)
        return self

    def add_transform_step(
        self,
        number: int,
        title: str,
        transform_type: str,
        input_key: str = "$history[-1]",
        output_format: str | None = None,
        expression: str | None = None,
        map_template: str | None = None,
        dependencies: list[int] | None = None,
        checkpoint: bool = False,
        checkpoint_name: str | None = None,
        replan_enabled: bool | None = None,
    ) -> "ChainBuilder":
        """
        Add a data transformation step to the chain (no LLM call).

        Args:
            number: Step number
            title: Step title
            transform_type: Type (extract, format, aggregate, filter, map)
            input_key: Source of input data
            output_format: Format template for 'format' type
            expression: Regex for 'extract' or 'filter' types
            map_template: Template for 'map' type
            dependencies: List of step numbers this depends on

        Returns:
            Self for method chaining
        """
        step = TransformStepDescription(
            number=number,
            title=title,
            dependencies=dependencies or [],
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=TransformStepConfig(
                transform_type=transform_type,  # type: ignore
                input_key=input_key,
                output_format=output_format,
                expression=expression,
                map_template=map_template,
            ),
        )
        self.steps.append(step)
        return self

    def add_conditional_step(
        self,
        number: int,
        title: str,
        branches: list[tuple[str, int] | ConditionalBranch],
        default_step: int | None = None,
        condition_context_key: str = "$history[-1]",
        dependencies: list[int] | None = None,
        checkpoint: bool = False,
        checkpoint_name: str | None = None,
        replan_enabled: bool | None = None,
    ) -> "ChainBuilder":
        """
        Add a conditional branching step to the chain.

        Args:
            number: Step number
            title: Step title
            branches: List of (condition, next_step) tuples OR ConditionalBranch objects
            default_step: Default step if no condition matches
            condition_context_key: Context key to evaluate
            dependencies: List of step numbers this depends on

        Returns:
            Self for method chaining
        """
        from .models import ConditionalBranch

        # Normalize branches to ConditionalBranch objects
        normalized_branches = []
        for branch in branches:
            if isinstance(branch, tuple):
                condition, next_step = branch
                normalized_branches.append(
                    ConditionalBranch(condition=condition, next_step=next_step)
                )
            elif isinstance(branch, ConditionalBranch):
                normalized_branches.append(branch)
            else:
                raise ValueError(
                    f"Branch must be tuple (condition, next_step) or ConditionalBranch, "
                    f"got {type(branch)}"
                )

        step = ConditionalStepDescription(
            number=number,
            title=title,
            dependencies=dependencies or [],
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=ConditionalStepConfig(
                branches=normalized_branches,
                default_step=default_step,
                condition_context_key=condition_context_key,
            ),
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

    def with_trace_name(self, trace_name: str) -> "ChainBuilder":
        """
        Set the trace name for LangFuse tracing.

        Args:
            trace_name: Name shown in LangFuse UI for this chain's trace

        Returns:
            Self for method chaining
        """
        self.trace_name = trace_name
        return self

    def with_session_id(self, session_id: str) -> "ChainBuilder":
        """
        Set the session ID for LangFuse tracing.

        Multiple chains sharing the same session_id are grouped
        as one session in LangFuse.

        Args:
            session_id: Session identifier for grouping traces

        Returns:
            Self for method chaining
        """
        self.session_id = session_id
        return self

    def with_timeout(self, timeout: float) -> "ChainBuilder":
        """
        Set maximum execution time for the chain.

        Args:
            timeout: Maximum execution time in seconds

        Returns:
            Self for method chaining
        """
        self.timeout = timeout
        return self

    def with_replan_policy(self, policy: ReplanPolicy | None) -> "ChainBuilder":
        """
        Set chain-level RE-PLAN policy.

        Args:
            policy: RE-PLAN policy configuration (None to disable)

        Returns:
            Self for method chaining
        """
        self.replan_policy = policy
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
            timeout=self.timeout,
            trace_name=self.trace_name,
            session_id=self.session_id,
            replan_policy=self.replan_policy,
        )


def create_chain_from_config(config: dict[str, Any]) -> ReasoningChain:
    """
    Create a reasoning chain from a configuration dictionary.

    This function delegates to ReasoningChain.from_dict() which properly
    handles all step types including tool, MCP, memory, transform, and conditional.

    Args:
        config: Configuration dictionary

    Returns:
        Constructed reasoning chain
    """
    return ReasoningChain.from_dict(config)

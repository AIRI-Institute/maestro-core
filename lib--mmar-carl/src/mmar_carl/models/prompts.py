"""
Prompt templates for CARL reasoning system.
"""

from typing import Union

from pydantic import BaseModel, Field

from .config import ContextQuery
from .enums import Language
from .search import ContextSearchConfig, SubstringSearchStrategy, VectorSearchStrategy
from .steps import AnyStepDescription, StepDescription, StepDescriptionBase


class PromptTemplate(BaseModel):
    """
    Template for generating prompts from reasoning steps.
    """

    system_prompt: str | None = Field(default=None, description="System-level instructions")
    search_config: ContextSearchConfig = Field(
        default_factory=ContextSearchConfig, description="Configuration for context search strategies"
    )

    # Russian templates
    ru_step_template: str = Field(
        default="Шаг {step_number}. {step_title}\nЦель: {aim}\nЗадача: {stage_action}\nВопросы: {reasoning_questions}\nКонтекстные запросы: {context_queries}\nПример рассуждений: {example_reasoning}",
        description="Template for individual step prompts in Russian",
    )
    ru_chain_template: str = Field(
        default="Данные для анализа:\n{outer_context}\n{step_prompt}\nОтвечай кратко, подумай какие можно сделать выводы о результатах. Ответ должен состоять из одного параграфа. Не задавай дополнительных вопросов и не передавай инструкций. Пиши только текстом, без математических формул.",
        description="Template for complete chain prompts in Russian",
    )
    ru_history_template: str = Field(
        default="История предыдущих шагов:\n{history}\nОсновываясь на результатах предыдущих шагов, выполни следующую задачу:\n{current_task}",
        description="Template for including history in prompts in Russian",
    )

    # English templates
    en_step_template: str = Field(
        default="Step {step_number}. {step_title}\nObjective: {aim}\nTask: {stage_action}\nQuestions: {reasoning_questions}\nContext Queries: {context_queries}\nExample reasoning: {example_reasoning}",
        description="Template for individual step prompts in English",
    )
    en_chain_template: str = Field(
        default="Data for analysis:\n{outer_context}\n{step_prompt}\nRespond concisely, consider what conclusions can be drawn from the results. Response should be one paragraph. Do not ask additional questions or provide instructions. Write in text only, without mathematical formulas.",
        description="Template for complete chain prompts in English",
    )
    en_history_template: str = Field(
        default="History of previous steps:\n{history}\nBased on the results of previous steps, perform the following task:\n{current_task}",
        description="Template for including history in prompts in English",
    )

    def extract_context_from_queries(self, outer_context: str, queries: list[Union[ContextQuery, str]]) -> str:
        """
        Extract relevant context from outer_context using queries (RAG-like functionality).

        Args:
            outer_context: The full context data to search through
            queries: List of queries to find relevant context (strings or ContextQuery objects)

        Returns:
            String containing relevant context found for each query
        """
        if not queries:
            return "No specific context queries defined"

        relevant_contexts = []
        for query_item in queries:
            # Handle both string queries and ContextQuery objects
            if isinstance(query_item, str):
                query_text = query_item
                query_strategy = None
                query_config = {}
            else:  # ContextQuery object
                query_text = query_item.query
                query_strategy = query_item.search_strategy
                query_config = query_item.search_config or {}

            # Use query-specific strategy or default chain strategy
            if query_strategy:
                if query_strategy == "vector":
                    strategy = VectorSearchStrategy(
                        embedding_model=query_config.get("embedding_model", self.search_config.embedding_model),
                        index_type=query_config.get("index_type", "flat"),
                        similarity_threshold=query_config.get("similarity_threshold", 0.7),
                        max_results=query_config.get("max_results", 5),
                    )
                else:  # substring
                    strategy = SubstringSearchStrategy(
                        case_sensitive=query_config.get("case_sensitive", False),
                        min_word_length=query_config.get("min_word_length", 2),
                        max_matches_per_query=query_config.get("max_matches_per_query", 3),
                    )
            else:
                # Use default chain strategy
                strategy = self.search_config.get_strategy()

            # Extract context for this specific query
            result = strategy.extract_context(outer_context, [query_text])
            relevant_contexts.append(result)

        return "\n\n".join(relevant_contexts)

    def format_step_prompt(
        self, step: StepDescription | StepDescriptionBase | AnyStepDescription, outer_context: str = "", language: Language = Language.RUSSIAN
    ) -> str:
        """Format a single step prompt with RAG-like context extraction."""
        # Extract relevant context using step_context_queries
        step_context_queries = getattr(step, "step_context_queries", [])
        context_queries_result = self.extract_context_from_queries(outer_context, step_context_queries)

        # Get LLM-specific fields with defaults
        aim = getattr(step, "aim", "")
        stage_action = getattr(step, "stage_action", "")
        reasoning_questions = getattr(step, "reasoning_questions", "")
        example_reasoning = getattr(step, "example_reasoning", "")

        if language == Language.ENGLISH:
            return self.en_step_template.format(
                step_number=step.number,
                step_title=step.title,
                aim=aim,
                stage_action=stage_action,
                reasoning_questions=reasoning_questions,
                context_queries=context_queries_result,
                example_reasoning=example_reasoning,
            )
        else:  # Russian
            return self.ru_step_template.format(
                step_number=step.number,
                step_title=step.title,
                aim=aim,
                stage_action=stage_action,
                reasoning_questions=reasoning_questions,
                context_queries=context_queries_result,
                example_reasoning=example_reasoning,
            )

    def format_chain_prompt(
        self,
        outer_context: str,
        current_task: str,
        history: str = "",
        language: Language = Language.RUSSIAN,
        system_prompt: str = "",
    ) -> str:
        """Format a complete chain prompt."""
        if language == Language.ENGLISH:
            if history:
                current_task = self.en_history_template.format(history=history, current_task=current_task)

            full_prompt = self.en_chain_template.format(outer_context=outer_context, step_prompt=current_task)
        else:  # Russian
            if history:
                current_task = self.ru_history_template.format(history=history, current_task=current_task)

            full_prompt = self.ru_chain_template.format(outer_context=outer_context, step_prompt=current_task)

        # Add system prompt at the beginning if provided
        if system_prompt:
            if language == Language.ENGLISH:
                return f"System Instructions:\n{system_prompt}\n\n{full_prompt}"
            else:  # Russian
                return f"Системные инструкции:\n{system_prompt}\n\n{full_prompt}"

        return full_prompt

"""
Core data models for CARL reasoning system.
"""

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class LLMClientBase(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def get_response(self, prompt: str) -> str:
        """
        Get a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM response as a string
        """
        pass

    @abstractmethod
    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """
        Get a response from the LLM with retry logic.

        Args:
            prompt: The prompt to send to the LLM
            retries: Maximum number of retry attempts

        Returns:
            The LLM response as a string
        """
        pass


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""

    @abstractmethod
    def extract_context(self, outer_context: str, queries: List[str], **kwargs) -> str:
        """
        Extract relevant context from outer_context using queries.

        Args:
            outer_context: The full context data to search through
            queries: List of queries to find relevant context
            **kwargs: Additional strategy-specific parameters

        Returns:
            String containing relevant context found for each query
        """
        pass


class SubstringSearchStrategy(SearchStrategy):
    """Substring-based search strategy."""

    def __init__(self, case_sensitive: bool = False, min_word_length: int = 2, max_matches_per_query: int = 3):
        """
        Initialize substring search strategy.

        Args:
            case_sensitive: Whether to perform case-sensitive search
            min_word_length: Minimum word length to consider for matching
            max_matches_per_query: Maximum number of matches per query
        """
        self.case_sensitive = case_sensitive
        self.min_word_length = min_word_length
        self.max_matches_per_query = max_matches_per_query

    def extract_context(self, outer_context: str, queries: List[str], **kwargs) -> str:
        """Extract context using substring search."""
        if not queries:
            return "No specific context queries defined"

        relevant_contexts = []
        for query in queries:
            lines = outer_context.split("\n")
            relevant_lines = []

            for line in lines:
                line_content = line.strip()
                if not line_content:
                    continue

                query_text = query if self.case_sensitive else query.lower()
                line_text = line_content if self.case_sensitive else line_content.lower()

                # Check if any word from query appears in the line
                query_words = query_text.split()
                if any(word in line_text for word in query_words if len(word) >= self.min_word_length):
                    relevant_lines.append(line_content)
                    if len(relevant_lines) >= self.max_matches_per_query:
                        break

            if relevant_lines:
                context_snippet = " | ".join(relevant_lines)
                relevant_contexts.append(f"Query '{query}': {context_snippet}")
            else:
                relevant_contexts.append(f"Query '{query}': No matches found")

        return "\n".join(relevant_contexts)


class VectorSearchStrategy(SearchStrategy):
    """Vector-based search strategy using FAISS."""

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        index_type: Literal["flat", "ivf"] = "flat",
        similarity_threshold: float = 0.7,
        max_results: int = 5,
    ):
        """
        Initialize vector search strategy.

        Args:
            embedding_model: Name of sentence-transformers model to use (e.g., "all-MiniLM-L6-v2")
            index_type: Type of FAISS index ("flat" or "ivf")
            similarity_threshold: Minimum similarity score (0-1)
            max_results: Maximum number of results to return
        """
        self.embedding_model = embedding_model
        self.index_type = index_type
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self._index = None
        self._documents = []

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        try:
            from fastembed import TextEmbedding

            model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
            # Convert generator to list and then to list of lists
            embeddings = list(model.embed(texts))
            return embeddings

        except ImportError:
            # Fallback to simple character embeddings if fastembed not available
            return self._fallback_embeddings(texts)

    def _fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Fallback character-level embeddings if sentence-transformers not available."""
        embeddings = []
        for text in texts:
            # Simple character-based embedding
            chars = [ord(c) for c in text[:500]]  # Limit to first 500 chars
            # Pad or truncate to fixed length
            embedding = chars + [0] * (100 - len(chars))
            embeddings.append(embedding)

        return embeddings

    def extract_context(self, outer_context: str, queries: List[str], **kwargs) -> str:
        """Extract context using vector similarity search."""
        if not queries:
            return "No specific context queries defined"

        try:
            import faiss
        except ImportError:
            # Fallback to substring search if FAISS not available
            print("Warning: FAISS not available, falling back to substring search")
            fallback_strategy = SubstringSearchStrategy()
            return fallback_strategy.extract_context(outer_context, queries, **kwargs)

        # Split context into chunks for indexing
        lines = outer_context.split("\n")
        self._documents = [line.strip() for line in lines if line.strip()]

        if not self._documents:
            return "No context available for vector search"

        # Create embeddings for documents
        doc_embeddings = self._get_embeddings(self._documents)
        import numpy as np

        doc_embeddings = np.array(doc_embeddings).astype("float32")

        # Build FAISS index
        dimension = doc_embeddings.shape[1]
        if self.index_type == "flat":
            self._index = faiss.IndexFlatL2(dimension)
        else:  # ivf
            quantizer = faiss.IndexFlatL2(dimension)
            self._index = faiss.IndexIVFFlat(quantizer, dimension, min(100, len(doc_embeddings) // 10))

        self._index.add(doc_embeddings)

        # Generate embeddings for queries and search
        query_embeddings = self._get_embeddings(queries)
        query_embeddings = np.array(query_embeddings).astype("float32")

        relevant_contexts = []
        for i, query in enumerate(queries):
            if self.index_type == "flat":
                distances, indices = self._index.search(query_embeddings[i : i + 1], self.max_results)
            else:
                distances, indices = self._index.search(query_embeddings[i : i + 1], self.max_results)

            # Filter by similarity threshold
            filtered_results = []
            for dist, idx in zip(distances[0], indices[0]):
                # Convert L2 distance to similarity (lower distance = higher similarity)
                similarity = max(0.0, 1.0 - (dist / (dist + 1e-8)))
                if similarity >= self.similarity_threshold:
                    filtered_results.append((similarity, idx))

            # Sort by similarity (descending) and format results
            filtered_results.sort(reverse=True)
            if filtered_results:
                context_parts = [f"{result:.3f}:{self._documents[idx]}" for result, idx in filtered_results]
                relevant_contexts.append(f"Query '{query}': {' | '.join(context_parts[:3])}")
            else:
                relevant_contexts.append(f"Query '{query}': No similar content found")

        return "\n".join(relevant_contexts)


class ContextSearchConfig(BaseModel):
    """Configuration for context search strategies."""

    strategy: Literal["substring", "vector"] = Field(default="substring", description="Search strategy to use")
    substring_config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration for substring search")
    vector_config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration for vector search")
    embedding_model: Optional[str] = Field(default=None, description="Embedding model name for vector search")

    def get_strategy(self) -> SearchStrategy:
        """Get the configured search strategy."""
        if self.strategy == "vector":
            vector_config = self.vector_config or {}
            return VectorSearchStrategy(
                embedding_model=self.embedding_model or vector_config.get("embedding_model"),
                index_type=vector_config.get("index_type", "flat"),
                similarity_threshold=vector_config.get("similarity_threshold", 0.7),
                max_results=vector_config.get("max_results", 5),
            )
        else:  # substring
            substring_config = self.substring_config or {}
            return SubstringSearchStrategy(
                case_sensitive=substring_config.get("case_sensitive", False),
                min_word_length=substring_config.get("min_word_length", 2),
                max_matches_per_query=substring_config.get("max_matches_per_query", 3),
            )


class Language(StrEnum):
    """Supported languages."""

    RUSSIAN = "ru"
    ENGLISH = "en"


class ContextQuery(BaseModel):
    """
    Individual context query with optional search configuration override.

    Allows fine-grained control over search strategy for specific queries.
    """

    query: str = Field(..., description="The query text for context extraction")
    search_strategy: Optional[Literal["substring", "vector"]] = Field(
        default=None, description="Override search strategy for this query"
    )
    search_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional search configuration for this query"
    )

    def __str__(self) -> str:
        return self.query


class StepDescription(BaseModel):
    """
    Defines a single reasoning step in a chain.

    This model encapsulates all the metadata needed for a reasoning step,
    including its dependencies, objectives, and execution guidance.
    """

    number: int = Field(..., description="Step number in the sequence")
    title: str = Field(..., description="Human-readable title of the step")
    aim: str = Field(..., description="Primary objective of this step")
    reasoning_questions: str = Field(..., description="Key questions to answer")
    dependencies: list[int] = Field(default_factory=list, description="List of step numbers this step depends on")
    step_context_queries: list[ContextQuery | str] = Field(
        default_factory=list, description="List of queries to extract relevant context from outer_context (RAG-like)"
    )
    stage_action: str = Field(..., description="Specific action to perform")
    example_reasoning: str = Field(..., description="Example of expert reasoning")

    def depends_on(self, step_number: int) -> bool:
        """Check if this step depends on a given step number."""
        return step_number in self.dependencies

    def has_dependencies(self) -> bool:
        """Check if this step has any dependencies."""
        return len(self.dependencies) > 0


class ReasoningContext(BaseModel):
    """
    Context object that maintains state during reasoning execution.

    Contains the input data, API object for LLM calls, execution history, and configuration.
    """

    outer_context: str = Field(..., description="Input data as string (it can be CSV or other text information)")
    api: Any = Field(..., description="API object for LLM execution (EntrypointsAccessor or LLMAccessorAPI)")
    entrypoint_key: str = Field(default="default", description="Key for the specific entrypoint to use")
    retry_max: int = Field(default=3, description="Maximum retry attempts")
    history: list[str] = Field(default_factory=list, description="Accumulated reasoning history")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata and state")
    language: Language = Field(default=Language.RUSSIAN, description="Language for reasoning prompts")
    system_prompt: str = Field(default="", description="System prompt to include in each reasoning step")

    # Internal LLM client (created automatically)
    _llm_client: LLMClientBase | None = None

    def model_post_init(self, __context: Any) -> None:
        """Create LLM client after model initialization."""
        from .llm import create_llm_client

        self._llm_client = create_llm_client(self.api, self.entrypoint_key)

    @property
    def llm_client(self) -> LLMClientBase:
        """Get the LLM client (creates it if not already created)."""
        if self._llm_client is None:
            from .llm import create_llm_client

            self._llm_client = create_llm_client(self.api, self.entrypoint_key)
        return self._llm_client

    def add_to_history(self, entry: str) -> None:
        """Add a new entry to the reasoning history."""
        self.history.append(entry)

    def get_current_history(self) -> str:
        """Get the current reasoning history as a single string."""
        return "\n".join(self.history)

    model_config = {"arbitrary_types_allowed": True}


class StepExecutionResult(BaseModel):
    """
    Result of executing a single reasoning step.
    """

    step_number: int = Field(..., description="Number of the executed step")
    step_title: str = Field(..., description="Title of the executed step")
    result: str = Field(..., description="Result content from LLM")
    success: bool = Field(..., description="Whether execution succeeded")
    error_message: str | None = Field(default=None, description="Error message if execution failed")
    execution_time: float | None = Field(default=None, description="Time taken for execution in seconds")
    updated_history: list[str] = Field(default_factory=list, description="History after this step's execution")


class ReasoningResult(BaseModel):
    """
    Final result of executing a complete reasoning chain.
    """

    success: bool = Field(..., description="Whether overall execution succeeded")
    history: list[str] = Field(..., description="Complete reasoning history")
    step_results: list[StepExecutionResult] = Field(..., description="Results from each step")
    total_execution_time: float | None = Field(default=None, description="Total execution time in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")

    def get_full_output(self) -> str:
        """Get the full reasoning output as a single string."""
        return "\n".join(self.history)

    def get_final_output(self) -> str:
        """Get the final reasoning output as a single string without step headers."""
        if not self.history:
            return ""
        last_entry = self.history[-1]
        # Check if it's a step result with header (Russian or English)
        if last_entry.startswith("Шаг ") and "\nРезультат: " in last_entry:
            # Extract content after "Результат: " for Russian steps
            return last_entry.split("\nРезультат: ", 1)[1].strip()
        elif last_entry.startswith("Step ") and "\nResult: " in last_entry:
            # Extract content after "Result: " for English steps
            return last_entry.split("\nResult: ", 1)[1].strip()
        else:
            # Return as-is if it doesn't match expected format
            return last_entry.strip()

    def get_successful_steps(self) -> list[StepExecutionResult]:
        """Get all successfully executed steps."""
        return [step for step in self.step_results if step.success]

    def get_failed_steps(self) -> list[StepExecutionResult]:
        """Get all failed steps."""
        return [step for step in self.step_results if not step.success]


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

    def extract_context_from_queries(self, outer_context: str, queries: list[ContextQuery | str]) -> str:
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
        self, step: StepDescription, outer_context: str = "", language: Language = Language.RUSSIAN
    ) -> str:
        """Format a single step prompt with RAG-like context extraction."""
        # Extract relevant context using step_context_queries
        context_queries_result = self.extract_context_from_queries(outer_context, step.step_context_queries)

        if language == Language.ENGLISH:
            return self.en_step_template.format(
                step_number=step.number,
                step_title=step.title,
                aim=step.aim,
                stage_action=step.stage_action,
                reasoning_questions=step.reasoning_questions,
                context_queries=context_queries_result,
                example_reasoning=step.example_reasoning,
            )
        else:  # Russian
            return self.ru_step_template.format(
                step_number=step.number,
                step_title=step.title,
                aim=step.aim,
                stage_action=step.stage_action,
                reasoning_questions=step.reasoning_questions,
                context_queries=context_queries_result,
                example_reasoning=step.example_reasoning,
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

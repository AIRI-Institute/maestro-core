"""
Search strategies for context extraction in CARL reasoning system.
"""

import warnings
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from .base import SearchStrategy

if TYPE_CHECKING:
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

        Raises:
            ImportError: If vector-search dependencies are not installed
        """
        # Check dependencies early
        try:
            import faiss  # noqa: F401
            import fastembed  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            raise ImportError(
                "Vector search requires additional dependencies. "
                "Install with: pip install 'mmar-carl[vector-search]'"
            ) from None

        self.embedding_model = embedding_model
        self.index_type = index_type
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self._index = None
        self._documents: List[str] = []

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
            warnings.warn(
                "FastEmbed not available. Using fallback character embeddings. "
                "Install 'mmar-carl[vector-search]' for better embeddings.",
                UserWarning,
                stacklevel=3
            )
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
            warnings.warn(
                "FAISS not available. Falling back to substring search. "
                "Install 'mmar-carl[vector-search]' for vector search capabilities.",
                UserWarning,
                stacklevel=2
            )
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

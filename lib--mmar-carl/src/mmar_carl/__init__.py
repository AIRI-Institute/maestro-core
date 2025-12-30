"""
MAESTRO CARL (Collaborative Agent Reasoning Library)

A library for building chain-of-thought reasoning systems with DAG-based parallel execution.
"""

from .chain import ChainBuilder, ReasoningChain
from .executor import DAGExecutor
from .llm import create_llm_client
from .models import (
    ContextQuery,
    ContextSearchConfig,
    Language,
    LLMClientBase,
    PromptTemplate,
    ReasoningContext,
    ReasoningResult,
    StepDescription,
    StepExecutionResult,
    SubstringSearchStrategy,
    VectorSearchStrategy,
)

__version__ = "0.0.11"
__all__ = [
    "Language",
    "StepDescription",
    "ReasoningContext",
    "StepExecutionResult",
    "ReasoningResult",
    "PromptTemplate",
    "ReasoningChain",
    "ChainBuilder",
    "DAGExecutor",
    "LLMClientBase",
    "create_llm_client",
    "ContextQuery",
    "ContextSearchConfig",
    "SubstringSearchStrategy",
    "VectorSearchStrategy",
]

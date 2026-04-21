"""
Optional LangFuse tracing integration for CARL reasoning chains.

Enable tracing by setting ENABLE_LANGFUSE=True in environment variables.
Required variables when enabled:
  LANGFUSE_PUBLIC_KEY - LangFuse project public key
  LANGFUSE_SECRET_KEY - LangFuse project secret key
  LANGFUSE_HOST       - LangFuse server URL (default: https://cloud.langfuse.com)

Example .env:
  ENABLE_LANGFUSE=True
  LANGFUSE_PUBLIC_KEY=pk-lf-...
  LANGFUSE_SECRET_KEY=sk-lf-...
  LANGFUSE_HOST=https://cloud.langfuse.com

Trace hierarchy:
  Trace (chain execution - root, named)
  └── Span (step execution)
      └── Generation (LLM API call, for LLM/StructuredOutput steps)
"""

import os
import warnings
from typing import Any, Optional


class _NoOpSpan:
    """No-op span/trace returned when LangFuse is disabled or unavailable."""

    def __init__(self, id: str = "noop") -> None:
        self.id = id
        self.trace_id = "noop-trace"

    def start_span(self, **kwargs) -> "_NoOpSpan":
        """Create a child span."""
        return _NoOpSpan()

    def start_generation(self, **kwargs) -> "_NoOpSpan":
        """Create a generation span."""
        return _NoOpSpan()

    def start_observation(self, **kwargs) -> "_NoOpSpan":
        """Create an observation (span, generation, etc.)."""
        return _NoOpSpan()

    def score(self, **kwargs) -> "_NoOpSpan":
        """Score a trace or span."""
        return self

    def score_trace(self, **kwargs) -> "_NoOpSpan":
        """Score the trace."""
        return self

    def start_as_current_span(self, **kwargs):
        """Context manager for span."""
        from contextlib import nullcontext

        return nullcontext()

    def start_as_current_generation(self, **kwargs):
        """Context manager for generation."""
        from contextlib import nullcontext

        return nullcontext()

    def start_as_current_observation(self, **kwargs):
        """Context manager for observation."""
        from contextlib import nullcontext

        return nullcontext()

    def end(self, **kwargs) -> None:
        """End the span."""
        pass

    def update(self, **kwargs) -> None:
        """Update span metadata."""
        pass

    def update_trace(self, **kwargs) -> None:
        """Update trace metadata."""
        pass


_langfuse_enabled: Optional[bool] = None
_langfuse_client: Any = None


def is_langfuse_enabled() -> bool:
    """Check if LangFuse tracing is enabled via ENABLE_LANGFUSE environment variable."""
    global _langfuse_enabled
    if _langfuse_enabled is None:
        val = os.environ.get("ENABLE_LANGFUSE", "").strip().lower()
        _langfuse_enabled = val in ("true", "1", "yes")
    return _langfuse_enabled


def _get_client() -> Any:
    """Get or lazily initialize the LangFuse client."""
    global _langfuse_client, _langfuse_enabled
    if _langfuse_client is None and is_langfuse_enabled():
        try:
            from langfuse import Langfuse  # noqa: PLC0415

            _langfuse_client = Langfuse()
        except ImportError:
            _langfuse_enabled = False
            warnings.warn(
                "ENABLE_LANGFUSE=True but 'langfuse' package is not installed. "
                "Install it with: pip install langfuse",
                stacklevel=3,
            )
    return _langfuse_client


def create_chain_trace(
    chain_name: str,
    context_preview: str,
    total_steps: int,
    language: str,
    session_id: Optional[str] = None,
) -> Any:
    """
    Create a LangFuse trace for a chain execution.

    When LangFuse is disabled or unavailable, returns a no-op object so calling
    code does not need to branch on whether tracing is active.

    Args:
        chain_name: Display name for the trace (shown in LangFuse UI)
        context_preview: Input context string (truncated to 500 chars)
        total_steps: Total number of steps in the chain
        language: Language setting for the chain
        session_id: Optional session ID to group multiple traces together

    Returns:
        LangfuseSpan or _NoOpSpan
    """
    client = _get_client()
    if client is None:
        return _NoOpSpan()

    span_kwargs: dict[str, Any] = {
        "name": chain_name,
        "input": {"outer_context": context_preview[:500]},
        "metadata": {
            "total_steps": total_steps,
            "language": language,
            "library": "mmar-carl",
        },
    }

    span = client.start_span(**span_kwargs)

    if session_id:
        span.update_trace(session_id=session_id)

    return span


def flush() -> None:
    """
    Flush pending LangFuse events to the server.

    Call this at the end of scripts/jobs to ensure all traces are sent before
    the process exits. No-op when LangFuse is disabled.
    """
    client = _get_client()
    if client is not None:
        client.flush()

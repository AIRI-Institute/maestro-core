import time
from contextlib import contextmanager
from typing import Any, Callable

import tiktoken
from langfuse import Langfuse as LangfuseClient
from loguru import logger
from mmar_mapi.services import (
    LCP,
    LLMCallProps,
    LLMHubAPI,
    LLMHubMetadata,
    LLMPayload,
    LLMRequest,
    LLMResponseExt,
    Message,
)
from mmar_ptag.ptag_framework import TRACE_ID_VAR

from llm_hub_monitoring.config import LangfuseConfig

LLMHubMetadata.EMPTY = LLMHubMetadata(endpoints=[], default_endpoint_key="")  # type: ignore[attr-defined]


def request_to_str(request: LLMRequest) -> str:
    """Convert request to string representation."""
    if isinstance(request, str):
        return request
    elif isinstance(request, list) and request and isinstance(request[0], Message):
        return "\n".join(f"{msg.role}: {msg.content}" for msg in request)
    elif isinstance(request, LLMPayload):
        return "\n".join(f"{msg.role}: {msg.content}" for msg in request.messages)
    return str(request)


def get_trace_url(host: str, trace_id: str) -> str:
    """Get the trace URL from host and trace_id."""
    return host.replace("/api/public/ingestion", "").replace("/api/public", "") + f"/trace/{trace_id}"


@contextmanager
def trace_span(langfuse: LangfuseClient, name: str, **kwargs):
    """Context manager for creating and managing a Langfuse span."""
    span = langfuse.start_span(name=name, **kwargs)
    logger.trace(f"Langfuse span created: {span.trace_id}")
    start_time = time.time()

    try:
        yield span, start_time
    except Exception as e:
        latency = time.time() - start_time
        span.update(
            level="ERROR",
            status_message=str(e),
            metadata={"error_type": type(e).__name__, "latency_seconds": latency},
        )
        span.end()
        raise


def log_trace_success(host: str, trace_id: str) -> None:
    """Log successful trace send with URL."""
    logger.trace(f"Langfuse trace sent: {get_trace_url(host, trace_id)}")


def log_trace_error(host: str, trace_id: str) -> None:
    """Log error trace send with URL."""
    logger.error(f"Langfuse error trace sent: {get_trace_url(host, trace_id)}")


def with_trace_id(metadata: dict | None) -> dict:
    """Add external_id to metadata if trace_id exists in context."""
    if metadata is None:
        metadata = {}
    trace_id = TRACE_ID_VAR.get()
    if trace_id:
        metadata = {**metadata, "external_id": trace_id}
    else:
        logger.warning("Not found trace_id...")
    return metadata


# todo introduce separate llm-hub-facade?
class LLMHubMonitoring(LLMHubAPI):
    def __init__(self, llm_hub: LLMHubAPI, config: LangfuseConfig):
        self.llm_hub = llm_hub
        self.config = config
        self.langfuse: LangfuseClient | None = None
        self.tokenizer: tiktoken.Encoding | None = None

        if config.enabled and config.public_key and config.secret_key:
            try:
                self.langfuse = LangfuseClient(
                    public_key=config.public_key,
                    secret_key=config.secret_key,
                    host=config.host,
                )
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info(f"Langfuse initialized: host={config.host}")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse: {e}. Monitoring disabled.")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken."""
        if not self.tokenizer or not text:
            return 0
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return 0

    def _get_request_metadata(self, request: LLMRequest, props: LLMCallProps) -> dict:
        """Extract metadata from request for tracing."""
        request_str = request_to_str(request)
        return {
            "endpoint_key": props.endpoint_key,
            "attempts": props.attempts,
            "request_length": len(request_str),
        }

    def get_metadata(self) -> LLMHubMetadata:
        logger.info("#get_metadata, forwarding")
        return self.llm_hub.get_metadata()

    def get_response(self, *, request: LLMRequest, props: LLMCallProps = LCP) -> str:
        logger.debug("#get_response, forwarding")
        if not self.langfuse:
            return self.llm_hub.get_response(request=request, props=props)

        request_str = request_to_str(request)
        input_tokens = self._estimate_tokens(request_str)

        metadata = with_trace_id({
            **self._get_request_metadata(request, props),
            **({"input_length": len(request_str)} if not self.config.capture_content else {}),
        })

        with trace_span(
            self.langfuse,
            "get_response",
            input=(request_str if self.config.capture_content else None),
            metadata=metadata,
        ) as (span, start_time):
            response = self.llm_hub.get_response(request=request, props=props)
            latency = time.time() - start_time
            output_tokens = self._estimate_tokens(response)

            span.update(
                output=(response if self.config.capture_content else None),
                metadata=(
                    {
                        "output_length": len(response),
                        "latency_seconds": latency,
                    }
                    if not self.config.capture_content
                    else {"latency_seconds": latency}
                ),
                usage_details={
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                },
            )
            span.end()
            log_trace_success(self.config.host, span.trace_id)
            return response

    def get_response_ext(self, *, request: LLMRequest, props: LLMCallProps = LCP) -> LLMResponseExt:
        logger.info("#get_response_ext, forwarding")
        if not self.langfuse:
            return self.llm_hub.get_response_ext(request=request, props=props)

        request_str = request_to_str(request)
        input_tokens = self._estimate_tokens(request_str)

        metadata = with_trace_id({
            **self._get_request_metadata(request, props),
            **({"input_length": len(request_str)} if not self.config.capture_content else {}),
        })

        with trace_span(
            self.langfuse,
            "get_response_ext",
            input=(request_str if self.config.capture_content else None),
            metadata=metadata,
        ) as (span, start_time):
            response = self.llm_hub.get_response_ext(request=request, props=props)
            latency = time.time() - start_time
            output_tokens = self._estimate_tokens(response.text)

            span_metadata: dict[str, str | int | float] = {"latency_seconds": latency}
            if response.resource_id:
                span_metadata["resource_id"] = response.resource_id

            if not self.config.capture_content:
                span_metadata["output_length"] = len(response.text)

            span.update(
                output=(response.text if self.config.capture_content else None),
                metadata=span_metadata,
                usage_details={
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                },
            )
            span.end()
            log_trace_success(self.config.host, span.trace_id)
            return response

    def get_embedding(self, *, prompt: str, props: LLMCallProps = LCP) -> list[float] | None:
        logger.info("#get_embedding, forwarding")
        if not self.langfuse:
            return self.llm_hub.get_embedding(prompt=prompt, props=props)

        metadata = with_trace_id({"prompt_length": len(prompt)} if not self.config.capture_content else None)

        with trace_span(
            self.langfuse,
            "get_embedding",
            input=(prompt if self.config.capture_content else None),
            metadata=metadata,
        ) as (span, start_time):
            embedding = self.llm_hub.get_embedding(prompt=prompt, props=props)
            latency = time.time() - start_time
            span.update(metadata={"latency_seconds": latency, "embedding_dim": len(embedding) if embedding else 0})
            span.end()
            log_trace_success(self.config.host, span.trace_id)
            return embedding

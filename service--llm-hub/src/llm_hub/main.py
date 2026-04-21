import sys

import uvicorn
from dishka import make_container
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from llm_hub.config import Config
from llm_hub.config_server import ConfigServer
from llm_hub.ioc import IOCLocal
from llm_hub.llm_hub import LLMHub
from llm_hub.utils_fastapi import extract_forwardable_headers
from mmar_mimpl import init_logger

def create_app(client: LLMHub) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        client: LLM Hub OpenAI client instance.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="LLM Hub",
        description="Unified LLM proxy service supporting OpenAI-compatible and Gigachat APIs",
        version="1.0.0",
    )

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "service": "llm-hub"}

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return client.models.list()

    @app.api_route("/v1/chat/completions", methods=["POST"])
    async def chat_completions(request: Request):
        """Handle chat completion requests."""
        data = await request.json()
        headers = extract_forwardable_headers(request)
        return await client.chat.completions.create(headers=headers, **data)

    @app.api_route("/v1/embeddings", methods=["POST"])
    async def embeddings(request: Request):
        """Handle embeddings requests."""
        data = await request.json()
        headers = extract_forwardable_headers(request)
        return await client.embeddings.create(headers=headers, **data)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "internal_error"}},
        )

    return app


def main() -> int:
    """Run the unified LLM proxy service."""
    load_dotenv()

    # Create IoC container
    container = make_container(IOCLocal())
    config = container.get(Config)
    config_server = container.get(ConfigServer)
    init_logger(config_server.logger.level)
    client = container.get(LLMHub)
    host, port = config_server.host, config_server.port
    
    logger.info(f"Config: {config!r}")
    logger.info(f"Server config: {config_server!r}")
    logger.info(f"Starting LLM Hub OpenAI on {host}:{port}")

    app = create_app(client)
    try:
        uvicorn.run(app, host=host, port=port, factory=False, access_log=True)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        return 0
    except Exception as e:
        logger.exception(f"Service error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

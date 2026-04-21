#!/usr/bin/env python3
"""Server with specific API - expects config as STR."""
from types import SimpleNamespace
from typing import Optional
from pydantic import BaseModel

from mmar_ptag import deploy_server


class Chat(BaseModel):
    model: str
    messages: list[dict]


class ServiceInterface:
    """Interface defining the server API."""
    def interpret(
        self,
        *,
        file_path: str,
        config: str,
        model: str,
        chat: Optional[Chat] = None,
        trace_id: str = ""
    ) -> dict:
        raise NotImplementedError


class Service(ServiceInterface):
    """Service that processes files."""

    def interpret(
        self,
        *,
        file_path: str,
        config: str,  # <-- SERVER EXPECTS STRING
        model: str,
        chat: Optional[Chat] = None,
        trace_id: str = ""
    ) -> dict:
        """Interpret a file with the given configuration."""
        return {
            "file": file_path,
            "config": config,
            "model": model,
            "chat_model": chat.model if chat else None,
            "result": "processed"
        }


if __name__ == "__main__":
    deploy_server(
        config_server=SimpleNamespace(port=50051, max_workers=10),
        service=Service()
    )

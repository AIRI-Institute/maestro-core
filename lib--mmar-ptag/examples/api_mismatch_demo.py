#!/usr/bin/env python3
"""
Demonstration of API mismatch error.

Run this script to see what happens when client and server
have different type definitions for the same RPC method.
"""

import json
from types import SimpleNamespace
from typing import Optional
from pydantic import BaseModel
from unittest.mock import MagicMock

from mmar_ptag.ptag_framework import WrappedPTAGService
from mmar_ptag.ptag_pb2 import PTAGRequest


class Chat(BaseModel):
    model: str
    messages: list[dict]


# Interface - defines the expected types
class ServiceInterface:
    def process(
        self,
        *,
        file_path: str,
        config: str,  # Interface expects STRING
        model: str,
        chat: Optional[Chat] = None,
        trace_id: str = ""
    ) -> dict:
        raise NotImplementedError


# Server implementation - follows the interface
class ServerService(ServiceInterface):
    def process(
        self,
        *,
        file_path: str,
        config: str,
        model: str,
        chat: Optional[Chat] = None,
        trace_id: str = ""
    ) -> dict:
        return {"status": "ok"}


def main():
    print("=" * 60)
    print("API Mismatch Error Demonstration")
    print("=" * 60)
    print()

    # Create server service
    service = ServerService()
    wrapped_service = WrappedPTAGService(service)

    # Mock gRPC context
    fake_context = MagicMock()
    fake_context.invocation_metadata.return_value = {}
    fake_context.set_code = MagicMock()
    fake_context.set_details = MagicMock()

    print("Server expects: config: str")
    print("Client sends:  config: dict (WRONG!)")
    print()

    # Create a request with DICT instead of STRING
    # This simulates what happens when client has wrong type definition
    args_payload = json.dumps([
        "/path/to/file.json",
        {"model": "extractor", "temperature": 0.5},  # DICT - but server expects STR!
        "giga-model",
        None,
        ""
    ])

    request = PTAGRequest(
        FunctionName="process",
        Payload=args_payload.encode()
    )

    print("Sending request with mismatched types...")
    print()

    # Try to invoke - this will fail with validation error
    response = wrapped_service.Invoke(request, fake_context)

    # Check what error was set
    print("-" * 60)
    print("ERROR DETAILS:")
    print("-" * 60)
    print(f"Status Code: {fake_context.set_code.call_args[0][0]}")
    print()
    print("Error Details:")
    print(fake_context.set_details.call_args[0][0])
    print("-" * 60)
    print()

    print("SUMMARY:")
    print("  - Expected type: str (string)")
    print("  - Received type: dict (dictionary)")
    print("  - The error clearly shows what went wrong!")
    print()


if __name__ == "__main__":
    main()

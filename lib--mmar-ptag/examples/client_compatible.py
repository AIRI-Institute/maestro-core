#!/usr/bin/env python3
"""Client with COMPATIBLE API - sends config as STR."""
import sys
import time

from mmar_ptag import ptag_client


# Client interface - sends config as STR (compatible!)
class ServiceClient:
    """Client interface - CORRECT: expects str for config."""

    def interpret(
        self,
        *,
        file_path: str,
        config: str,  # <-- CLIENT SENDS STRING (CORRECT!)
        model: str,
        trace_id: str = ""
    ) -> dict:
        """Interpret a file with the given configuration."""
        ...


def main():
    # Wait for server to be ready
    print("Waiting for server...", flush=True)
    time.sleep(2)

    print("Creating client with COMPATIBLE API (config: str)...", flush=True)

    client = ptag_client(ServiceClient, "localhost:50051")

    print("Sending request with string config...", flush=True)

    try:
        # This will SUCCEED because types match
        result = client.interpret(
            file_path="/mnt/data/file.json",
            config='{"model": "extractor"}',  # STRING!
            model="giga-max"
        )
        print(f"\n{'='*60}", flush=True)
        print("SUCCESS - Types match!", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Result: {result}\n", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Client with INCOMPATIBLE API - sends config as DICT."""
import sys
import time

from mmar_ptag import ptag_client


# Client interface - sends config as DICT (incompatible!)
class ServiceClient:
    """Client interface - WRONG: expects dict for config."""

    def interpret(
        self,
        *,
        file_path: str,
        config: dict,  # <-- CLIENT SENDS DICT (WRONG!)
        model: str,
        trace_id: str = ""
    ) -> dict:
        """Interpret a file with the given configuration."""
        ...


def main():
    # Wait for server to be ready
    print("Waiting for server...", flush=True)
    time.sleep(2)

    print("Creating client with INCOMPATIBLE API (config: dict)...", flush=True)

    # Create client - this will succeed because Python doesn't check types at creation
    client = ptag_client(ServiceClient, "localhost:50051")

    print("Sending request with dict config...", flush=True)

    try:
        # This will FAIL because server expects str but we send dict
        result = client.interpret(
            file_path="/mnt/data/file.json",
            config={"model": "extractor", "provider": "giga"},  # DICT!
            model="giga-max"
        )
        print(f"Unexpected success: {result}", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n{'='*60}", flush=True)
        print("API MISMATCH ERROR:", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"{e}\n", flush=True)
        print("Expected: config: str", flush=True)
        print("Sent:     config: dict", flush=True)
        print(f"{'='*60}\n", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()

"""Code snippets for README.md documentation.

These are executable examples that demonstrate the library's usage.
They are tested in tests/test_snippets.py to ensure they stay up-to-date.

IMPORTANT: These snippets must match the Python code blocks in README.md exactly.
Run 'make check-readme' to verify they match.
"""

# -----------------------------------------------------------------------------
# Snippet: define_service (Quick Start - 1. Define the Service)
# -----------------------------------------------------------------------------

define_service = [
    "from types import SimpleNamespace",
    "from mmar_ptag import deploy_server",
    "",
    "class Greeter:",
    "    def say_hello(self, *, name: str, count: int = 1, trace_id: str = \"\") -> dict:",
    "        return {\"message\": f\"Hello, {name} x{count}\"}",
]

# -----------------------------------------------------------------------------
# Snippet: run_server (Quick Start - 2. Run the Server)
# -----------------------------------------------------------------------------

run_server = [
    "deploy_server(",
    "    config_server=SimpleNamespace(port=50051, max_workers=10),",
    "    service=Greeter()",
    ")",
]

# -----------------------------------------------------------------------------
# Snippet: create_client (Quick Start - 3. Create a Client)
# -----------------------------------------------------------------------------

create_client = [
    "from mmar_ptag import ptag_client",
    "",
    "class Greeter:",
    "    def say_hello(self, *, name: str, count: int = 1) -> dict:",
    "        ...",
    "",
    "client = ptag_client(Greeter, \"localhost:50051\")",
    "result = client.say_hello(name=\"World\", count=3)",
    "print(result)  # {'message': 'Hello, World x3'}",
]

# -----------------------------------------------------------------------------
# Snippet: trace_id (Trace ID Support)
# -----------------------------------------------------------------------------

trace_id = [
    "from mmar_ptag import ptag_client",
    "",
    "class UserService:",
    "    def get_user(self, *, user_id: int, trace_id: str = \"\") -> dict:",
    "        ...",
    "",
    "client = ptag_client(UserService, \"localhost:50051\")",
    "",
    "# Trace ID is automatically propagated through the call chain",
    "result = client.get_user(user_id=123, trace_id=\"request-abc-123\")",
]

# -----------------------------------------------------------------------------
# All snippets dictionary for validation
# -----------------------------------------------------------------------------

ALL_SNIPPETS = {
    "define_service": define_service,
    "run_server": run_server,
    "create_client": create_client,
    "trace_id": trace_id,
}

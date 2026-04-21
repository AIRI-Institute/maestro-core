#!/usr/bin/env python3
"""LLM Hub CLI - Command-line interface for testing LLM Hub.

Supports all providers: OpenAI, Anthropic, GigaChat, and others.

Usage:
    # Remote mode (via HTTP):
    python tools/llm_cli.py models

    # Direct mode (local config):
    python tools/llm_cli.py --env=llm-config.toml models

    # Specify env file:
    python tools/llm_cli.py --env=.env models

Examples:
    python tools/llm_cli.py models              # List available models
    python tools/llm_cli.py hello               # Test all models
    python tools/llm_cli.py hello --f=claude    # Test models containing 'claude'
    python tools/llm_cli.py structured          # Test structured output
    python tools/llm_cli.py tools               # Test tool calling
    python tools/llm_cli.py embeddings          # Test embeddings
    python tools/llm_cli.py image path.jpg      # Test image understanding
    python tools/llm_cli.py parallel --n=10     # Test 10 parallel requests
"""

import asyncio
import base64
import inspect
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import fire
from dotenv import load_dotenv
from openai import APIConnectionError, OpenAI

# Tool definitions for testing
get_weather = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
            },
            "required": ["location"],
        },
    },
}
weather_tools = [get_weather]


class LLMCli:
    """LLM Hub command-line interface."""

    def __init__(self, env: str = ""):
        """Initialize LLM CLI.

        Args:
            env: Path to environment file (.env) or config file (.toml)
                 If empty, uses environment variables or defaults
        """
        self.env_path = env
        self.client = None
        self._direct_mode = False
        self._setup_client()

    def _setup_client(self):
        """Set up the OpenAI-compatible client based on configuration."""
        # Load environment file if provided
        if self.env_path:
            env_file = Path(self.env_path)
            if env_file.exists():
                # Check file suffix or name
                if env_file.suffix == '.env' or env_file.name.endswith('.env'):
                    load_dotenv(env_file)
                    print(f"Loaded environment from: {env_file}")
                elif env_file.suffix == '.toml':
                    # Direct mode: use config file
                    self._direct_mode = True
                    os.environ["LLM_CONFIG_PATH"] = str(env_file)
                    print(f"Direct mode: {env_file}")
                else:
                    warnings.warn(f"Unknown file type: {env_file}. Expected .env or .toml")
            else:
                warnings.warn(f"File not found: {env_file}")

        # Get configuration from environment
        base_url = os.environ.get("LLM_HUB_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        api_key = os.environ.get("LLM_HUB_API_KEY") or os.environ.get("OPENAI_API_KEY", "dummy")

        # Check if base_url is a path to a config file
        if base_url and Path(base_url).exists() and base_url.endswith('.toml'):
            self._direct_mode = True
            os.environ["LLM_CONFIG_PATH"] = base_url

        if self._direct_mode:
            # Direct mode: Create LLMHub instance
            from dishka import make_container

            from llm_hub.ioc import IOCLocal
            from llm_hub.llm_hub import LLMHub

            container = make_container(IOCLocal())
            self.client = container.get(LLMHub)
            print(f"Created LLMHub instance from config")
        else:
            # Remote mode: Create OpenAI client
            if not base_url:
                base_url = "http://localhost:40631/v1"
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            print(f"Remote mode: {base_url}")

    @staticmethod
    def _await_if_needed(result):
        """Await result if it's a coroutine (for direct mode compatibility)."""
        if inspect.iscoroutine(result):
            return asyncio.run(result)
        return result

    @staticmethod
    def _parse_response(response):
        """Parse response from LLMHub or OpenAI client.

        LLMHub returns FastAPI Response or dict, OpenAI client returns Pydantic models.
        Convert to a dict-like object with .choices attribute.
        """
        # If it's a FastAPI Response, parse JSON
        if hasattr(response, 'body'):
            import json
            body = response.body.decode('utf-8') if isinstance(response.body, bytes) else response.body
            data = json.loads(body)
            # Parse choices properly
            choices_data = data.get('choices', [])
            parsed_choices = []
            for choice in choices_data:
                if isinstance(choice, dict):
                    # Parse message within choice
                    message = choice.get('message', {})
                    parsed_message = SimpleNamespace(**message) if isinstance(message, dict) else message
                    parsed_choice = SimpleNamespace(**{**choice, 'message': parsed_message})
                    parsed_choices.append(parsed_choice)
                else:
                    parsed_choices.append(choice)
            # Remove choices from data and add parsed_choices
            result_data = {k: v for k, v in data.items() if k != 'choices'}
            return SimpleNamespace(**result_data, choices=parsed_choices)

        # If it's already a dict, convert to namespace
        if isinstance(response, dict):
            choices_data = response.get('choices', [])
            parsed_choices = []
            for choice in choices_data:
                if isinstance(choice, dict):
                    message = choice.get('message', {})
                    parsed_message = SimpleNamespace(**message) if isinstance(message, dict) else message
                    parsed_choice = SimpleNamespace(**{**choice, 'message': parsed_message})
                    parsed_choices.append(parsed_choice)
                else:
                    parsed_choices.append(choice)
            result_data = {k: v for k, v in response.items() if k != 'choices'}
            return SimpleNamespace(**result_data, choices=parsed_choices)

        # OpenAI client response - return as is
        return response

    def _filtered_models(self, filter_str: str) -> list:
        """Get list of models, optionally filtered by string."""
        try:
            models = self.client.models.list()
        except APIConnectionError as ex:
            warnings.warn(f"Failed to access models: {ex}")
            return []

        # Handle dict response (LLMHub direct mode)
        if isinstance(models, dict):
            models_data = models.get("data", [])
            model_list = [SimpleNamespace(id=m.get("id", m) if isinstance(m, dict) else m) for m in models_data]
        else:
            # Handle Pydantic response (OpenAI client remote mode)
            model_list = list(models)

        if filter_str:
            model_list = [m for m in model_list if filter_str in m.id]
        return model_list

    def models(self, raw: bool = False, join: bool = False):
        """List available models from LLM Hub.

        Args:
            raw: Show raw model objects instead of formatted list
            join: Join models on one line with commas instead of newlines
        """
        try:
            response = self.client.models.list()
        except APIConnectionError as ex:
            print(f"ERROR: {ex}")
            return

        # Handle both OpenAI client and LLMHub responses
        if isinstance(response, dict):
            models_data = response.get("data", [])
        else:
            models_data = response.model_dump().get("data", [])

        if raw:
            for model in models_data:
                print(model)
        else:
            model_lines = []
            for model in models_data:
                if isinstance(model, dict):
                    model_id = model.get("id", "")
                    default_mark = " *" if model.get("default") else ""
                    caption = model.get("caption", "")
                    owned_by = model.get("owned_by", "")
                else:
                    model_id = model.id
                    default_mark = " *" if getattr(model, "default", False) else ""
                    caption = getattr(model, "caption", "")
                    owned_by = getattr(model, "owned_by", "")

                parts = [model_id + default_mark]
                if owned_by:
                    parts.append(f"[{owned_by}]")
                if caption:
                    parts.append(f"- {caption}")

                model_lines.append(" ".join(parts))

            separator = ", " if join else "\n"
            print(separator.join(model_lines))
            print(f"\nTotal: {len(model_lines)} model(s)")
            print("(* = default model)")

    def hello(self, text: str = "", f: str = ""):
        """Test basic chat completion with all or filtered models.

        Args:
            text: Custom message to send
            f: Filter string to match in model ID
        """
        text = text or "Hello, what is your name?"
        messages = [{"role": "user", "content": text}]
        model_list = self._filtered_models(f)

        if not model_list:
            print(f"No models found{' matching filter: ' + f if f else ''}")
            return

        print(f"Testing {len(model_list)} model(s)...\n")

        for model in model_list:
            try:
                response = self.client.chat.completions.create(model=model.id, messages=messages)
                response = self._await_if_needed(response)
                response = self._parse_response(response)
                message = response.choices[0].message
                content = message.content or getattr(message, 'reasoning', None)
                if content:
                    content = content.strip()
                    print(f"✓ {model.id}: {content}")
                else:
                    print(f"✗ {model.id}: Empty response")
            except Exception as e:
                print(f"✗ {model.id}: ERROR: {e}")

    def structured(self, f: str = ""):
        """Test structured output (JSON schema) with models.

        Args:
            f: Filter string to match in model ID
        """
        model_list = self._filtered_models(f)
        if not model_list:
            print(f"No models found{' matching filter: ' + f if f else ''}")
            return

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "language": {"type": "string"},
                "capabilities": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "language"],
            "additionalProperties": False,
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "model_info", "schema": schema},
        }
        text = "Provide your model name, primary language, and main capabilities."
        messages = [{"role": "user", "content": text}]

        for model in model_list:
            print(f"\n=== {model.id} (structured output) ===")
            try:
                response = self.client.chat.completions.create(
                    model=model.id,
                    messages=messages,
                    response_format=response_format,
                )
                response = self._await_if_needed(response)
                response = self._parse_response(response)
                content_raw = response.choices[0].message.content
                content = json.loads(content_raw)
                print(json.dumps(content, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"ERROR: {e}")

    def tools(self, f: str = ""):
        """Test tool/function calling with models.

        Args:
            f: Filter string to match in model ID
        """
        model_list = self._filtered_models(f)
        if not model_list:
            print(f"No models found{' matching filter: ' + f if f else ''}")
            return

        messages = [{"role": "user", "content": "What's the weather in Paris?"}]

        for model in model_list:
            print(f"\n=== {model.id} (tool calling) ===")
            try:
                response = self.client.chat.completions.create(
                    model=model.id,
                    messages=messages,
                    tools=weather_tools,
                )
                response = self._await_if_needed(response)
                response = self._parse_response(response)
                msg = response.choices[0].message
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  ✓ Tool: {tc.function.name}")
                        print(f"    Args: {tc.function.arguments}")
                else:
                    print(f"  ✗ No tool call, response: {msg.content}")
            except Exception as e:
                print(f"ERROR: {e}")

    def embeddings(self, f: str = ""):
        """Test embedding generation with models.

        Args:
            f: Filter string to match in model ID (or exact model name if no matches)
        """
        model_list = self._filtered_models(f)
        if f and not model_list:
            model_list = [SimpleNamespace(id=f)]

        if not model_list:
            print("No embedding models found. Specify with --f=model_name")
            return

        text = "Hello, world!"

        for model in model_list:
            print(f"\n=== {model.id} (embeddings) ===")
            try:
                response = self.client.embeddings.create(model=model.id, input=text)
                response = self._await_if_needed(response)
                response = self._parse_response(response)
                embedding = response.data[0]
                print(f"  ✓ Dimension: {len(embedding.embedding)}")
                print(f"    First 5 values: {embedding.embedding[:5]}")
            except Exception as e:
                print(f"  ✗ ERROR: {e}")

    def image(self, path: str, f: str = ""):
        """Test image understanding with models.

        Args:
            path: Path to image file
            f: Filter string to match in model ID
        """
        model_list = self._filtered_models(f)
        if not model_list:
            print(f"No models found{' matching filter: ' + f if f else ''}")
            return

        image_path = Path(path)
        if not image_path.exists():
            print(f"ERROR: Image file not found: {path}")
            return

        image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            }
        ]

        for model in model_list:
            print(f"\n=== {model.id} (image understanding) ===")
            try:
                response = self.client.chat.completions.create(
                    model=model.id,
                    messages=messages,
                )
                response = self._await_if_needed(response)
                content = response.choices[0].message.content.strip()
                print(f"  {content}")
            except Exception as e:
                print(f"ERROR: {e}")

    def info(self):
        """Show LLM Hub connection information."""
        print("LLM Hub CLI Configuration")
        print("=" * 60)

        if self._direct_mode:
            print(f"Mode:      Direct (local config)")
            config_path = os.environ.get("LLM_CONFIG_PATH", "N/A")
            print(f"Config:    {config_path}")
            print("\nDirect mode: LLMHub instance loaded from TOML config")
            print("No HTTP connection required")
        else:
            base_url = os.environ.get("LLM_HUB_BASE_URL") or os.environ.get("OPENAI_API_BASE", "http://localhost:40631/v1")
            api_key = os.environ.get("LLM_HUB_API_KEY") or os.environ.get("OPENAI_API_KEY", "dummy")
            print(f"Mode:      Remote (HTTP)")
            print(f"Base URL:  {base_url}")
            print(f"API Key:   {api_key[:10]}... (masked)")

        print()
        print("Environment variables:")
        print("  LLM_HUB_BASE_URL  - Base URL for LLM Hub OR path to llm-config.toml")
        print("  LLM_HUB_API_KEY   - API key for LLM Hub (remote mode only)")
        print()
        print("Legacy (still supported):")
        print("  OPENAI_API_BASE")
        print("  OPENAI_API_KEY")

    def parallel(self, n: int = 10, f: str = ""):
        """Test parallel requests to a model.

        Args:
            n: Number of parallel requests to make
            f: Filter string to match in model ID
        """
        model_list = self._filtered_models(f)
        if not model_list:
            print(f"No models found{' matching filter: ' + f if f else ''}")
            return

        if len(model_list) > 1:
            print(f"Multiple models found, testing first one: {model_list[0].id}")

        model = model_list[0]
        print(f"Testing {n} parallel requests to model: {model.id}")

        def extract_content(response) -> bool:
            """Extract and validate content from response."""
            try:
                response = self._parse_response(response)
                message = response.choices[0].message
                content = message.content or getattr(message, 'reasoning', None)
                return bool(content)
            except Exception:
                return False

        if self._direct_mode:
            # Direct mode: use asyncio.gather for async LLMHub
            async def make_request(i: int) -> bool:
                try:
                    messages = [{"role": "user", "content": f"What is {i} * {i}?"}]
                    response = await self.client.chat.completions.create(model=model.id, messages=messages)
                    return extract_content(response)
                except Exception:
                    return False

            async def run_parallel():
                return await asyncio.gather(*[make_request(i) for i in range(n)])

            start = time.time()
            results = asyncio.run(run_parallel())
            elapsed = time.time() - start
        else:
            # Remote mode: use ThreadPoolExecutor for sync OpenAI client
            def make_request(i: int) -> bool:
                try:
                    messages = [{"role": "user", "content": f"What is {i} * {i}?"}]
                    response = self.client.chat.completions.create(model=model.id, messages=messages)
                    return extract_content(response)
                except Exception:
                    return False

            start = time.time()
            with ThreadPoolExecutor(max_workers=n) as executor:
                results = list(executor.map(make_request, range(n)))
            elapsed = time.time() - start

        total_ok = sum(results)
        print(f"Model: {model.id}")
        print(f"Success rate: {total_ok}/{n}")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Average time per request: {elapsed/n:.2f} seconds")


if __name__ == "__main__":
    fire.Fire(LLMCli)

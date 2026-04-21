# LLM Hub OpenAI

Unified LLM proxy service that provides a single OpenAI-compatible API for multiple LLM providers.

## Responsible
tagin

## Overview

LLM Hub OpenAI is a FastAPI-based service that acts as a unified proxy for various LLM providers. It presents an OpenAI-compatible API, allowing clients to use different providers (OpenAI, GigaChat, DeepSeek, etc.) through a single endpoint with consistent request/response formats.

## Features

- **OpenAI-compatible API** - Drop-in replacement for OpenAI clients
- **Multiple provider support** - OpenAI, GigaChat (personal/corporate), DeepSeek, and any OpenAI-compatible provider
- **Model routing** - Map custom model names to actual provider models
- **Chat completions** - Full support for chat completion API
- **Embeddings** - Full support for embeddings API
- **Model listing** - Dynamic model listing with metadata
- **Health checks** - `/health` endpoint for service monitoring
- **Configuration-based** - TOML configuration for easy setup

## Installation

```bash
# Install dependencies
pip install -e .
```

## Configuration

The service uses a TOML configuration file. Copy `config.toml.example` to `config.toml` and adjust for your environment:

```bash
cp config.toml.example config.toml
```

### Configuration Structure

```toml
# Server configuration
host = "0.0.0.0"
port = 40631

# Connection definitions
[connections.openai]
api_type = "openai"
api_base = "https://api.openai.com/v1"
api_key = "your-api-key"

[connections.gigachat]
api_type = "gigachat"
api_base = "https://gigachat.devices.sberbank.ru/api/v1"
scope = "GIGACHAT_API_PERS"

# Model routing: model_id -> "connection.target_model"
[routing]
"gpt-4o" = "openai.gpt-4o"
"deepseek-chat" = "deepseek.DeepChat-V3"
"gigachat" = "gigachat.GigaChat"

# Model metadata
[model_info.gpt-4o]
owned_by = "openai"
default = true
```

### Connection Types

**OpenAI-compatible:**
- `api_type`: "openai"
- `api_base`: API base URL
- `api_key`: API key

**GigaChat:**
- `api_type`: "gigachat"
- `scope`: "GIGACHAT_API_PERS" or "GIGACHAT_API_CORP"
- `api_base`: API base URL (optional, has default)
- `auth_url`: OAuth authorization URL (for corporate)
- `access_token`: Direct access token (optional)
- `authorization`: OAuth authorization code (optional)
- `user`/`password`: Client credentials (optional)
- `verify_ssl`: SSL verification (default: true)

**Model metadata options:**
- `caption`: Human-readable model description
- `owned_by`: Provider/owner identifier
- `max_concurrent`: Maximum concurrent requests for this model
- `default`: Mark as default chat model
- `default_embeddings`: Mark as default embedding model

## Usage

### Running the service

```bash
python -m llm_hub.main
```

The service will start on the configured host/port (default: `0.0.0.0:40631`).

### API Endpoints

**Health Check:**
```bash
GET /health
```

**List Models:**
```bash
GET /v1/models
```

**Chat Completions:**
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "gpt-4o",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7
}
```

**Embeddings:**
```bash
POST /v1/embeddings
Content-Type: application/json

{
  "model": "text-embedding-3-small",
  "input": "Hello world"
}
```

### Using with OpenAI Client

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:40631/v1",
    api_key="any-key",  # Not used by proxy
)

# Chat completion
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Embeddings
response = await client.embeddings.create(
    model="text-embedding-3-small",
    input="Hello world"
)

# List models
models = await client.models.list()
```

## Development

### Running tests

```bash
pytest
```

### Code style

The project uses Ruff for linting and formatting:
```bash
ruff check .
ruff format .
```

## License

Internal project - Maestro Core services

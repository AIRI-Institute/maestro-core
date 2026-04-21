# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAESTRO Core is a microservice-based conversational AI framework demonstrating the MAESTRO architecture. It consists of independently deployable services orchestrated via Docker Compose, communicating through a shared API protocol (mmar-* packages).

**Key Technologies:** Python 3.12+, FastAPI, Docker Compose, uv (package manager)

## Development Commands

### Environment Setup
```bash
make setup-env     # Create data directory and copy default configs
make build         # Build all Docker images
make up            # Start all services in detached mode
```

### Running Examples
```bash
make run-dummy     records='dummy=hello dummy=test dummy=exit'   # Basic test track
make run-chatbot   records='start="Какая ты языковая модель?"'    # LLM-powered chat
make run-describer # Describer example
make run-wizard    # LLM configuration wizard
```

### Service Management
```bash
make logs                    # Follow all service logs
make logs-llm-hub            # Follow llm-hub logs specifically
make ps                      # Show service status
make stop                    # Stop all services
```

### Document Processing (Experimental)
```bash
make document-extractor-up       # Start document extractor (CPU)
make document-extractor-up-on-gpu # Start document extractor (GPU, requires CUDA >= 12.4)
make run-document-describer      # Run document describer example
```

### LLM Configuration
```bash
# Option 1: Manual
cp llm_config.json.example llm_config.json
# Edit llm_config.json to fill ??? placeholders with API keys
make update-llm-config
make restart-llm-hub

# Option 2: Wizard
make run-wizard  # Follow prompts, then copy resulting llm_config.json to ./data
make update-llm-config
make restart-llm-hub
```

### Testing via CLI
The main CLI (`maestro_client_cli.py`) supports running tracks:
```bash
uv run --prerelease=allow --refresh-package=mmar-llm --refresh-package=mmar-mapi --refresh-package=mmar-mcli python maestro_client_cli.py track <TrackName> <records...>
```

Records format: `expected_state=message` (e.g., `dummy=hello`)

## Architecture

### Service Flow
```
Client Request → Gateway (port 7732) → Chat Manager → Domain Services (LLM Hub, Text Extractor, etc.)
```

### Core Services

| Service | Port | Purpose |
|---------|------|---------|
| `service--gateway` | 7732 | FastAPI entry point, routing, chat storage |
| `service--chat-manager-examples` | 17231 | Business logic, track implementations |
| `service--llm-hub` | 40631 | Unified LLM access (OpenRouter, GigaChat) |
| `service--question-detector` | 31611 | Question detection in messages |
| `service--text-extractor` | 9681 | Text extraction from files |
| `service--document-extractor` | - | Experimental document parsing (CPU/GPU) |
| `service--frontend-telegram` | - | Telegram bot frontend |

### Key Concepts

**Tracks:** Individual conversation flows implemented in `service--chat-manager-examples/src/chat_manager_examples/tracks/`. Each track:
- Inherits from `SimpleTrack` or implements `TrackI`
- Has a `DOMAIN` and `CAPTION` class attribute
- Implements `generate_response(chat, user_message) -> TrackResponse`

**Chat Flow:**
1. Gateway creates/retrieves `Chat` with `Context(client_id, track_id)`
2. Gateway forwards to ChatManager via `get_response(chat)`
3. ChatManager routes to appropriate Track
4. Track returns `AIMessage` with state and content
5. Gateway stores messages and returns response

**Content:** Messages use `make_content(text=..., resource=...)` where resource is `{"resource_id": ..., "resource_name": ...}`. File uploads via `mc.upload_resource((name, bytes), client_id)`.

**File Storage:** Resources stored in `./data/maestro/files/` via `FileStorage`.

### Shared Libraries (mmar-*)

These are external packages defining the MAESTRO protocol:
- `mmar-mapi`: Core API definitions (`Chat`, `Context`, `HumanMessage`, `AIMessage`, services)
- `mmar-mcli`: Client library (`MaestroClient`)
- `mmar-llm`: LLM abstractions and configuration
- `mmar-ptag`: Protocol tags
- `mmar-utils`: Shared utilities

## Configuration

- **Environment:** Copy `.env.default` to `data/.env` and modify
- **LLM Config:** `llm_config.json` (see `llm_config.json.default` for structure)
- **Service Discovery:** Services communicate via Docker network `network-maestro-core`
- **Configuration Files:** Services read `ENV_FILE` environment variable pointing to `/mnt/data/.env`

## Adding a New Track

1. Create file in `service--chat-manager-examples/src/chat_manager_examples/tracks/your_track.py`
2. Inherit from `SimpleTrack` and implement `generate_response`:
   ```python
   from mmar_mapi.tracks import SimpleTrack, TrackResponse

   class YourTrack(SimpleTrack):
       DOMAIN = DOMAINS.your_domain
       CAPTION = "Display Name"

       def generate_response(self, chat: Chat, user_message: HumanMessage) -> TrackResponse:
           return "state", "response text"
   ```
3. Register in `service--chat-manager-examples/src/chat_manager_examples/ioc.py`
4. Rebuild: `docker compose build chat-manager-examples`

## Code Style

- **Line length:** 120 characters
- **Type checking:** mypy with strict mode enabled (most services)
- **Linting:** ruff with specific rule sets (see service `pyproject.toml`)
- **Python version:** Services require Python >= 3.13 (root project >= 3.12)

## Important Notes

- All services mount `./data` to `/mnt/data` for persistence
- Gateway generates trace IDs for request tracking
- Some services (question-detector) have missing binaries - check functionality before依赖
- Document extractor on CPU is significantly slower than GPU
- Russian language support is primary (Tesseract OCR packages, Russian comments in code)

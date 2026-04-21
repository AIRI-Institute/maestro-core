# MAESTRO Core

This is Minimal subset of MAESTRO to demonstrate MAESTRO architecture.

Framework description is available here: https://airi-institute.github.io/maestro-cover

It consist of:
- gateway :: backend gateway for all requests
- chat-manager-examples :: component which manages bots business-logic
- llm-hub :: service to interact with LLM
# Usage: steps
## Start MAESTRO:
```sh
make build setup-env up
```
## Check basic track:
```sh
make run-dummy records='dummy=hello dummy=test dummy=exit'
```
## Setup LLM
LLM integrations supported via OpenAI-compatible API:
- **OpenAI-compatible** providers (OpenAI, Anthropic, DeepSeek, OpenRouter, etc.)
- **GigaChat** (Sber)

Configuration is done via TOML format in `service--llm-hub/`:
### Manual Configuration
- navigate to `service--llm-hub/`
- copy `llm-config.toml.example` to `llm-config.toml`
- edit `llm-config.toml`:
  - configure `[connections.*]` sections with your API keys
  - set up `[routing]` to map model names
  - configure `[model_info.*]` with captions and defaults
- restart llm-hub: `make restart-llm-hub`

See `service--llm-hub/llm-config.toml.example` for detailed configuration examples.
## Run LLM example
```sh
make run-chatbot records='start="Какая ты языковая модель?"'
```
Questions forwarded to your configured LLM.
## Run `Describer` example
```sh
make run-describer
```
# document-extractor (EXPERIMENTAL)
## Build and up container with document-extractor
- on CPU :: `make document-extractor-up`
- on GPU :: `make document-extractor-up-on-gpu`
  - assumed CUDA with version >= 12.4 available
  - run `nvidia-smi | grep -o 'CUDA Version.*'` to check it
## Prepare your document
```sh
make prepare-document
```
## Run `DocumentDescriber` example
```sh
make run-document-describer
```

**Note**: `document-extractor` works pretty slow on CPU.
# frontend-telegram
## Create chatbot
Via @BotFather: https://t.me/BotFather
## Setup .env
Update `./data/.env`:
```env
TG_APPLICATION__HANDLE=@your-bot-handle
TG_APPLICATION__TOKEN=your-bot-token
bot__commands={"start": "Dummy"}

AUTH__TG_PASSWORD=password-to-yourbot
# optional
```
## Run
```sh
docker compose --file=compose--frontend-telegram.yaml --file=compose.yaml up
```

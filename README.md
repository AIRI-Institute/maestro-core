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
LLM integrations supported. (Right now only `OpenRouter` or `GigaChat`)

If you have keys, you need to setup `llm_config.json`, configuration for LLM.
### Variant 1
- copy `llm_config.json.example` to `llm_config.json`
- edit `llm_config.json` :: fill `???` placeholders with your keys, fix `model_id` if you need, remove excess llm_config
- copy `llm_config.json` to `./data`, run `make update-llm-config`
- restart llm-hub, run `make restart-llm-hub`
### Variant 2
- use wizard to setup `llm_config.json`: run `make run-wizard`
- follow MAESTRO CLI steps.
- copy `llm_config.json` to `./data`, run `make update-llm-config`
- restart llm-hub, run `make restart-llm-hub`
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

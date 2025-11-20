# Maestro Core

This is Minimal subset of Maestro to demonstrate Maestro architecture.

Framework description is available here: https://airi-institute.github.io/maestro-cover

It consist of:
- gateway :: backend gateway for all requests
- chat-manager-examples :: component which manages bots business-logic
- llm-accessor :: service to interact with LLM
# Usage: steps
## Start Maestro:
```sh
make build setup-env up
```
## Check basic track:
```sh
make run-dummy records='dummy=hello dummy=test dummy=exit'
```
## Setup LLM
LLM integrations supported. (Right now only `OpenRouter` or `Gigachat`)

If you have keys, you need to setup `entrypoints.json`, configuration for LLM.
### Variant 1
- copy `entrypoints.json.example` to `entrypoints.json`
- edit `entrypoints.json` :: fill `???` placeholders with your keys, fix `model_id` if you need, remove excess entrypoints
- copy `entrypoints.json` to `./data`, run `make update-entrypoints`
- restart llm-accessor, run `make restart-llm-accessor`
### Variant 2
- use wizard to setup `entrypoints.json`: run `make run-wizard`
- follow Maestro CLI steps.
- copy `entrypoints.json` to `./data`, run `make update-entrypoints`
- restart llm-accessor, run `make restart-llm-accessor`
## Run LLM example
```sh
make run-chatbot records='start="Какая ты языковая модель?"'
```
Questions forwarded to your configured LLM.
## Run Describer example
```sh
make run-describer
```

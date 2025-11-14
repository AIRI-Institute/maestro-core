# Maestro Core

This is Minimal subset of Maestro to demonstrate Maestro architecture.

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

If you have keys, run wizard to setup `entrypoints.json`: environment for LLM:
```sh
make run-wizard
```
Then follow Maestro CLI steps.

When done, copy `entrypoints.json` to `./data` and restart llm-accessor:
```sh
make update-entrypoints restart-llm-accessor
```
## Run LLM example
```sh
make run-chatbot 
```
Questions forwarded to your configured LLM.


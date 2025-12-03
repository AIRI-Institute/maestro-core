DOC=docker compose --progress='plain'
records=
env_params=
RUN_MAESTRO_CLI_TRACK=$(env_params) uv run --prerelease=allow  --refresh-package=mmar-llm --refresh-package=mmar-mapi --refresh-package=mmar-mcli python maestro_client_cli.py track
service=

build:
	 $(DOC) build

setup-env:
	mkdir -p data
	cp .env.default data/.env
	cp llm_config.json.default data/llm_config.json

up:
	$(DOC) up --detach

logs:
	$(DOC) logs --follow $(service)

logs-llm-hub:
	$(DOC) logs --follow llm-hub

tail:
	$(DOC) logs --follow --tail=1

t: tail

stop:
	$(DOC) stop

run-dummy:
	$(RUN_MAESTRO_CLI_TRACK) Dummy $(records)

run-wizard:
	$(RUN_MAESTRO_CLI_TRACK) LLMConfigWizard $(records)

run-describer:
	$(RUN_MAESTRO_CLI_TRACK) Describer $(records)

update-llm-config:
	cp llm_config.json data/llm_config.json

restart-llm-hub:
	$(DOC) restart llm-hub

run-chatbot:
	$(RUN_MAESTRO_CLI_TRACK) Chatbot $(records)

ps:
	$(DOC) ps

document-extractor-up:
	$(DOC) --file compose--document-extractor--cpu.yaml build
	$(DOC) --file compose--document-extractor--cpu.yaml up --detach

document-extractor-up-on-gpu:
	$(DOC) --file compose--document-extractor--cu124.yaml build
	$(DOC) --file compose--document-extractor--cu124.yaml up --detach

run-document-describer:
	$(RUN_MAESTRO_CLI_TRACK) DocumentDescriber =@test_document.pdf =exit

s1: build setup-env up
s2:
	$(MAKE) run-dummy records='dummy=hello dummy=test dummy=exit'
s3: run-wizard
s4: update-llm-config
s5: restart-llm-hub
s6:
	$(MAKE) run-chatbot records='start="Какая ты языковая модель?"'

s7: document-extractor-up
s7-gpu: document-extractor-up-on-gpu
s8: run-document-describer

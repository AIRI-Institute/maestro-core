DOC=docker compose --progress='plain'
records=
env_params=
RUN_MAESTRO_CLI_TRACK=$(env_params) uv run --prerelease=allow --refresh-package=mmar-mapi --refresh-package=mmar-mcli python maestro_client_cli.py track
service=

build:
	 $(DOC) build

setup-env:
	mkdir -p data/maestro
	cp .env.default data/.env
	@echo "ℹ️  To configure LLM Hub:"
	@echo "   Option 1: cp service--llm-hub/llm-config.toml.example service--llm-hub/llm-config.toml"
	@echo "   Option 2: make run-wizard && make update-llm-config"

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
	@if [ -f llm-config.toml ]; then \
		echo "Copying llm-config.toml to service--llm-hub/"; \
		cp llm-config.toml service--llm-hub/llm-config.toml; \
		echo "✓ Updated service--llm-hub/llm-config.toml"; \
	else \
		echo "Error: llm-config.toml not found"; \
		echo "Run 'make run-wizard' to generate it, or create it manually"; \
		exit 1; \
	fi

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

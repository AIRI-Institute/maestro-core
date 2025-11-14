DOC=docker compose --progress='plain'
records=
env_params=
RUN_MAESTRO_CLI=$(env_params) uv run --prerelease=allow  --refresh-package=mmar-llm --refresh-package=mmar-mapi --refresh-package=mmar-mcli python maestro_client_cli.py

build:
	 $(DOC) build

setup-env:
	mkdir -p data
	cp .env.default data/.env
	cp entrypoints.json.default data/entrypoints.json

up:
	$(DOC) up --detach

logs:
	$(DOC) logs --follow

logs-llm-accessor:
	$(DOC) logs --follow llm-accessor

tail:
	$(DOC) logs --follow --tail=1

t: tail

stop:
	$(DOC) stop

run-dummy:
	$(RUN_MAESTRO_CLI) Dummy $(records)

run-wizard:
	$(RUN_MAESTRO_CLI) EntrypointsWizard $(records)

update-entrypoints:
	cp entrypoints.json data/entrypoints.json

restart-llm-accessor:
	$(DOC) restart llm-accessor

run-chatbot:
	$(RUN_MAESTRO_CLI) Chatbot $(records)

ps:
	$(DOC) ps

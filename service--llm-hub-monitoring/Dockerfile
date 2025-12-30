FROM python:3.13-slim-bullseye
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app/

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv sync --no-dev --prerelease=allow --refresh-package=mmar-ptag --refresh-package=mmar-mapi --refresh-package=mmar-llm --refresh-package=mmar-utils --refresh-package=mmar-mimpl --refresh-package=langfuse

COPY pyproject.toml pyproject.toml
COPY src/llm_hub_monitoring llm_hub_monitoring

ENV UV_NO_SYNC=1
CMD ["uv", "run", "-m", "llm_hub_monitoring.main"]

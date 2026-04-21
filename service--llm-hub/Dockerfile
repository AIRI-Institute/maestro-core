FROM python:3.13-slim-bullseye@sha256:e98b521460ee75bca92175c16247bdf7275637a8faaeb2bcfa19d879ae5c4b9a
COPY --from=ghcr.io/astral-sh/uv@sha256:c4089b0085cf4d38e38d5cdaa5e57752c1878a6f41f2e3a3a234dc5f23942cb4 /uv /uvx /bin/

WORKDIR /app/

COPY pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --prerelease=allow

COPY src/llm_hub llm_hub

ENV UV_NO_SYNC=1
CMD ["uv", "run", "-m", "llm_hub.main"]

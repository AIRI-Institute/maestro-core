FROM python:3.13-slim@sha256:079601253d5d25ae095110937ea8cfd7403917b53b077870bccd8b026dc7c42f
COPY --from=ghcr.io/astral-sh/uv@sha256:c4089b0085cf4d38e38d5cdaa5e57752c1878a6f41f2e3a3a234dc5f23942cb4 /uv /uvx /bin/

WORKDIR /app/text-extractor
RUN apt update -y \
  && apt install -y tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng --no-install-recommends \
  && rm -rf /var/lib/apt/lists/*

# Set Tesseract path as env variable
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
  UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  PATH="/app/document-extractor/.venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv sync --no-dev --prerelease=allow --refresh-package=mmar-ptag --refresh-package=mmar-mapi --refresh-package=mmar-utils
ENV UV_NO_SYNC=1

COPY pyproject.toml pyproject.toml
COPY src/ src/

CMD ["uv", "run", "-m", "src.main"]

FROM python:3.13-slim@sha256:079601253d5d25ae095110937ea8cfd7403917b53b077870bccd8b026dc7c42f AS base
COPY --from=ghcr.io/astral-sh/uv@sha256:ecfea7316b266ba82a5e9efb052339ca410dd774dc01e134a30890e6b85c7cd1 /uv /uvx /bin/

ARG EXTRA
RUN [ "$EXTRA" = "cpu" ] || [ "$EXTRA" = "cu124" ] || (echo "ERROR: EXTRA must be 'cpu' or 'cu124'. Found: $EXTRA" && exit 1)

WORKDIR /app/
RUN apt update -y \
  && apt install -y tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng --no-install-recommends \
  && rm -rf /var/lib/apt/lists/*

# Set Tesseract path as env variable
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
  UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  PATH="/app/.venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv sync --no-dev --extra=${EXTRA} --prerelease=allow --refresh-package=mmar-ptag --refresh-package=mmar-mapi --refresh-package=mmar-utils
ENV UV_NO_SYNC=1

COPY pyproject.toml pyproject.toml
COPY src/document_extractor document_extractor

FROM base AS single-run
CMD ["uv", "run", "-m", "document_extractor.main_single_run"]

FROM base AS service
CMD ["uv", "run", "-m", "document_extractor.main"]

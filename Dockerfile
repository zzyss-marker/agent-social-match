FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir sqlalchemy[asyncio] aiosqlite pydantic-settings structlog alembic httpx

FROM python:3.12-slim AS runtime

RUN groupadd -r app && useradd -r -g app app
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

RUN mkdir -p /app/data && chown -R app:app /app

USER app
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

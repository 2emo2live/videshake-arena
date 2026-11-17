FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev --compile-bytecode

COPY vibesnake ./vibesnake
COPY run_local.py ./

ENV PORT=8000
ENV SNAKE_MOVE_BUDGET=0.33
EXPOSE 8000

CMD ["uv", "run", "--no-sync", "python", "-m", "vibesnake.api"]

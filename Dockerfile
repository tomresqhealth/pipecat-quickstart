FROM python:3.11-slim-bookworm

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# --- THE FIX: Bind to all interfaces for cloud deployment ---
ENV HOST=0.0.0.0

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Add the virtual environment to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application code
COPY ./prompts.py prompts.py
COPY ./bot.py bot.py

ENTRYPOINT ["python", "bot.py"]
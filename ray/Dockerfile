FROM ubuntu:25.10
RUN apt update && apt -y install wget libgl1-mesa-dev libglib2.0-0
WORKDIR /root
RUN --mount=type=cache,sharing=locked,mode=0777,target=/root/.cache/uv,id=uv \
    --mount=from=ghcr.io/astral-sh/uv:0.7.6,source=/uv,target=/usr/bin/uv \
    --mount=type=bind,target=uv.lock,src=uv.lock \
    --mount=type=bind,target=pyproject.toml,src=pyproject.toml \
    uv sync -p 3.12
ENV PATH="/root/.venv/bin:$PATH" \
    UV_PYTHON=/root/.venv/bin/python

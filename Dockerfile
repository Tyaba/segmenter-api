#syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

WORKDIR /app

# install basic dependencies
RUN <<EOF
apt-get update
apt-get upgrade -y
apt-get install -y --no-install-recommends sudo git curl gcc \
openssh-client software-properties-common
apt-get clean
rm -rf /var/lib/apt/lists/*
EOF

# install python using uv
RUN uv python install 3.11
# cache用にsrc以外を先にinstall
COPY pyproject.toml /app/
RUN uv sync --all-groups
# srcをコピーしてinstall
COPY src /app/src
RUN uv pip install -e .
COPY . /app/

CMD ["uv", "run", "gunicorn", "-c", "src/segmenter_api/gunicorn.conf.py"]
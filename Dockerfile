FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install uv from distroless image
# uv is not mandatory, but it makes everything easier
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Python
RUN uv python install 3.12

# Create a virtual environment
RUN uv venv --seed --python=3.12 /venv

# Install the package without building it, then remove the project from venv
RUN mkdir /project
COPY pyproject.toml /project/
COPY src /project/src
RUN . /venv/bin/activate \
    && uv pip install -e /project -f "https://data.pyg.org/whl/torch-2.3.1%2Bcu121.html"
RUN rm -rf /project && mkdir /project

# Entrypoint with the virtual environment activated
WORKDIR /project
ENTRYPOINT ["/usr/bin/sh", "-c", ". /venv/bin/activate && \"$@\"", "--"]
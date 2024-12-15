FROM python:3.12

RUN curl -sSL https://install.python-poetry.org | python

ENV PATH="${PATH}:/root/.local/bin"
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=0

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root --without dev

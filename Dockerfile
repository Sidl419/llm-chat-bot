FROM python:3.10

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_VERSION=1.6.1

ENV PATH="$PATH:$POETRY_HOME/bin"\
    MODEL_FILE="llama-2-7b-chat.Q2_K.gguf"

RUN pip install -U setuptools &&\
    # install redis 
    apt-get update &&\
    apt-get -y install lsb-release curl gpg &&\
    curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg &&\
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list &&\
    apt-get -y install redis &&\
    # install poetry
    curl -sSL https://install.python-poetry.org | POETRY_VERSION=$POETRY_VERSION python3 -

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root && rm -rf "${HOME}/.cache/pypoetry"

COPY ./app ./app
COPY ./$MODEL_FILE ./$MODEL_FILE

#WORKDIR /src

CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

FROM python:3.11-slim

WORKDIR /app

RUN pip install pipenv

COPY Pipfile* ./
RUN pipenv install --deploy --system

COPY src/ ./src/
COPY .env ./
RUN mkdir -p /app/data/database

CMD ["python", "-m", "transcribai.main"]
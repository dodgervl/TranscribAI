# TranscribAI
Coolest transcriber at most convenient location - telegram bot
Telegram bot for receiving audio/video and preparing for lecture transcription.

## Setup

- Install [Pipenv](https://pipenv.pypa.io/)
- Install dependencies: `pipenv install`
- Activate environment: `pipenv shell`
- Create `.env` file
- Add your Telegram bot token to `.env` (see sample.env for reference)

## Run locally

```sh
PYTHONPATH=src python -m transcribai.main
```

## Run with Docker

```sh
docker build -t transcribai .
docker run --env-file .env transcribai
```
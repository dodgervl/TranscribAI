import asyncio
import logging
import os
from dotenv import load_dotenv

load_dotenv()

from .db import init_db

init_db()

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from .handlers import router

logging.basicConfig(level=logging.INFO)
TOKEN = os.getenv("BOT_TOKEN")


async def main():
    if not TOKEN:
        print("No BOT_TOKEN found. Please check your .env file.")
        return
    bot = Bot(token=TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

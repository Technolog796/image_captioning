import os
import asyncio

from aiogram import Bot, Dispatcher, executor, types

import bot_mp_utils

TOKEN = 'TOKEN'

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


async def starup(*args):
    print("Бот успешно запущен")


async def shutdown(*args):
    print("Бот успешно отключен")


@dp.message_handler(commands=["start", "help"])
async def help_handler(message: types.Message):
    await bot.send_message(chat_id=message.from_user.id,
                           text="Отправь боту изображение, что бы он ответил, что на нём изображено")


@dp.message_handler(content_types=types.ContentTypes.PHOTO)
async def photo_handler(message: types.Message):
    a = await bot.send_message(chat_id=message.from_user.id, text="Модель загружается...⏳")
    photo = message.photo[-1]
    file_id = photo.file_id
    path = f"../data/tests/{file_id}.jpg"
    await bot.download_file_by_id(file_id, path)
    ans = bot_mp_utils.bot_inference(path)

    await bot.edit_message_text(chat_id=message.from_user.id, message_id=a.message_id,
                                text=f"На этом изображении: {ans.lower()}")
    os.remove(path)


if __name__ == '__main__':
    executor.start_polling(dp, on_startup=starup, on_shutdown=shutdown, skip_updates=True)

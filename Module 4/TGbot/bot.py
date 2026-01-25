import asyncio
import os
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, CallbackQuery
from aiogram.filters import Command
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np


BOT_TOKEN = "пупупу" #токен получеаем из тг-бота BotFather
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


model = YOLO("/Users/veronikadenisenko/Documents/Preparing_Rea2026/Module 4/TGbot/dogs_yolo_model.pt") #уже обученна модель

# АВТОМАТИЧЕСКИ получаем классы из модели для интерфейса
CLASSES = list(model.names.values())  
NUM_CLASSES = len(CLASSES)

# создадим клавиатуру с кнопками для удобства
def get_main_keyboard():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🐶 Классифицировать"), KeyboardButton(text="📋 Породы")],
            [KeyboardButton(text="ℹ️ Инфо"), KeyboardButton(text="🔄 Новый поиск")]
        ],
        resize_keyboard=True
    )
    return keyboard


# ===== СТАРТОВАЯ КОМАНДА С КНОПКАМИ =====
@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        f"🐶 Dog Classifier Bot\n\n"
        f"📊 {NUM_CLASSES} пород из модели\n\n"
        "Отправь фото собаки или используй кнопки ниже:",
        reply_markup=get_main_keyboard()
    )


# ===== ОБРАБОТКА КНОПОК =====
@dp.message(F.text == "🐶 Классифицировать")
async def classify_btn(message: Message):
    await message.answer(
        "📸 Отправь фото собаки для классификации!",
        reply_markup=get_main_keyboard()
    )


@dp.message(F.text == "📋 Породы")
async def classes_btn(message: Message):
    text = f"🐕 {NUM_CLASSES} пород из модели:\n\n"
    text += "\n".join([f"{i+1}. {CLASSES[i]}" for i in range(min(25, NUM_CLASSES))])
    if NUM_CLASSES > 25:
        text += f"\n\n... +{NUM_CLASSES-25} пород"
    await message.answer(text, reply_markup=get_main_keyboard())


@dp.message(F.text == "ℹ️ Инфо")
async def info_btn(message: Message):
    await message.answer(
        f"🤖 YOLO11 Classification\n\n"
        f"• {NUM_CLASSES} пород собак\n"
        f"• Классы: {CLASSES[0]}, {CLASSES[1]}, ...\n"
        f"• Точность 97%+\n"
        f"• Работает с любыми фото\n\n"
        "🐶 Отправь фото!",
        reply_markup=get_main_keyboard()
    )


@dp.message(F.text == "🔄 Новый поиск")
async def new_search_btn(message: Message):
    await message.answer(
        "📸 Отправь новое фото для классификации!",
        reply_markup=get_main_keyboard()
    )


# ===== ОСНОВНАЯ ЛОГИКА КЛАССИФИКАЦИИ =====
@dp.message(F.photo)
async def handle_photo(message: Message):
    await message.answer("🔍 Классифицирую породу...")
    
    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        photo_bytes = await bot.download_file(file.file_path)
        
        photo_data = photo_bytes.getvalue() if isinstance(photo_bytes, io.BytesIO) else photo_bytes
        
        image = Image.open(io.BytesIO(photo_data)).convert('RGB') #предобработка для yolo
        if image.size[0] < 224:
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        image_np = np.array(image)
        
        results = model(image_np, task='classify')[0]
        
        top1 = model.names[results.probs.top1]
        top1_conf = results.probs.top1conf.item()
        
        top5 = results.probs.top5
        top5_confs = results.probs.top5conf.tolist()
        
        response = f"🐶 ТОП-1: {top1} ({top1_conf:.1%})\n\nТОП-3:\n"
        for i, cls_id in enumerate(top5[:3]):
            class_name = model.names[cls_id]
            conf = top5_confs[i]
            response += f"{i+1}. {class_name} ({conf:.1%})\n"
        
        await message.answer(response, reply_markup=get_main_keyboard())
        print(f"✅ {top1} ({top1_conf:.1%})")
        
    except Exception as e:
        await message.answer(f"Ошибка: {str(e)}", reply_markup=get_main_keyboard())


# ===== ОСТАЛЬНОЕ =====
@dp.message()
async def unknown(message: Message):
    await message.answer(
        "❓ Используй кнопки или отправь фото!",
        reply_markup=get_main_keyboard()
    )


# ЗАПУСК
async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

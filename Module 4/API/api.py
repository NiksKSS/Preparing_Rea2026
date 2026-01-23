from fastapi import FastAPI
from fastapi import UploadFile, File, HTTPException, Form
from pydantic import BaseModel

import pandas as pd
import os
import logging

# Импорт функций инференса и дообучения
from tables import inference_table, fine_tuning_regression
from audio import inference_audio, fine_tuning_audio
from image import inference_image, fine_tuning_fruit

app = FastAPI()

# -----------------------------
# Pydantic модели для валидации входных данных
# -----------------------------
class TableData(BaseModel):
    CO: float
    NO2: float
    SO2: float
    O3: float
    PM25: float
    PM10: float
    AQI: float


class AudioData(BaseModel):
    label: str


@app.get("/")
async def root():
    # Тестовый эндпоинт для проверки работы сервера
    return {"message": "It works!"}


# -----------------------------
# Инференс табличной модели
# -----------------------------
@app.get("/table_inference")
async def get_table_inference(
    CO: float = 300.3, NO2: float = 20.5,
    SO2: float = 3.1, O3: float = 32.2,
    PM25: float = 15.0, PM10: float = 16.6
):
    # Собираем вход в DataFrame
    test = pd.DataFrame({
        'CO': [CO], 'NO2': [NO2], 
        'SO2': [SO2], 'O3': [O3],
        'PM2.5': [PM25], 'PM10': [PM10]
    })

    # Получаем предсказание
    y_pred = inference_table(test)

    logging.error(y_pred)  # Логируем предикт (пока как error для видимости)

    return y_pred


# -----------------------------
# Дообучение табличной модели (один объект)
# -----------------------------
@app.post('/finetuning_table_single')
async def finetuning_table_single(data: TableData):

    # Преобразуем данные в DataFrame для обучения
    data2ft = pd.DataFrame({
        'CO': [data.CO], 'NO2': [data.NO2], 
        'SO2': [data.SO2], 'O3': [data.O3],
        'PM2.5': [data.PM25], 'PM10': [data.PM10],
        'AQI': [data.AQI]
    })

    # Запускаем дообучение
    metrics = fine_tuning_regression(data2ft)
    return metrics


# -----------------------------
# Дообучение табличной модели (батч)
# -----------------------------
@app.post("/finetuning_table_batch")
async def finetuning_table_batch(file: UploadFile = File(...)):

    filename = file.filename.lower()
    if not filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Формат файла не поддерживается: {filename}. Ожидается CSV."
        )

    # Читаем CSV в DataFrame
    df = pd.read_csv(file.file)

    # Обучаем модель
    metrics = fine_tuning_regression(df)
    return metrics


# -----------------------------
# Инференс аудио модели
# -----------------------------
@app.post('/audio_inference')
async def audio_inference_api(file: UploadFile = File(...)):

    filename = file.filename.lower()
    if not filename.endswith(".wav"):
        raise HTTPException(400, f"Ожидается WAV, получено: {filename}")

    # Сохраняем файл для инференса
    os.makedirs('audio_models/cats_dogs/inference', exist_ok=True)
    filepath = f'audio_models/cats_dogs/inference/{filename}'

    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    # Получаем предсказание
    y_pred = inference_audio([filepath])
    return {'label': y_pred[0]}


# -----------------------------
# Дообучение аудио модели
# -----------------------------
@app.post('/finetuning_audio')
async def finetuning_audio_api(file: UploadFile = File(...),
                               label: str = Form(...)):

    filename = file.filename.lower()
    if not filename.endswith(".wav"):
        raise HTTPException(400, f"Ожидается WAV, получено: {filename}")

    # Сохраняем аудиофайл в папку класса
    os.makedirs(f'audio_models/cats_dogs/train/{label}', exist_ok=True)
    filepath = f'audio_models/cats_dogs/train/{label}/{filename}'

    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    # Дообучаем модель
    metrics = fine_tuning_audio([filepath], [label])
    return metrics


# -----------------------------
# Инференс изображения
# -----------------------------
@app.post('/image_inference')
async def image_inference_api(file: UploadFile = File(...)):

    filename = file.filename.lower()
    if not filename.endswith(".jpg"):
        raise HTTPException(400, f"Ожидается JPG, получено: {filename}")

    # Сохраняем изображение
    os.makedirs('image_models/data_fruits/inference', exist_ok=True)
    filepath = f'image_models/data_fruits/inference/{filename}'

    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    # Предсказание класса
    y_pred = inference_image(filepath)
    return {'label': y_pred}


# -----------------------------
# Дообучение модели изображений
# -----------------------------
@app.post('/finetuning_image')
async def finetuning_image_api(file: UploadFile = File(...),
                               label: str = Form(...)):

    filename = file.filename.lower()
    if not filename.endswith(".jpg"):
        raise HTTPException(400, f"Ожидается JPG, получено: {filename}")

    # Сохраняем изображение в директорию класса
    os.makedirs(f'image_models/data_fruits/train/{label}', exist_ok=True)
    filepath = f'image_models/data_fruits/train/{label}/{filename}'

    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    # Запуск дообучения
    metrics = fine_tuning_fruit(filepath, label)
    return metrics


# -----------------------------
# Локальный запуск через uvicorn (для разработки)
# -----------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
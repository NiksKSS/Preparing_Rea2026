from fastapi import FastAPI
from fastapi import UploadFile, File, HTTPException, Form
from pydantic import BaseModel

import pandas as pd
import os
import logging

# ------------------------------------------
# Базовая настройка логирования
# ------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Импорт функций инференса и дообучения
from tables import inference_table, fine_tuning_regression
from audio import inference_audio, fine_tuning_audio
from image import inference_image, fine_tuning_fruit
from geo import inference_geo

app = FastAPI()

# -----------------------------
# Pydantic модели для валидации
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

    # Формируем DataFrame
    test = pd.DataFrame({
        'CO': [CO], 'NO2': [NO2],
        'SO2': [SO2], 'O3': [O3],
        'PM2.5': [PM25], 'PM10': [PM10]
    })

    # Запуск инференса
    y_pred = inference_table(test)

    logging.info(f"Предсказание табличной модели: {y_pred}")

    return y_pred


# -----------------------------
# Дообучение табличной модели (1 объект)
# -----------------------------
@app.post('/finetuning_table_single')
async def finetuning_table_single(data: TableData):

    data2ft = pd.DataFrame({
        'CO': [data.CO], 'NO2': [data.NO2],
        'SO2': [data.SO2], 'O3': [data.O3],
        'PM2.5': [data.PM25], 'PM10': [data.PM10],
        'AQI': [data.AQI]
    })

    metrics = fine_tuning_regression(data2ft)

    logging.info(f"Метрики дообучения табличной модели: {metrics}")

    return metrics


# -----------------------------
# Дообучение табличной модели (батч)
# -----------------------------
@app.post("/finetuning_table_batch")
async def finetuning_table_batch(file: UploadFile = File(...)):

    filename = file.filename.lower()
    if not filename.endswith(".csv"):
        raise HTTPException(400, f"Ожидается CSV, получено: {filename}")

    df = pd.read_csv(file.file)
    metrics = fine_tuning_regression(df)

    logging.info(f"Метрики батч дообучения табличной модели: {metrics}")

    return metrics


# -----------------------------
# Инференс аудио модели
# -----------------------------
@app.post('/audio_inference')
async def audio_inference_api(file: UploadFile = File(...)):

    filename = file.filename.lower()
    if not filename.endswith(".wav"):
        raise HTTPException(400, f"Ожидается WAV, получено: {filename}")

    os.makedirs('audio_models/cats_dogs/inference', exist_ok=True)
    filepath = f'audio_models/cats_dogs/inference/{filename}'

    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    y_pred = inference_audio([filepath])

    logging.info(f"Предсказание аудио модели: {y_pred}")

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

    os.makedirs(f'audio_models/cats_dogs/train/{label}', exist_ok=True)
    filepath = f'audio_models/cats_dogs/train/{label}/{filename}'

    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    metrics = fine_tuning_audio([filepath], [label])

    logging.info(f"Метрики дообучения аудио модели: {metrics}")

    return metrics


# -----------------------------
# Инференс изображения
# -----------------------------
@app.post('/image_inference')
async def image_inference_api(file: UploadFile = File(...)):

    filename = file.filename.lower()
    if not filename.endswith(".jpg"):
        raise HTTPException(400, f"Ожидается JPG, получено: {filename}")

    os.makedirs('image_models/data_fruits/inference', exist_ok=True)
    filepath = f'image_models/data_fruits/inference/{filename}'

    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    y_pred = inference_image(filepath)

    logging.info(f"Предсказание изображения: {y_pred}")

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

    os.makedirs(f'image_models/data_fruits/train/{label}', exist_ok=True)
    filepath = f'image_models/data_fruits/train/{label}/{filename}'

    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    metrics = fine_tuning_fruit(filepath, label)

    logging.info(f"Метрики дообучения изображения: {metrics}")

    return metrics


# -----------------------------
# Инференс геомодели
# -----------------------------
@app.get("/geo_inference")
async def get_geo_inference(
    Amz2: float = 25.0, H2: float = 15.5,
    D2: float = 12.1, Skal2: float = 0.0,
    Tur1h2: float = 10.0
):

    data = pd.DataFrame({
        'Amz2': [Amz2], 'H2': [H2],
        'D2': [D2], 'Skal2': [Skal2],
        'Tur1h2': [Tur1h2]
    })

    y_pred = inference_geo(data)

    logging.info(f"Предсказание геомодели: {y_pred}")

    return {'predicted': y_pred}


# -----------------------------
# Локальный запуск сервера
# -----------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
import requests
import json

# -----------------------------
# 1) Тест инференса табличной модели
# -----------------------------
url = "http://127.0.0.1:8002/table_inference"

# Параметры передаются через query string (?CO=...&NO2=...)
params = {
    "CO": 300.3,
    "NO2": 20.5,
    "SO2": 3.1,
    "O3": 32.2,
    "PM25": 15.0,
    "PM10": 16.6,
}

response = requests.get(url, params=params)
print("Response:", response.text)


# -----------------------------
# 2) Дообучение табличной модели (одна строка)
# -----------------------------
url = "http://127.0.0.1:8002/finetuning_table_single"

# Передаем JSON с одной записью
payload = {
    "CO": 330.3,
    "NO2": 20.5,
    "SO2": 3.1,
    "O3": 32.2,
    "PM25": 15.0,
    "PM10": 16.6,
    "AQI": 80
}

response = requests.post(url, json=payload)
print("Status:", response.status_code)
print("Response:", response.text)


# -----------------------------
# 3) Дообучение табличной модели батчем (CSV)
# -----------------------------
url = "http://127.0.0.1:8002/finetuning_table_batch"

# Локальный CSV-файл с табличными данными
csv_path = "table_models/data2ft.csv"

# Отправляем CSV через multipart/form-data
with open(csv_path, "rb") as f:
    files = {
        "file": ("test_data.csv", f, "text/csv")
    }
    response = requests.post(url, files=files)

print("Status:", response.status_code)
print("Response:", response.json())


# -----------------------------
# 4) Дообучение аудио модели
# -----------------------------
url = "http://127.0.0.1:8002/finetuning_audio"

wav_path = "audio_models/cats_dogs/inference/frog.wav" 
label = "frog"  # класс, под которым сохранить файл

with open(wav_path, "rb") as f:
    files = {
        "file": ("necat_0441313224.wav", f, "audio/wav")
    }
    data = {
        "label": label
    }

    # POST multipart/form-data + поле label
    response = requests.post(url, files=files, data=data)

print("Status:", response.status_code)
print("Response:", response.json())


# -----------------------------
# 5) Инференс аудио модели
# -----------------------------
url = "http://127.0.0.1:8002/audio_inference"

wav_path = "audio_models/cats_dogs/train/dog/dog_barking_101.wav"

with open(wav_path, "rb") as f:
    files = {
        "file": ("hz.wav", f, "audio/wav")
    }

    # Модель вернёт label
    response = requests.post(url, files=files)

print("Status:", response.status_code)
print("Response:", response.json())


# -----------------------------
# 6) Инференс изображения
# -----------------------------
url = "http://127.0.0.1:8002/image_inference"

img_path = "image_models/data_fruits/banana.jpg"

with open(img_path, "rb") as f:
    files = {
        "file": ("img.jpg", f, "image/jpeg")
    }

    response = requests.post(url, files=files)

print("Status:", response.status_code)
print("Response:", response.json())


# -----------------------------
# 7) Дообучение модели изображений
# -----------------------------
url = "http://127.0.0.1:8002/finetuning_image"

wav_path = "image_models/data_fruits/banana.jpg"
label = "banana fruit"  # название класса

with open(wav_path, "rb") as f:
    files = {
        "file": ("banana_banana.jpg", f, "image/jpeg")
    }
    data = {
        "label": label
    }

    # Передаём изображение + label
    response = requests.post(url, files=files, data=data)

print("Status:", response.status_code)
print("Response:", response.json())


# -----------------------------
# 8) Тест инференса геомодели
# -----------------------------
url = "http://127.0.0.1:8002/geo_inference"

# Параметры передаются через query string (?CO=...&NO2=...)
params = {
    "Amz2": 25.3,
    "H2": 15.5,
    "D2": 12.1,
    "Skal2": 0.0,
    "Tur1h2": 10.0,
}

response = requests.get(url, params=params)
print("Response:", response.text)
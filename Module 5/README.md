# 🔹 Полная документация работы API

## 🔹 Общая информация

**Назначение системы:** Полнофункциональное REST API для инференса и онлайн-дообучения четырех типов ML-моделей.

**Поддерживаемые задачи:**
- Табличная регрессия AQI (CatBoost + MLPRegressor)
- Аудиоклассификация (MFCC + CatBoostClassifier)  
- Классификация изображений (ResNet18)
- Георегрессия (joblib.Regressor)

**Технический стек:** FastAPI, PyTorch, CatBoost, scikit-learn, librosa  
**Порт сервера:** 8002  
**Автодокументация:** http://localhost:8002/docs

***

## 🔹 Содержание

1. [🔹 Технические требования](#-технические-требования)
2. [🔹 Установка и запуск](#-установка-и-запуск)
3. [🔹 Архитектура системы](#-архитектура-системы)
4. [🔹 Структура проекта](#-структура-проекта)
5. [🔹 Спецификация API](#-спецификация-api)
6. [🔹 Описание моделей](#-описание-моделей)
7. [🔹 Тестирование](#-тестирование)
8. [🔹 Устранение неисправностей](#-устранение-неисправностей)
9. [🔹 Развертывание](#-развертывание)

***

## 🔹 Технические требования

### 🔹 Программное обеспечение
```
Python                 3.10+
FastAPI                0.115+
Uvicorn                0.30+
PyTorch                2.0+
CatBoost               1.2+
```

### 🔹 Зависимости Python
```
fastapi==0.115.0
uvicorn[standard]==0.30.1
torch>=2.0.0
torchvision>=0.15.0
catboost>=1.2.0
scikit-learn>=1.3.0
librosa>=0.10.0
pandas>=2.0.0
pillow>=10.0.0
joblib>=1.3.0
soundfile>=0.12.0
```

***

## 🔹 Установка и запуск

### 🔹 Шаг 1: Подготовка окружения
```bash
git clone <repository_url>
cd <project_directory>
```

### 🔹 Шаг 2: Виртуальное окружение
```bash
# Linux/macOS
python -m venv venv
source venv/bin/activate

# Windows  
python -m venv venv
venv\Scripts\activate
```

### 🔹 Шаг 3: Установка зависимостей
```bash
pip install -r requirements.txt
```

### 🔹 Шаг 4: Проверка структуры
```
audio_models/     ✅ scaler.pkl, audio_model.pkl
image_models/     ✅ fruit_model.pth  
table_models/     ✅ catboost_model_regression.cbm
geo_models/       ✅ geo_model.pkl
```

### 🔹 Шаг 5: Запуск сервера
```bash
# Разработка
uvicorn api:app --host 0.0.0.0 --port 8002 --reload

# Продакшен
uvicorn api:app --host 0.0.0.0 --port 8002 --workers 4
```

**Доступ:**
- API: http://localhost:8002
- Swagger: http://localhost:8002/docs  
- ReDoc: http://localhost:8002/redoc

***

## 🔹 Архитектура системы

### 🔹 Компоненты системы

```
┌─────────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│      HTTP Клиент    │───▶│    FastAPI сервер    │───▶│    ML Модули     │
│   (curl/Postman)    │    │     api.py:8002      │    │ (audio/image/...)│
└─────────────────────┘    └──────────────────────┘    └──────────────────┘
                                                         │
                                                 ┌──────────────────┐
                                                 │   Модели данных  │
                                                 │  (.pkl/.cbm/.pth)│
                                                 └──────────────────┘
```

### 🔹 Поток обработки запроса
```
1. Валидация (Pydantic) → 2. Маршрутизация → 3. ML-модуль
     ↓                           ↓                      ↓
4. Инференс/Дообучение → 5. Сохранение → 6. JSON-ответ
```

***

## 🔹 Структура проекта

```
📁 .
├── 🔹 api.py                      # FastAPI сервер (9 эндпоинтов)
├── 🔹 audio.py                    # MFCC + CatBoostClassifier
├── 🔹 geo.py                      # GeoRegressor
├── 🔹 image.py                    # ResNet18 + PyTorch
├── 🔹 tables.py                   # CatBoost + MLPRegressor  
├── 🔹 test_api.py                 # Тесты всех эндпоинтов
├── 🔹 requirements.txt

📁 audio_models/                   # Аудио ML
│   ├── 🔹 scaler.pkl
│   ├── 🔹 encoder.pkl
│   ├── 🔹 audio_model.pkl
│   └── 📁 cats_dogs/
│       ├── 📁 train/
│       └── 📁 inference/

📁 image_models/                   # Компьютерное зрение
│   ├── 🔹 fruit_model.pth
│   └── 📁 data_fruits/

📁 table_models/                   # Табличная регрессия
│   ├── 🔹 catboost_model_regression.cbm
│   └── 🔹 nn_model_regression.pkl

📁 geo_models/                     # Геоданные
│   └── 🔹 geo_model.pkl
```

***

## 🔹 Спецификация API

### 🔹 Общие характеристики
| Параметр | Значение |
|----------|----------|
| Базовый URL | http://localhost:8002 |
| Формат запросов | JSON, multipart/form-data |
| Кодировка | UTF-8 |
| Эндпоинтов | 9 |

### 🔹 Таблица эндпоинтов

| Метод | Эндпоинт | Операция | Вход | Выход |
|-------|----------|----------|------|-------|
| `GET` | `/` | Health | - | JSON |
| `GET` | `/table_inference` | Инференс | 6x float | 2x предсказания |
| `POST` | `/finetuning_table_single` | Дообучение | JSON (7x float) | Метрики |
| `POST` | `/finetuning_table_batch` | Дообучение | CSV | Метрики |
| `POST` | `/audio_inference` | Инференс | WAV | Класс |
| `POST` | `/finetuning_audio` | Дообучение | WAV+label | Метрики |
| `POST` | `/image_inference` | Инференс | JPG | Класс |
| `POST` | `/finetuning_image` | Дообучение | JPG+label | Метрики |
| `GET` | `/geo_inference` | Инференс | 5x float | Предсказание |

### 🔹 Примеры запросов

#### 🔹 1. Health Check
```bash
curl -X GET "http://localhost:8002/"
```
```json
{"message": "It works!"}
```

#### 🔹 2. Табличный инференс AQI
```bash
curl "http://localhost:8002/table_inference?CO=300.3&NO2=20.5&SO2=3.1&O3=32.2&PM25=15.0&PM10=16.6"
```
```json
{"catboost": 75.123, "neuralnet": 78.456}
```

#### 🔹 3. Дообучение таблиц (одиночный)
```bash
curl -X POST "http://localhost:8002/finetuning_table_single" \
  -H "Content-Type: application/json" \
  -d '{"CO":330.3,"NO2":20.5,"SO2":3.1,"O3":32.2,"PM25":15.0,"PM10":16.6,"AQI":80}'
```

#### 🔹 4. Аудио инференс
```bash
curl -X POST "http://localhost:8002/audio_inference" -F "file=@dog.wav"
```
```json
{"label": "dog"}
```

#### 🔹 5. Изображение инференс
```bash
curl -X POST "http://localhost:8002/image_inference" -F "file=@banana.jpg"
```
```json
{"label": "banana"}
```

***

## 🔹 Описание моделей

### 🔹 Табличная регрессия AQI
| Характеристика | Значение |
|----------------|----------|
| Входные данные | CO, NO2, SO2, O3, PM2.5, PM10 |
| Задача | Регрессия AQI |
| Модели | CatBoostRegressor, MLPRegressor |
| Метрики | R²=0.91, RMSE=6.9 |

### 🔹 Аудиоклассификация
| Характеристика | Значение |
|----------------|----------|
| Признаки | MFCC(30) + delta + delta-delta |
| Классы | cat, dog (расширяемо) |
| Модель | CatBoostClassifier |
| Метрика | Accuracy=94.2% |

### 🔹 Классификация изображений
| Характеристика | Значение |
|----------------|----------|
| Архитектура | ResNet18 |
| Классы | Фрукты (автоопределение) |
| Метрика | F1-macro=0.91 |

### 🔹 Георегрессия
| Характеристика | Значение |
|----------------|----------|
| Вход | Amz2, H2, D2, Skal2, Tur1h2 |
| Модель | joblib.Regressor |

***

## 🔹 Тестирование

### 🔹 Автоматическое тестирование
```bash
python test_api.py
```
**Выполняет:** 8 запросов ко всем эндпоинтам

### 🔹 Интерактивное тестирование
1. Откройте http://localhost:8002/docs
2. Выберите эндпоинт → "Try it out"
3. Заполните параметры → "Execute"

***

## 🔹 Устранение неисправностей

| Ошибка | Решение |
|--------|---------|
| `Address already in use` | `lsof -ti:8002 | xargs kill -9` |
| `No module 'librosa'` | `pip install librosa soundfile` |
| `CUDA out of memory` | CPU режим или `torch.cuda.empty_cache()` |
| `Model file not found` | Проверьте `*_models/` директории |
| `Invalid file format` | WAV/JPG/CSV по назначению |

***

## 🔹 Развертывание в продакшен

### 🔹 Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8002
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "4"]
```

### 🔹 Gunicorn
```bash
pip install gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8002
```

***

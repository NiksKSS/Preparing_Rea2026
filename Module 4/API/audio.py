import numpy as np
import librosa
from sklearn import metrics
import joblib


def extract_features_mfcc(
    file_path,
    n_mfcc=30,
    n_fft=2048,
    hop_length=512,
    add_delta=True
):
    """
    Читает аудио-файл и извлекает признаки MFCC.
    Возвращает один вектор: усреднение MFCC + std, 
    а также (опционально) дельта и дельта2.
    """

    # Загружаем аудио (sr=None — использовать исходную частоту)
    y, sr = librosa.load(file_path, sr=None)

    # Базовые MFCC-признаки
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Среднее и стандартное отклонение по времени
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc, axis=1)

    features = list(mfcc_mean) + list(mfcc_std)

    # Добавляем дельта-признаки: первая и вторая производные
    if add_delta:
        delta = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta, axis=1)
        delta_std  = np.std(delta, axis=1)

        delta2 = librosa.feature.delta(mfcc, order=2)
        delta2_mean = np.mean(delta2, axis=1)
        delta2_std  = np.std(delta2, axis=1)

        features += list(delta_mean) + list(delta_std)
        features += list(delta2_mean) + list(delta2_std)

    return np.array(features)


def update_label_encoder(encoder, new_labels):
    """
    Проверяет, появились ли новые классы.
    Если да — расширяет encoder.classes_.
    """

    new_unique = np.unique(new_labels)
    old_unique = encoder.classes_

    # ищем новые классы
    added = [cls for cls in new_unique if cls not in old_unique]

    if added:
        print("Добавлены новые классы:", added)
        # обновляем список классов
        encoder.classes_ = np.concatenate([old_unique, added])

    return encoder



def inference_audio(new_audio_path):
    """
    Инференс аудио-классификатора.
    - Загружает модели и scaler
    - Извлекает MFCC
    - Масштабирует
    - Предсказывает класс
    """

    scaler = joblib.load("audio_models/scaler.pkl")
    encoder = joblib.load("audio_models/encoder.pkl")
    model = joblib.load("audio_models/audio_model.pkl")
    
    # Извлекаем признаки из первого файла
    feats = extract_features_mfcc(new_audio_path[0])

    # Применяем scaler
    feats = scaler.transform([feats])

    # Предсказание класса (в виде индекса)
    y_pred = model.predict(feats)

    import logging
    logging.error(y_pred)  # временный лог для дебага

    # Конвертация индекса обратно в строковую метку
    response = encoder.inverse_transform([int(y_pred[0])])

    return response


def fine_tuning_audio(new_audio_paths, new_labels) -> dict:
    """
    Fine-tuning аудио-классификатора:
    - добавление новых классов в encoder,
    - извлечение признаков новых аудио,
    - добавление данных к старому train,
    - повторное обучение модели,
    - оценка качества,
    - сохранение обновлённого пайплайна.
    """

    # 1. Загрузка старых артефактов (модели, scaler, encoder, данные)
    scaler = joblib.load("audio_models/scaler.pkl")
    encoder = joblib.load("audio_models/encoder.pkl")

    old_X = joblib.load("audio_models/old_X_audio.pkl")
    old_y = joblib.load("audio_models/old_y_audio.pkl")

    model = joblib.load("audio_models/audio_model.pkl")

    # 2. Проверяем, появились ли новые классы — обновляем encoder
    encoder = update_label_encoder(encoder, new_labels)

    # Сохраняем encoder сразу (важно!)
    joblib.dump(encoder, "audio_models/encoder.pkl")

    # 3. Извлечение признаков для всех новых аудио
    X_new = []
    for path in new_audio_paths:
        feats = extract_features_mfcc(path)
        X_new.append(feats)

    X_new = np.array(X_new)

    # Преобразуем метки в числовой формат
    y_new = encoder.transform(new_labels)

    # Масштабируем новые признаки тем же scaler
    X_new_scaled = scaler.transform(X_new)

    # 4. Добавляем новые данные к старому train
    X_full = np.vstack([old_X, X_new_scaled])
    y_full = np.concatenate([old_y, y_new])

    # 5. Переобучаем модель на расширенном датасете
    model.fit(X_full, y_full)

    # 6. Тестирование модели на сохранённом test-наборе
    X_test = joblib.load("audio_models/X_test_audio.pkl")
    y_test = joblib.load("audio_models/y_test_audio.pkl")

    y_pred = model.predict(X_test)

    # Метрики качества
    metrics_dict = {
        "accuracy": round(metrics.accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(metrics.f1_score(y_test, y_pred, average="macro"), 4),
        "precision_macro": round(metrics.precision_score(y_test, y_pred, average="macro"), 4),
        "recall_macro": round(metrics.recall_score(y_test, y_pred, average="macro"), 4),
    }

    # 7. Сохраняем обновлённую модель и новые train-данные
    joblib.dump(model, "audio_models/audio_model.pkl")
    joblib.dump(X_full, "audio_models/old_X_audio.pkl")
    joblib.dump(y_full, "audio_models/old_y_audio.pkl")

    return metrics_dict
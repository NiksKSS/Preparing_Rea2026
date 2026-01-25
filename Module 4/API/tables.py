import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from sklearn import metrics


def inference_table(test: pd.DataFrame):
    # Загружаем обученные модели + scaler при каждом вызове 
    cb_model = CatBoostRegressor()
    cb_model.load_model('table_models/catboost_model_regression.cbm')
    nn_model = joblib.load('table_models/nn_model_regression.pkl')
    scaler = joblib.load('table_models/scaler.pkl')

    # Скейлим входные данные тем же scaler, что использовался при обучении
    test = scaler.transform(test)

    # Предсказания обеих моделей
    y_pred_cb = cb_model.predict(test)
    y_pred_nn = nn_model.predict(test)

    # Возвращаем предсказания в удобном виде
    return {
        'catboost': y_pred_cb.round(3)[0],
        'neuralnet': y_pred_nn.round(3)[0]
    }


# -------------------------------------------------
# Дообучение регрессионных моделей (CatBoost + MLP)
# -------------------------------------------------
def fine_tuning_regression(new_data: pd.DataFrame) -> dict:
    """
    Fine-tuning моделей на новых данных:
    - загружаем старые train-данные и модели,
    - добавляем новые примеры,
    - заново обучаем обе модели,
    - пересчитываем метрики на сохранённом тестовом наборе,
    - сохраняем обновлённые артефакты.
    """

    # 1. Загружаем старые артефакты (train-данные, scaler, модели)
    scaler = joblib.load('table_models/scaler.pkl')
    old_X = joblib.load('table_models/old_X_train_reg.pkl')
    old_y = joblib.load('table_models/old_y_train_reg.pkl')

    cb_model = CatBoostRegressor()
    cb_model.load_model('table_models/catboost_model_regression.cbm')

    nn_model = joblib.load('table_models/nn_model_regression.pkl')

    # 2. Подготовка новых данных
    if 'AQI' not in new_data.columns:
        raise ValueError("В new_data должен быть столбец 'AQI'.")

    X_new = new_data.drop('AQI', axis=1)
    y_new = new_data['AQI'].astype(float)

    # Скейлим новые данные тем же scaler
    X_new_scaled = scaler.transform(X_new)

    # 3. Объединение старых + новых данных
    X_full = np.vstack([old_X, X_new_scaled])
    y_full = np.concatenate([old_y, y_new])

    # 4. Переобучение моделей на расширенном датасете
    nn_model.fit(X_full, y_full)
    cb_model.fit(X_full, y_full)

    # 5. Тестирование на сохранённом тестовом наборе
    X_test_scaled = joblib.load('table_models/X_test_reg.pkl')
    y_test = joblib.load('table_models/y_test_reg.pkl')

    y_pred_cb = cb_model.predict(X_test_scaled)
    y_pred_nn = nn_model.predict(X_test_scaled)

    # 6. Расчёт метрик качества
    model_metrics = {
        "CatBoostRegressor": {
            "MAE": round(metrics.mean_absolute_error(y_test, y_pred_cb), 3),
            "RMSE": round(metrics.root_mean_squared_error(y_test, y_pred_cb), 3),
            "R2": round(metrics.r2_score(y_test, y_pred_cb), 3),
        },
        "MLPRegressor": {
            "MAE": round(metrics.mean_absolute_error(y_test, y_pred_nn), 3),
            "RMSE": round(metrics.root_mean_squared_error(y_test, y_pred_nn), 3),
            "R2": round(metrics.r2_score(y_test, y_pred_nn), 3),
        }
    }

    # 7. Сохраняем обновлённые модели и данные для дальнейших дообучений
    cb_model.save_model('table_models/catboost_model_regression.cbm')
    joblib.dump(nn_model, 'table_models/nn_model_regression.pkl')

    joblib.dump(X_full, 'table_models/old_X_train_reg.pkl')
    joblib.dump(y_full, 'table_models/old_y_train_reg.pkl')

    return model_metrics

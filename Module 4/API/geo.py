import joblib
import pandas as pd


def inference_geo(data: pd.DataFrame) -> int:

    model = joblib.load('geo_models/geo_model.pkl')
    y_pred = model.predict(data)
    return int(y_pred[0])
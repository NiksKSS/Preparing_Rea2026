import streamlit as st
import requests
import geopandas as gpd
import folium
from streamlit_folium import st_folium


API_URL = "http://127.0.0.1:8002"

# Заголовки интерфейса
st.header('Графический интерфейс пользователя')
st.subheader('Для каждого типа данных')
st.divider()

# =======================================================
#                 TABULAR DATA INFERENCE
# =======================================================
st.subheader("Анализ табличных данных")

# --- форма для инференса таблиц ---
with st.form("table_inference_form"):
    st.write("Введите параметры загрязнителей:")

    # Поля ввода значений загрязнителей
    CO = st.number_input("CO", value=300.3)
    NO2 = st.number_input("NO2", value=20.5)
    SO2 = st.number_input("SO2", value=3.1)
    O3 = st.number_input("O3", value=32.2)
    PM25 = st.number_input("PM2.5", value=15.0)
    PM10 = st.number_input("PM10", value=16.6)

    submit = st.form_submit_button("Получить предсказание")

    # Запрос к API
    if submit:
        params = {
            "CO": CO, "NO2": NO2, "SO2": SO2,
            "O3": O3, "PM25": PM25, "PM10": PM10
        }

        r = requests.get(f"{API_URL}/table_inference", params=params)
        st.success(r.json())

# --- Дообучение на одном объекте ---
st.write("### Дообучение (1 объект)")
with st.form("table_ft_single"):
    CO = st.number_input("CO:", value=200.0)
    NO2 = st.number_input("NO2:", value=10.0)
    SO2 = st.number_input("SO2:", value=2.0)
    O3 = st.number_input("O3:", value=30.0)
    PM25 = st.number_input("PM25:", value=11.0)
    PM10 = st.number_input("PM10:", value=18.0)
    AQI = st.number_input("AQI (истинная метка):", value=150.0)

    ft_submit = st.form_submit_button("Дообучить")

    # POST запрос с одним экземпляром
    if ft_submit:
        payload = {
            "CO": CO, "NO2": NO2, "SO2": SO2,
            "O3": O3, "PM25": PM25, "PM10": PM10,
            "AQI": AQI
        }
        r = requests.post(f"{API_URL}/finetuning_table_single", json=payload)
        st.success(r.json())

# --- Дообучение батчем CSV ---
st.write("### Дообучение (батч CSV)")

with st.form('batch_ft_table'):
    csv_file = st.file_uploader("Выберите CSV", type=["csv"])
    button = st.form_submit_button("Отправить CSV")

    # Отправка CSV в API
    if csv_file:
        if button:
            files = {"file": (csv_file.name, csv_file, "text/csv")}
            r = requests.post(f"{API_URL}/finetuning_table_batch", files=files)
            st.success(r.json())

st.divider()

# =======================================================
#                       IMAGE SECTION
# =======================================================

st.subheader("Анализ изображений")

# Сопоставление русских названий классов с английскими метками
fruits = {
    "Яблоко": "apple fruit",
    "Банан": "banana fruit",
    "Вишня": "cherry fruit",
    "Чику": "chickoo fruit",
    "Кокос": "coconut fruit",
    "Виноград": "grapes fruit",
    "Киви": "kiwi fruit",
    "Манго": "mango fruit",
    "Апельсин": "orange fruit",
    "Клубника": "strawberry fruit"
}

# --- Инференс изображений ---
with st.form('image_inference'):
    image_file = st.file_uploader("Изображение JPG", type=["jpg"])
    image_inference_button = st.form_submit_button('Предсказать класс')

    if image_file and image_inference_button:
        files = {"file": (image_file.name, image_file, "image/jpeg")}
        r = requests.post(f"{API_URL}/image_inference", files=files)
        st.success(r.json())

# --- Дообучение изображений ---
with st.form('image_ft'):
    st.write("### Дообучение изображения")

    image_train_file = st.file_uploader("JPG для дообучения", type=["jpg"], key="img_train")
    label_img = st.selectbox(
        "Метка класса",
        ("Яблоко", "Банан", "Вишня",
         "Чику", "Кокос", "Виноград", "Киви",
         "Манго", "Апельсин", "Клубника"),
    )

    ft_button_img = st.form_submit_button("Дообучить изображение")

    if image_train_file and label_img and ft_button_img:
        files = {"file": (image_train_file.name, image_train_file, "image/jpeg")}
        data = {"label": fruits.get(label_img)}
        r = requests.post(f"{API_URL}/finetuning_image", files=files, data=data)
        st.success(r.json())

st.divider()

# =======================================================
#                   AUDIO SECTION
# =======================================================

# --- Инференс аудио ---
with st.form('audio_inference'):
    audio_file = st.file_uploader("Аудио WAV", type=["wav"])
    audio_inference_button = st.form_submit_button('Предсказать класс аудио')

    if audio_file and audio_inference_button:
        files = {"file": (audio_file.name, audio_file, "audio/wav")}
        r = requests.post(f"{API_URL}/audio_inference", files=files)
        st.success(r.json())

# --- Дообучение аудио ---
st.write("### Дообучение аудио")
audio_train_file = st.file_uploader("WAV для дообучения", type=["wav"], key="wav_train")

# Сопоставление животных с английскими метками
animals = {
    'Собака': 'dog',
    'Кот': 'cat',
    'Лягушка': 'frog'
}

# Выбор метки класса
label_audio_raw = st.radio(
    "Метка класса аудио",
    ["Кот", "Собака", "Лягушка", "Новый класс"]
)

# Возможность задать новый класс
if label_audio_raw == 'Новый класс':
    label_audio = st.text_input('Название нового класса')
else:
    label_audio = animals[label_audio_raw]

audio_ft_button = st.button("Дообучить аудио")

# POST запрос на дообучение
if audio_train_file and label_audio and audio_ft_button:
    files = {"file": (audio_train_file.name, audio_train_file, "audio/wav")}
    data = {"label": label_audio}
    r = requests.post(f"{API_URL}/finetuning_audio", files=files, data=data)
    st.success(r.json())

st.divider()

# =======================================================
#                   GEO DATA SECTION
# =======================================================

st.subheader("Анализ геоданных")
st.write("### Загрузка геоданных")

# Загрузка GeoJSON
geo_file = st.file_uploader("Выберите GeoJSON / SHP", type=["geojson"])

# Чтение геоданных
if geo_file:
    try:
        gdf = gpd.read_file(geo_file)
        st.success(f"Файл загружен. Объектов: {len(gdf)}")
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        gdf = None
else:
    gdf = None

st.write("### Включаемые слои")

# Опции отображения слоёв
show_raw = st.checkbox("Исходные участки", True)
show_prediction = st.checkbox("Прогноз модели (Kf2)", True)
show_anomalies = st.checkbox("Аномалии", False)
show_regions = st.checkbox("Границы районов", False)

# --- Прогноз по участкам ---
if gdf is not None and "Amz2" in gdf.columns:

    if st.button("Выполнить прогноз по участкам"):

        preds = []
        # Поочередный запрос прогноза по каждому объекту
        for _, row in gdf.iterrows():
            params = {
                "Amz2": float(row["Amz2"]),
                "H2": float(row["H2"]),
                "D2": float(row["D2"]),
                "Skal2": float(row["Skal2"]),
                "Tur1h2": float(row["Tur1h2"])
            }
            r = requests.get(f"{API_URL}/geo_inference", params=params)
            preds.append(r.json()["predicted"])

        gdf["prediction"] = preds
        st.success("Прогнозы сохранены в колонку 'prediction'")
        st.dataframe(gdf)

# --- Определение аномалий ---
def detect_anomaly(x):
    return 1 if x > gdf["Amz2"].quantile(0.95) else 0

if gdf is not None:
    if show_anomalies and "prediction" in gdf.columns:
        gdf["anomaly"] = gdf["Amz2"].apply(detect_anomaly)

# --- Визуализация карты ---
if gdf is not None:

    st.write("### Карта участков")

    # Центр карты
    center = [
        gdf.geometry.centroid.y.mean(),
        gdf.geometry.centroid.x.mean()
    ]

    m = folium.Map(location=center, zoom_start=10)

    # Список колонок без geometry
    base_columns = [c for c in gdf.columns if c != "geometry"]

    # Универсальная функция добавления слоя
    def add_layer(data, name, color, fields):
        folium.GeoJson(
            data,
            name=name,
            style_function=lambda _: {
                "color": color,
                "fillColor": color,
                "weight": 1,
                "fillOpacity": 0.5,
            },
            tooltip=folium.GeoJsonTooltip(fields=fields),
        ).add_to(m)

    # --- Добавление слоёв ---

    if show_raw:
        add_layer(gdf, "Исходные участки", "blue", base_columns)

    if show_prediction and "prediction" in gdf.columns:
        add_layer(gdf, "Прогноз модели", "orange", ["prediction"])

    if show_anomalies and "anomaly" in gdf.columns:
        anomalies = gdf[gdf["anomaly"] == 1]
        add_layer(anomalies, "Аномалии", "red", ["prediction", "anomaly"])

    if show_regions and "region" in gdf.columns:
        add_layer(gdf, "Границы районов", "green", ["region"])

    # Управление слоями на карте
    folium.LayerControl().add_to(m)

    st_folium(m, width=1200, height=700)

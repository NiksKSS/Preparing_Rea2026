import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import pytz

# Заголовок приложения
st.title("🌍 Аналитический дашборд загрязнённости воздуха")
st.markdown("---")

# --- Загрузка и подготовка данных ---
@st.cache_data
def load_data():

    df = pd.read_csv('data/Air_Quality.csv')
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    return df

df = load_data()

# --- Фильтры ---
col1, col2 = st.columns([1, 1])
with col1:
    cities = st.multiselect(
        "Выберите города",
        options=df['City'].unique(),
        default=df['City'].unique()
    )
with col2:
    date_range = st.date_input(
        "Выберите диапазон дат",
        value=(df['Date'].min().date(), df['Date'].max().date()),
        min_value=df['Date'].min().date(),
        max_value=df['Date'].max().date()
    )
print(date_range)
if len(date_range) != 2:
    st.warning("Пожалуйста, выберите начальную и конечную даты.")
    st.stop()

start_date, end_date = date_range
start_date = pd.Timestamp(start_date, tz='UTC')
end_date = pd.Timestamp(end_date, tz='UTC') + timedelta(days=1) - timedelta(seconds=1)

filtered_df = df[
    (df['City'].isin(cities)) &
    (df['Date'] >= start_date) &
    (df['Date'] <= end_date)
]

if filtered_df.empty:
    st.warning("Нет данных для выбранных фильтров.")
    st.stop()

# --- ПДК ---
pdk_values = {
    "CO": 5.0,  # mg/m³
    "NO2": 0.04, # mg/m³
    "SO2": 0.05, # mg/m³
    "O3": 0.06,  # mg/m³
    "PM2.5": 15.0, # μg/m³
    "PM10": 45.0  # μg/m³
}

# --- Вычисления ---
# 1. Уровень загрязнения в % от ПДК
pollutants = ["CO", "NO2", "SO2", "O3", "PM2.5", "PM10"]
for p in pollutants:
    filtered_df[f"{p}_pct_PDK"] = (filtered_df[p] / pdk_values[p]) * 100

# 3. Топ загрязнённых станций (городов)
city_avg_pm25 = filtered_df.groupby('City')['PM2.5'].mean().sort_values(ascending=False).head(10)
top_polluted_cities = city_avg_pm25.index.tolist()

# 4. Средние значения по станциям
averages = filtered_df.groupby('City')[pollutants].mean()

# 5. Количество станций с превышением ПДК
exceedance_mask = pd.DataFrame()
for p in pollutants:
    exceedance_mask[p] = filtered_df[p] > pdk_values[p]
filtered_df['Exceedance_Flag'] = exceedance_mask.any(axis=1)
stations_exceeding = filtered_df[filtered_df['Exceedance_Flag']]['City'].nunique()

# --- Отображение метрик ---
st.subheader("📊 Ключевые метрики")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Всего записей", value=f"{len(filtered_df):,}")
with col2:
    st.metric(label="Всего локаций", value=filtered_df['City'].nunique())
with col3:
    st.metric(label="Локаций с превышением", value=stations_exceeding)

st.markdown("---")

# --- 1. Уровень загрязнения в % от ПДК ---
st.subheader("1. Уровень загрязнения в % от ПДК")
avg_pct_pdk = {p: (filtered_df[p].mean() / pdk_values[p]) * 100 for p in pollutants}
pct_df = pd.DataFrame(list(avg_pct_pdk.items()), columns=['Pollutant', '% от ПДК'])
fig_pct = px.bar(pct_df, x='Pollutant', y='% от ПДК', color='% от ПДК',
                 title="Средний уровень загрязнения в % от ПДК",
                 color_continuous_scale="Bluered_r")
fig_pct.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="ПДК (100%)")
st.plotly_chart(fig_pct)

# --- 2. Фактические концентрации ---
st.subheader("2. Фактические концентрации загрязнителей")
selected_pollutant = st.selectbox("Выберите загрязнитель", options=pollutants)
fig_time = px.line(filtered_df.sort_values(by='Date'), 
                   x='Date', y=selected_pollutant, color='City',
                   title=f"Изменение концентрации {selected_pollutant} по времени")
st.plotly_chart(fig_time)

# --- 3. Топ загрязнённых станций ---
st.subheader("3. Топ наиболее загрязнённых локаций (по PM2.5)")
fig_top = px.bar(city_avg_pm25, x=city_avg_pm25.values, y=city_avg_pm25.index,
                 orientation='h', title="Топ-10 локаций по среднему PM2.5",
                 labels={'y': 'Город', 'x': 'Среднее PM2.5'})
st.plotly_chart(fig_top)

# --- 4. Средние значения по станциям ---
st.subheader("4. Средние значения концентраций по локациям")
st.dataframe(averages.style.format("{:.2f}"))

# --- 5. Количество станций с превышением ---
st.subheader("5. Количество локаций с превышением хотя бы одного показателя")
st.metric(label=" ", value=stations_exceeding)

# --- Дополнительная визуализация: Карта рассеивания ---
st.subheader("Дополнительная визуализация: Корреляция PM2.5 vs PM10")
fig_scatter = px.scatter(filtered_df, x='PM2.5', y='PM10', color='City',
                         title="Корреляция между PM2.5 и PM10",
                         opacity=0.6)
st.plotly_chart(fig_scatter)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Загрузка модели
try:
    with open('hits_service.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Файл модели 'final_model.pkl' не найден. Пожалуйста, сохраните вашу модель в этот файл.")

# Определение схемы входных данных
class UserAttributes(BaseModel):
    utm_medium: str
    device_category: str
    device_os: str
    device_browser: str
    geo_city: str
    city_importance: str
    day_of_week: int
    visit_number: int
    visit_time_hour: int
    visit_time_minute: int
    device_screen_resolution_width: float
    device_screen_resolution_height: float
    device_screen_resolution_ratio: float

# Функция преобразования входных данных в числовой формат
def preprocess_input(data: UserAttributes):
    # Используем хеширование строк для преобразования в числа
    feature_vector = []

    #feature_vector.append(abs(hash(data.utm_medium) % 1000))
    #feature_vector.append(abs(hash(data.device_category) % 1000))
    #feature_vector.append(abs(hash(data.device_os) % 1000))
    #feature_vector.append(abs(hash(data.device_browser) % 1000))
    #feature_vector.append(abs(hash(data.geo_city) % 1000))
    #feature_vector.append(abs(hash(data.city_importance) % 1000))

    # Добавляем числовые признаки
    feature_vector.extend([
        data.utm_medium,
        data.device_category,
        data.device_os,
        data.device_browser,
        data.geo_city,
        data.city_importance,
        data.day_of_week,
        data.visit_number,
        data.visit_time_hour,
        data.visit_time_minute,
        data.device_screen_resolution_width,
        data.device_screen_resolution_height,
        data.device_screen_resolution_ratio
    ])

    columns = [
        'utm_medium', 'device_category', 'device_os', 'device_browser', 'geo_city', 'city_importance',
        'day_of_week', 'visit_number', 'visit_time_hour', 'visit_time_minute',
        'device_screen_resolution_width', 'device_screen_resolution_height', 'device_screen_resolution_ratio'
    ]

    return pd.DataFrame([feature_vector], columns=columns)

# API-метод для предсказания
@app.post("/predict/")
async def predict(user_data: UserAttributes):
    try:
        features = preprocess_input(user_data)
        prediction = model.predict(features)
        result = int(prediction[0])
        return {"target": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
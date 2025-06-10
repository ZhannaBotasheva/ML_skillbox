import requests

url = "http://127.0.0.1:8000/predict/"

data = {
    "utm_medium": "cpm",
    'device_category': 'mobile',
    "device_os": "Android",
    'device_browser': 'Samsung',
    'geo_city': 'Moscow',
    'city_importance': 'high',
    'day_of_week': 1,
    "visit_number": 2,
    "visit_time_hour": 10,
    "visit_time_minute": 20,
    "device_screen_resolution_width": 385,
    "device_screen_resolution_height": 854,
    "device_screen_resolution_ratio": 0.45
}

response = requests.post(url, json=data)

# Выводим ответ сервера
print(response.status_code)
print(response.json())

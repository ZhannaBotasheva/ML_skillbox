import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pickle

# Загрузка данных
def load_data():
    df = pd.read_csv('df_final.csv', low_memory=False)  # замените на путь к вашему файлу
    return df

# Предобработка данных
def preprocess_data(df):
    # Выбираем признаки и целевую переменную
    X = df[['utm_medium', 'device_category', 'device_os', 'device_browser', 'geo_city',
            'city_importance', 'day_of_week', 'visit_number', 'visit_time_hour', 'visit_time_minute',
            'device_screen_resolution_width', 'device_screen_resolution_height', 'device_screen_resolution_ratio']]
    y = df['target']

    # Категориальные признаки
    categorical_features = ['utm_medium', 'device_category', 'device_os', 'device_browser', 'geo_city', 'city_importance']
    for col in categorical_features:
        X[col] = X[col].astype('category')

    # Числовые признаки
    numeric_features = ['visit_number', 'visit_time_hour', 'visit_time_minute', 'day_of_week',
                        'device_screen_resolution_width', 'device_screen_resolution_height', 'device_screen_resolution_ratio']

    return X, y, categorical_features, numeric_features

# Создание предобработчика
def create_preprocessor(categorical_features, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ])
    return preprocessor

# Обучение модели
def train_model(X_train, y_train, preprocessor):
    model = lgb.LGBMClassifier(objective='binary', class_weight='balanced', verbose=-1)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

# Основная функция
def main():
    df = load_data()
    X, y, cat_features, num_features = preprocess_data(df)
    preprocessor = create_preprocessor(cat_features, num_features)

    # Разделение данных
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    # Обучение базовой модели
    pipeline = train_model(X_train, y_train, preprocessor)

    # Лучшие гиперпараметры (замените на результаты оптимизации, если есть)
    best_params = {
        'bagging_fraction': 0.843,
        'feature_fraction': 0.668,
        'learning_rate': 0.029,
        'max_depth': 14,
        'min_data_in_leaf': 97,
        'num_leaves': 125
    }

    # Обучение финальной модели с лучшими гиперпараметрами
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(**best_params))
    ])
    final_pipeline.fit(X_train, y_train)

    # Сохранение модели
    with open('hits_service.pkl', 'wb') as f:
        pickle.dump(final_pipeline, f)

    # Оценка на тестовой выборке
    y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
    print('Test ROC-AUC:', roc_auc_score(y_test, y_pred_proba))

if __name__ == '__main__':
    main()

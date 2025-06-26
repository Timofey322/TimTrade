#!/usr/bin/env python3
"""
Скрипт для анализа модели и понимания проблемы с признаками
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_model():
    """Анализ модели для понимания проблемы"""
    print("🔍 Анализ модели...")
    
    try:
        # Загружаем модель
        model_path = "models/improved_xgboost_BTC_USDT_latest.pkl"
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"📊 Информация о модели:")
        print(f"   Модель обучена: {model_data.get('is_trained', False)}")
        print(f"   Основная модель: {model_data.get('class_model') is not None}")
        print(f"   Регрессионная модель: {model_data.get('reg_model') is not None}")
        print(f"   Ансамбль классификаторов: {len(model_data.get('ensemble_class_models', []))}")
        print(f"   Ансамбль регрессоров: {len(model_data.get('ensemble_reg_models', []))}")
        
        # Анализируем признаки
        selected_features = model_data.get('selected_features')
        if selected_features:
            print(f"\n🎯 Выбранные признаки ({len(selected_features)}):")
            for i, feature in enumerate(selected_features[:10]):  # Показываем первые 10
                print(f"   {i+1}. {feature}")
            if len(selected_features) > 10:
                print(f"   ... и еще {len(selected_features) - 10} признаков")
        else:
            print("❌ Признаки не найдены в модели")
        
        # Анализируем параметры
        default_params = model_data.get('default_params', {})
        print(f"\n⚙️ Параметры модели:")
        for key, value in default_params.items():
            print(f"   {key}: {value}")
        
        # Тестируем предсказания на разных данных
        print(f"\n🧪 Тестирование предсказаний:")
        
        # Тест 1: Случайные данные с правильными признаками
        if selected_features:
            X_test1 = pd.DataFrame(np.random.randn(10, len(selected_features)), columns=selected_features)
            class_model = model_data.get('class_model')
            if class_model:
                pred1 = class_model.predict(X_test1)
                proba1 = class_model.predict_proba(X_test1)
                print(f"   Тест 1 (правильные признаки):")
                print(f"     Предсказания: {pred1}")
                print(f"     Уникальные классы: {np.unique(pred1)}")
                print(f"     Распределение: {np.bincount(pred1)}")
                print(f"     Средняя уверенность: {np.mean(np.max(proba1, axis=1)):.3f}")
        
        # Тест 2: Данные с feature_0, feature_1 и т.д.
        X_test2 = pd.DataFrame(np.random.randn(10, 10), columns=[f'feature_{i}' for i in range(10)])
        if class_model:
            try:
                pred2 = class_model.predict(X_test2)
                proba2 = class_model.predict_proba(X_test2)
                print(f"   Тест 2 (feature_0, feature_1, ...):")
                print(f"     Предсказания: {pred2}")
                print(f"     Уникальные классы: {np.unique(pred2)}")
                print(f"     Распределение: {np.bincount(pred2)}")
                print(f"     Средняя уверенность: {np.mean(np.max(proba2, axis=1)):.3f}")
            except Exception as e:
                print(f"   Тест 2: Ошибка - {e}")
        
        # Тест 3: Данные с реальными признаками (rsi, macd и т.д.)
        real_features = ['rsi_14', 'macd_line', 'bb_upper_20', 'sma_20', 'ema_12', 'volatility_20', 'volume', 'close', 'high', 'low']
        X_test3 = pd.DataFrame(np.random.randn(10, len(real_features)), columns=real_features)
        if class_model:
            try:
                pred3 = class_model.predict(X_test3)
                proba3 = class_model.predict_proba(X_test3)
                print(f"   Тест 3 (реальные признаки):")
                print(f"     Предсказания: {pred3}")
                print(f"     Уникальные классы: {np.unique(pred3)}")
                print(f"     Распределение: {np.bincount(pred3)}")
                print(f"     Средняя уверенность: {np.mean(np.max(proba3, axis=1)):.3f}")
            except Exception as e:
                print(f"   Тест 3: Ошибка - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка анализа модели: {e}")
        return False

if __name__ == "__main__":
    analyze_model() 
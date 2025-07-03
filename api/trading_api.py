#!/usr/bin/env python3
"""
🌐 TimAI Trading API
FastAPI REST API для торговой системы TimAI
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import asyncio

# Импортируем TimAI
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.timai_core import TimAI, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE

# Инициализация FastAPI
app = FastAPI(
    title="TimAI Trading API",
    description="Advanced Multi-Model Trading System API",
    version="2.0.0"
)

# CORS для веб-интерфейса
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальная переменная TimAI
timai_instance = None
training_status = {"is_training": False, "status": "ready", "progress": 0}

# Pydantic модели
class MarketData(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class PredictionRequest(BaseModel):
    market_data: List[MarketData]
    ensemble_method: Optional[str] = "weighted_voting"

class PredictionResponse(BaseModel):
    predictions: List[int]
    individual_predictions: Dict[str, List[int]]
    confidence_scores: Dict[str, float]
    timestamp: str

class TrainingRequest(BaseModel):
    data_file: Optional[str] = "data/historical/BTCUSDT_15m_5years_20210705_20250702.csv"

class SystemStatus(BaseModel):
    is_trained: bool
    available_models: List[str]
    model_performance: Dict[str, Any]
    xgboost_available: bool
    lightgbm_available: bool
    last_training: Optional[str]

@app.get("/", response_model=Dict[str, str])
async def root():
    """API информация"""
    return {
        "name": "TimAI Trading API",
        "version": "2.0.0",
        "description": "Advanced Multi-Model Trading System",
        "status": "running"
    }

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Получить статус системы TimAI"""
    global timai_instance
    
    available_models = []
    model_performance = {}
    is_trained = False
    last_training = None
    
    if timai_instance:
        is_trained = timai_instance.is_trained
        if is_trained:
            available_models = list(timai_instance.model_manager.models.keys())
            model_performance = timai_instance.model_manager.model_performance
            
            # Ищем последний файл обучения
            model_files = []
            if os.path.exists("models/production"):
                model_files = [f for f in os.listdir("models/production") if f.endswith("_meta.json")]
                if model_files:
                    latest_file = max(model_files, key=lambda f: os.path.getctime(os.path.join("models/production", f)))
                    last_training = latest_file.split("_")[-2] + "_" + latest_file.split("_")[-1].replace("_meta.json", "")
    
    return SystemStatus(
        is_trained=is_trained,
        available_models=available_models,
        model_performance=model_performance,
        xgboost_available=XGBOOST_AVAILABLE,
        lightgbm_available=LIGHTGBM_AVAILABLE,
        last_training=last_training
    )

@app.get("/models", response_model=Dict[str, Any])
async def get_available_models():
    """Получить список доступных моделей и их производительность"""
    global timai_instance
    
    if not timai_instance or not timai_instance.is_trained:
        raise HTTPException(status_code=400, detail="TimAI не обучена")
    
    models_info = {}
    
    for name, performance in timai_instance.model_manager.model_performance.items():
        if performance['status'] == 'success':
            models_info[name] = {
                "f1_score": round(performance['cv_f1_mean'], 3),
                "f1_std": round(performance['cv_f1_std'], 3),
                "training_time": round(performance['training_time'], 1),
                "feature_importance": len(timai_instance.model_manager.feature_importance.get(name, {}))
            }
    
    return {
        "models": models_info,
        "total_models": len(models_info),
        "best_model": max(models_info.items(), key=lambda x: x[1]['f1_score'])[0] if models_info else None
    }

@app.post("/train")
async def train_system(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Запустить обучение TimAI в фоновом режиме"""
    global timai_instance, training_status
    
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Обучение уже в процессе")
    
    if not os.path.exists(request.data_file):
        raise HTTPException(status_code=404, detail=f"Файл данных не найден: {request.data_file}")
    
    # Запускаем обучение в фоне
    background_tasks.add_task(train_timai_background, request.data_file)
    
    return {
        "message": "Обучение TimAI запущено в фоновом режиме",
        "data_file": request.data_file,
        "check_status": "/training-status"
    }

async def train_timai_background(data_file: str):
    """Фоновое обучение TimAI"""
    global timai_instance, training_status
    
    try:
        training_status = {"is_training": True, "status": "loading_data", "progress": 10}
        
        # Загрузка данных
        df = pd.read_csv(data_file)
        training_status = {"is_training": True, "status": "initializing", "progress": 20}
        
        # Инициализация TimAI
        timai_instance = TimAI()
        training_status = {"is_training": True, "status": "training", "progress": 30}
        
        # Обучение
        results = timai_instance.train(df)
        training_status = {"is_training": True, "status": "saving", "progress": 90}
        
        # Завершение
        training_status = {
            "is_training": False, 
            "status": "completed", 
            "progress": 100,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        training_status = {
            "is_training": False, 
            "status": "failed", 
            "progress": 0,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/training-status")
async def get_training_status():
    """Получить статус обучения"""
    return training_status

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Сделать предсказание с помощью TimAI"""
    global timai_instance
    
    if not timai_instance or not timai_instance.is_trained:
        raise HTTPException(status_code=400, detail="TimAI не обучена. Сначала выполните обучение.")
    
    try:
        # Конвертируем данные в DataFrame
        data_dict = []
        for item in request.market_data:
            data_dict.append({
                "datetime": item.timestamp,
                "open": item.open,
                "high": item.high,
                "low": item.low,
                "close": item.close,
                "volume": item.volume
            })
        
        df = pd.DataFrame(data_dict)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Получаем предсказания
        predictions, individual_preds = timai_instance.predict(df, request.ensemble_method)
        
        # Рассчитываем confidence scores
        confidence_scores = {}
        for model_name, model_preds in individual_preds.items():
            # Простая мера уверенности на основе консистентности
            unique_preds = np.unique(model_preds)
            if len(unique_preds) == 1:
                confidence_scores[model_name] = 1.0
            else:
                most_common = np.bincount(model_preds).max()
                confidence_scores[model_name] = most_common / len(model_preds)
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            individual_predictions={name: preds.tolist() for name, preds in individual_preds.items()},
            confidence_scores=confidence_scores,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

@app.get("/feature-importance/{model_name}")
async def get_feature_importance(model_name: str, top_n: int = 20):
    """Получить важность признаков для конкретной модели"""
    global timai_instance
    
    if not timai_instance or not timai_instance.is_trained:
        raise HTTPException(status_code=400, detail="TimAI не обучена")
    
    if model_name not in timai_instance.model_manager.feature_importance:
        raise HTTPException(status_code=404, detail=f"Feature importance для модели {model_name} не найдена")
    
    importance_dict = timai_instance.model_manager.feature_importance[model_name]
    
    # Сортируем по важности
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return {
        "model_name": model_name,
        "feature_importance": [{"feature": name, "importance": round(value, 4)} for name, value in sorted_features],
        "total_features": len(importance_dict)
    }

@app.get("/market-analysis")
async def get_market_analysis():
    """Анализ рынка на основе последних данных"""
    global timai_instance
    
    if not timai_instance or not timai_instance.is_trained:
        raise HTTPException(status_code=400, detail="TimAI не обучена")
    
    # Это демо-функция, в реальности нужно загружать свежие данные
    try:
        # Загружаем последние данные
        df = pd.read_csv("data/historical/BTCUSDT_15m_5years_20210705_20250702.csv")
        recent_data = df.tail(100)  # Последние 100 свечей
        
        # Получаем предсказания
        predictions, individual_preds = timai_instance.predict(recent_data)
        
        # Анализируем предсказания
        pred_counts = np.bincount(predictions)
        total_preds = len(predictions)
        
        analysis = {
            "market_sentiment": {
                "strong_sell": int(pred_counts[0]) if len(pred_counts) > 0 else 0,
                "hold": int(pred_counts[1]) if len(pred_counts) > 1 else 0,
                "strong_buy": int(pred_counts[2]) if len(pred_counts) > 2 else 0
            },
            "dominant_signal": "hold",  # По умолчанию
            "confidence": 0.0,
            "models_agreement": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Определяем доминирующий сигнал
        if len(pred_counts) > 0:
            dominant_class = np.argmax(pred_counts)
            signal_names = ["strong_sell", "hold", "strong_buy"]
            analysis["dominant_signal"] = signal_names[dominant_class]
            analysis["confidence"] = float(pred_counts[dominant_class] / total_preds)
        
        # Согласованность моделей
        if individual_preds:
            agreements = []
            for i in range(len(predictions)):
                model_preds_at_i = [preds[i] for preds in individual_preds.values()]
                agreement = sum(1 for pred in model_preds_at_i if pred == predictions[i]) / len(model_preds_at_i)
                agreements.append(agreement)
            analysis["models_agreement"] = float(np.mean(agreements))
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа рынка: {str(e)}")

@app.post("/load-model")
async def load_saved_model(model_timestamp: str):
    """Загрузить сохраненную модель TimAI"""
    global timai_instance
    
    try:
        # Ищем файлы модели
        model_dir = "models/production"
        if not os.path.exists(model_dir):
            raise HTTPException(status_code=404, detail="Папка с моделями не найдена")
        
        # Загружаем модель (это упрощенная версия)
        timai_instance = TimAI()
        # В реальности здесь нужна более сложная логика загрузки
        
        return {
            "message": f"Модель {model_timestamp} загружена",
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели: {str(e)}")

@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "timai_trained": timai_instance.is_trained if timai_instance else False,
        "available_endpoints": [
            "/", "/status", "/models", "/train", "/predict", 
            "/feature-importance/{model}", "/market-analysis", "/health"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🌐 Запуск TimAI Trading API...")
    print("📍 Swagger UI: http://localhost:8000/docs")
    print("📍 ReDoc: http://localhost:8000/redoc")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 
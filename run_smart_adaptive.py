#!/usr/bin/env python3
"""
Упрощенный Smart Adaptive Trading Pipeline
Использует готовые рекомендации из JSON файлов без повторной оптимизации
"""

import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import yaml

# Добавляем src в path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_collection.collector import DataCollector
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.multi_timeframe import MultiTimeframeProcessor
from ml_models.advanced_xgboost_model import AdvancedEnsembleModel

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/smart_adaptive.log', encoding='utf-8')
    ]
)

def setup_logger():
    """Настройка логгера"""
    logger = logging.getLogger(__name__)
    # Создаем директорию для логов
    Path('logs').mkdir(exist_ok=True)
    return logger

def load_config(config_path: str) -> dict:
    """Загружает конфигурацию из YAML файла"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.warning(f"Не удалось загрузить конфигурацию {config_path}: {e}")
        return {}

def load_recommendations(timeframe: str) -> dict:
    """Загружает рекомендации для указанного таймфрейма"""
    filename = f"recommendations_{timeframe}_3years.json"
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Дефолтные рекомендации если файл не найден
        return {
            "best_threshold": 0.001 if timeframe == "5m" else 0.0015 if timeframe == "15m" else 0.003,
            "top_features": ["sma_20", "rsi", "macd", "bb_upper", "bb_lower", "volume"],
            "best_params": {
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.02
            }
        }

def get_best_indicators(timeframe: str) -> list:
    """Возвращает лучшие индикаторы для таймфрейма на основе рекомендаций"""
    # Фиксированные лучшие индикаторы на основе наших исследований
    indicator_mapping = {
        "5m": ["macd", "obv", "vwap"],
        "15m": ["rsi", "obv", "vwap"], 
        "1h": ["rsi", "obv", "vwap"]
    }
    return indicator_mapping.get(timeframe, ["macd", "rsi", "bollinger"])

def main():
    """Основная функция pipeline"""
    
    # Настройка логирования
    logger = setup_logger()
    logger.info("🚀 Запуск упрощенного Smart Adaptive Pipeline")
    
    try:
        # Загрузка конфигурации (теперь с enhanced возможностями)
        config = load_config('config/smart_adaptive_config.yaml')
        
        # Параметры
        data_config = config.get('data_collection', {})
        symbols = data_config.get('symbols', ['BTC/USDT'])
        timeframes = data_config.get('timeframes', ['5m', '15m', '1h'])
        
        # === СБОР ДАННЫХ ===
        logger.info("=== СБОР ДАННЫХ ===")
        
        # Создаем конфигурацию для DataCollector
        collector_config = {
            'exchange': 'binance',
            'symbols': symbols,
            'timeframes': timeframes,
            'primary_timeframe': timeframes[0],
            'limits': {tf: 200000 for tf in timeframes}
        }
        collector = DataCollector(collector_config)
        
        all_data = {}
        for symbol in symbols:
            symbol_data = {}
            for timeframe in timeframes:
                logger.info(f"Сбор данных для {symbol} на {timeframe}")
                data = collector.fetch_ohlcv_with_history(
                    symbol=symbol,
                    timeframe=timeframe,
                    target_limit=200000  # Максимум данных
                )
                if data is not None and not data.empty:
                    symbol_data[timeframe] = data
                    logger.info(f"Собрано {len(data)} записей для {timeframe}")
                else:
                    logger.warning(f"Не удалось собрать данные для {symbol} на {timeframe}")
            
            if symbol_data:
                all_data[symbol] = symbol_data
        
        # === ПРЕДОБРАБОТКА ДАННЫХ ===
        logger.info("=== ПРЕДОБРАБОТКА ДАННЫХ ===")
        
        # Инициализация процессоров
        multi_config = {
            'enabled': True,
            'primary_timeframe': timeframes[0],
            'features_per_timeframe': {tf: [] for tf in timeframes},
            'prefixes': {tf: f"{tf}_" for tf in timeframes if tf != timeframes[0]}
        }
        multi_processor = MultiTimeframeProcessor(multi_config)
        
        # Инициализация с enhanced конфигурацией
        feature_engineer = FeatureEngineer(config)
        
        processed_data = {}
        
        for symbol in symbols:
            logger.info(f"Обработка данных для {symbol}")
            symbol_clean = symbol.replace('/', '_')
            
            if len(timeframes) > 1:
                logger.info("Используем мульти-таймфрейм обработку")
                
                # Предварительная валидация
                if not multi_processor.validate_timeframe_data(all_data[symbol]):
                    logger.error(f"Ошибка валидации данных для {symbol}")
                    continue
                
                # Обработка каждого таймфрейма с фиксированными индикаторами
                tf_data = {}
                for timeframe in timeframes:
                    logger.info(f"Обрабатываем таймфрейм {timeframe}")
                    
                    # Получаем лучшие индикаторы для этого таймфрейма
                    best_indicators = get_best_indicators(timeframe)
                    logger.info(f"✅ Используем индикаторы: {best_indicators}")
                    
                    # Горизонт предсказаний
                    horizon_map = {"5m": 12, "15m": 10, "1h": 8}
                    horizon = horizon_map.get(timeframe, 12)
                    
                    # Обработка признаков с фиксированными индикаторами
                    processed = feature_engineer.process_features(
                        data=all_data[symbol][timeframe],
                        symbol=symbol_clean,
                        timeframe=timeframe,
                        prediction_horizon=horizon,
                        adaptive_indicators_enabled=False,  # Отключаем адаптивную оптимизацию
                        fixed_indicators=best_indicators    # Используем фиксированные индикаторы
                    )
                    
                    tf_data[timeframe] = processed
                    
                    # Проверяем наличие sentiment фичей
                    sentiment_cols = [col for col in processed.columns if any(
                        keyword in col.lower() for keyword in ['sentiment', 'bybit', 'regime']
                    )]
                    
                    logger.info(f"Таймфрейм {timeframe}: {len(processed)} строк, {len(processed.columns)} колонок")
                    if sentiment_cols:
                        logger.info(f"✅ Enhanced: найдено {len(sentiment_cols)} sentiment фичей")
                        
                        # Показываем текущий market regime
                        if 'market_regime_bullish' in processed.columns:
                            regime = "🐂 БЫЧИЙ" if processed['market_regime_bullish'].iloc[-1] else \
                                    "🐻 МЕДВЕЖИЙ" if processed['market_regime_bearish'].iloc[-1] else \
                                    "😐 НЕЙТРАЛЬНЫЙ"
                            sentiment_score = processed.get('sentiment_composite_score', pd.Series([0.5])).iloc[-1]
                            logger.info(f"📊 Market Regime: {regime}, Sentiment: {sentiment_score:.3f}")
                    else:
                        logger.info("ℹ️ Sentiment фичи недоступны (используются только технические индикаторы)")
                
                processed_data[symbol] = tf_data
            else:
                # Обработка одного таймфрейма
                timeframe = timeframes[0]
                best_indicators = get_best_indicators(timeframe)
                logger.info(f"✅ Используем индикаторы: {best_indicators}")
                
                processed = feature_engineer.process_features(
                    data=all_data[symbol][timeframe],
                    symbol=symbol_clean,
                    timeframe=timeframe,
                    adaptive_indicators_enabled=False,
                    fixed_indicators=best_indicators
                )
                processed_data[symbol] = {timeframe: processed}
        
        # === ОБУЧЕНИЕ МОДЕЛЕЙ ===
        logger.info("=== ОБУЧЕНИЕ МОДЕЛЕЙ ===")
        
        models = {}
        for symbol in symbols:
            symbol_clean = symbol.replace('/', '_')
            symbol_models = {}
            
            for timeframe in timeframes:
                logger.info(f"Обучение модели для {symbol} на {timeframe}")
                
                data = processed_data[symbol][timeframe]
                if len(data) < 1000:
                    logger.warning(f"Недостаточно данных для {symbol} {timeframe}: {len(data)}")
                    continue
                
                # Загружаем параметры из конфигурации
                timeframe_config = config.get('timeframe_params', {}).get(timeframe, {})
                
                # Подготавливаем параметры модели
                xgb_params = timeframe_config.get("xgboost", {
                    "n_estimators": 300,
                    "max_depth": 8,
                    "learning_rate": 0.02,
                    "random_state": 42,
                    "n_jobs": -1
                })
                
                # Обучение
                try:
                    # Инициализация модели
                    model = AdvancedEnsembleModel(config={
                        'timeframe_params': {
                            timeframe: {
                                'xgboost': xgb_params,
                                'random_forest': timeframe_config.get('random_forest', {
                                    'n_estimators': 200,
                                    'max_depth': 10,
                                    'random_state': 42,
                                    'n_jobs': -1
                                }),
                                'lightgbm': timeframe_config.get('lightgbm', {
                                    'n_estimators': 300,
                                    'max_depth': 8,
                                    'learning_rate': 0.02,
                                    'random_state': 42,
                                    'n_jobs': -1
                                })
                            }
                        },
                        'feature_selection': {
                            'enabled': True,
                            'n_features': 20
                        }
                    })
                    
                    # Обучение модели
                    success = model.train(
                        df=data,
                        symbol=symbol_clean,
                        timeframe=timeframe
                    )
                    
                    if success:
                        symbol_models[timeframe] = model
                        logger.info(f"✅ Модель для {symbol} {timeframe} обучена успешно")
                    else:
                        logger.error(f"❌ Ошибка обучения модели для {symbol} {timeframe}")
                        continue
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка обучения модели для {symbol} {timeframe}: {e}")
                    continue
            
            if symbol_models:
                models[symbol] = symbol_models
        
        # === СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
        logger.info("=== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===")
        
        # Создаем директорию для моделей
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Сохраняем каждую модель
        for symbol, symbol_models in models.items():
            symbol_clean = symbol.replace('/', '_')
            
            for timeframe, model in symbol_models.items():
                model_path = models_dir / f"{symbol_clean}_{timeframe}_model.pkl"
                
                try:
                    model.save_model(str(model_path))
                    logger.info(f"✅ Модель сохранена: {model_path}")
                except Exception as e:
                    logger.error(f"❌ Ошибка сохранения модели {model_path}: {e}")
        
        # === ЗАПУСК БЭКТЕСТА ===
        logger.info("=== ЗАПУСК БЭКТЕСТА ===")
        
        try:
            # Запускаем продвинутый бэктест
            logger.info("Запускаем продвинутый бэктест...")
            
            # Импортируем и запускаем бэктест
            import advanced_backtest
            advanced_backtest.main()
            
            logger.info("✅ Бэктест завершен успешно!")
            
        except Exception as e:
            logger.error(f"❌ Ошибка бэктеста: {e}")
            logger.warning("Продолжаем без бэктеста...")
        
        logger.info("✅ Pipeline завершен успешно!")
        logger.info(f"Обучено моделей: {sum(len(sm) for sm in models.values())}")
        logger.info("Готово к торговле!")
        
        return {
            'models': models,
            'processed_data': processed_data
        }
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка в pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 
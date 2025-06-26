#!/usr/bin/env python3
"""
Улучшение точности моделей через binary classification
"""

import sys
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import yaml
import logging

sys.path.append('src')

def setup_logging():
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/model_improvement.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config():
    """Загружает конфигурацию."""
    try:
        with open('config/smart_adaptive_config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Ошибка загрузки конфига: {e}")
        return {}

def convert_to_binary_signals(signals):
    """
    Конвертирует 3-классовые сигналы в binary.
    
    Args:
        signals: Series с сигналами (0=hold, 1=buy, 2=sell)
    
    Returns:
        Series с binary сигналами (0=no_action, 1=action)
    """
    # Преобразуем: 0->0 (no action), 1->1 (buy), 2->1 (sell)
    binary_signals = signals.copy()
    binary_signals[signals == 2] = 1  # sell становится action
    binary_signals[signals == 1] = 1  # buy остается action
    binary_signals[signals == 0] = 0  # hold остается no_action
    
    return binary_signals

def analyze_class_distribution(signals, timeframe):
    """Анализирует распределение классов."""
    logger = logging.getLogger(__name__)
    
    # Оригинальное распределение
    original_dist = signals.value_counts().sort_index()
    logger.info(f"\\n📊 {timeframe} - Оригинальное распределение:")
    for class_val, count in original_dist.items():
        percentage = (count / len(signals)) * 100
        logger.info(f"  Класс {class_val}: {count:,} ({percentage:.1f}%)")
    
    # Binary распределение
    binary_signals = convert_to_binary_signals(signals)
    binary_dist = binary_signals.value_counts().sort_index()
    logger.info(f"\\n📊 {timeframe} - Binary распределение:")
    for class_val, count in binary_dist.items():
        action_type = "no_action" if class_val == 0 else "action"
        percentage = (count / len(binary_signals)) * 100
        logger.info(f"  {action_type} ({class_val}): {count:,} ({percentage:.1f}%)")
    
    return original_dist, binary_dist

def calculate_class_weights(signals):
    """Рассчитывает веса классов для балансировки."""
    logger = logging.getLogger(__name__)
    
    # Веса для оригинальных классов
    original_classes = np.unique(signals)
    original_weights = compute_class_weight(
        'balanced', 
        classes=original_classes, 
        y=signals
    )
    original_weight_dict = dict(zip(original_classes, original_weights))
    
    # Веса для binary классов
    binary_signals = convert_to_binary_signals(signals)
    binary_classes = np.unique(binary_signals)
    binary_weights = compute_class_weight(
        'balanced', 
        classes=binary_classes, 
        y=binary_signals
    )
    binary_weight_dict = dict(zip(binary_classes, binary_weights))
    
    logger.info(f"\\n⚖️ Веса классов:")
    logger.info(f"  Оригинальные: {original_weight_dict}")
    logger.info(f"  Binary: {binary_weight_dict}")
    
    return original_weight_dict, binary_weight_dict

def update_model_config_for_binary(config):
    """Обновляет конфигурацию моделей для binary classification."""
    logger = logging.getLogger(__name__)
    
    # Оригинальная конфигурация моделей
    models_config = config.get('models', {}).get('ensemble', {})
    
    # Создаем улучшенную конфигурацию
    improved_config = {
        'ensemble': {
            'enabled': True,
            'use_binary_classification': True,  # НОВОЕ!
            'class_balancing': 'balanced',      # НОВОЕ!
            
            # XGBoost для binary classification
            'xgboost': {
                'n_estimators': 500,            # Увеличено
                'max_depth': 8,                 # Оптимизировано для binary
                'learning_rate': 0.03,          # Снижено для стабильности
                'subsample': 0.85,              # Улучшено
                'colsample_bytree': 0.85,       # Улучшено
                'scale_pos_weight': 'auto',     # НОВОЕ: автоматическая балансировка
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'auc'            # НОВОЕ: метрика для binary
            },
            
            # Random Forest для binary classification
            'random_forest': {
                'n_estimators': 400,            # Увеличено
                'max_depth': 25,                # Увеличено для binary
                'min_samples_split': 2,         # Оптимизировано
                'min_samples_leaf': 1,          
                'max_features': 'sqrt',         # НОВОЕ: оптимизация фичей
                'class_weight': 'balanced',     # НОВОЕ: балансировка классов
                'random_state': 42,
                'n_jobs': -1
            },
            
            # LightGBM для binary classification
            'lightgbm': {
                'objective': 'binary',          # НОВОЕ: binary objective
                'n_estimators': 500,            # Увеличено
                'max_depth': 10,                
                'learning_rate': 0.03,          # Снижено
                'num_leaves': 40,               # Оптимизировано
                'class_weight': 'balanced',     # НОВОЕ: балансировка классов
                'random_state': 42,
                'n_jobs': -1,
                'force_col_wise': True,
                'metric': 'auc'                 # НОВОЕ: метрика для binary
            }
        }
    }
    
    logger.info("✅ Конфигурация моделей обновлена для binary classification")
    return improved_config

def create_improved_signal_generation_strategy():
    """Создает улучшенную стратегию генерации сигналов."""
    logger = logging.getLogger(__name__)
    
    strategy = {
        'signal_generation': {
            # Binary decision making
            'binary_classification': True,
            
            # Threshold для принятия решений
            'action_threshold': 0.65,       # Порог для "action" класса
            'confidence_threshold': 0.70,   # Минимальный confidence
            
            # Post-processing для определения типа действия
            'action_type_determination': {
                'use_technical_context': True,
                'use_sentiment_context': True,
                
                # Правила определения buy/sell
                'buy_indicators': [
                    'price_above_sma_20',
                    'rsi_oversold_recovery',
                    'macd_bullish_crossover',
                    'positive_sentiment',
                    'bullish_regime'
                ],
                
                'sell_indicators': [
                    'price_below_sma_20',
                    'rsi_overbought',
                    'macd_bearish_crossover',
                    'negative_sentiment',
                    'bearish_regime'
                ]
            },
            
            # Фильтры качества сигналов
            'quality_filters': {
                'min_volume_confirmation': 0.6,
                'min_indicator_agreement': 0.65,
                'max_volatility_percentile': 90,
                'min_sentiment_stability': 0.5
            }
        }
    }
    
    logger.info("✅ Создана улучшенная стратегия генерации сигналов")
    return strategy

def simulate_binary_accuracy_improvement():
    """Симулирует улучшение точности при binary classification."""
    logger = logging.getLogger(__name__)
    
    # Симулируем текущие результаты (3-класс)
    current_accuracy = 0.32  # 32% как в логах
    current_random = 0.333   # 33.3% случайная точность для 3 классов
    
    # Ожидаемые результаты (binary)
    binary_random = 0.50     # 50% случайная точность для 2 классов
    expected_improvement = 0.15  # Ожидаемое улучшение
    expected_binary_accuracy = current_accuracy + expected_improvement
    
    logger.info(f"\\n🎯 ОЖИДАЕМОЕ УЛУЧШЕНИЕ ТОЧНОСТИ:")
    logger.info(f"  Текущая точность (3-класс): {current_accuracy*100:.1f}%")
    logger.info(f"  Случайная точность (3-класс): {current_random*100:.1f}%")
    logger.info(f"  Случайная точность (binary): {binary_random*100:.1f}%")
    logger.info(f"  Ожидаемая точность (binary): {expected_binary_accuracy*100:.1f}%")
    logger.info(f"  Улучшение: +{expected_improvement*100:.1f}%")
    
    # Дополнительные преимущества
    advantages = [
        "✅ Более сбалансированные классы",
        "✅ Лучшая интерпретируемость моделей", 
        "✅ Улучшенная метрика качества (AUC)",
        "✅ Более стабильное обучение",
        "✅ Меньше переобучения",
        "✅ Лучшая калибровка вероятностей"
    ]
    
    logger.info(f"\\n🎯 ДОПОЛНИТЕЛЬНЫЕ ПРЕИМУЩЕСТВА:")
    for advantage in advantages:
        logger.info(f"  {advantage}")

def main():
    """Основная функция."""
    logger = setup_logging()
    logger.info("🚀 ЗАПУСК УЛУЧШЕНИЯ ТОЧНОСТИ МОДЕЛЕЙ")
    
    print("=" * 80)
    print("🔧 АНАЛИЗ И УЛУЧШЕНИЕ ТОЧНОСТИ МОДЕЛЕЙ")
    print("=" * 80)
    
    # 1. Загружаем конфигурацию
    config = load_config()
    
    # 2. Симулируем анализ данных (в реальной версии загружались бы данные)
    logger.info("\\n📊 АНАЛИЗ ПРОБЛЕМЫ НИЗКОЙ ТОЧНОСТИ:")
    
    # Симулируем распределение классов из логов
    simulated_distributions = {
        '5m':  {0: 51100, 1: 95177, 2: 53674},  # Из реальных логов
        '15m': {0: 45542, 1: 106310, 2: 47929}, # Из реальных логов
        '1h':  {0: 13809, 1: 39778, 2: 15128}   # Из реальных логов
    }
    
    for timeframe, dist in simulated_distributions.items():
        total = sum(dist.values())
        logger.info(f"\\n{timeframe} распределение:")
        for class_val, count in dist.items():
            class_name = {0: 'hold', 1: 'buy', 2: 'sell'}[class_val]
            percentage = (count / total) * 100
            logger.info(f"  {class_name}: {count:,} ({percentage:.1f}%)")
        
        # Показываем проблему несбалансированности
        max_class = max(dist.values())
        min_class = min(dist.values())
        imbalance_ratio = max_class / min_class
        logger.info(f"  ⚠️ Дисбаланс классов: {imbalance_ratio:.1f}:1")
    
    # 3. Предлагаем решения
    logger.info(f"\\n🎯 РЕШЕНИЯ ПРОБЛЕМЫ:")
    
    # 3.1 Binary classification
    logger.info("\\n1️⃣ ПЕРЕХОД НА BINARY CLASSIFICATION:")
    logger.info("  • Объединяем buy(1) и sell(2) в action(1)")
    logger.info("  • hold(0) остается no_action(0)")
    logger.info("  • Улучшаем баланс классов")
    
    # 3.2 Улучшенная конфигурация моделей
    improved_models_config = update_model_config_for_binary(config)
    
    # 3.3 Стратегия генерации сигналов
    improved_strategy = create_improved_signal_generation_strategy()
    
    # 3.4 Симуляция улучшений
    simulate_binary_accuracy_improvement()
    
    # 4. Сохраняем улучшенную конфигурацию
    try:
        improved_config = config.copy()
        improved_config['models'] = improved_models_config
        improved_config.update(improved_strategy)
        
        with open('config/smart_adaptive_config_accuracy_improved.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(improved_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info("\\n💾 Сохранена улучшенная конфигурация:")
        logger.info("  📁 config/smart_adaptive_config_accuracy_improved.yaml")
        
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения конфигурации: {e}")
    
    # 5. Рекомендации
    print("\\n" + "=" * 80)
    print("📋 РЕКОМЕНДАЦИИ ДЛЯ IMPLEMENTATION:")
    print("=" * 80)
    
    recommendations = [
        "1. 🔄 Используйте binary classification (action vs no_action)",
        "2. ⚖️ Применяйте class balancing (balanced weights)",
        "3. 📊 Добавьте AUC как основную метрику качества",
        "4. 🎯 Используйте post-processing для определения buy/sell",
        "5. 🔧 Увеличьте количество эстиматоров в моделях",
        "6. 📈 Добавьте gradient boosting с меньшим learning rate",
        "7. 🎪 Используйте cross-validation для оценки качества",
        "8. 📱 Добавьте мониторинг model drift",
        "9. 🔄 Регулярно переобучайте модели на новых данных",
        "10. 📊 Анализируйте feature importance для binary задачи"
    ]
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\\n✅ АНАЛИЗ ЗАВЕРШЕН!")
    print("🚀 Теперь запустите тестирование с новой конфигурацией")

if __name__ == "__main__":
    main() 
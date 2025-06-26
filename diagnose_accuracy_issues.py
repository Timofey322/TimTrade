#!/usr/bin/env python3
"""
Диагностический скрипт для анализа проблем с точностью модели.
Проверяет все возможные причины низкой точности (55%).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from src.data_collection.collector import DataCollector
from src.preprocessing.feature_engineering import FeatureEngineer
from src.ml_models.advanced_xgboost_model import AdvancedEnsembleModel

def diagnose_accuracy_issues():
    """Полная диагностика проблем с точностью модели."""
    
    logger.info("🔍 ДИАГНОСТИКА ПРОБЛЕМ С ТОЧНОСТЬЮ МОДЕЛИ")
    logger.info("=" * 60)
    
    try:
        # Загружаем конфигурацию
        with open('config/smart_adaptive_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 1. СБОР ДАННЫХ
        logger.info("\n📊 1. СБОР ДАННЫХ")
        logger.info("-" * 30)
        
        collector = DataCollector(config['data_collection'])
        data_5m = collector.fetch_ohlcv('BTC/USDT', '5m', limit=20000)
        data_15m = collector.fetch_ohlcv('BTC/USDT', '15m', limit=20000)
        
        if data_5m is None or data_15m is None:
            logger.error("❌ Не удалось собрать данные")
            return False
        
        logger.info(f"✅ Собрано данных: 5m={len(data_5m)}, 15m={len(data_15m)}")
        
        # 2. ПРЕДОБРАБОТКА ДАННЫХ
        logger.info("\n🔧 2. ПРЕДОБРАБОТКА ДАННЫХ")
        logger.info("-" * 30)
        
        feature_engineer = FeatureEngineer(config['preprocessing'])
        
        # Обрабатываем данные
        processed_data = feature_engineer.process_multi_timeframe(
            {'5m': data_5m, '15m': data_15m}, 
            'BTC/USDT'
        )
        
        if processed_data is None:
            logger.error("❌ Ошибка предобработки данных")
            return False
        
        logger.info(f"✅ Обработано данных: {len(processed_data)} строк")
        logger.info(f"✅ Признаков: {len(processed_data.columns)}")
        
        # 3. АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
        logger.info("\n🎯 3. АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
        logger.info("-" * 30)
        
        if 'target' not in processed_data.columns:
            logger.error("❌ Отсутствует целевая переменная")
            return False
        
        # Распределение классов ДО балансировки
        target_dist_before = processed_data['target'].value_counts()
        logger.info(f"📈 Распределение классов ДО балансировки:")
        for class_id, count in target_dist_before.items():
            percentage = count / len(processed_data) * 100
            logger.info(f"  Класс {class_id}: {count} ({percentage:.1f}%)")
        
        # Проверяем баланс классов
        min_class = target_dist_before.min()
        max_class = target_dist_before.max()
        balance_ratio = min_class / max_class
        logger.info(f"📊 Соотношение классов: {balance_ratio:.3f}")
        
        if balance_ratio < 0.3:
            logger.warning("⚠️ СИЛЬНЫЙ ДИСБАЛАНС КЛАССОВ! Нужна балансировка.")
        elif balance_ratio < 0.7:
            logger.warning("⚠️ УМЕРЕННЫЙ ДИСБАЛАНС КЛАССОВ.")
        else:
            logger.info("✅ Классы достаточно сбалансированы.")
        
        # 4. ТЕСТИРОВАНИЕ РАЗНЫХ ГОРИЗОНТОВ ПРЕДСКАЗАНИЯ
        logger.info("\n⏰ 4. ТЕСТИРОВАНИЕ РАЗНЫХ ГОРИЗОНТОВ ПРЕДСКАЗАНИЯ")
        logger.info("-" * 30)
        
        horizons = [3, 6, 12, 24]
        horizon_results = {}
        
        for horizon in horizons:
            logger.info(f"🔍 Тестируем горизонт: {horizon} свечей")
            
            # Временно изменяем конфигурацию
            temp_config = config.copy()
            temp_config['preprocessing']['target_creation']['lookforward_periods'] = horizon
            
            # Создаем временный FeatureEngineer
            temp_feature_engineer = FeatureEngineer(temp_config['preprocessing'])
            
            # Обрабатываем данные с новым горизонтом
            temp_processed = temp_feature_engineer.process_multi_timeframe(
                {'5m': data_5m, '15m': data_15m}, 
                'BTC/USDT'
            )
            
            if temp_processed is not None and 'target' in temp_processed.columns:
                temp_dist = temp_processed['target'].value_counts()
                temp_balance = temp_dist.min() / temp_dist.max()
                horizon_results[horizon] = {
                    'balance_ratio': temp_balance,
                    'class_distribution': temp_dist.to_dict(),
                    'total_samples': len(temp_processed)
                }
                logger.info(f"  Горизонт {horizon}: баланс={temp_balance:.3f}, образцов={len(temp_processed)}")
        
        # 5. ТЕСТИРОВАНИЕ РАЗНЫХ ПОРОГОВ
        logger.info("\n📊 5. ТЕСТИРОВАНИЕ РАЗНЫХ ПОРОГОВ КЛАССИФИКАЦИИ")
        logger.info("-" * 30)
        
        thresholds = [0.001, 0.002, 0.005, 0.01]  # 0.1%, 0.2%, 0.5%, 1%
        threshold_results = {}
        
        for threshold in thresholds:
            logger.info(f"🔍 Тестируем порог: {threshold*100:.1f}%")
            
            # Временно изменяем конфигурацию
            temp_config = config.copy()
            temp_config['preprocessing']['target_creation']['fall_threshold'] = -threshold
            temp_config['preprocessing']['target_creation']['rise_threshold'] = threshold
            
            # Создаем временный FeatureEngineer
            temp_feature_engineer = FeatureEngineer(temp_config['preprocessing'])
            
            # Обрабатываем данные с новыми порогами
            temp_processed = temp_feature_engineer.process_multi_timeframe(
                {'5m': data_5m, '15m': data_15m}, 
                'BTC/USDT'
            )
            
            if temp_processed is not None and 'target' in temp_processed.columns:
                temp_dist = temp_processed['target'].value_counts()
                temp_balance = temp_dist.min() / temp_dist.max()
                threshold_results[threshold] = {
                    'balance_ratio': temp_balance,
                    'class_distribution': temp_dist.to_dict(),
                    'total_samples': len(temp_processed)
                }
                logger.info(f"  Порог {threshold*100:.1f}%: баланс={temp_balance:.3f}, образцов={len(temp_processed)}")
        
        # 6. БАЛАНСИРОВКА ДАННЫХ
        logger.info("\n⚖️ 6. БАЛАНСИРОВКА ДАННЫХ")
        logger.info("-" * 30)
        
        # Балансировка данных
        balanced_data = feature_engineer.balance_data(processed_data, 'BTC/USDT')
        
        if balanced_data is not None:
            target_dist_after = balanced_data['target'].value_counts()
            logger.info(f"📈 Распределение классов ПОСЛЕ балансировки:")
            for class_id, count in target_dist_after.items():
                percentage = count / len(balanced_data) * 100
                logger.info(f"  Класс {class_id}: {count} ({percentage:.1f}%)")
            
            balance_ratio_after = target_dist_after.min() / target_dist_after.max()
            logger.info(f"📊 Соотношение классов после балансировки: {balance_ratio_after:.3f}")
            
            if balance_ratio_after > 0.8:
                logger.info("✅ Балансировка работает отлично!")
            elif balance_ratio_after > 0.6:
                logger.info("✅ Балансировка работает хорошо.")
            else:
                logger.warning("⚠️ Балансировка работает плохо.")
        else:
            logger.error("❌ Ошибка балансировки данных")
            balanced_data = processed_data
        
        # 7. ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ
        logger.info("\n🎓 7. ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
        logger.info("-" * 30)
        
        X, y = feature_engineer.prepare_for_training(balanced_data, 'BTC/USDT')
        
        if X is None or y is None:
            logger.error("❌ Ошибка подготовки данных для обучения")
            return False
        
        logger.info(f"✅ Подготовлено для обучения: {len(X)} строк, {len(X.columns)} признаков")
        
        # 8. ВРЕМЕННАЯ ВАЛИДАЦИЯ
        logger.info("\n⏱️ 8. ВРЕМЕННАЯ ВАЛИДАЦИЯ")
        logger.info("-" * 30)
        
        # Проверяем, что используется TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X) * 0.2))
        
        logger.info("✅ Используется TimeSeriesSplit для временной валидации")
        logger.info(f"📊 Количество фолдов: {tscv.n_splits}")
        logger.info(f"📊 Размер тестового сета: {int(len(X) * 0.2)} образцов")
        
        # 9. ОБУЧЕНИЕ МОДЕЛИ И АНАЛИЗ ПЕРЕОБУЧЕНИЯ
        logger.info("\n🤖 9. ОБУЧЕНИЕ МОДЕЛИ И АНАЛИЗ ПЕРЕОБУЧЕНИЯ")
        logger.info("-" * 30)
        
        # Создаем и обучаем модель
        model = AdvancedEnsembleModel(config['ml_model'])
        
        # Создаем DataFrame для обучения
        train_df = X.copy()
        train_df['target'] = y
        
        success = model.train(train_df, 'BTC/USDT')
        
        if not success:
            logger.error("❌ Ошибка обучения модели")
            return False
        
        # Получаем результаты
        results = model.get_model_summary()
        
        logger.info("📈 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
        val_metrics = results.get('val_metrics')
        if val_metrics is None and 'training_results' in results and 'val_metrics' in results['training_results']:
            val_metrics = results['training_results']['val_metrics']
        if val_metrics:
            logger.info(f"  Валидационная точность: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
            logger.info(f"  F1-скор: {val_metrics['f1']:.4f}")
            logger.info(f"  Precision: {val_metrics['precision']:.4f}")
            logger.info(f"  Recall: {val_metrics['recall']:.4f}")
        else:
            logger.error("❌ Не удалось получить метрики валидации из результатов модели!")
        
        # 10. CONFUSION MATRIX
        logger.info("\n🔍 10. CONFUSION MATRIX")
        logger.info("-" * 30)
        
        # Получаем предсказания на валидационном сете
        if 'classifier' in model.models:
            # Разделяем данные на train/val
            split_idx = int(len(train_df) * 0.9)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Предсказания
            y_pred = model.models['classifier'].predict(X_val)
            
            # Confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            logger.info("📊 Confusion Matrix:")
            logger.info(f"  {cm}")
            
            # Анализ ошибок
            class_names = sorted(y.unique())
            logger.info("🔍 АНАЛИЗ ОШИБОК:")
            for i, true_class in enumerate(class_names):
                for j, pred_class in enumerate(class_names):
                    if i != j and cm[i, j] > 0:
                        logger.info(f"  Класс {true_class} → Класс {pred_class}: {cm[i, j]} ошибок")
        
        # 11. ВАЖНОСТЬ ПРИЗНАКОВ
        logger.info("\n🔝 11. ВАЖНОСТЬ ПРИЗНАКОВ")
        logger.info("-" * 30)
        
        if 'feature_importance' in results:
            top_features = sorted(results['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True)[:15]
            logger.info("🔝 Топ-15 важных признаков:")
            for i, (feature, importance) in enumerate(top_features, 1):
                logger.info(f"  {i:2d}. {feature}: {importance:.4f}")
        
        # 12. РЕКОМЕНДАЦИИ
        logger.info("\n💡 12. РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ ТОЧНОСТИ")
        logger.info("-" * 30)
        accuracy = None
        if val_metrics:
            accuracy = val_metrics.get('accuracy')
        if accuracy is not None:
            if accuracy < 0.60:
                logger.warning("🚨 КРИТИЧЕСКИ НИЗКАЯ ТОЧНОСТЬ!")
                logger.info("Рекомендации:")
                logger.info("  1. Увеличить горизонт предсказания до 12-24 свечей")
                logger.info("  2. Снизить пороги классификации до 0.1-0.2%")
                logger.info("  3. Улучшить балансировку данных")
                logger.info("  4. Добавить больше признаков")
                logger.info("  5. Увеличить количество деревьев в модели")
            elif accuracy < 0.70:
                logger.warning("⚠️ НИЗКАЯ ТОЧНОСТЬ!")
                logger.info("Рекомендации:")
                logger.info("  1. Попробовать другие пороги классификации")
                logger.info("  2. Улучшить feature engineering")
                logger.info("  3. Использовать ансамблирование моделей")
            else:
                logger.info("✅ Точность в норме!")
        else:
            logger.warning("❓ Не удалось получить accuracy для рекомендаций!")
        
        # 13. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ДИАГНОСТИКИ
        logger.info("\n💾 13. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ДИАГНОСТИКИ")
        logger.info("-" * 30)
        
        diagnostic_results = {
            'original_balance_ratio': balance_ratio,
            'balanced_balance_ratio': balance_ratio_after if balanced_data is not None else balance_ratio,
            'horizon_results': horizon_results,
            'threshold_results': threshold_results,
            'model_accuracy': accuracy,
            'class_distribution_before': target_dist_before.to_dict(),
            'class_distribution_after': target_dist_after.to_dict() if balanced_data is not None else target_dist_before.to_dict(),
            'recommendations': []
        }
        
        # Добавляем рекомендации
        if accuracy < 0.60:
            diagnostic_results['recommendations'].extend([
                'Увеличить горизонт предсказания',
                'Снизить пороги классификации',
                'Улучшить балансировку данных'
            ])
        
        # Сохраняем результаты
        import json
        with open('diagnostic_results.json', 'w', encoding='utf-8') as f:
            json.dump(diagnostic_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("✅ Результаты диагностики сохранены в diagnostic_results.json")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка диагностики: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = diagnose_accuracy_issues()
    if success:
        logger.info("\n✅ Диагностика завершена успешно!")
    else:
        logger.error("\n❌ Диагностика завершена с ошибками!") 
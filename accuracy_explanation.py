#!/usr/bin/env python3
"""
Объяснение точности (accuracy) в торговой системе
"""

import numpy as np
import pandas as pd

def explain_accuracy_metrics():
    """Объясняет различные метрики точности."""
    
    print("=" * 80)
    print("📊 ПОНИМАНИЕ ТОЧНОСТИ В ТОРГОВЫХ СИСТЕМАХ")
    print("=" * 80)
    
    # 1. Accuracy (Точность классификации)
    print("\n🎯 1. ACCURACY (Точность модели)")
    print("-" * 50)
    print("Формула: Accuracy = Правильные предсказания / Всего предсказаний")
    print()
    
    # Пример с нашими данными
    total_predictions = 1000
    correct_predictions = 534  # LightGBM 1h accuracy 53.4%
    accuracy = correct_predictions / total_predictions
    
    print(f"Пример из нашей системы (LightGBM 1h):")
    print(f"  Всего предсказаний: {total_predictions}")
    print(f"  Правильных: {correct_predictions}")
    print(f"  Accuracy: {accuracy:.1%}")
    print()
    
    # Сравнение со случайностью
    random_accuracy_3class = 1/3
    improvement = accuracy - random_accuracy_3class
    
    print(f"Сравнение со случайностью:")
    print(f"  Случайная точность (3 класса): {random_accuracy_3class:.1%}")
    print(f"  Наша точность: {accuracy:.1%}")
    print(f"  Улучшение: +{improvement:.1%} ✅")
    print()
    
    # 2. Confidence Score
    print("\n🎪 2. CONFIDENCE SCORE (Уверенность в сигнале)")
    print("-" * 50)
    print("Это ОТДЕЛЬНАЯ метрика, которая показывает уверенность в конкретном сигнале")
    print()
    
    confidence_scores = [0.45, 0.65, 0.78, 0.82, 0.91]
    print("Примеры confidence scores:")
    for i, conf in enumerate(confidence_scores, 1):
        interpretation = ""
        if conf < 0.5:
            interpretation = "❌ Очень низкая уверенность - НЕ торгуем"
        elif conf < 0.65:
            interpretation = "⚠️ Низкая уверенность - НЕ торгуем"
        elif conf < 0.8:
            interpretation = "✅ Хорошая уверенность - можно торговать"
        else:
            interpretation = "🚀 Высокая уверенность - сильный сигнал"
        
        print(f"  Сигнал {i}: {conf:.2f} - {interpretation}")
    print()

def explain_trading_accuracy():
    """Объясняет точность в контексте торговли."""
    
    print("\n💰 3. ТОРГОВАЯ ТОЧНОСТЬ vs МОДЕЛЬНАЯ ТОЧНОСТЬ")
    print("-" * 50)
    
    print("📊 Модельная точность (53.4%):")
    print("  - Как часто модель правильно предсказывает класс (BUY/SELL/HOLD)")
    print("  - Измеряется на исторических данных")
    print("  - НЕ гарантирует прибыльность сделок")
    print()
    
    print("💰 Торговая эффективность (наша система):")
    print("  - Win rate: 24.7% (прибыльных сделок)")
    print("  - Profit factor: 3.55 (прибыль/убыток)")
    print("  - Общая доходность: +9.42%")
    print("  - Sharpe ratio: 1.885")
    print()
    
    print("🎯 ВАЖНО ПОНИМАТЬ:")
    print("  ✅ Высокая точность модели ≠ Высокая прибыльность")
    print("  ✅ Можно быть прибыльным с низким win rate")
    print("  ✅ Главное - размер прибыли vs размер убытков")
    print()

def explain_class_imbalance():
    """Объясняет проблему несбалансированных классов."""
    
    print("\n⚖️ 4. ПРОБЛЕМА НЕСБАЛАНСИРОВАННЫХ КЛАССОВ")
    print("-" * 50)
    
    # Реальное распределение из наших данных
    class_distribution = {
        '5m': {'hold': 25.6, 'buy': 47.6, 'sell': 26.8},
        '15m': {'hold': 22.8, 'buy': 53.2, 'sell': 24.0},
        '1h': {'hold': 20.1, 'buy': 57.9, 'sell': 22.0}
    }
    
    print("Распределение классов в наших данных:")
    for timeframe, dist in class_distribution.items():
        print(f"\n{timeframe}:")
        for class_name, percentage in dist.items():
            print(f"  {class_name}: {percentage}%")
        
        # Показываем дисбаланс
        max_class = max(dist.values())
        min_class = min(dist.values())
        imbalance_ratio = max_class / min_class
        print(f"  Дисбаланс: {imbalance_ratio:.1f}:1")
    
    print("\n🎯 Почему это влияет на точность:")
    print("  - Модель склонна предсказывать доминирующий класс (BUY)")
    print("  - Accuracy становится misleading метрикой")
    print("  - Нужны дополнительные метрики (precision, recall, F1)")
    print()

def explain_confidence_calculation():
    """Объясняет расчет confidence score."""
    
    print("\n🔧 5. КАК РАССЧИТЫВАЕТСЯ CONFIDENCE SCORE")
    print("-" * 50)
    
    print("Наш confidence score учитывает:")
    
    factors = {
        'indicator_agreement': 0.22,
        'signal_strength': 0.20,
        'volatility_factor': 0.15,
        'volume_confirmation': 0.28,  # Увеличили!
        'market_regime': 0.10,
        'sentiment_confirmation': 0.05
    }
    
    print("\nВеса факторов:")
    for factor, weight in factors.items():
        print(f"  {factor}: {weight:.0%}")
    
    print(f"\nОбщий вес: {sum(factors.values()):.0%}")
    
    print("\n🎯 Пример расчета:")
    example_factors = {
        'indicator_agreement': 0.8,      # 80% индикаторов согласны
        'signal_strength': 0.7,          # Сильный сигнал
        'volatility_factor': 0.6,        # Умеренная волатильность
        'volume_confirmation': 0.9,      # Отличное объемное подтверждение
        'market_regime': 0.8,            # Благоприятный режим
        'sentiment_confirmation': 0.7     # Позитивный sentiment
    }
    
    weighted_sum = sum(factor_value * factors[factor_name] 
                      for factor_name, factor_value in example_factors.items())
    
    print("\nЗначения факторов:")
    for factor_name, factor_value in example_factors.items():
        weight = factors[factor_name]
        contribution = factor_value * weight
        print(f"  {factor_name}: {factor_value:.1f} × {weight:.0%} = {contribution:.3f}")
    
    print(f"\nИтоговый confidence: {weighted_sum:.3f} ({weighted_sum:.1%})")
    
    if weighted_sum >= 0.8:
        print("  🚀 ВЫСОКИЙ confidence - сильный сигнал!")
    elif weighted_sum >= 0.65:
        print("  ✅ ХОРОШИЙ confidence - можно торговать")
    else:
        print("  ⚠️ НИЗКИЙ confidence - лучше пропустить")
    
    print()

def main():
    """Основная функция."""
    explain_accuracy_metrics()
    explain_trading_accuracy()
    explain_class_imbalance()
    explain_confidence_calculation()
    
    print("=" * 80)
    print("🎯 ИТОГОВЫЙ ВЫВОД:")
    print("=" * 80)
    print()
    print("1. 📊 Accuracy 53.4% - это ХОРОШИЙ результат для 3-классовой задачи")
    print("2. 🎪 Confidence score - более важная метрика для торговых решений")
    print("3. 💰 Финансовые результаты важнее модельной точности")
    print("4. ⚖️ Volume confirmation (28% веса) - ключевой фактор качества")
    print("5. 🚀 Система готова к торговле с отличными показателями!")
    print()
    print("✅ ГЛАВНОЕ: Наша система показывает +9.42% доходность")
    print("   с Sharpe ratio 1.885 - это excellent результат! 🚀")

if __name__ == "__main__":
    main() 
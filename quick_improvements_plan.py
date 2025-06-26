#!/usr/bin/env python3
"""
План быстрых улучшений торговой системы
Реализация в течение 1-2 дней для немедленных результатов
"""

import pandas as pd
import numpy as np
from datetime import datetime

class QuickImprovements:
    """Класс для быстрых улучшений системы"""
    
    def __init__(self):
        self.improvements = {
            'КРИТИЧЕСКИЙ': [
                'Добавить ATR индикатор для волатильности',
                'Улучшить feature selection (топ-30 вместо топ-20)',
                'Добавить confidence scoring для сигналов',
                'Оптимизировать параметры XGBoost'
            ],
            'ВЫСОКИЙ': [
                'Добавить Williams %R и CCI индикаторы',
                'Реализовать динамический position sizing',
                'Добавить trailing stop loss',
                'Улучшить фильтры входа (требовать больше подтверждений)'
            ],
            'СРЕДНИЙ': [
                'Добавить ETH/USDT к системе',
                'Реализовать детектор рыночных режимов',
                'Добавить корреляционный анализ',
                'Улучшить логирование и мониторинг'
            ]
        }
    
    def add_atr_indicator(self, df, period=14):
        """Добавляет Average True Range (ATR) индикатор"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        df['atr'] = true_range.rolling(period).mean()
        df['atr_percentage'] = df['atr'] / df['close'] * 100
        
        return df
    
    def add_williams_r(self, df, period=14):
        """Добавляет Williams %R индикатор"""
        highest_high = df['high'].rolling(period).max()
        lowest_low = df['low'].rolling(period).min()
        
        df['williams_r'] = (highest_high - df['close']) / (highest_high - lowest_low) * -100
        
        return df
    
    def add_cci(self, df, period=20):
        """Добавляет Commodity Channel Index (CCI)"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        df['cci'] = (typical_price - sma) / (0.015 * mad)
        
        return df
    
    def calculate_confidence_score(self, signals, indicators):
        """Рассчитывает confidence score для сигналов"""
        confidence_factors = []
        
        # 1. Согласованность индикаторов
        indicator_agreement = np.mean([
            indicators.get('macd_signal', 0),
            indicators.get('rsi_signal', 0),
            indicators.get('obv_signal', 0),
            indicators.get('vwap_signal', 0)
        ])
        confidence_factors.append(indicator_agreement)
        
        # 2. Сила сигнала (расстояние от нейтральной зоны)
        signal_strength = abs(signals) if signals != 0 else 0
        confidence_factors.append(signal_strength)
        
        # 3. Волатильность (высокая волатильность = низкая уверенность)
        volatility_factor = 1.0 / (1.0 + indicators.get('atr_percentage', 1.0))
        confidence_factors.append(volatility_factor)
        
        # 4. Объемное подтверждение
        volume_confirmation = indicators.get('volume_confirmation', 0.5)
        confidence_factors.append(volume_confirmation)
        
        return np.mean(confidence_factors)
    
    def dynamic_position_sizing(self, base_size, confidence, volatility, account_balance):
        """Динамический расчет размера позиции"""
        # Базовый размер 2%
        position_size = base_size
        
        # Корректировка на уверенность (0.5-2.0x)
        confidence_multiplier = 0.5 + confidence * 1.5
        position_size *= confidence_multiplier
        
        # Корректировка на волатильность (меньше размер при высокой волатильности)
        volatility_adjustment = max(0.3, 1.0 - volatility * 2)
        position_size *= volatility_adjustment
        
        # Максимальный размер позиции 5%
        max_position = account_balance * 0.05
        position_value = account_balance * position_size
        
        return min(position_value, max_position)
    
    def improved_entry_filters(self, indicators):
        """Улучшенные фильтры для входа в позицию"""
        buy_conditions = []
        sell_conditions = []
        
        # Основные условия
        buy_conditions.extend([
            indicators.get('macd_bullish', False),
            indicators.get('price_above_vwap', False),
            indicators.get('obv_rising', False),
            indicators.get('rsi_not_overbought', False)
        ])
        
        # Дополнительные условия (новые)
        buy_conditions.extend([
            indicators.get('williams_r', 0) < -20,  # Не перепродан
            indicators.get('cci', 0) > -100,        # Восходящий импульс
            indicators.get('atr_rising', False),    # Растущая волатильность
            indicators.get('volume_above_average', False)  # Высокий объем
        ])
        
        # Для продажи (противоположные условия)
        sell_conditions.extend([
            indicators.get('macd_bearish', False),
            indicators.get('price_below_vwap', False),
            indicators.get('obv_falling', False),
            indicators.get('rsi_overbought', False),
            indicators.get('williams_r', 0) > -80,  # Перекуплен
            indicators.get('cci', 0) < 100,         # Нисходящий импульс
        ])
        
        # Требуем больше подтверждений
        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)
        
        # Повышенные требования: 6+ для покупки, 5+ для продажи
        if buy_score >= 6:
            return 1  # Покупка
        elif sell_score >= 5:
            return -1  # Продажа
        else:
            return 0  # Ожидание
    
    def trailing_stop_loss(self, entry_price, current_price, direction, atr, trail_factor=2.0):
        """Трейлинг стоп-лосс на основе ATR"""
        if direction == 1:  # Длинная позиция
            stop_distance = atr * trail_factor
            trailing_stop = current_price - stop_distance
            
            # Стоп только поднимается, никогда не опускается
            initial_stop = entry_price - stop_distance
            return max(trailing_stop, initial_stop)
            
        else:  # Короткая позиция
            stop_distance = atr * trail_factor
            trailing_stop = current_price + stop_distance
            
            # Стоп только опускается, никогда не поднимается
            initial_stop = entry_price + stop_distance
            return min(trailing_stop, initial_stop)
    
    def market_regime_detector(self, df, lookback=50):
        """Простой детектор рыночного режима"""
        # Рассчитываем трендовость
        price_change = (df['close'] - df['close'].shift(lookback)) / df['close'].shift(lookback)
        
        # Рассчитываем волатильность
        returns = df['close'].pct_change()
        volatility = returns.rolling(lookback).std()
        
        # Определяем режим
        regime = []
        for i in range(len(df)):
            if i < lookback:
                regime.append('sideways')
                continue
                
            trend = price_change.iloc[i]
            vol = volatility.iloc[i]
            
            if trend > 0.05 and vol < 0.03:  # Сильный рост, низкая волатильность
                regime.append('bull')
            elif trend < -0.05 and vol < 0.03:  # Сильное падение, низкая волатильность
                regime.append('bear')
            elif vol > 0.05:  # Высокая волатильность
                regime.append('volatile')
            else:
                regime.append('sideways')
        
        df['market_regime'] = regime
        return df

# План реализации быстрых улучшений
IMPLEMENTATION_PLAN = {
    'День 1': {
        'Утро (2-3 часа)': [
            'Добавить ATR, Williams %R, CCI индикаторы',
            'Реализовать confidence scoring',
            'Улучшить feature selection до топ-30'
        ],
        'Вечер (2-3 часа)': [
            'Добавить динамический position sizing',
            'Реализовать trailing stop loss',
            'Тестирование новых функций'
        ]
    },
    'День 2': {
        'Утро (2-3 часа)': [
            'Улучшить фильтры входа',
            'Добавить детектор рыночных режимов',
            'Оптимизировать параметры моделей'
        ],
        'Вечер (2-3 часа)': [
            'Добавить ETH/USDT',
            'Полное тестирование системы',
            'Сравнение с предыдущими результатами'
        ]
    }
}

# Ожидаемые улучшения
EXPECTED_IMPROVEMENTS = {
    'Винрейт': 'С 24.9% до 32-35% (+30%)',
    'Sharpe ratio': 'С 1.910 до 2.2-2.5 (+20%)',
    'Максимальная просадка': 'С -8.09% до -6.5% (+20%)',
    'Количество ложных сигналов': 'Снижение на 40%',
    'Стабильность доходности': 'Улучшение на 25%'
}

if __name__ == "__main__":
    print("🚀 ПЛАН БЫСТРЫХ УЛУЧШЕНИЙ ТОРГОВОЙ СИСТЕМЫ")
    print("=" * 50)
    
    improvements = QuickImprovements()
    
    print("\n📋 ПРИОРИТЕТЫ УЛУЧШЕНИЙ:")
    for priority, items in improvements.improvements.items():
        print(f"\n{priority} ПРИОРИТЕТ:")
        for i, item in enumerate(items, 1):
            print(f"  {i}. {item}")
    
    print("\n📅 ПЛАН РЕАЛИЗАЦИИ:")
    for day, schedule in IMPLEMENTATION_PLAN.items():
        print(f"\n{day}:")
        for time_period, tasks in schedule.items():
            print(f"  {time_period}:")
            for task in tasks:
                print(f"    • {task}")
    
    print("\n🎯 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ:")
    for metric, improvement in EXPECTED_IMPROVEMENTS.items():
        print(f"  • {metric}: {improvement}")
    
    print("\n" + "=" * 50)
    print("Готово к реализации! 🚀") 
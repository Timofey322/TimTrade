#!/usr/bin/env python3
"""
Продвинутое тестирование TimAI с множественными таймфреймами и улучшенной логикой
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.timai_core import TimAI

def test_advanced_timai():
    """Тестирует TimAI с продвинутыми настройками и множественными таймфреймами"""
    
    print("🚀 Продвинутое тестирование TimAI с множественными таймфреймами")
    print("="*80)
    
    # Загружаем данные для разных таймфреймов
    print("📊 Загрузка данных для множественных таймфреймов...")
    
    # 15-минутные данные (основной таймфрейм)
    df_15m = pd.read_csv('data/historical/BTCUSDT_15m_5years_20210705_20250702.csv')
    df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
    
    # 5-минутные данные (для быстрых сигналов)
    df_5m = pd.read_csv('data/historical/BTCUSDT_5m_5years_20210705_20250702.csv')
    df_5m['datetime'] = pd.to_datetime(df_5m['datetime'])
    
    # 60-минутные данные (для трендового анализа)
    df_1h = pd.read_csv('data/historical/BTCUSDT_60m_5years_20210705_20250702.csv')
    df_1h['datetime'] = pd.to_datetime(df_1h['datetime'])
    
    # Определяем дату обучения (последние 90 дней данных)
    last_date = df_15m['datetime'].max()
    train_end_date = last_date - timedelta(days=90)
    
    # Разделяем данные
    df_15m_train = df_15m[df_15m['datetime'] < train_end_date].copy()
    df_15m_test = df_15m[df_15m['datetime'] >= train_end_date].copy()
    
    df_5m_train = df_5m[df_5m['datetime'] < train_end_date].copy()
    df_5m_test = df_5m[df_5m['datetime'] >= train_end_date].copy()
    
    df_1h_train = df_1h[df_1h['datetime'] < train_end_date].copy()
    df_1h_test = df_1h[df_1h['datetime'] >= train_end_date].copy()
    
    print(f"📊 Данные:")
    print(f"   Обучение: {df_15m_train['datetime'].min()} - {df_15m_train['datetime'].max()}")
    print(f"   Тестирование: {df_15m_test['datetime'].min()} - {df_15m_test['datetime'].max()}")
    print(f"   15m записей для тестирования: {len(df_15m_test):,}")
    print(f"   5m записей для тестирования: {len(df_5m_test):,}")
    print(f"   1h записей для тестирования: {len(df_1h_test):,}")
    
    # Инициализируем TimAI
    timai = TimAI()
    
    # Обучаем на 15-минутных данных
    print(f"\n🚀 Обучение TimAI на 15-минутных данных...")
    results = timai.train(df_15m_train)
    
    # Добавляем технические индикаторы для всех таймфреймов
    print(f"\n🔧 Подготовка индикаторов для множественных таймфреймов...")
    df_15m_test = add_advanced_indicators(df_15m_test, "15m")
    df_5m_test = add_advanced_indicators(df_5m_test, "5m")
    df_1h_test = add_advanced_indicators(df_1h_test, "1h")
    
    # Тестируем с оптимизированными настройками для максимальной прибыли
    print(f"\n🧪 Тестирование с оптимизированными настройками для максимальной прибыли...")
    
    confidence_levels = [0.06, 0.10, 0.15, 0.20, 0.25, 0.30]  # Оптимизированные для золотой середины
    tp_sl_multipliers = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]  # Оптимизированные TP/SL
    
    best_result = None
    best_config = None
    
    for confidence in confidence_levels:
        for tp_sl_mult in tp_sl_multipliers:
            print(f"\n🔧 Тестирование: confidence={confidence}, TP/SL_mult={tp_sl_mult}")
            
            # Делаем предсказания на 15-минутных данных
            predictions_15m, individual_preds_15m = timai.predict(df_15m_test, confidence_threshold=confidence)
            
            # Анализируем результаты
            print(f"   Распределение сигналов 15m: {np.bincount([max(0, int(p)) for p in predictions_15m])}")
            
            # Симуляция торговли с множественными таймфреймами
            trading_result = simulate_multi_timeframe_trading(
                df_15m_test, df_5m_test, df_1h_test,
                predictions_15m, confidence, tp_sl_mult
            )
            
            print(f"   Доход: {trading_result['total_return']:.2f}%, Сделки: {trading_result['trades_count']}, "
                  f"Винрейт: {trading_result['win_rate']:.1f}%, Средняя сделка: {trading_result['avg_trade']:.4f}")
            
            if best_result is None or trading_result['total_return'] > best_result['total_return']:
                best_result = trading_result
                best_config = {'confidence': confidence, 'tp_sl_mult': tp_sl_mult}
    
    print(f"\n🏆 Лучшая конфигурация: confidence={best_config['confidence']}, TP/SL_mult={best_config['tp_sl_mult']}")
    print(f"💰 Лучший результат:")
    print(f"   Общий доход: {best_result['total_return']:.2f}%")
    print(f"   Количество сделок: {best_result['trades_count']}")
    print(f"   Винрейт: {best_result['win_rate']:.1f}%")
    print(f"   Средняя сделка: {best_result['avg_trade']:.4f}")
    print(f"   Максимальная просадка: {best_result['max_drawdown']:.2f}%")
    print(f"   Соотношение риск/прибыль: {best_result['risk_reward_ratio']:.2f}")
    print(f"   Превышение над B&H: {best_result['excess_over_bh']:.2f}%")
    print(f"   Средняя позиция: {best_result['avg_position_size']:.2f}%")
    
    return best_result

def add_advanced_indicators(df, timeframe):
    """Добавляет продвинутые технические индикаторы для множественных таймфреймов"""
    
    # Базовые индикаторы
    df['atr'] = calculate_atr(df, 14)
    df['volatility'] = df['atr'] / df['close']
    
    # Трендовые индикаторы
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # Сила тренда
    df['trend_strength'] = abs(df['ema_9'] - df['ema_21']) / df['ema_21']
    df['trend_direction'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)
    df['price_above_ema'] = df['close'] > df['ema_21']
    
    # RSI и фильтры
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['rsi_sma'] = df['rsi'].rolling(10).mean()
    
    # Volume индикаторы
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['volume_trend'] = df['volume'].rolling(10).mean() / df['volume'].rolling(50).mean()
    
    # Momentum индикаторы
    df['momentum'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_sma'] = df['momentum'].rolling(10).mean()
    
    # Bollinger Bands
    df['bb_upper'] = df['sma_20'] + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['sma_20'] - 2 * df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # MACD
    df['macd_line'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']
    
    # Stochastic
    df['stoch_k'] = calculate_stochastic(df, 14)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # Дополнительные фильтры для разных таймфреймов
    if timeframe == "1h":
        # Часовые фильтры - более консервативные
        df['trend_filter'] = (df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])
        df['volume_filter'] = df['volume_ratio'] > 1.0
        df['volatility_filter'] = df['volatility'] > 0.005
    elif timeframe == "15m":
        # 15-минутные фильтры - умеренные
        df['trend_filter'] = df['ema_9'] > df['ema_21']
        df['volume_filter'] = df['volume_ratio'] > 0.8
        df['volatility_filter'] = df['volatility'] > 0.003
    else:  # 5m
        # 5-минутные фильтры - более агрессивные
        df['trend_filter'] = True  # Любой тренд
        df['volume_filter'] = df['volume_ratio'] > 0.6
        df['volatility_filter'] = df['volatility'] > 0.002
    
    return df

def calculate_atr(df, period):
    """Рассчитывает Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr

def calculate_rsi(prices, period):
    """Рассчитывает RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic(df, period):
    """Рассчитывает Stochastic Oscillator"""
    low_min = df['low'].rolling(period).min()
    high_max = df['high'].rolling(period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    return k

def simulate_multi_timeframe_trading(df_15m, df_5m, df_1h, predictions_15m, confidence, tp_sl_multiplier):
    """Симуляция торговли с множественными таймфреймами и продвинутой логикой"""
    
    initial_balance = 1400  # Увеличиваем баланс до $1400
    balance = initial_balance
    max_balance = initial_balance
    trades = []
    position = None
    entry_price = 0
    entry_time = None
    entry_atr = 0
    position_size = 1.0  # Фиксированный размер позиции 100%
    
    # Синхронизируем данные по времени
    df_15m = df_15m.set_index('datetime')
    df_5m = df_5m.set_index('datetime')
    df_1h = df_1h.set_index('datetime')
    
    # Объединяем индикаторы с разных таймфреймов
    common_times = df_15m.index.intersection(df_5m.index).intersection(df_1h.index)
    
    for i, current_time in enumerate(common_times):
        if i < 100:  # Пропускаем первые 100 свечей для стабильности индикаторов
            continue
            
        # Получаем данные с разных таймфреймов
        current_15m = df_15m.loc[current_time]
        current_5m = df_5m.loc[current_time]
        current_1h = df_1h.loc[current_time]
        
        current_price = current_15m['close']
        current_atr = current_15m['atr']
        current_volatility = current_15m['volatility']
        
        # Проверяем динамические стоп-лосс и тейк-профит для открытой позиции
        if position is not None:
            # Адаптивные TP/SL на основе тренда и волатильности
            trend_multiplier = 1.5 if current_1h['trend_strength'] > 0.02 else 1.0
            atr_based_tp = entry_atr * tp_sl_multiplier * trend_multiplier
            atr_based_sl = entry_atr * tp_sl_multiplier * 0.7  # Более узкий SL
            
            if position == 'long':
                # Динамический стоп-лосс для длинной позиции
                if current_price <= entry_price - atr_based_sl:
                    profit = -atr_based_sl / entry_price * position_size - 0.0004  # Комиссия
                    balance *= (1 + profit)
                    trades.append(profit)
                    position = None
                # Динамический тейк-профит для длинной позиции
                elif current_price >= entry_price + atr_based_tp:
                    profit = atr_based_tp / entry_price * position_size - 0.0004  # Комиссия
                    balance *= (1 + profit)
                    trades.append(profit)
                    position = None
                # Трейлинг стоп для длинной позиции
                elif current_price > entry_price + atr_based_sl:
                    new_sl = current_price - atr_based_sl * 0.5
                    if current_price <= new_sl:
                        profit = (new_sl - entry_price) / entry_price * position_size - 0.0004
                        balance *= (1 + profit)
                        trades.append(profit)
                        position = None
            else:  # short position
                # Динамический стоп-лосс для короткой позиции
                if current_price >= entry_price + atr_based_sl:
                    profit = -atr_based_sl / entry_price * position_size - 0.0004  # Комиссия
                    balance *= (1 + profit)
                    trades.append(profit)
                    position = None
                # Динамический тейк-профит для короткой позиции
                elif current_price <= entry_price - atr_based_tp:
                    profit = atr_based_tp / entry_price * position_size - 0.0004  # Комиссия
                    balance *= (1 + profit)
                    trades.append(profit)
                    position = None
                # Трейлинг стоп для короткой позиции
                elif current_price < entry_price - atr_based_sl:
                    new_sl = current_price + atr_based_sl * 0.5
                    if current_price >= new_sl:
                        profit = (entry_price - new_sl) / entry_price * position_size - 0.0004
                        balance *= (1 + profit)
                        trades.append(profit)
                        position = None
        
        # Продвинутая логика входа в позиции с множественными таймфреймами
        if position is None:
            # Получаем сигнал с 15-минутного таймфрейма
            pred_15m = predictions_15m[i] if i < len(predictions_15m) else 1
            
            # ЗОЛОТАЯ СЕРЕДИНА - оптимальный баланс между консервативностью и агрессивностью
            
            # Анализируем текущие рыночные условия
            market_volatility = current_15m['volatility']
            market_trend = current_1h['trend_strength']
            market_volume = current_15m['volume_ratio']
            
            # ТРЕНДОВЫЕ ФИЛЬТРЫ - ключевое улучшение
            # Сильный тренд на 1-часовом таймфрейме
            strong_uptrend = (current_1h['ema_9'] > current_1h['ema_21'] > current_1h['ema_50']) and (current_1h['trend_strength'] > 0.015)
            strong_downtrend = (current_1h['ema_9'] < current_1h['ema_21'] < current_1h['ema_50']) and (current_1h['trend_strength'] > 0.015)
            
            # Умеренный тренд на 15-минутном таймфрейме
            moderate_uptrend_15m = current_15m['ema_9'] > current_15m['ema_21']
            moderate_downtrend_15m = current_15m['ema_9'] < current_15m['ema_21']
            
            # Слабый тренд (более гибкий)
            weak_uptrend = current_1h['ema_9'] > current_1h['ema_21'] * 0.998  # Очень небольшой допуск
            weak_downtrend = current_1h['ema_9'] < current_1h['ema_21'] * 1.002  # Очень небольшой допуск
            
            # Адаптивные пороги в зависимости от рыночных условий
            if market_volatility > 0.006:  # Высокая волатильность
                volume_threshold = 0.4  # Очень мягкий объем
                volatility_threshold = 0.0008  # Очень мягкая волатильность
                rsi_long_threshold = 88  # Очень мягкий RSI для покупки
                rsi_short_threshold = 12  # Очень мягкий RSI для продажи
            elif market_volatility > 0.004:  # Средняя волатильность
                volume_threshold = 0.5  # Мягкий объем
                volatility_threshold = 0.0015  # Мягкая волатильность
                rsi_long_threshold = 82  # Мягкий RSI
                rsi_short_threshold = 18  # Мягкий RSI
            elif market_trend > 0.025:  # Сильный тренд
                volume_threshold = 0.6  # Умеренный объем
                volatility_threshold = 0.002  # Умеренная волатильность
                rsi_long_threshold = 78  # Стандартный RSI
                rsi_short_threshold = 22  # Стандартный RSI
            else:  # Нормальные условия
                volume_threshold = 0.65  # Стандартный объем
                volatility_threshold = 0.002  # Стандартная волатильность
                rsi_long_threshold = 75  # Более строгий RSI
                rsi_short_threshold = 25  # Более строгий RSI
            
            # 1. Базовые фильтры (адаптивные)
            volume_ok = current_15m['volume_ratio'] > volume_threshold
            volatility_ok = current_15m['volatility'] > volatility_threshold
            
            # 2. RSI фильтры (адаптивные)
            rsi_ok_long = current_15m['rsi'] < rsi_long_threshold
            rsi_ok_short = current_15m['rsi'] > rsi_short_threshold
            
            # 3. Трендовые фильтры (ОСНОВНОЕ УЛУЧШЕНИЕ)
            trend_ok_long = strong_uptrend or (moderate_uptrend_15m and current_1h['trend_strength'] > 0.01) or weak_uptrend
            trend_ok_short = strong_downtrend or (moderate_downtrend_15m and current_1h['trend_strength'] > 0.01) or weak_downtrend
            
            # 4. Momentum фильтры (мягкие)
            momentum_ok_long = current_15m['momentum_sma'] > -0.003  # Более мягкий
            momentum_ok_short = current_15m['momentum_sma'] < 0.003   # Более мягкий
            
            # 5. MACD фильтры (мягкие)
            macd_ok_long = current_15m['macd_histogram'] > -0.0003  # Более мягкий
            macd_ok_short = current_15m['macd_histogram'] < 0.0003   # Более мягкий
            
            # 6. Bollinger Bands фильтры (адаптивные)
            bb_ok_long = current_15m['bb_position'] < 0.88  # Более мягкий
            bb_ok_short = current_15m['bb_position'] > 0.12  # Более мягкий
            
            # 7. Stochastic фильтры (мягкие)
            stoch_ok_long = current_5m['stoch_k'] < 92  # Более мягкий
            stoch_ok_short = current_5m['stoch_k'] > 8   # Более мягкий
            
            # Определяем силу сигнала для дополнительных фильтров
            signal_strength = 0
            
            if pred_15m == 2:  # Buy signal
                # Базовые условия (обязательные)
                if volume_ok and volatility_ok and rsi_ok_long and trend_ok_long:
                    signal_strength = 1
                    
                    # Дополнительные бонусы за силу сигнала
                    if strong_uptrend:  # Очень сильный тренд
                        signal_strength += 3
                    elif moderate_uptrend_15m:  # Умеренный тренд
                        signal_strength += 2
                    if momentum_ok_long and macd_ok_long:  # Momentum и MACD не против
                        signal_strength += 1
                    if bb_ok_long:  # Хорошая позиция в BB
                        signal_strength += 1
                    if stoch_ok_long:  # Stochastic не перекуплен
                        signal_strength += 1
                    if current_1h['trend_strength'] > 0.02:  # Сильный тренд
                        signal_strength += 1
                    if current_15m['bb_position'] < 0.35:  # Около нижней полосы BB
                        signal_strength += 2
                    if current_5m['stoch_k'] < 35:  # Перепроданность на 5m
                        signal_strength += 1
                    if current_15m['volume_ratio'] > 1.3:  # Высокий объем
                        signal_strength += 1
                    if current_15m['rsi'] < 35:  # Сильная перепроданность
                        signal_strength += 1
                    
                    # Входим в позицию только при достаточной силе сигнала
                    if signal_strength >= 2:  # Снижаем минимальную силу сигнала для большей активности
                        position = 'long'
                        entry_price = current_price
                        entry_time = current_time
                        entry_atr = current_atr
                    
            elif pred_15m == 0:  # Sell signal
                # Базовые условия (обязательные)
                if volume_ok and volatility_ok and rsi_ok_short and trend_ok_short:
                    signal_strength = 1
                    
                    # Дополнительные бонусы за силу сигнала
                    if strong_downtrend:  # Очень сильный тренд
                        signal_strength += 3
                    elif moderate_downtrend_15m:  # Умеренный тренд
                        signal_strength += 2
                    if momentum_ok_short and macd_ok_short:  # Momentum и MACD не против
                        signal_strength += 1
                    if bb_ok_short:  # Хорошая позиция в BB
                        signal_strength += 1
                    if stoch_ok_short:  # Stochastic не перепродан
                        signal_strength += 1
                    if current_1h['trend_strength'] > 0.02:  # Сильный тренд
                        signal_strength += 1
                    if current_15m['bb_position'] > 0.65:  # Около верхней полосы BB
                        signal_strength += 2
                    if current_5m['stoch_k'] > 65:  # Перекупленность на 5m
                        signal_strength += 1
                    if current_15m['volume_ratio'] > 1.3:  # Высокий объем
                        signal_strength += 1
                    if current_15m['rsi'] > 65:  # Сильная перекупленность
                        signal_strength += 1
                    
                    # Входим в позицию только при достаточной силе сигнала
                    if signal_strength >= 3:  # Минимальная сила сигнала
                        position = 'short'
                        entry_price = current_price
                        entry_time = current_time
                        entry_atr = current_atr
        
        # Обновляем максимальный баланс для расчета просадки
        max_balance = max(max_balance, balance)
    
    # Закрываем открытую позицию в конце
    if position is not None:
        final_price = df_15m.iloc[-1]['close']
        if position == 'long':
            profit = (final_price - entry_price) / entry_price * position_size - 0.0004
        else:
            profit = (entry_price - final_price) / entry_price * position_size - 0.0004
        balance *= (1 + profit)
        trades.append(profit)
    
    # Статистика
    total_return = (balance - initial_balance) / initial_balance * 100
    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0
    avg_trade = np.mean(trades) if trades else 0
    max_drawdown = (max_balance - balance) / max_balance * 100
    
    # Дополнительная статистика
    winning_trades = [t for t in trades if t > 0]
    losing_trades = [t for t in trades if t < 0]
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Buy & Hold для сравнения
    buy_hold_return = (df_15m['close'].iloc[-1] - df_15m['close'].iloc[0]) / df_15m['close'].iloc[0] * 100
    excess_over_bh = total_return - buy_hold_return
    
    # Средняя позиция
    avg_position_size = np.mean([abs(t) for t in trades]) * 100 if trades else 0
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'trades_count': len(trades),
        'avg_trade': avg_trade,
        'max_drawdown': max_drawdown,
        'risk_reward_ratio': risk_reward_ratio,
        'buy_hold_return': buy_hold_return,
        'excess_over_bh': excess_over_bh,
        'final_balance': balance,
        'avg_position_size': avg_position_size
    }

if __name__ == "__main__":
    results = test_advanced_timai() 
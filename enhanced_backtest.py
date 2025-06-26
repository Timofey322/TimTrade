#!/usr/bin/env python3
"""
Улучшенный бэктест с новыми индикаторами и confidence scoring
Включает:
- ATR, Williams %R, CCI индикаторы
- Confidence scoring для сигналов
- Динамический position sizing
- Trailing stop loss
- Улучшенные фильтры входа (6+ подтверждений)
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings
from datetime import datetime

# Добавляем путь к src
sys.path.append(str(Path(__file__).parent / "src"))

from data_collection.collector import DataCollector
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.confidence_scorer import ConfidenceScorer

def load_config():
    """Загружает конфигурацию."""
    config_path = "config/smart_adaptive_config_clean.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def calculate_enhanced_indicators(df):
    """
    Расчет улучшенных технических индикаторов.
    """
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    df['atr_percentage'] = df['atr'] / df['close'] * 100
    
    # Williams %R
    highest_high = df['high'].rolling(14).max()
    lowest_low = df['low'].rolling(14).min()
    df['williams_r'] = (highest_high - df['close']) / (highest_high - lowest_low) * -100
    
    # CCI (Commodity Channel Index)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(20).mean()
    mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # Базовые индикаторы
    df['rsi'] = calculate_rsi(df['close'])
    df['bb_upper'], df['bb_lower'], df['bb_middle'] = calculate_bollinger_bands(df['close'])
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd_line'] = ema_12 - ema_26
    df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']
    
    # VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwap'] = vwap
    
    # OBV
    price_change = df['close'].diff()
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = df['volume'].iloc[0]
    for i in range(1, len(df)):
        if price_change.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif price_change.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    df['obv'] = obv
    
    # Money Flow Index (MFI)
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(df['close'] > df['close'].shift(), 0).rolling(14).sum()
    negative_flow = money_flow.where(df['close'] < df['close'].shift(), 0).rolling(14).sum()
    df['mfi'] = 100 - (100 / (1 + positive_flow / negative_flow))
    
    return df

def calculate_rsi(prices, period=14):
    """Рассчитывает RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Рассчитывает полосы Боллинджера."""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band, sma

def calculate_confidence_score(row):
    """
    Расчет confidence score для сигнала на основе множественных факторов.
    """
    factors = []
    
    # 1. Согласованность основных индикаторов (30%)
    macd_bullish = row['macd_line'] > row['macd_signal']
    price_above_vwap = row['close'] > row['vwap']
    obv_momentum = row['obv'] > row['obv'].shift(1) if hasattr(row['obv'], 'shift') else True
    rsi_not_extreme = 30 < row['rsi'] < 70
    
    agreement_score = sum([macd_bullish, price_above_vwap, obv_momentum, rsi_not_extreme]) / 4
    factors.append(('agreement', agreement_score, 0.3))
    
    # 2. Сила сигнала (25%)
    macd_strength = min(abs(row['macd_histogram']) / 0.01, 1.0)
    rsi_strength = abs(row['rsi'] - 50) / 50
    williams_strength = abs(row['williams_r'] + 50) / 50
    
    signal_strength = np.mean([macd_strength, rsi_strength, williams_strength])
    factors.append(('strength', signal_strength, 0.25))
    
    # 3. Волатильность (20%) - высокая волатильность снижает уверенность
    atr_factor = 1.0 / (1.0 + row['atr_percentage'] / 2.0)
    factors.append(('volatility', atr_factor, 0.2))
    
    # 4. Объемное подтверждение (15%)
    volume_avg = row['volume'].rolling(20).mean() if hasattr(row['volume'], 'rolling') else row['volume']
    volume_factor = min(row['volume'] / volume_avg, 2.0) / 2.0 if volume_avg > 0 else 0.5
    factors.append(('volume', volume_factor, 0.15))
    
    # 5. Рыночный режим (10%)
    trend_factor = 1.0 if row['sma_20'] > row['sma_50'] else 0.6
    factors.append(('trend', trend_factor, 0.1))
    
    # Взвешенный расчет
    confidence = sum(score * weight for _, score, weight in factors)
    return max(0.0, min(1.0, confidence))

def enhanced_strategy_signals(df):
    """
    Улучшенная стратегия с новыми индикаторами и confidence scoring.
    """
    signals = np.zeros(len(df))
    confidence_scores = np.zeros(len(df))
    
    for i in range(50, len(df)):  # Начинаем с 50 для всех индикаторов
        # === УСЛОВИЯ ДЛЯ ПОКУПКИ ===
        buy_conditions = []
        
        # 1. MACD - бычий сигнал
        macd_bullish = (
            df['macd_line'].iloc[i] > df['macd_signal'].iloc[i] and 
            df['macd_line'].iloc[i-1] <= df['macd_signal'].iloc[i-1]
        )
        buy_conditions.append(macd_bullish)
        
        # 2. Цена выше VWAP
        price_above_vwap = df['close'].iloc[i] > df['vwap'].iloc[i]
        buy_conditions.append(price_above_vwap)
        
        # 3. OBV растет
        obv_rising = df['obv'].iloc[i] > df['obv'].iloc[i-1]
        buy_conditions.append(obv_rising)
        
        # 4. RSI в зоне покупки
        rsi_buy_zone = 30 < df['rsi'].iloc[i] < 70
        buy_conditions.append(rsi_buy_zone)
        
        # 5. Williams %R не перекуплен
        williams_buy = df['williams_r'].iloc[i] < -20
        buy_conditions.append(williams_buy)
        
        # 6. CCI показывает импульс
        cci_buy = df['cci'].iloc[i] > -100 and df['cci'].iloc[i] < 100
        buy_conditions.append(cci_buy)
        
        # 7. SMA тренд восходящий
        sma_trend_up = df['sma_20'].iloc[i] > df['sma_50'].iloc[i]
        buy_conditions.append(sma_trend_up)
        
        # 8. Не слишком высоко от BB нижней границы
        bb_position = (df['close'].iloc[i] - df['bb_lower'].iloc[i]) / (df['bb_upper'].iloc[i] - df['bb_lower'].iloc[i])
        bb_not_high = bb_position < 0.8
        buy_conditions.append(bb_not_high)
        
        # 9. MFI подтверждение
        mfi_buy = 20 < df['mfi'].iloc[i] < 80
        buy_conditions.append(mfi_buy)
        
        # 10. ATR показывает умеренную волатильность
        atr_moderate = df['atr_percentage'].iloc[i] < 4.0
        buy_conditions.append(atr_moderate)
        
        # === УСЛОВИЯ ДЛЯ ПРОДАЖИ ===
        sell_conditions = []
        
        # 1. MACD - медвежий сигнал
        macd_bearish = (
            df['macd_line'].iloc[i] < df['macd_signal'].iloc[i] and 
            df['macd_line'].iloc[i-1] >= df['macd_signal'].iloc[i-1]
        )
        sell_conditions.append(macd_bearish)
        
        # 2. Цена ниже VWAP
        price_below_vwap = df['close'].iloc[i] < df['vwap'].iloc[i]
        sell_conditions.append(price_below_vwap)
        
        # 3. OBV падает
        obv_falling = df['obv'].iloc[i] < df['obv'].iloc[i-1]
        sell_conditions.append(obv_falling)
        
        # 4. RSI перекуплен или перепродан
        rsi_extreme = df['rsi'].iloc[i] > 70 or df['rsi'].iloc[i] < 30
        sell_conditions.append(rsi_extreme)
        
        # 5. Williams %R перекуплен
        williams_sell = df['williams_r'].iloc[i] > -20
        sell_conditions.append(williams_sell)
        
        # 6. CCI экстремальные значения
        cci_extreme = df['cci'].iloc[i] > 100 or df['cci'].iloc[i] < -100
        sell_conditions.append(cci_extreme)
        
        # 7. SMA тренд нисходящий
        sma_trend_down = df['sma_20'].iloc[i] < df['sma_50'].iloc[i]
        sell_conditions.append(sma_trend_down)
        
        # 8. MFI показывает перекупленность
        mfi_sell = df['mfi'].iloc[i] > 80 or df['mfi'].iloc[i] < 20
        sell_conditions.append(mfi_sell)
        
        # УЛУЧШЕННАЯ ЛОГИКА СИГНАЛОВ
        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)
        
        # Требуем больше подтверждений: 6+ для покупки, 5+ для продажи
        if buy_score >= 6:
            signal = 1
        elif sell_score >= 5:
            signal = -1
        else:
            signal = 0
        
        signals[i] = signal
        
        # Рассчитываем confidence score для сигнала
        if signal != 0:
            confidence = calculate_confidence_score(df.iloc[i])
            confidence_scores[i] = confidence
    
    return signals, confidence_scores

def dynamic_position_sizing(base_size, confidence, atr_percentage, account_balance):
    """
    Динамический расчет размера позиции на основе confidence и волатильности.
    """
    # Базовый размер 2%
    position_size = base_size
    
    # Корректировка на уверенность (0.5-2.0x)
    confidence_multiplier = 0.5 + confidence * 1.5
    position_size *= confidence_multiplier
    
    # Корректировка на волатильность
    volatility_adjustment = max(0.3, 1.0 - atr_percentage / 5.0)
    position_size *= volatility_adjustment
    
    # Максимальный размер позиции 5%
    max_position = account_balance * 0.05
    position_value = account_balance * position_size
    
    return min(position_value, max_position)

def trailing_stop_loss(entry_price, current_price, direction, atr, trail_factor=2.0):
    """
    Трейлинг стоп-лосс на основе ATR.
    """
    if direction == 1:  # Длинная позиция
        stop_distance = atr * trail_factor
        trailing_stop = current_price - stop_distance
        initial_stop = entry_price - stop_distance
        return max(trailing_stop, initial_stop)
    else:  # Короткая позиция
        stop_distance = atr * trail_factor
        trailing_stop = current_price + stop_distance
        initial_stop = entry_price + stop_distance
        return min(trailing_stop, initial_stop)

def run_enhanced_backtest(df, initial_capital=10000, min_confidence=0.6):
    """
    Запускает улучшенный бэктест с новыми возможностями.
    """
    logger.info("=== ЗАПУСК УЛУЧШЕННОГО БЭКТЕСТА ===")
    
    # Генерируем торговые сигналы с confidence
    signals, confidence_scores = enhanced_strategy_signals(df)
    df['signal'] = signals
    df['confidence'] = confidence_scores
    
    # Фильтруем сигналы по минимальной уверенности
    filtered_signals = signals.copy()
    low_confidence_mask = confidence_scores < min_confidence
    filtered_signals[low_confidence_mask] = 0
    df['filtered_signal'] = filtered_signals
    
    logger.info(f"Исходных сигналов: {len(signals[signals != 0])}")
    logger.info(f"После фильтрации по confidence {min_confidence}: {len(filtered_signals[filtered_signals != 0])}")
    
    # Параметры риск-менеджмента
    base_risk_per_trade = 0.02  # 2% базовый риск
    stop_loss_atr_multiplier = 2.5  # Стоп-лосс = 2.5 * ATR
    take_profit_atr_multiplier = 5.0  # Тейк-профит = 5.0 * ATR
    max_position_size = 0.95  # Максимум 95% капитала в позиции
    
    # Инициализируем переменные
    capital = initial_capital
    position = 0
    position_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    trailing_stop = 0
    trades = []
    portfolio_values = [initial_capital]
    
    # Статистика
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    total_profit = 0
    max_consecutive_losses = 0
    current_consecutive_losses = 0
    high_confidence_trades = 0
    
    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        current_signal = df['filtered_signal'].iloc[i]
        current_confidence = df['confidence'].iloc[i]
        current_atr = df['atr'].iloc[i]
        current_atr_pct = df['atr_percentage'].iloc[i]
        
        # Обновляем trailing stop для открытых позиций
        if position > 0:  # Длинная позиция
            new_trailing_stop = trailing_stop_loss(position_price, current_price, 1, current_atr)
            trailing_stop = max(trailing_stop, new_trailing_stop)
            
            # Проверяем trailing stop
            if current_price <= trailing_stop:
                profit = (current_price - position_price) * position
                capital += profit + (position * position_price)
                total_profit += profit
                total_trades += 1
                
                if profit > 0:
                    winning_trades += 1
                    current_consecutive_losses = 0
                else:
                    losing_trades += 1
                    current_consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
                
                trades.append({
                    'type': 'trailing_stop',
                    'price': current_price,
                    'profit': profit,
                    'capital': capital,
                    'confidence': current_confidence,
                    'reason': 'trailing_stop'
                })
                position = 0
                continue
            
            # Проверяем стоп-лосс и тейк-профит
            if current_price <= stop_loss_price:
                profit = (current_price - position_price) * position
                capital += profit + (position * position_price)
                total_profit += profit
                total_trades += 1
                current_consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
                losing_trades += 1
                
                trades.append({
                    'type': 'stop_loss',
                    'price': current_price,
                    'profit': profit,
                    'capital': capital,
                    'confidence': current_confidence,
                    'reason': 'stop_loss'
                })
                position = 0
                continue
                
            elif current_price >= take_profit_price:
                profit = (current_price - position_price) * position
                capital += profit + (position * position_price)
                total_profit += profit
                total_trades += 1
                current_consecutive_losses = 0
                winning_trades += 1
                
                trades.append({
                    'type': 'take_profit',
                    'price': current_price,
                    'profit': profit,
                    'capital': capital,
                    'confidence': current_confidence,
                    'reason': 'take_profit'
                })
                position = 0
                continue
        
        # Обрабатываем новые сигналы
        if current_signal == 1 and position <= 0:  # Покупка
            if position < 0:  # Закрываем короткую позицию
                profit = (position_price - current_price) * abs(position)
                capital += profit
                total_profit += profit
                total_trades += 1
                if profit > 0:
                    winning_trades += 1
                    current_consecutive_losses = 0
                else:
                    losing_trades += 1
                    current_consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
                
                trades.append({
                    'type': 'close_short',
                    'price': current_price,
                    'profit': profit,
                    'capital': capital,
                    'confidence': current_confidence,
                    'reason': 'signal'
                })
            
            # Динамический расчет размера позиции
            position_value = dynamic_position_sizing(
                base_risk_per_trade, current_confidence, current_atr_pct, capital
            )
            position = position_value / current_price
            
            # Ограничиваем размер позиции
            max_shares = (capital * max_position_size) / current_price
            position = min(position, max_shares)
            
            position_price = current_price
            stop_loss_price = current_price - (current_atr * stop_loss_atr_multiplier)
            take_profit_price = current_price + (current_atr * take_profit_atr_multiplier)
            trailing_stop = stop_loss_price
            
            capital -= position * current_price
            
            if current_confidence > 0.7:
                high_confidence_trades += 1
            
            trades.append({
                'type': 'buy',
                'price': current_price,
                'position': position,
                'capital': capital,
                'confidence': current_confidence,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'reason': 'signal'
            })
            
        elif current_signal == -1 and position >= 0:  # Продажа
            if position > 0:  # Закрываем длинную позицию
                profit = (current_price - position_price) * position
                capital += profit + (position * position_price)
                total_profit += profit
                total_trades += 1
                if profit > 0:
                    winning_trades += 1
                    current_consecutive_losses = 0
                else:
                    losing_trades += 1
                    current_consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
                
                trades.append({
                    'type': 'close_long',
                    'price': current_price,
                    'profit': profit,
                    'capital': capital,
                    'confidence': current_confidence,
                    'reason': 'signal'
                })
                position = 0
        
        # Рассчитываем текущую стоимость портфеля
        if position > 0:
            portfolio_value = capital + (position * current_price)
        elif position < 0:
            portfolio_value = capital + (position_price - current_price) * abs(position)
        else:
            portfolio_value = capital
            
        portfolio_values.append(portfolio_value)
    
    # Закрываем открытую позицию в конце
    if position != 0:
        final_price = df['close'].iloc[-1]
        if position > 0:
            profit = (final_price - position_price) * position
            capital += profit + (position * position_price)
        else:
            profit = (position_price - final_price) * abs(position)
            capital += profit
        
        total_profit += profit
        total_trades += 1
        if profit > 0:
            winning_trades += 1
        else:
            losing_trades += 1
    
    # Рассчитываем продвинутую статистику
    final_capital = capital
    total_return = (final_capital - initial_capital) / initial_capital * 100
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    # Рассчитываем максимальную просадку
    portfolio_series = pd.Series(portfolio_values)
    peak = portfolio_series.expanding().max()
    drawdown = (portfolio_series - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    # Рассчитываем коэффициент Шарпа
    if len(portfolio_values) > 1:
        returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 12) if returns.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Статистика по сделкам
    if total_trades > 0:
        profits = [t['profit'] for t in trades if 'profit' in t and t['profit'] > 0]
        losses = [t['profit'] for t in trades if 'profit' in t and t['profit'] < 0]
        
        avg_win = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    else:
        avg_win = avg_loss = profit_factor = 0
    
    # Выводим результаты
    logger.info("=== РЕЗУЛЬТАТЫ УЛУЧШЕННОГО БЭКТЕСТА ===")
    logger.info(f"Начальный капитал: ${initial_capital:,.2f}")
    logger.info(f"Конечный капитал: ${final_capital:,.2f}")
    logger.info(f"Общая доходность: {total_return:.2f}%")
    logger.info(f"Общая прибыль: ${total_profit:,.2f}")
    logger.info(f"Количество сделок: {total_trades}")
    logger.info(f"Прибыльных сделок: {winning_trades} ({win_rate:.1f}%)")
    logger.info(f"Убыточных сделок: {losing_trades}")
    logger.info(f"Сделок с высокой уверенностью: {high_confidence_trades}")
    logger.info(f"Максимальная просадка: {max_drawdown:.2f}%")
    logger.info(f"Коэффициент Шарпа: {sharpe_ratio:.3f}")
    logger.info(f"Коэффициент прибыли: {profit_factor:.2f}")
    logger.info(f"Средняя прибыльная сделка: ${avg_win:.2f}")
    logger.info(f"Средняя убыточная сделка: ${avg_loss:.2f}")
    logger.info(f"Макс. подряд убыточных: {max_consecutive_losses}")
    
    return {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return': total_return,
        'total_profit': total_profit,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'high_confidence_trades': high_confidence_trades,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_consecutive_losses': max_consecutive_losses,
        'trades': trades,
        'portfolio_values': portfolio_values
    }

def main():
    """Основная функция."""
    warnings.filterwarnings('ignore')
    
    logger.info("🚀 Запуск УЛУЧШЕННОГО бэктеста с новыми индикаторами")
    
    # Загружаем конфигурацию
    config = load_config()
    
    # Собираем данные
    logger.info("=== СБОР ДАННЫХ ===")
    data_config = config.get('data_collection', {})
    collector = DataCollector(data_config)
    
    # Тестируем на обеих парах
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframe = '5m'
    
    all_results = {}
    
    for symbol in symbols:
        logger.info(f"\n=== ТЕСТИРОВАНИЕ {symbol} ===")
        
        # Загружаем данные
        data = collector.fetch_ohlcv_with_history(
            symbol=symbol,
            timeframe=timeframe,
            target_limit=100000
        )
        
        if data is None or data.empty:
            logger.error(f"Не удалось загрузить данные для {symbol}")
            continue
        
        logger.info(f"Загружено {len(data)} свечей для {symbol} на {timeframe}")
        
        # Обрабатываем данные
        processed_data = calculate_enhanced_indicators(data)
        
        # Используем последние 30k записей для бэктеста
        test_data = processed_data.tail(30000).copy()
        logger.info(f"Используем {len(test_data)} записей для улучшенного бэктеста {symbol}")
        
        # Запускаем улучшенный бэктест
        results = run_enhanced_backtest(test_data, initial_capital=10000, min_confidence=0.6)
        all_results[symbol] = results
        
        # Сохраняем результаты
        results_df = pd.DataFrame(results['trades'])
        if not results_df.empty:
            results_df.to_csv(f'backtest_results/enhanced_{symbol.replace("/", "_")}_trades.csv', index=False)
            logger.info(f"Сделки {symbol} сохранены в backtest_results/enhanced_{symbol.replace('/', '_')}_trades.csv")
        
        # Сохраняем кривую капитала
        portfolio_df = pd.DataFrame({
            'timestamp': test_data.index[:len(results['portfolio_values'])],
            'portfolio_value': results['portfolio_values']
        })
        portfolio_df.to_csv(f'backtest_results/enhanced_{symbol.replace("/", "_")}_portfolio.csv', index=False)
    
    # Сравнительный анализ
    logger.info("\n=== СРАВНИТЕЛЬНЫЙ АНАЛИЗ ===")
    for symbol, results in all_results.items():
        logger.info(f"{symbol}: Доходность {results['total_return']:.2f}%, "
                   f"Винрейт {results['win_rate']:.1f}%, "
                   f"Sharpe {results['sharpe_ratio']:.3f}")
    
    logger.info("🎉 Улучшенный бэктест завершен успешно!")

if __name__ == "__main__":
    main() 
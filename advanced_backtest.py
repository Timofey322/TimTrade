#!/usr/bin/env python3
"""
Продвинутый бэктест адаптивной стратегии с улучшенными методами.
Включает управление рисками, множественные стратегии и оптимизацию.
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings

# Добавляем путь к src
sys.path.append(str(Path(__file__).parent / "src"))

from data_collection.collector import DataCollector
from preprocessing.feature_engineering import FeatureEngineer

def load_config():
    """Загружает конфигурацию."""
    config_path = "config/smart_adaptive_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def advanced_strategy_signals(df):
    """
    Продвинутая стратегия с использованием множественных индикаторов и фильтров.
    """
    signals = np.zeros(len(df))
    
    # Вычисляем дополнительные индикаторы
    df['rsi'] = calculate_rsi(df['close'])
    df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    for i in range(50, len(df)):  # Начинаем с 50 для всех индикаторов
        # === УСЛОВИЯ ДЛЯ ПОКУПКИ ===
        buy_conditions = []
        
        # 1. MACD - бычий сигнал
        macd_bullish = (
            df['macd_line'].iloc[i] > df['macd_signal'].iloc[i] and 
            df['macd_line'].iloc[i-1] <= df['macd_signal'].iloc[i-1]
        )
        buy_conditions.append(macd_bullish)
        
        # 2. Цена выше VWAP (тренд)
        price_above_vwap = df['close'].iloc[i] > df['vwap'].iloc[i]
        buy_conditions.append(price_above_vwap)
        
        # 3. OBV растет (объем подтверждает)
        obv_rising = df['obv'].iloc[i] > df['obv'].iloc[i-1]
        buy_conditions.append(obv_rising)
        
        # 4. RSI не перекуплен (< 70)
        rsi_not_overbought = df['rsi'].iloc[i] < 70
        buy_conditions.append(rsi_not_overbought)
        
        # 5. Цена выше SMA20 (краткосрочный тренд)
        price_above_sma20 = df['close'].iloc[i] > df['sma_20'].iloc[i]
        buy_conditions.append(price_above_sma20)
        
        # 6. SMA20 выше SMA50 (восходящий тренд)
        sma_trend_up = df['sma_20'].iloc[i] > df['sma_50'].iloc[i]
        buy_conditions.append(sma_trend_up)
        
        # 7. Цена не слишком высоко от нижней BB (не перекуплена)
        bb_position = (df['close'].iloc[i] - df['bb_lower'].iloc[i]) / (df['bb_upper'].iloc[i] - df['bb_lower'].iloc[i])
        bb_not_high = bb_position < 0.8
        buy_conditions.append(bb_not_high)
        
        # === УСЛОВИЯ ДЛЯ ПРОДАЖИ ===
        sell_conditions = []
        
        # 1. MACD - медвежий сигнал
        macd_bearish = (
            df['macd_line'].iloc[i] < df['macd_signal'].iloc[i] and 
            df['macd_line'].iloc[i-1] >= df['macd_signal'].iloc[i-1]
        )
        sell_conditions.append(macd_bearish)
        
        # 2. Цена ниже VWAP (нисходящий тренд)
        price_below_vwap = df['close'].iloc[i] < df['vwap'].iloc[i]
        sell_conditions.append(price_below_vwap)
        
        # 3. OBV падает (объем подтверждает падение)
        obv_falling = df['obv'].iloc[i] < df['obv'].iloc[i-1]
        sell_conditions.append(obv_falling)
        
        # 4. RSI перекуплен (> 80) или перепродан (< 20)
        rsi_extreme = df['rsi'].iloc[i] > 80 or df['rsi'].iloc[i] < 20
        sell_conditions.append(rsi_extreme)
        
        # 5. Цена ниже SMA20 (нарушение краткосрочного тренда)
        price_below_sma20 = df['close'].iloc[i] < df['sma_20'].iloc[i]
        sell_conditions.append(price_below_sma20)
        
        # 6. SMA20 ниже SMA50 (нисходящий тренд)
        sma_trend_down = df['sma_20'].iloc[i] < df['sma_50'].iloc[i]
        sell_conditions.append(sma_trend_down)
        
        # СИГНАЛЫ С ВЕСАМИ
        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)
        
        # Покупка если сильный бычий сигнал (5+ условий)
        if buy_score >= 5:
            signals[i] = 1
        # Продажа если сильный медвежий сигнал (4+ условий)
        elif sell_score >= 4:
            signals[i] = -1
    
    return signals

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
    return upper_band, lower_band

def run_advanced_backtest(df, initial_capital=10000):
    """Запускает продвинутый бэктест с управлением рисками."""
    logger.info("=== ЗАПУСК ПРОДВИНУТОГО БЭКТЕСТА ===")
    
    # Генерируем торговые сигналы
    signals = advanced_strategy_signals(df)
    df['signal'] = signals
    
    # Параметры риск-менеджмента
    risk_per_trade = 0.02  # 2% риска за сделку
    stop_loss_pct = 0.03   # 3% стоп-лосс
    take_profit_pct = 0.06 # 6% тейк-профит (соотношение риск/прибыль 1:2)
    max_position_size = 0.9 # Максимум 90% капитала в позиции
    
    # Инициализируем переменные
    capital = initial_capital
    position = 0
    position_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    trades = []
    portfolio_values = [initial_capital]
    
    # Статистика
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    total_profit = 0
    max_consecutive_losses = 0
    current_consecutive_losses = 0
    
    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        current_signal = df['signal'].iloc[i]
        
        # Проверяем стоп-лосс и тейк-профит для открытых позиций
        if position > 0:  # Длинная позиция
            # Стоп-лосс
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
                    'reason': 'stop_loss'
                })
                position = 0
                continue
                
            # Тейк-профит
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
                    'reason': 'signal'
                })
            
            # Рассчитываем размер позиции на основе риска
            risk_amount = capital * risk_per_trade
            position_size = min(risk_amount / (current_price * stop_loss_pct), capital * max_position_size / current_price)
            
            # Открываем длинную позицию
            position = position_size
            position_price = current_price
            stop_loss_price = current_price * (1 - stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)
            capital -= position * current_price
            
            trades.append({
                'type': 'buy',
                'price': current_price,
                'position': position,
                'capital': capital,
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
    
    # Рассчитываем коэффициент Шарпа (упрощенно)
    if len(portfolio_values) > 1:
        returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 12) if returns.std() > 0 else 0  # Для 5-минутных данных
    else:
        sharpe_ratio = 0
    
    # Прибыльные vs убыточные сделки
    if total_trades > 0:
        avg_win = sum(t['profit'] for t in trades if t.get('profit', 0) > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(t['profit'] for t in trades if t.get('profit', 0) < 0) / losing_trades if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    else:
        avg_win = avg_loss = profit_factor = 0
    
    # Выводим результаты
    logger.info("=== РЕЗУЛЬТАТЫ ПРОДВИНУТОГО БЭКТЕСТА ===")
    logger.info(f"Начальный капитал: ${initial_capital:,.2f}")
    logger.info(f"Конечный капитал: ${final_capital:,.2f}")
    logger.info(f"Общая доходность: {total_return:.2f}%")
    logger.info(f"Общая прибыль: ${total_profit:,.2f}")
    logger.info(f"Количество сделок: {total_trades}")
    logger.info(f"Прибыльных сделок: {winning_trades} ({win_rate:.1f}%)")
    logger.info(f"Убыточных сделок: {losing_trades}")
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
    
    logger.info("Запуск продвинутого бэктеста адаптивной стратегии")
    
    # Загружаем конфигурацию
    config = load_config()
    
    # Собираем данные
    logger.info("=== СБОР ДАННЫХ ===")
    data_config = config.get('data_collection', {})
    collector = DataCollector(data_config)
    
    symbol = 'BTC/USDT'
    timeframe = '5m'
    
    # Загружаем больше данных для продвинутого теста
    data = collector.fetch_ohlcv_with_history(
        symbol=symbol,
        timeframe=timeframe,
        target_limit=100000  # 100k свечей для лучшей статистики
    )
    
    if data is None or data.empty:
        logger.error("Не удалось загрузить данные")
        return
    
    logger.info(f"Загружено {len(data)} свечей для {symbol} на {timeframe}")
    
    # Предобрабатываем данные
    logger.info("=== ПРЕДОБРАБОТКА ДАННЫХ ===")
    preprocessing_config = config.get('preprocessing', {})
    feature_engineer = FeatureEngineer(preprocessing_config)
    
    processed_data = feature_engineer.process_single_timeframe(data, symbol, timeframe)
    
    if processed_data is None or processed_data.empty:
        logger.error("Не удалось предобработать данные")
        return
    
    logger.info(f"Предобработано {len(processed_data)} записей")
    
    # Используем последние 20k записей для бэктеста
    test_data = processed_data.tail(20000).copy()
    logger.info(f"Используем {len(test_data)} записей для продвинутого бэктеста")
    
    # Запускаем продвинутый бэктест
    results = run_advanced_backtest(test_data, initial_capital=10000)
    
    # Сохраняем результаты
    results_df = pd.DataFrame(results['trades'])
    if not results_df.empty:
        results_df.to_csv('backtest_results/advanced_backtest_trades.csv', index=False)
        logger.info("Сделки сохранены в backtest_results/advanced_backtest_trades.csv")
    
    # Сохраняем кривую капитала
    portfolio_df = pd.DataFrame({
        'timestamp': test_data.index[:len(results['portfolio_values'])],
        'portfolio_value': results['portfolio_values']
    })
    portfolio_df.to_csv('backtest_results/advanced_portfolio_curve.csv', index=False)
    logger.info("Кривая капитала сохранена в backtest_results/advanced_portfolio_curve.csv")
    
    logger.info("Продвинутый бэктест завершен успешно!")

if __name__ == "__main__":
    main() 
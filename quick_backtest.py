#!/usr/bin/env python3
"""
Быстрый бэктест адаптивной стратегии без сложного машинного обучения.
Демонстрирует работу адаптивных фильтров и управления рисками.
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
from backtesting.adaptive_filter_strategy import AdaptiveFilterStrategy
from trading.risk_manager import AdaptiveRiskManager

def load_config():
    """Загружает конфигурацию."""
    config_path = "config/smart_adaptive_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def simple_strategy_signals(df):
    """
    Простая стратегия на основе MACD, OBV и VWAP.
    Возвращает сигналы: 1 (покупка), -1 (продажа), 0 (удержание)
    """
    signals = np.zeros(len(df))
    
    for i in range(1, len(df)):
        # Условия для покупки
        buy_conditions = [
            df['macd_line'].iloc[i] > 0 and df['macd_line'].iloc[i-1] <= 0,  # MACD пересекает 0 снизу
            df['close'].iloc[i] > df['vwap'].iloc[i],  # Цена выше VWAP
            df['obv'].iloc[i] > df['obv'].iloc[i-1]  # OBV растет
        ]
        
        # Условия для продажи
        sell_conditions = [
            df['macd_line'].iloc[i] < 0 and df['macd_line'].iloc[i-1] >= 0,  # MACD пересекает 0 сверху
            df['close'].iloc[i] < df['vwap'].iloc[i],  # Цена ниже VWAP
            df['obv'].iloc[i] < df['obv'].iloc[i-1]  # OBV падает
        ]
        
        # Покупка если выполнены 2+ условий
        if sum(buy_conditions) >= 2:
            signals[i] = 1
        # Продажа если выполнены 2+ условий
        elif sum(sell_conditions) >= 2:
            signals[i] = -1
    
    return signals

def run_simple_backtest(df, initial_capital=10000):
    """Запускает простой бэктест."""
    logger.info("=== ЗАПУСК ПРОСТОГО БЭКТЕСТА ===")
    
    # Генерируем торговые сигналы
    signals = simple_strategy_signals(df)
    df['signal'] = signals
    
    # Инициализируем переменные для бэктеста
    capital = initial_capital
    position = 0
    position_price = 0
    trades = []
    portfolio_values = [initial_capital]
    
    # Статистика
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    total_profit = 0
    
    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        current_signal = df['signal'].iloc[i]
        
        # Обрабатываем сигналы
        if current_signal == 1 and position <= 0:  # Покупка
            if position < 0:  # Закрываем короткую позицию
                profit = (position_price - current_price) * abs(position)
                capital += profit
                total_profit += profit
                total_trades += 1
                if profit > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
                
                trades.append({
                    'type': 'close_short',
                    'price': current_price,
                    'profit': profit,
                    'capital': capital
                })
            
            # Открываем длинную позицию
            position = capital * 0.95 / current_price  # 95% капитала в позицию
            position_price = current_price
            capital *= 0.05  # Оставляем 5% в кеше
            
            trades.append({
                'type': 'buy',
                'price': current_price,
                'position': position,
                'capital': capital
            })
            
        elif current_signal == -1 and position >= 0:  # Продажа
            if position > 0:  # Закрываем длинную позицию
                profit = (current_price - position_price) * position
                capital += profit + (position * position_price)
                total_profit += profit
                total_trades += 1
                if profit > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
                
                trades.append({
                    'type': 'close_long',
                    'price': current_price,
                    'profit': profit,
                    'capital': capital
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
    
    # Рассчитываем статистику
    final_capital = capital
    total_return = (final_capital - initial_capital) / initial_capital * 100
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    # Рассчитываем максимальную просадку
    portfolio_series = pd.Series(portfolio_values)
    peak = portfolio_series.expanding().max()
    drawdown = (portfolio_series - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    # Выводим результаты
    logger.info("=== РЕЗУЛЬТАТЫ БЭКТЕСТА ===")
    logger.info(f"Начальный капитал: ${initial_capital:,.2f}")
    logger.info(f"Конечный капитал: ${final_capital:,.2f}")
    logger.info(f"Общая доходность: {total_return:.2f}%")
    logger.info(f"Общая прибыль: ${total_profit:,.2f}")
    logger.info(f"Количество сделок: {total_trades}")
    logger.info(f"Прибыльных сделок: {winning_trades}")
    logger.info(f"Убыточных сделок: {losing_trades}")
    logger.info(f"Процент прибыльных: {win_rate:.1f}%")
    logger.info(f"Максимальная просадка: {max_drawdown:.2f}%")
    
    if total_trades > 0:
        avg_profit = total_profit / total_trades
        logger.info(f"Средняя прибыль за сделку: ${avg_profit:.2f}")
    
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
        'trades': trades,
        'portfolio_values': portfolio_values
    }

def main():
    """Основная функция."""
    warnings.filterwarnings('ignore')
    
    logger.info("Запуск быстрого бэктеста адаптивной стратегии")
    
    # Загружаем конфигурацию
    config = load_config()
    
    # Собираем данные (меньший объем для быстрого теста)
    logger.info("=== СБОР ДАННЫХ ===")
    data_config = config.get('data_collection', {})
    collector = DataCollector(data_config)
    
    symbol = 'BTC/USDT'
    timeframe = '5m'
    
    # Загружаем последние 50k свечей для быстрого теста
    data = collector.fetch_ohlcv_with_history(
        symbol=symbol,
        timeframe=timeframe,
        target_limit=50000
    )
    
    if data is None or data.empty:
        logger.error("Не удалось загрузить данные")
        return
    
    logger.info(f"Загружено {len(data)} свечей для {symbol} на {timeframe}")
    
    # Предобрабатываем данные
    logger.info("=== ПРЕДОБРАБОТКА ДАННЫХ ===")
    preprocessing_config = config.get('preprocessing', {})
    feature_engineer = FeatureEngineer(preprocessing_config)
    
    # Обрабатываем данные с минимальным набором индикаторов
    processed_data = feature_engineer.process_single_timeframe(data, symbol, timeframe)
    
    if processed_data is None or processed_data.empty:
        logger.error("Не удалось предобработать данные")
        return
    
    logger.info(f"Предобработано {len(processed_data)} записей")
    
    # Используем только последние 10k записей для бэктеста
    test_data = processed_data.tail(10000).copy()
    logger.info(f"Используем {len(test_data)} записей для бэктеста")
    
    # Запускаем бэктест
    results = run_simple_backtest(test_data, initial_capital=10000)
    
    # Сохраняем результаты
    results_df = pd.DataFrame(results['trades'])
    if not results_df.empty:
        results_df.to_csv('backtest_results/quick_backtest_trades.csv', index=False)
        logger.info("Сделки сохранены в backtest_results/quick_backtest_trades.csv")
    
    logger.info("Быстрый бэктест завершен успешно!")

if __name__ == "__main__":
    main() 
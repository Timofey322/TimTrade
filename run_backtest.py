#!/usr/bin/env python3
"""
Скрипт для запуска бэктестинга торговой модели.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent))

from src.backtesting import BacktestRunner


def main():
    """Основная функция запуска бэктестинга."""
    
    # Настройка логирования
    logger.add("logs/backtest.log", rotation="1 day", retention="7 days")
    
    try:
        # Создаем runner
        runner = BacktestRunner()
        
        # Конфигурация бэктестинга
        backtest_config = {
            'initial_capital': 10000,  # Начальный капитал
            'commission': 0.001,       # Комиссия 0.1%
            'slippage': 0.0005,        # Проскальзывание 0.05%
            'position_size': 0.05,     # Уменьшаем размер позиции до 5% от капитала
            'min_confidence': 0.75,    # Увеличиваем минимальную уверенность до 75%
            'stop_loss': 0.015,        # Уменьшаем Stop Loss до 1.5%
            'take_profit': 0.03,       # Уменьшаем Take Profit до 3%
            'max_positions': 2         # Уменьшаем максимум позиций до 2
        }
        
        # Параметры бэктестинга
        symbol = "BTC/USDT"
        model_path = "models/improved_xgboost_BTC_USDT_latest.pkl"
        
        # Период бэктестинга (последние 2 месяца для 5m/15m таймфреймов)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        
        print(f"Запуск бэктестинга для {symbol}")
        print(f"Период: {start_date} - {end_date}")
        print(f"Модель: {model_path}")
        print(f"Начальный капитал: ${backtest_config['initial_capital']:,.2f}")
        print("-" * 50)
        
        # Запускаем бэктестинг
        results = runner.run_full_backtest(
            symbol=symbol,
            model_path=model_path,
            start_date=start_date,
            end_date=end_date,
            backtest_config=backtest_config
        )
        
        if results:
            print("\n✅ Бэктестинг завершен успешно!")
            print(f"📊 Результаты сохранены в папке backtest_results/")
        else:
            print("\n❌ Ошибка в бэктестинге")
            return 1
            
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        print(f"\n❌ Критическая ошибка: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
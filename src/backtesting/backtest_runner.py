"""
Модуль для запуска бэктестинга с интеграцией в основную торговую систему.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import yaml
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from .backtester import Backtester
from src.data_collection.collector import DataCollector
from src.preprocessing.indicators import TechnicalIndicators


class BacktestRunner:
    """
    Класс для запуска бэктестинга с полной интеграцией в систему.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Инициализация runner'а бэктестинга.
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.logger = logger.bind(name="BacktestRunner")
        
        # Инициализация компонентов
        self.data_collector = DataCollector(self.config)
        self.backtester = None
        
    def load_config(self) -> dict:
        """Загрузка конфигурации."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Ошибка загрузки конфигурации: {e}")
            return {}
    
    def collect_backtest_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Сбор данных для бэктестинга.
        
        Args:
            symbol: Торговая пара
            start_date: Дата начала в формате YYYY-MM-DD
            end_date: Дата окончания в формате YYYY-MM-DD
            
        Returns:
            DataFrame с данными для бэктестинга
        """
        try:
            self.logger.info(f"Сбор данных для бэктестинга: {symbol} с {start_date} по {end_date}")
            
            # Собираем данные только на 5m таймфрейме (как при обучении)
            data = self.data_collector.fetch_ohlcv_with_history(
                symbol=symbol,
                timeframe='5m',
                target_limit=20000,  # Больше данных для 5m
                batch_size=1000
            )
            
            if data is None or data.empty:
                self.logger.error("Не удалось собрать данные для 5m")
                return pd.DataFrame()
            
            # Фильтруем данные по датам
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            data = data[(data.index >= start_dt) & (data.index <= end_dt)]
            
            if data.empty:
                self.logger.error(f"Нет данных в указанном периоде {start_date} - {end_date}")
                return pd.DataFrame()
            
            # Добавляем технические индикаторы (как при обучении)
            indicators = TechnicalIndicators()
            processed_data = indicators.add_all_indicators(data)
            
            # Удаляем NaN значения
            processed_data = processed_data.dropna()
            
            self.logger.info(f"Собрано {len(processed_data)} записей для бэктестинга на 5m таймфрейме")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Ошибка сбора данных: {e}")
            return pd.DataFrame()
    
    def setup_backtester(self, backtest_config: dict = None) -> Backtester:
        """
        Настройка бэктестера.
        
        Args:
            backtest_config: Конфигурация бэктестинга
            
        Returns:
            Настроенный экземпляр Backtester
        """
        try:
            # Конфигурация по умолчанию
            default_config = {
                'initial_capital': 10000,
                'commission': 0.001,
                'slippage': 0.0005,
                'position_size': 0.1,
                'min_confidence': 0.6,
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'max_positions': 3
            }
            
            # Обновляем конфигурацию
            if backtest_config:
                default_config.update(backtest_config)
            
            self.backtester = Backtester(default_config)
            self.logger.info("Бэктестер настроен")
            
            return self.backtester
            
        except Exception as e:
            self.logger.error(f"Ошибка настройки бэктестера: {e}")
            return None
    
    def run_full_backtest(self, symbol: str, model_path: str, 
                         start_date: str, end_date: str,
                         backtest_config: dict = None) -> dict:
        """
        Запуск полного бэктестинга.
        
        Args:
            symbol: Торговая пара
            model_path: Путь к модели
            start_date: Дата начала
            end_date: Дата окончания
            backtest_config: Конфигурация бэктестинга
            
        Returns:
            Результаты бэктестинга
        """
        try:
            self.logger.info(f"Запуск полного бэктестинга для {symbol}")
            
            # Собираем данные
            data = self.collect_backtest_data(symbol, start_date, end_date)
            
            if data.empty:
                self.logger.error("Нет данных для бэктестинга")
                return {}
            
            # Настраиваем бэктестер
            backtester = self.setup_backtester(backtest_config)
            
            if not backtester:
                self.logger.error("Не удалось настроить бэктестер")
                return {}
            
            # Запускаем бэктестинг
            results = backtester.run_backtest(data, model_path)
            
            if results:
                # Визуализируем результаты
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_symbol = symbol.replace('/', '_')  # Заменяем / на _ для безопасности путей
                plot_path = f"backtest_results/backtest_plot_{safe_symbol}_{timestamp}.png"
                backtester.plot_results(plot_path)
                
                # Сохраняем результаты
                results_path = f"backtest_results/backtest_results_{safe_symbol}_{timestamp}.json"
                backtester.save_results(results_path)
                
                self.logger.info("Бэктестинг завершен успешно")
            else:
                self.logger.error("Бэктестинг не дал результатов")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка полного бэктестинга: {e}")
            return {}
    
    def run_multiple_backtests(self, symbols: list, model_paths: dict,
                              start_date: str, end_date: str,
                              backtest_config: dict = None) -> dict:
        """
        Запуск бэктестинга для нескольких пар.
        
        Args:
            symbols: Список торговых пар
            model_paths: Словарь {symbol: model_path}
            start_date: Дата начала
            end_date: Дата окончания
            backtest_config: Конфигурация бэктестинга
            
        Returns:
            Результаты всех бэктестингов
        """
        try:
            all_results = {}
            
            for symbol in symbols:
                if symbol not in model_paths:
                    self.logger.warning(f"Нет модели для {symbol}")
                    continue
                
                self.logger.info(f"Бэктестинг для {symbol}")
                
                results = self.run_full_backtest(
                    symbol=symbol,
                    model_path=model_paths[symbol],
                    start_date=start_date,
                    end_date=end_date,
                    backtest_config=backtest_config
                )
                
                all_results[symbol] = results
            
            # Создаем сводный отчет
            self.create_summary_report(all_results)
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Ошибка множественного бэктестинга: {e}")
            return {}
    
    def create_summary_report(self, results: dict):
        """
        Создание сводного отчета по всем бэктестингам.
        
        Args:
            results: Результаты всех бэктестингов
        """
        try:
            if not results:
                return
            
            # Создаем сводную таблицу
            summary_data = []
            
            for symbol, result in results.items():
                if not result:
                    continue
                
                summary_data.append({
                    'Symbol': symbol,
                    'Total Return': f"{result.get('total_return', 0):.2%}",
                    'Annual Return': f"{result.get('annual_return', 0):.2%}",
                    'Sharpe Ratio': f"{result.get('sharpe_ratio', 0):.2f}",
                    'Max Drawdown': f"{result.get('max_drawdown', 0):.2%}",
                    'Total Trades': result.get('total_trades', 0),
                    'Win Rate': f"{result.get('win_rate', 0):.2%}",
                    'Profit Factor': f"{result.get('profit_factor', 0):.2f}"
                })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                
                # Сохраняем сводный отчет
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_path = f"backtest_results/summary_report_{timestamp}.csv"
                summary_df.to_csv(summary_path, index=False)
                
                # Выводим сводку
                print("\n" + "="*80)
                print("СВОДНЫЙ ОТЧЕТ ПО БЭКТЕСТИНГУ")
                print("="*80)
                print(summary_df.to_string(index=False))
                print("="*80)
                
                self.logger.info(f"Сводный отчет сохранен в {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка создания сводного отчета: {e}")

    def run_backtest(self, data: pd.DataFrame, model, symbol: str) -> dict:
        """
        Запуск бэктеста с переданными данными и моделью.
        
        Args:
            data: Данные для бэктеста (уже разделенные в main.py)
            model: Обученная модель
            symbol: Торговая пара
            
        Returns:
            Результаты бэктеста
        """
        try:
            self.logger.info(f"Запуск бэктеста для {symbol}")
            
            if data.empty:
                self.logger.error("Нет данных для бэктеста")
                return {}
            
            # Получаем настройки бэктеста
            backtest_config = self.config.get('backtesting', {})
            
            # Используем переданные данные напрямую (они уже разделены в main.py)
            backtest_data = data.copy()
            self.logger.info(f"Используем переданные данные для бэктеста: {len(backtest_data)} записей")
            self.logger.info(f"Период бэктеста: {backtest_data.index[0]} - {backtest_data.index[-1]}")
            
            # Настраиваем бэктестер
            backtester = self.setup_backtester(backtest_config)
            
            if not backtester:
                self.logger.error("Не удалось настроить бэктестер")
                return {}
            
            # Запускаем бэктестинг с моделью в памяти
            results = backtester.run_backtest_with_model(backtest_data, model, symbol)
            
            if results:
                # Визуализируем результаты
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_symbol = symbol.replace('/', '_')
                plot_path = f"backtest_results/backtest_plot_{safe_symbol}_{timestamp}.png"
                backtester.plot_results(plot_path)
                
                # Сохраняем результаты
                results_path = f"backtest_results/backtest_results_{safe_symbol}_{timestamp}.json"
                backtester.save_results(results_path)
                
                self.logger.info("Бэктест завершен успешно")
            else:
                self.logger.error("Бэктест не дал результатов")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка бэктеста: {e}")
            return {}


def run_backtest_example():
    """Пример запуска бэктестинга."""
    # Создаем runner
    runner = BacktestRunner()
    
    # Конфигурация бэктестинга
    backtest_config = {
        'initial_capital': 10000,
        'commission': 0.001,
        'slippage': 0.0005,
        'position_size': 0.1,
        'min_confidence': 0.6
    }
    
    # Параметры бэктестинга
    symbol = "BTC_USDT"
    model_path = "models/xgboost_BTC_USDT.pkl"
    start_date = "2024-01-01"
    end_date = "2024-06-30"
    
    # Запускаем бэктестинг
    results = runner.run_full_backtest(
        symbol=symbol,
        model_path=model_path,
        start_date=start_date,
        end_date=end_date,
        backtest_config=backtest_config
    )
    
    if results:
        print("Бэктестинг завершен успешно!")
    else:
        print("Ошибка в бэктестинге")


if __name__ == "__main__":
    run_backtest_example() 
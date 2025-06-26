"""
Пакет для бэктестинга торговых стратегий.
"""

from .backtester import Backtester
from .backtest_runner import BacktestRunner

__all__ = ['Backtester', 'BacktestRunner'] 
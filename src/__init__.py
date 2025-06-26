"""
Умная адаптивная торговая система для криптовалют.

Этот пакет содержит все компоненты для сбора данных, предобработки,
обучения моделей машинного обучения и бэктестинга торговых стратегий.
"""

__version__ = "2.0.0"
__author__ = "Trading System Team"

# Основные модули
from . import data_collection
from . import preprocessing
from . import ml_models
from . import backtesting
from . import trading
from . import utils

__all__ = [
    'data_collection',
    'preprocessing', 
    'ml_models',
    'backtesting',
    'trading',
    'utils'
] 
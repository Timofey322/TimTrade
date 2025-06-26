"""
Модуль для сбора данных с криптобирж.

Этот модуль содержит классы и функции для:
- Сбора исторических данных OHLCV
- Кэширования данных
- Валидации и очистки данных
- Интеграции с различными биржами
"""

from .collector import DataCollector
from .cache import DataCache

__all__ = ['DataCollector', 'DataCache'] 
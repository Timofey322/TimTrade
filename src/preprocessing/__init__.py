"""
Модуль для предобработки данных.

Этот модуль содержит классы и функции для:
- Создания технических индикаторов
- Feature engineering
- Нормализации и масштабирования данных
- Создания целевых переменных для ML
"""

from .feature_engineering import FeatureEngineer
from .indicators import TechnicalIndicators
from .target_creation import TargetCreator

__all__ = ['FeatureEngineer', 'TechnicalIndicators', 'TargetCreator'] 
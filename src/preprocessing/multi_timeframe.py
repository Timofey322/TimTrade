"""
Модуль для обработки множественных таймфреймов.

Этот модуль предоставляет функциональность для:
- Объединения данных с разных таймфреймов
- Создания признаков на основе множественных таймфреймов
- Синхронизации данных между таймфреймами
- Создания префиксов для признаков разных таймфреймов
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from datetime import datetime, timedelta

class MultiTimeframeProcessor:
    """
    Класс для обработки данных с множественных таймфреймов.
    
    Поддерживает:
    - Объединение данных с разных таймфреймов
    - Создание признаков для каждого таймфрейма
    - Синхронизацию данных по времени
    - Префиксы для различения признаков
    """
    
    def __init__(self, config: Dict):
        """
        Инициализация процессора множественных таймфреймов.
        
        Args:
            config: Конфигурация обработки множественных таймфреймов
        """
        self.config = config
        self.logger = logger.bind(name="MultiTimeframeProcessor")
        
        # Настройки множественных таймфреймов
        self.enabled = config.get('enabled', True)
        self.features_per_timeframe = config.get('features_per_timeframe', {})
        self.prefixes = config.get('prefixes', {})
        
        # Основной таймфрейм для синхронизации
        self.primary_timeframe = config.get('primary_timeframe', '1h')
        
        self.logger.info(f"MultiTimeframeProcessor инициализирован. Включен: {self.enabled}")
        if self.enabled:
            self.logger.info(f"Таймфреймы: {list(self.features_per_timeframe.keys())}")
            self.logger.info(f"Основной таймфрейм: {self.primary_timeframe}")
    
    def process_multi_timeframe_data(self, 
                                   data_dict: Dict[str, pd.DataFrame],
                                   symbol: str) -> Optional[pd.DataFrame]:
        """
        Обработка данных с множественных таймфреймов.
        
        Args:
            data_dict: Словарь с данными для каждого таймфрейма
            symbol: Торговая пара
        
        Returns:
            DataFrame с объединенными признаками или None при ошибке
        """
        if not self.enabled or not data_dict:
            return None
        
        try:
            self.logger.info(f"Обрабатываем множественные таймфреймы для {symbol}")
            
            # Проверка наличия основного таймфрейма
            if self.primary_timeframe not in data_dict:
                self.logger.error(f"Основной таймфрейм {self.primary_timeframe} отсутствует в данных")
                return None
            
            # Получение основного DataFrame
            primary_df = data_dict[self.primary_timeframe].copy()
            self.logger.info(f"Основной DataFrame: {len(primary_df)} записей")
            
            # Обработка каждого таймфрейма
            for timeframe, df in data_dict.items():
                if timeframe == self.primary_timeframe:
                    continue
                
                self.logger.info(f"Обрабатываем таймфрейм {timeframe}")
                
                # Синхронизация с основным таймфреймом
                synced_df = self._synchronize_timeframes(primary_df, df, timeframe)
                
                if synced_df is not None:
                    # Добавление признаков с префиксом
                    prefix = self.prefixes.get(timeframe, f"{timeframe}_")
                    primary_df = self._add_timeframe_features(primary_df, synced_df, prefix)
            
            self.logger.info(f"Обработка завершена. Итоговый DataFrame: {len(primary_df)} записей, {len(primary_df.columns)} колонок")
            
            return primary_df
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки множественных таймфреймов для {symbol}: {e}")
            return None
    
    def _synchronize_timeframes(self, 
                               primary_df: pd.DataFrame, 
                               secondary_df: pd.DataFrame, 
                               timeframe: str) -> Optional[pd.DataFrame]:
        """
        Синхронизация данных между таймфреймами.
        
        Args:
            primary_df: DataFrame основного таймфрейма
            secondary_df: DataFrame вторичного таймфрейма
            timeframe: Название вторичного таймфрейма
        
        Returns:
            Синхронизированный DataFrame или None при ошибке
        """
        try:
            # Определение направления синхронизации
            if self._is_higher_timeframe(primary_df, secondary_df):
                # Основной таймфрейм выше - агрегируем вторичный
                synced_df = self._aggregate_to_higher_timeframe(primary_df, secondary_df, timeframe)
            else:
                # Основной таймфрейм ниже - интерполируем вторичный
                synced_df = self._interpolate_to_lower_timeframe(primary_df, secondary_df, timeframe)
            
            return synced_df
            
        except Exception as e:
            self.logger.error(f"Ошибка синхронизации таймфрейма {timeframe}: {e}")
            return None
    
    def _is_higher_timeframe(self, primary_df: pd.DataFrame, secondary_df: pd.DataFrame) -> bool:
        """
        Определение, является ли основной таймфрейм выше вторичного.
        
        Args:
            primary_df: DataFrame основного таймфрейма
            secondary_df: DataFrame вторичного таймфрейма
        
        Returns:
            True если основной таймфрейм выше
        """
        try:
            # Сравнение временных интервалов между записями
            primary_interval = primary_df.index[1] - primary_df.index[0]
            secondary_interval = secondary_df.index[1] - secondary_df.index[0]
            
            return primary_interval > secondary_interval
            
        except Exception:
            # Если не удалось определить, считаем что основной выше
            return True
    
    def _aggregate_to_higher_timeframe(self, 
                                     primary_df: pd.DataFrame, 
                                     secondary_df: pd.DataFrame, 
                                     timeframe: str) -> pd.DataFrame:
        """
        Агрегация данных вторичного таймфрейма к основному.
        
        Args:
            primary_df: DataFrame основного таймфрейма
            secondary_df: DataFrame вторичного таймфрейма
            timeframe: Название вторичного таймфрейма
        
        Returns:
            Агрегированный DataFrame
        """
        try:
            # Создание копии основного DataFrame
            result_df = primary_df.copy()
            
            # Для каждого временного интервала основного таймфрейма
            for i in range(len(primary_df)):
                start_time = primary_df.index[i]
                
                # Определение конца интервала
                if i < len(primary_df) - 1:
                    end_time = primary_df.index[i + 1]
                else:
                    # Для последней записи используем максимальное время вторичного таймфрейма
                    end_time = secondary_df.index[-1] + timedelta(hours=1)
                
                # Фильтрация данных вторичного таймфрейма в пределах интервала
                mask = (secondary_df.index >= start_time) & (secondary_df.index < end_time)
                interval_data = secondary_df[mask]
                
                if not interval_data.empty:
                    # Агрегация данных
                    aggregated = self._aggregate_interval_data(interval_data, timeframe)
                    
                    # Добавление агрегированных данных к результату
                    for col, value in aggregated.items():
                        result_df.loc[start_time, f"{timeframe}_{col}"] = value
            
            self.logger.debug(f"Агрегировано {len(result_df)} интервалов для таймфрейма {timeframe}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Ошибка агрегации таймфрейма {timeframe}: {e}")
            return primary_df.copy()
    
    def _interpolate_to_lower_timeframe(self, 
                                      primary_df: pd.DataFrame, 
                                      secondary_df: pd.DataFrame, 
                                      timeframe: str) -> pd.DataFrame:
        """
        Интерполяция данных вторичного таймфрейма к основному.
        
        Args:
            primary_df: DataFrame основного таймфрейма
            secondary_df: DataFrame вторичного таймфрейма
            timeframe: Название вторичного таймфрейма
        
        Returns:
            Интерполированный DataFrame
        """
        try:
            # Создание копии основного DataFrame
            result_df = primary_df.copy()
            
            # Интерполяция числовых колонок
            numeric_columns = secondary_df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                # Создание временного ряда для интерполяции
                temp_series = pd.Series(secondary_df[col].values, index=secondary_df.index)
                
                # Интерполяция к временным меткам основного таймфрейма
                interpolated = temp_series.reindex(primary_df.index, method='ffill')
                
                # Добавление к результату
                result_df[f"{timeframe}_{col}"] = interpolated
            
            self.logger.debug(f"Интерполировано {len(numeric_columns)} колонок для таймфрейма {timeframe}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Ошибка интерполяции таймфрейма {timeframe}: {e}")
            return primary_df.copy()
    
    def _aggregate_interval_data(self, interval_data: pd.DataFrame, timeframe: str) -> Dict:
        """
        Агрегация данных за временной интервал.
        
        Args:
            interval_data: Данные за интервал
            timeframe: Название таймфрейма
        
        Returns:
            Словарь с агрегированными значениями
        """
        aggregated = {}
        
        try:
            # OHLC агрегация
            if 'open' in interval_data.columns:
                aggregated['open'] = interval_data['open'].iloc[0]
            if 'high' in interval_data.columns:
                aggregated['high'] = interval_data['high'].max()
            if 'low' in interval_data.columns:
                aggregated['low'] = interval_data['low'].min()
            if 'close' in interval_data.columns:
                aggregated['close'] = interval_data['close'].iloc[-1]
            if 'volume' in interval_data.columns:
                aggregated['volume'] = interval_data['volume'].sum()
            
            # Дополнительные агрегации для технических индикаторов
            for col in interval_data.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume']:
                    # Для технических индикаторов используем последнее значение
                    aggregated[col] = interval_data[col].iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Ошибка агрегации данных интервала для {timeframe}: {e}")
        
        return aggregated
    
    def _add_timeframe_features(self, 
                               primary_df: pd.DataFrame, 
                               synced_df: pd.DataFrame, 
                               prefix: str) -> pd.DataFrame:
        """
        Добавление признаков с префиксом таймфрейма.
        
        Args:
            primary_df: Основной DataFrame
            synced_df: Синхронизированный DataFrame
            prefix: Префикс для признаков
        
        Returns:
            DataFrame с добавленными признаками
        """
        try:
            # Добавление колонок с префиксом
            for col in synced_df.columns:
                # Пропускаем колонку target - она должна остаться только из основного таймфрейма
                if col == 'target':
                    continue
                    
                if col not in primary_df.columns:
                    prefixed_col = f"{prefix}{col}"
                    primary_df[prefixed_col] = synced_df[col]
            
            self.logger.debug(f"Добавлено {len([col for col in synced_df.columns if col != 'target'])} признаков с префиксом {prefix}")
            return primary_df
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления признаков с префиксом {prefix}: {e}")
            return primary_df
    
    def get_timeframe_features(self, timeframe: str) -> List[str]:
        """
        Получение списка признаков для таймфрейма.
        
        Args:
            timeframe: Название таймфрейма
        
        Returns:
            Список признаков
        """
        return self.features_per_timeframe.get(timeframe, [])
    
    def get_all_timeframes(self) -> List[str]:
        """
        Получение списка всех таймфреймов.
        
        Returns:
            Список таймфреймов
        """
        return list(self.features_per_timeframe.keys())
    
    def get_timeframe_prefix(self, timeframe: str) -> str:
        """
        Получение префикса для таймфрейма.
        
        Args:
            timeframe: Название таймфрейма
        
        Returns:
            Префикс
        """
        return self.prefixes.get(timeframe, f"{timeframe}_")
    
    def validate_timeframe_data(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """
        Валидация данных множественных таймфреймов.
        
        Args:
            data_dict: Словарь с данными для каждого таймфрейма
        
        Returns:
            True если данные валидны
        """
        try:
            if not data_dict:
                self.logger.error("Пустой словарь данных")
                return False
            
            # Проверка наличия основного таймфрейма
            if self.primary_timeframe not in data_dict:
                self.logger.error(f"Основной таймфрейм {self.primary_timeframe} отсутствует")
                return False
            
            # Проверка каждого таймфрейма
            for timeframe, df in data_dict.items():
                if df is None or df.empty:
                    self.logger.error(f"Пустые данные для таймфрейма {timeframe}")
                    return False
                
                # Проверка наличия необходимых колонок
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    self.logger.error(f"Отсутствуют колонки для {timeframe}: {missing_columns}")
                    return False
            
            self.logger.info("Валидация данных множественных таймфреймов пройдена")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации данных: {e}")
            return False
    
    def get_processing_statistics(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Получение статистики обработки множественных таймфреймов.
        
        Args:
            data_dict: Словарь с данными для каждого таймфрейма
        
        Returns:
            Словарь со статистикой
        """
        stats = {
            'enabled': self.enabled,
            'timeframes_count': len(data_dict),
            'timeframes': list(data_dict.keys()),
            'primary_timeframe': self.primary_timeframe,
            'data_per_timeframe': {}
        }
        
        for timeframe, df in data_dict.items():
            stats['data_per_timeframe'][timeframe] = {
                'rows': len(df),
                'columns': len(df.columns),
                'start_date': df.index[0].isoformat() if len(df) > 0 else None,
                'end_date': df.index[-1].isoformat() if len(df) > 0 else None
            }
        
        return stats 
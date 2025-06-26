"""
Модуль для кэширования данных.

Этот модуль предоставляет функциональность для:
- Кэширования OHLCV данных на диск
- Поддержки множественных таймфреймов
- Управления метаданными и истечением кэша
- Автоматической очистки устаревших данных
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import hashlib

class DataCache:
    """
    Класс для кэширования данных на диск.
    
    Поддерживает:
    - Кэширование данных для множественных таймфреймов
    - Метаданные с информацией о данных
    - Автоматическое истечение кэша
    - Управление размером кэша
    """
    
    def __init__(self, config: Dict):
        """
        Инициализация кэша данных.
        
        Args:
            config: Конфигурация кэширования
        """
        self.config = config
        self.logger = logger.bind(name="DataCache")
        
        # Настройки кэша
        self.enabled = config.get('enabled', True)
        self.cache_dir = Path(config.get('directory', 'data/cache'))
        self.expiration_hours = config.get('expiration_hours', 24)
        self.max_cache_size_mb = config.get('max_cache_size_mb', 1000)
        
        # Создание директории кэша
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Кэш инициализирован: {self.cache_dir}")
        
        # Статистика кэша
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0,
            'deletions': 0,
            'total_size_mb': 0
        }
    
    def _generate_cache_key(self, symbol: str, timeframe: str) -> str:
        """
        Генерация ключа кэша для символа и таймфрейма.
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
        
        Returns:
            Уникальный ключ кэша
        """
        # Создание хэша для уникального ключа
        key_string = f"{symbol}_{timeframe}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_paths(self, symbol: str, timeframe: str) -> Tuple[Path, Path]:
        """
        Получение путей к файлам кэша.
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
        
        Returns:
            Tuple с путями к данным и метаданным
        """
        cache_key = self._generate_cache_key(symbol, timeframe)
        data_path = self.cache_dir / f"{cache_key}_data.pkl"
        metadata_path = self.cache_dir / f"{cache_key}_metadata.json"
        return data_path, metadata_path
    
    def get_data(self, symbol: str, timeframe: str = None) -> Optional[pd.DataFrame]:
        """
        Получение данных из кэша.
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм (если None, используется primary_timeframe)
        
        Returns:
            DataFrame с данными или None если данные не найдены/устарели
        """
        if not self.enabled:
            return None
        
        try:
            # Использование primary_timeframe если не указан
            if timeframe is None:
                timeframe = "1h"  # По умолчанию
            
            data_path, metadata_path = self._get_cache_paths(symbol, timeframe)
            
            # Проверка существования файлов
            if not data_path.exists() or not metadata_path.exists():
                self.cache_stats['misses'] += 1
                self.logger.debug(f"Кэш miss для {symbol} на {timeframe}")
                return None
            
            # Загрузка метаданных
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Проверка истечения кэша
            if self._is_cache_expired(metadata):
                self.logger.info(f"Кэш истек для {symbol} на {timeframe}")
                self.delete_data(symbol, timeframe)
                self.cache_stats['misses'] += 1
                return None
            
            # Загрузка данных
            with open(data_path, 'rb') as f:
                df = pickle.load(f)
            
            self.cache_stats['hits'] += 1
            self.logger.info(f"Кэш hit для {symbol} на {timeframe}: {len(df)} записей")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки из кэша для {symbol} на {timeframe}: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    def save_data(self, symbol: str, df: pd.DataFrame, timeframe: str = None) -> bool:
        """
        Сохранение данных в кэш.
        
        Args:
            symbol: Торговая пара
            df: DataFrame с данными
            timeframe: Таймфрейм (если None, используется primary_timeframe)
        
        Returns:
            True если сохранение успешно
        """
        if not self.enabled or df is None or df.empty:
            return False
        
        try:
            # Использование primary_timeframe если не указан
            if timeframe is None:
                timeframe = "1h"  # По умолчанию
            
            data_path, metadata_path = self._get_cache_paths(symbol, timeframe)
            
            # Создание метаданных
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'cached_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=self.expiration_hours)).isoformat(),
                'rows': len(df),
                'columns': list(df.columns),
                'start_date': df.index[0].isoformat() if len(df) > 0 else None,
                'end_date': df.index[-1].isoformat() if len(df) > 0 else None,
                'size_bytes': 0  # Будет обновлено после сохранения
            }
            
            # Сохранение данных
            with open(data_path, 'wb') as f:
                pickle.dump(df, f)
            
            # Обновление размера в метаданных
            metadata['size_bytes'] = data_path.stat().st_size
            
            # Сохранение метаданных
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.cache_stats['saves'] += 1
            self.logger.info(f"Данные сохранены в кэш: {symbol} на {timeframe} ({len(df)} записей)")
            
            # Проверка размера кэша
            self._check_cache_size()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения в кэш для {symbol} на {timeframe}: {e}")
            return False
    
    def delete_data(self, symbol: str, timeframe: str = None) -> bool:
        """
        Удаление данных из кэша.
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм (если None, используется primary_timeframe)
        
        Returns:
            True если удаление успешно
        """
        try:
            # Использование primary_timeframe если не указан
            if timeframe is None:
                timeframe = "1h"  # По умолчанию
            
            data_path, metadata_path = self._get_cache_paths(symbol, timeframe)
            
            # Удаление файлов
            if data_path.exists():
                data_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            self.cache_stats['deletions'] += 1
            self.logger.debug(f"Данные удалены из кэша: {symbol} на {timeframe}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка удаления из кэша для {symbol} на {timeframe}: {e}")
            return False
    
    def get_multi_timeframe_data(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Получение данных для множественных таймфреймов.
        
        Args:
            symbol: Торговая пара
            timeframes: Список таймфреймов
        
        Returns:
            Словарь с данными для каждого таймфрейма
        """
        multi_tf_data = {}
        
        for timeframe in timeframes:
            df = self.get_data(symbol, timeframe)
            if df is not None and not df.empty:
                multi_tf_data[timeframe] = df
        
        self.logger.info(f"Загружено {len(multi_tf_data)} таймфреймов для {symbol}")
        return multi_tf_data
    
    def save_multi_timeframe_data(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """
        Сохранение данных для множественных таймфреймов.
        
        Args:
            symbol: Торговая пара
            data_dict: Словарь с данными для каждого таймфрейма
        
        Returns:
            True если сохранение успешно
        """
        success_count = 0
        
        for timeframe, df in data_dict.items():
            if self.save_data(symbol, df, timeframe):
                success_count += 1
        
        self.logger.info(f"Сохранено {success_count}/{len(data_dict)} таймфреймов для {symbol}")
        return success_count == len(data_dict)
    
    def _is_cache_expired(self, metadata: Dict) -> bool:
        """
        Проверка истечения кэша.
        
        Args:
            metadata: Метаданные кэша
        
        Returns:
            True если кэш истек
        """
        try:
            expires_at = datetime.fromisoformat(metadata['expires_at'])
            return datetime.now() > expires_at
        except Exception as e:
            self.logger.error(f"Ошибка проверки истечения кэша: {e}")
            return True
    
    def _check_cache_size(self):
        """
        Проверка размера кэша и очистка при необходимости.
        """
        try:
            total_size = 0
            cache_files = []
            
            # Подсчет размера всех файлов кэша
            for file_path in self.cache_dir.glob("*_data.pkl"):
                size = file_path.stat().st_size
                total_size += size
                cache_files.append((file_path, size))
            
            total_size_mb = total_size / (1024 * 1024)
            self.cache_stats['total_size_mb'] = total_size_mb
            
            # Если размер превышает лимит, удаляем старые файлы
            if total_size_mb > self.max_cache_size_mb:
                self.logger.warning(f"Размер кэша превышает лимит: {total_size_mb:.2f}MB > {self.max_cache_size_mb}MB")
                self._cleanup_cache(cache_files)
                
        except Exception as e:
            self.logger.error(f"Ошибка проверки размера кэша: {e}")
    
    def _cleanup_cache(self, cache_files: List[Tuple[Path, int]]):
        """
        Очистка кэша от старых файлов.
        
        Args:
            cache_files: Список файлов с их размерами
        """
        try:
            # Сортировка файлов по времени модификации (старые первыми)
            cache_files.sort(key=lambda x: x[0].stat().st_mtime)
            
            # Удаление старых файлов до достижения лимита
            current_size = sum(size for _, size in cache_files)
            target_size = self.max_cache_size_mb * 1024 * 1024 * 0.8  # 80% от лимита
            
            for file_path, size in cache_files:
                if current_size <= target_size:
                    break
                
                # Удаление файла данных и метаданных
                if file_path.exists():
                    file_path.unlink()
                
                metadata_path = file_path.with_name(file_path.name.replace('_data.pkl', '_metadata.json'))
                if metadata_path.exists():
                    metadata_path.unlink()
                
                current_size -= size
                self.cache_stats['deletions'] += 1
            
            self.logger.info(f"Кэш очищен. Новый размер: {current_size / (1024 * 1024):.2f}MB")
            
        except Exception as e:
            self.logger.error(f"Ошибка очистки кэша: {e}")
    
    def clear_all_cache(self) -> bool:
        """
        Полная очистка кэша.
        
        Returns:
            True если очистка успешна
        """
        try:
            deleted_count = 0
            
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    deleted_count += 1
            
            self.cache_stats['deletions'] += deleted_count
            self.cache_stats['total_size_mb'] = 0
            
            self.logger.info(f"Кэш полностью очищен. Удалено файлов: {deleted_count}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка полной очистки кэша: {e}")
            return False
    
    def get_cache_statistics(self) -> Dict:
        """
        Получение статистики кэша.
        
        Returns:
            Словарь со статистикой
        """
        stats = self.cache_stats.copy()
        
        # Расчет дополнительных метрик
        total_requests = stats['hits'] + stats['misses']
        if total_requests > 0:
            stats['hit_rate'] = stats['hits'] / total_requests
        else:
            stats['hit_rate'] = 0
        
        # Информация о кэше
        stats['enabled'] = self.enabled
        stats['cache_dir'] = str(self.cache_dir)
        stats['expiration_hours'] = self.expiration_hours
        stats['max_cache_size_mb'] = self.max_cache_size_mb
        
        return stats
    
    def list_cached_symbols(self) -> List[str]:
        """
        Получение списка символов в кэше.
        
        Returns:
            Список символов
        """
        symbols = set()
        
        try:
            for metadata_path in self.cache_dir.glob("*_metadata.json"):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    symbols.add(metadata.get('symbol', 'unknown'))
        except Exception as e:
            self.logger.error(f"Ошибка получения списка символов: {e}")
        
        return list(symbols)
    
    def list_cached_timeframes(self, symbol: str) -> List[str]:
        """
        Получение списка таймфреймов для символа в кэше.
        
        Args:
            symbol: Торговая пара
        
        Returns:
            Список таймфреймов
        """
        timeframes = []
        
        try:
            for metadata_path in self.cache_dir.glob("*_metadata.json"):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if metadata.get('symbol') == symbol:
                        timeframes.append(metadata.get('timeframe', 'unknown'))
        except Exception as e:
            self.logger.error(f"Ошибка получения списка таймфреймов для {symbol}: {e}")
        
        return timeframes 
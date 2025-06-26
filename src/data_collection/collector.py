"""
Модуль для сбора данных с криптовалютных бирж.

Этот модуль предоставляет функциональность для:
- Сбора OHLCV данных с различных бирж
- Поддержки множественных таймфреймов
- Валидации и очистки данных
- Обработки ошибок и повторных попыток
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import ccxt
from datetime import datetime, timedelta
from loguru import logger
import time

class DataCollector:
    """
    Класс для сбора данных с криптовалютных бирж.
    
    Поддерживает:
    - Множественные биржи через CCXT
    - Множественные таймфреймы
    - Валидацию данных
    - Обработку ошибок
    """
    
    def __init__(self, config: Dict):
        """
        Инициализация коллектора данных.
        
        Args:
            config: Конфигурация сбора данных
        """
        self.config = config
        self.logger = logger.bind(name="DataCollector")
        
        # Настройки биржи
        self.exchange_name = config.get('exchange', 'binance')
        self.symbols = config.get('symbols', ['BTC/USDT'])
        
        # Попытка получить таймфреймы из глобального конфига (если есть)
        global_cfg = config.get('global_config') if 'global_config' in config else None
        if global_cfg:
            mtf_cfg = global_cfg.get('preprocessing', {}).get('multi_timeframe', {})
            self.timeframes = mtf_cfg.get('timeframes', config.get('timeframes', ['1h']))
            self.primary_timeframe = mtf_cfg.get('primary_timeframe', config.get('primary_timeframe', '1h'))
        else:
            self.timeframes = config.get('timeframes', ['1h'])
            self.primary_timeframe = config.get('primary_timeframe', '1h')
        self.limits = config.get('limits', {'1h': 1000})
        
        # Инициализация биржи
        self.exchange = self._initialize_exchange()
        
        # Статистика сбора
        self.collection_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_candles': 0,
            'timeframes_collected': []
        }
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """
        Инициализация биржи через CCXT.
        
        Returns:
            Объект биржи
        """
        try:
            # Создание объекта биржи
            exchange_class = getattr(ccxt, self.exchange_name)
            exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000,
                'rateLimit': 1000
            })
            
            # Загрузка рынков
            exchange.load_markets()
            
            self.logger.info(f"Биржа {self.exchange_name} инициализирована")
            self.logger.info(f"Доступные таймфреймы: {list(exchange.timeframes.keys())}")
            
            return exchange
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации биржи {self.exchange_name}: {e}")
            raise
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = None, limit: int = None) -> Optional[pd.DataFrame]:
        """
        Сбор OHLCV данных для одного символа и таймфрейма.
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм (если None, используется primary_timeframe)
            limit: Количество свечей (если None, берется из конфигурации)
        
        Returns:
            DataFrame с OHLCV данными или None при ошибке
        """
        try:
            # Использование значений по умолчанию
            if timeframe is None:
                timeframe = self.primary_timeframe
            
            if limit is None:
                limit = self.limits.get(timeframe, 1000)
            
            self.logger.info(f"Собираем данные для {symbol} на таймфрейме {timeframe} (лимит: {limit})")
            
            # Проверка доступности символа
            if symbol not in self.exchange.markets:
                self.logger.error(f"Символ {symbol} недоступен на {self.exchange_name}")
                return None
            
            # Проверка доступности таймфрейма
            if timeframe not in self.exchange.timeframes:
                self.logger.error(f"Таймфрейм {timeframe} недоступен на {self.exchange_name}")
                return None
            
            # Сбор данных
            start_time = time.time()
            ohlcv_data = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv_data:
                self.logger.warning(f"Нет данных для {symbol} на таймфрейме {timeframe}")
                return None
            
            # Преобразование в DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Валидация данных
            if not self._validate_data(df, symbol, timeframe):
                return None
            
            # Обновление статистики
            collection_time = time.time() - start_time
            self.collection_stats['total_requests'] += 1
            self.collection_stats['successful_requests'] += 1
            self.collection_stats['total_candles'] += len(df)
            
            self.logger.info(f"Собрано {len(df)} свечей за {collection_time:.2f} секунд")
            self.logger.info(f"Период: {df.index[0]} - {df.index[-1]}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка сбора данных для {symbol} на {timeframe}: {e}")
            self.collection_stats['failed_requests'] += 1
            return None
    
    def fetch_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Сбор данных для всех таймфреймов для одного символа.
        
        Args:
            symbol: Торговая пара
        
        Returns:
            Словарь с данными для каждого таймфрейма
        """
        try:
            self.logger.info(f"Собираем данные для {symbol} на всех таймфреймах")
            
            multi_tf_data = {}
            
            for timeframe in self.timeframes:
                self.logger.info(f"Обрабатываем таймфрейм {timeframe}")
                
                # Получение лимита для данного таймфрейма
                limit = self.limits.get(timeframe, 1000)
                
                # Сбор данных
                df = self.fetch_ohlcv(symbol, timeframe, limit)
                
                if df is not None and not df.empty:
                    multi_tf_data[timeframe] = df
                    self.collection_stats['timeframes_collected'].append(timeframe)
                else:
                    self.logger.warning(f"Не удалось получить данные для {timeframe}")
            
            self.logger.info(f"Собраны данные для {len(multi_tf_data)} таймфреймов")
            
            return multi_tf_data
            
        except Exception as e:
            self.logger.error(f"Ошибка сбора мультитаймфреймных данных для {symbol}: {e}")
            return {}
    
    def fetch_all_symbols_data(self, timeframes: List[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Сбор данных для всех символов и таймфреймов.
        
        Args:
            timeframes: Список таймфреймов (если None, используются все из конфигурации)
        
        Returns:
            Словарь с данными для всех символов и таймфреймов
        """
        try:
            if timeframes is None:
                timeframes = self.timeframes
            
            self.logger.info(f"Собираем данные для {len(self.symbols)} символов на {len(timeframes)} таймфреймах")
            
            all_data = {}
            
            for symbol in self.symbols:
                self.logger.info(f"Обрабатываем символ {symbol}")
                
                symbol_data = {}
                
                for timeframe in timeframes:
                    limit = self.limits.get(timeframe, 1000)
                    df = self.fetch_ohlcv(symbol, timeframe, limit)
                    
                    if df is not None and not df.empty:
                        symbol_data[timeframe] = df
                
                if symbol_data:
                    all_data[symbol] = symbol_data
                    self.logger.info(f"Собраны данные для {symbol}: {list(symbol_data.keys())}")
                else:
                    self.logger.warning(f"Не удалось собрать данные для {symbol}")
            
            self.logger.info(f"Сбор данных завершен. Обработано символов: {len(all_data)}")
            
            return all_data
            
        except Exception as e:
            self.logger.error(f"Ошибка сбора данных для всех символов: {e}")
            return {}
    
    def _validate_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Валидация собранных данных.
        
        Args:
            df: DataFrame с данными
            symbol: Торговая пара
            timeframe: Таймфрейм
        
        Returns:
            True если данные валидны
        """
        try:
            # Проверка на пустые данные
            if df.empty:
                self.logger.error(f"Пустые данные для {symbol} на {timeframe}")
                return False
            
            # Проверка на минимальное количество свечей
            min_candles = 50
            if len(df) < min_candles:
                self.logger.warning(f"Мало данных для {symbol} на {timeframe}: {len(df)} < {min_candles}")
            
            # Проверка на NaN значения
            nan_count = df.isnull().sum().sum()
            if nan_count > 0:
                self.logger.warning(f"Найдено {nan_count} NaN значений в данных {symbol} на {timeframe}")
            
            # Проверка на отрицательные цены
            negative_prices = ((df[['open', 'high', 'low', 'close']] < 0).any().any())
            if negative_prices:
                self.logger.error(f"Найдены отрицательные цены в данных {symbol} на {timeframe}")
                return False
            
            # Проверка на логические ошибки в OHLC
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['open'] > df['high']) |
                (df['close'] > df['high']) |
                (df['open'] < df['low']) |
                (df['close'] < df['low'])
            ).any()
            
            if invalid_ohlc:
                self.logger.error(f"Найдены логические ошибки в OHLC данных {symbol} на {timeframe}")
                return False
            
            # Проверка на дубликаты
            duplicates = df.index.duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"Найдено {duplicates} дубликатов в данных {symbol} на {timeframe}")
                df = df[~df.index.duplicated(keep='first')]
            
            # Проверка на правильную сортировку
            if not df.index.is_monotonic_increasing:
                self.logger.warning(f"Данные {symbol} на {timeframe} не отсортированы по времени")
                df.sort_index(inplace=True)
            
            self.logger.info(f"Валидация данных {symbol} на {timeframe} пройдена успешно")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации данных {symbol} на {timeframe}: {e}")
            return False
    
    def get_collection_statistics(self) -> Dict:
        """
        Получение статистики сбора данных.
        
        Returns:
            Словарь со статистикой
        """
        stats = self.collection_stats.copy()
        
        # Расчет дополнительных метрик
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
        else:
            stats['success_rate'] = 0
        
        stats['exchange'] = self.exchange_name
        stats['symbols'] = self.symbols
        stats['timeframes'] = self.timeframes
        
        return stats
    
    def get_available_timeframes(self) -> List[str]:
        """
        Получение доступных таймфреймов на бирже.
        
        Returns:
            Список доступных таймфреймов
        """
        try:
            return list(self.exchange.timeframes.keys())
        except Exception as e:
            self.logger.error(f"Ошибка получения доступных таймфреймов: {e}")
            return []
    
    def get_available_symbols(self) -> List[str]:
        """
        Получение доступных символов на бирже.
        
        Returns:
            Список доступных символов
        """
        try:
            return list(self.exchange.markets.keys())
        except Exception as e:
            self.logger.error(f"Ошибка получения доступных символов: {e}")
            return []
    
    def test_connection(self) -> bool:
        """
        Тестирование подключения к бирже.
        
        Returns:
            True если подключение успешно
        """
        try:
            # Попытка получить время сервера
            server_time = self.exchange.fetch_time()
            self.logger.info(f"Подключение к {self.exchange_name} успешно. Время сервера: {datetime.fromtimestamp(server_time/1000)}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка подключения к {self.exchange_name}: {e}")
            return False
    
    def fetch_ohlcv_with_history(self, symbol: str, timeframe: str = None, 
                                target_limit: int = None, batch_size: int = None) -> Optional[pd.DataFrame]:
        """
        Поэтапная подгрузка исторических данных для получения последних N свечей.
        Загружает данные партиями, начиная с самых новых и двигаясь к старым.
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            target_limit: Целевое количество свечей (последние N свечей)
            batch_size: Размер одной партии (максимум 1000 для большинства бирж)
            
        Returns:
            DataFrame с последними N свечами
        """
        try:
            # Использование значений по умолчанию
            if timeframe is None:
                timeframe = self.primary_timeframe
            
            if target_limit is None:
                target_limit = self.limits.get(timeframe, 1000)
            
            if batch_size is None:
                batch_size = 1000  # Максимальный размер для большинства бирж
            
            self.logger.info(f"Начинаем загрузку последних {target_limit} свечей для {symbol} на {timeframe}")
            self.logger.info(f"Размер партии: {batch_size}")
            
            # Проверка доступности символа и таймфрейма
            if symbol not in self.exchange.markets:
                self.logger.error(f"Символ {symbol} недоступен на {self.exchange_name}")
                return None
            
            if timeframe not in self.exchange.timeframes:
                self.logger.error(f"Таймфрейм {timeframe} недоступен на {self.exchange_name}")
                return None
            
            # Список для хранения всех данных
            all_data = []
            since = None  # Начинаем с самых новых данных
            batch_count = 0
            max_batches = 200  # Увеличиваем лимит для загрузки большего количества данных
            
            start_time = time.time()
            
            # Сначала загружаем самую свежую партию для определения временных рамок
            try:
                initial_batch = self.exchange.fetch_ohlcv(symbol, timeframe, limit=batch_size)
                if not initial_batch:
                    self.logger.error("Не удалось загрузить начальную партию данных")
                    return None
                
                # Добавляем начальную партию
                all_data.extend(initial_batch)
                batch_count += 1
                
                # Определяем временной интервал для одной свечи (в миллисекундах)
                if len(initial_batch) >= 2:
                    time_interval = initial_batch[1][0] - initial_batch[0][0]
                else:
                    # Если только одна свеча, используем стандартные интервалы
                    timeframes_ms = {
                        '1m': 60 * 1000,
                        '3m': 3 * 60 * 1000,
                        '5m': 5 * 60 * 1000,
                        '15m': 15 * 60 * 1000,
                        '30m': 30 * 60 * 1000,
                        '1h': 60 * 60 * 1000,
                        '2h': 2 * 60 * 60 * 1000,
                        '4h': 4 * 60 * 60 * 1000,
                        '6h': 6 * 60 * 60 * 1000,
                        '8h': 8 * 60 * 60 * 1000,
                        '12h': 12 * 60 * 60 * 1000,
                        '1d': 24 * 60 * 60 * 1000,
                    }
                    time_interval = timeframes_ms.get(timeframe, 5 * 60 * 1000)  # По умолчанию 5m
                
                self.logger.info(f"Временной интервал между свечами: {time_interval}ms")
                
            except Exception as e:
                self.logger.error(f"Ошибка при загрузке начальной партии: {e}")
                return None
            
            # Продолжаем загружать данные, пока не достигнем целевого количества
            while len(all_data) < target_limit and batch_count < max_batches:
                batch_count += 1
                self.logger.info(f"Загружаем партию {batch_count}/{max_batches} (всего свечей: {len(all_data)})")
                
                try:
                    # Вычисляем since для следующей партии (идем назад во времени)
                    if since is None:
                        # Для первой итерации используем timestamp самой старой свечи из начальной партии
                        since = initial_batch[0][0] - (batch_size * time_interval)
                    else:
                        # Для последующих итераций идем еще дальше назад
                        since = since - (batch_size * time_interval)
                    
                    # Загружаем партию данных
                    batch_data = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_size)
                    
                    if not batch_data:
                        self.logger.info("Больше данных нет, завершаем загрузку")
                        break
                    
                    # Добавляем данные в общий список
                    all_data.extend(batch_data)
                    
                    # Добавляем небольшую задержку между запросами
                    time.sleep(0.1)
                    
                    # Проверяем, есть ли еще данные (если получили меньше batch_size)
                    if len(batch_data) < batch_size:
                        self.logger.info("Получено меньше данных чем размер партии, завершаем")
                        break
                    
                except Exception as e:
                    self.logger.error(f"Ошибка при загрузке партии {batch_count}: {e}")
                    break
            
            if not all_data:
                self.logger.warning(f"Не удалось загрузить данные для {symbol} на {timeframe}")
                return None
            
            # Преобразование в DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Сортируем по времени (от старых к новым)
            df.sort_index(inplace=True)
            
            # Удаляем дубликаты если есть
            df = df[~df.index.duplicated(keep='first')]
            
            # Оставляем только последние target_limit свечей
            if len(df) > target_limit:
                df = df.tail(target_limit)
                self.logger.info(f"Оставлено последних {target_limit} свечей из {len(all_data)}")
            
            # Валидация данных
            if not self._validate_data(df, symbol, timeframe):
                return None
            
            # Обновление статистики
            collection_time = time.time() - start_time
            self.collection_stats['total_requests'] += batch_count
            self.collection_stats['successful_requests'] += batch_count
            self.collection_stats['total_candles'] += len(df)
            
            self.logger.info(f"✅ Загрузка завершена!")
            self.logger.info(f"   Загружено партий: {batch_count}")
            self.logger.info(f"   Всего свечей: {len(df)}")
            self.logger.info(f"   Время загрузки: {collection_time:.2f} секунд")
            self.logger.info(f"   Период: {df.index[0]} - {df.index[-1]}")
            self.logger.info(f"   Скорость: {len(df)/collection_time:.1f} свечей/сек")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки для {symbol} на {timeframe}: {e}")
            self.collection_stats['failed_requests'] += 1
            return None 
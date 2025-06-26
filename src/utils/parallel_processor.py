"""
Модуль для параллельной обработки данных
Максимальное использование производительности компьютера
"""

import os
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger
import multiprocessing as mp


class ParallelProcessor:
    """
    Класс для параллельной обработки данных с максимальной производительностью
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация параллельного процессора
        
        Args:
            config: Конфигурация производительности
        """
        self.config = config or {}
        
        # Настройки многопоточности
        self.max_workers = self.config.get('max_workers', -1)
        if self.max_workers == -1:
            self.max_workers = psutil.cpu_count()
        
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.use_threadpool = self.config.get('use_threadpool', True)
        
        # Настройки кэширования
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.memory_cache = {}
        
        logger.info(f"🚀 ParallelProcessor инициализирован: {self.max_workers} ядер")
    
    def parallel_map(self, func: Callable, data: List, 
                    use_processes: bool = False, 
                    chunk_size: Optional[int] = None) -> List:
        """
        Параллельное применение функции к данным
        
        Args:
            func: Функция для применения
            data: Данные для обработки
            use_processes: Использовать процессы вместо потоков
            chunk_size: Размер чанка (None для автоматического)
        
        Returns:
            Результаты обработки
        """
        if not data:
            return []
        
        # Определяем размер чанка
        if chunk_size is None:
            chunk_size = max(1, len(data) // (self.max_workers * 4))
        
        # Разбиваем данные на чанки
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        logger.info(f"🔧 Параллельная обработка: {len(chunks)} чанков по {chunk_size} элементов")
        
        # Выбираем executor
        if use_processes:
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor
        
        # Параллельная обработка
        with executor_class(max_workers=self.max_workers) as executor:
            futures = [executor.submit(func, chunk) for chunk in chunks]
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Ошибка параллельной обработки: {e}")
                    results.append(None)
        
        # Объединяем результаты
        if isinstance(results[0], (list, tuple)):
            return [item for sublist in results if sublist is not None for item in sublist]
        else:
            return results
    
    def parallel_dataframe_processing(self, df: pd.DataFrame, 
                                    func: Callable,
                                    column_chunks: bool = False,
                                    use_processes: bool = False) -> pd.DataFrame:
        """
        Параллельная обработка DataFrame
        
        Args:
            df: DataFrame для обработки
            func: Функция обработки
            column_chunks: Разбивать по колонкам вместо строк
            use_processes: Использовать процессы
        
        Returns:
            Обработанный DataFrame
        """
        if df.empty:
            return df
        
        if column_chunks:
            # Разбиваем по колонкам
            columns = list(df.columns)
            chunk_size = max(1, len(columns) // self.max_workers)
            column_chunks = [columns[i:i+chunk_size] for i in range(0, len(columns), chunk_size)]
            
            logger.info(f"🔧 Параллельная обработка колонок: {len(column_chunks)} чанков")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(func, df[cols]): cols 
                    for cols in column_chunks
                }
                
                results = {}
                for future in as_completed(futures):
                    cols = futures[future]
                    try:
                        result = future.result()
                        results.update({col: result[col] for col in cols})
                    except Exception as e:
                        logger.error(f"❌ Ошибка обработки колонок {cols}: {e}")
                
                return pd.DataFrame(results)
        
        else:
            # Разбиваем по строкам
            chunk_size = max(1, len(df) // (self.max_workers * 4))
            row_chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
            
            logger.info(f"🔧 Параллельная обработка строк: {len(row_chunks)} чанков")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(func, chunk) for chunk in row_chunks]
                
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"❌ Ошибка обработки чанка: {e}")
                        results.append(pd.DataFrame())
                
                return pd.concat(results, ignore_index=True)
    
    def parallel_feature_engineering(self, df: pd.DataFrame, 
                                   feature_functions: List[Callable],
                                   use_processes: bool = False) -> pd.DataFrame:
        """
        Параллельный инжиниринг признаков
        
        Args:
            df: Исходный DataFrame
            feature_functions: Список функций для создания признаков
            use_processes: Использовать процессы
        
        Returns:
            DataFrame с новыми признаками
        """
        if not feature_functions:
            return df
        
        logger.info(f"🔧 Параллельный инжиниринг {len(feature_functions)} признаков")
        
        # Разбиваем функции на чанки
        chunk_size = max(1, len(feature_functions) // self.max_workers)
        function_chunks = [feature_functions[i:i+chunk_size] 
                          for i in range(0, len(feature_functions), chunk_size)]
        
        def process_feature_chunk(funcs):
            """Обработка чанка функций признаков"""
            result_df = df.copy()
            for func in funcs:
                try:
                    result_df = func(result_df)
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка функции признака: {e}")
            return result_df
        
        # Параллельная обработка
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_feature_chunk, funcs) 
                      for funcs in function_chunks]
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Ошибка обработки признаков: {e}")
                    results.append(df)
        
        # Объединяем результаты (берем последний, так как каждая функция модифицирует DataFrame)
        if results:
            return results[-1]
        else:
            return df
    
    def parallel_model_training(self, models: List, 
                              train_data: Tuple,
                              use_processes: bool = False) -> List:
        """
        Параллельное обучение моделей
        
        Args:
            models: Список моделей для обучения
            train_data: Данные для обучения (X, y)
            use_processes: Использовать процессы
        
        Returns:
            Список обученных моделей
        """
        if not models:
            return []
        
        logger.info(f"🧠 Параллельное обучение {len(models)} моделей")
        
        def train_single_model(model):
            """Обучение одной модели"""
            try:
                if hasattr(model, 'fit'):
                    model.fit(*train_data)
                elif hasattr(model, 'train'):
                    model.train(*train_data)
                return model
            except Exception as e:
                logger.error(f"❌ Ошибка обучения модели: {e}")
                return None
        
        # Параллельное обучение
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(models))) as executor:
            futures = [executor.submit(train_single_model, model) for model in models]
            
            trained_models = []
            for future in as_completed(futures):
                try:
                    model = future.result()
                    if model is not None:
                        trained_models.append(model)
                except Exception as e:
                    logger.error(f"❌ Ошибка получения результата обучения: {e}")
        
        logger.info(f"✅ Обучено {len(trained_models)}/{len(models)} моделей")
        return trained_models
    
    def parallel_hyperparameter_optimization(self, 
                                           model_class,
                                           param_spaces: List[Dict],
                                           train_data: Tuple,
                                           eval_func: Callable,
                                           use_processes: bool = False) -> List[Dict]:
        """
        Параллельная оптимизация гиперпараметров
        
        Args:
            model_class: Класс модели
            param_spaces: Список пространств параметров
            train_data: Данные для обучения
            eval_func: Функция оценки
            use_processes: Использовать процессы
        
        Returns:
            Список лучших параметров для каждого пространства
        """
        if not param_spaces:
            return []
        
        logger.info(f"⚙️ Параллельная оптимизация {len(param_spaces)} пространств параметров")
        
        def optimize_single_space(param_space):
            """Оптимизация одного пространства параметров"""
            try:
                # Простая случайная оптимизация (можно заменить на более сложную)
                best_score = -np.inf
                best_params = None
                
                for _ in range(10):  # 10 попыток
                    # Случайные параметры
                    params = {}
                    for key, value_range in param_space.items():
                        if isinstance(value_range, (list, tuple)):
                            params[key] = np.random.choice(value_range)
                        else:
                            params[key] = value_range
                    
                    # Создаем и обучаем модель
                    model = model_class(**params)
                    model.fit(*train_data)
                    
                    # Оцениваем
                    score = eval_func(model, *train_data)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                
                return {'params': best_params, 'score': best_score}
                
            except Exception as e:
                logger.error(f"❌ Ошибка оптимизации пространства: {e}")
                return None
        
        # Параллельная оптимизация
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(param_spaces))) as executor:
            futures = [executor.submit(optimize_single_space, space) for space in param_spaces]
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"❌ Ошибка получения результата оптимизации: {e}")
        
        logger.info(f"✅ Оптимизировано {len(results)}/{len(param_spaces)} пространств")
        return results
    
    def parallel_backtesting(self, 
                           models: List,
                           data: pd.DataFrame,
                           backtest_configs: List[Dict],
                           use_processes: bool = False) -> List[Dict]:
        """
        Параллельный бэктестинг
        
        Args:
            models: Список моделей
            data: Данные для бэктестинга
            backtest_configs: Конфигурации бэктестинга
            use_processes: Использовать процессы
        
        Returns:
            Список результатов бэктестинга
        """
        if not models or not backtest_configs:
            return []
        
        logger.info(f"📈 Параллельный бэктестинг {len(models)} моделей с {len(backtest_configs)} конфигурациями")
        
        def run_single_backtest(model, config):
            """Запуск одного бэктеста"""
            try:
                # Упрощенный бэктест (замените на реальную реализацию)
                predictions = model.predict(data)
                
                # Простая симуляция торговли
                returns = []
                position = 0
                
                for i, pred in enumerate(predictions):
                    if pred > 0.6 and position == 0:  # Покупка
                        position = 1
                    elif pred < 0.4 and position == 1:  # Продажа
                        position = 0
                        returns.append(0.1)  # Упрощенная прибыль
                
                total_return = sum(returns) if returns else 0
                
                return {
                    'model_id': id(model),
                    'config': config,
                    'total_return': total_return,
                    'num_trades': len(returns),
                    'sharpe_ratio': np.mean(returns) / np.std(returns) if len(returns) > 1 else 0
                }
                
            except Exception as e:
                logger.error(f"❌ Ошибка бэктеста: {e}")
                return None
        
        # Создаем задачи для бэктестинга
        tasks = []
        for model in models:
            for config in backtest_configs:
                tasks.append((model, config))
        
        # Параллельный бэктестинг
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tasks))) as executor:
            futures = [executor.submit(run_single_backtest, model, config) 
                      for model, config in tasks]
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"❌ Ошибка получения результата бэктеста: {e}")
        
        logger.info(f"✅ Выполнено {len(results)}/{len(tasks)} бэктестов")
        return results
    
    def get_performance_stats(self) -> Dict:
        """
        Получение статистики производительности
        
        Returns:
            Словарь со статистикой
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            stats = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'max_workers': self.max_workers,
                'active_threads': threading.active_count()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {}


class MemoryCache:
    """
    Кэш в памяти для ускорения обработки
    """
    
    def __init__(self, max_size_mb: int = 1024):
        """
        Инициализация кэша
        
        Args:
            max_size_mb: Максимальный размер кэша в МБ
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = {}
        self.access_times = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Any:
        """
        Получение значения из кэша
        
        Args:
            key: Ключ
        
        Returns:
            Значение или None
        """
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any):
        """
        Установка значения в кэш
        
        Args:
            key: Ключ
            value: Значение
        """
        with self.lock:
            # Проверяем размер кэша
            current_size = sum(self._get_size(v) for v in self.cache.values())
            value_size = self._get_size(value)
            
            # Если кэш переполнен, удаляем старые записи
            while current_size + value_size > self.max_size_bytes and self.cache:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                current_size -= self._get_size(self.cache[oldest_key])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            # Добавляем новое значение
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def _get_size(self, obj: Any) -> int:
        """
        Примерная оценка размера объекта в байтах
        
        Args:
            obj: Объект
        
        Returns:
            Размер в байтах
        """
        try:
            if isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum()
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, (list, tuple)):
                return sum(self._get_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._get_size(v) for v in obj.values())
            else:
                return len(str(obj).encode('utf-8'))
        except:
            return 1024  # Примерный размер по умолчанию
    
    def clear(self):
        """Очистка кэша"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict:
        """
        Статистика кэша
        
        Returns:
            Словарь со статистикой
        """
        with self.lock:
            total_size = sum(self._get_size(v) for v in self.cache.values())
            return {
                'entries': len(self.cache),
                'size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'usage_percent': (total_size / self.max_size_bytes) * 100
            }


# Глобальный экземпляр для использования в проекте
parallel_processor = ParallelProcessor()
memory_cache = MemoryCache() 
"""
Кэширование результатов оптимизации индикаторов.

Этот модуль сохраняет и загружает результаты оптимизации индикаторов,
чтобы избежать повторных вычислений при каждом запуске системы.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from loguru import logger
import pandas as pd


class IndicatorCache:
    """
    Кэш для результатов оптимизации индикаторов.
    
    Сохраняет и загружает результаты оптимизации, чтобы избежать
    повторных вычислений при каждом запуске системы.
    """
    
    def __init__(self, cache_dir: str = "models/indicator_cache"):
        """
        Инициализация кэша индикаторов.
        
        Args:
            cache_dir: Директория для сохранения кэша
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(name="IndicatorCache")
        
        # Файлы кэша
        self.results_file = self.cache_dir / "optimization_results.json"
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        self.logger.info(f"Кэш индикаторов инициализирован: {self.cache_dir}")
    
    def save_optimization_results(self, 
                                timeframe: str, 
                                results: Dict, 
                                symbol: str = "BTC_USDT") -> bool:
        """
        Сохранение результатов оптимизации индикаторов.
        
        Args:
            timeframe: Таймфрейм
            results: Результаты оптимизации
            symbol: Торговая пара
            
        Returns:
            True если сохранение успешно
        """
        try:
            # Загружаем существующие результаты
            all_results = self.load_all_results()
            
            # Создаем ключ для результатов
            key = f"{symbol}_{timeframe}"
            
            # Добавляем метаданные
            results_with_metadata = {
                'results': results,
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'version': '1.0'
            }
            
            all_results[key] = results_with_metadata
            
            # Сохраняем результаты
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            # Обновляем метаданные кэша
            self._update_cache_metadata()
            
            self.logger.info(f"✅ Результаты оптимизации сохранены для {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения результатов: {e}")
            return False
    
    def load_optimization_results(self, 
                                timeframe: str, 
                                symbol: str = "BTC_USDT") -> Optional[Dict]:
        """
        Загрузка результатов оптимизации индикаторов.
        
        Args:
            timeframe: Таймфрейм
            symbol: Торговая пара
            
        Returns:
            Результаты оптимизации или None
        """
        try:
            if not self.results_file.exists():
                self.logger.info("Файл результатов не найден")
                return None
            
            all_results = self.load_all_results()
            key = f"{symbol}_{timeframe}"
            
            if key not in all_results:
                self.logger.info(f"Результаты для {key} не найдены")
                return None
            
            result_data = all_results[key]
            
            # Проверяем актуальность результатов (не старше 7 дней)
            # if self._is_results_outdated(result_data):
            #     self.logger.warning(f"Результаты для {key} устарели")
            #     return None
            
            self.logger.info(f"✅ Загружены результаты оптимизации для {key}")
            return result_data['results']
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки результатов: {e}")
            return None
    
    def load_all_results(self) -> Dict:
        """
        Загрузка всех сохраненных результатов.
        
        Returns:
            Словарь со всеми результатами
        """
        try:
            if not self.results_file.exists():
                return {}
            
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Ошибка загрузки всех результатов: {e}")
            return {}
    
    def get_available_results(self) -> List[Tuple[str, str, str]]:
        """
        Получение списка доступных результатов.
        
        Returns:
            Список кортежей (symbol, timeframe, timestamp)
        """
        try:
            all_results = self.load_all_results()
            available = []
            
            for key, data in all_results.items():
                if isinstance(data, dict) and 'results' in data:
                    symbol = data.get('symbol', 'Unknown')
                    timeframe = data.get('timeframe', 'Unknown')
                    timestamp = data.get('timestamp', 'Unknown')
                    available.append((symbol, timeframe, timestamp))
            
            return available
            
        except Exception as e:
            self.logger.error(f"Ошибка получения списка результатов: {e}")
            return []
    
    def get_best_results(self) -> Dict:
        """
        Получение лучших результатов для каждого таймфрейма.
        
        Returns:
            Словарь с лучшими результатами
        """
        try:
            all_results = self.load_all_results()
            best_results = {}
            
            for key, data in all_results.items():
                if isinstance(data, dict) and 'results' in data:
                    timeframe = data.get('timeframe')
                    symbol = data.get('symbol')
                    
                    if timeframe and symbol:
                        # Если для этого таймфрейма уже есть результат, сравниваем по score
                        if timeframe in best_results:
                            current_score = best_results[timeframe]['score']
                            new_score = data['results'].get('best_score', 0)
                            
                            if new_score > current_score:
                                best_results[timeframe] = {
                                    'symbol': symbol,
                                    'score': new_score,
                                    'data': data['results']
                                }
                        else:
                            best_results[timeframe] = {
                                'symbol': symbol,
                                'score': data['results'].get('best_score', 0),
                                'data': data['results']
                            }
            
            return best_results
            
        except Exception as e:
            self.logger.error(f"Ошибка получения лучших результатов: {e}")
            return {}
    
    def clear_cache(self) -> bool:
        """
        Очистка всего кэша.
        
        Returns:
            True если очистка успешна
        """
        try:
            if self.results_file.exists():
                os.remove(self.results_file)
            if self.metadata_file.exists():
                os.remove(self.metadata_file)
            
            self.logger.info("✅ Кэш очищен")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка очистки кэша: {e}")
            return False
    
    def _is_results_outdated(self, result_data: Dict) -> bool:
        """
        Проверка, устарели ли результаты.
        
        Args:
            result_data: Данные результатов
            
        Returns:
            True если результаты устарели
        """
        try:
            timestamp_str = result_data.get('timestamp')
            if not timestamp_str:
                return True
            
            timestamp = datetime.fromisoformat(timestamp_str)
            age_days = (datetime.now() - timestamp).days
            
            # Считаем результаты устаревшими через 7 дней
            return age_days > 7
            
        except Exception:
            return True
    
    def _update_cache_metadata(self):
        """Обновление метаданных кэша."""
        try:
            all_results = self.load_all_results()
            
            metadata = {
                'last_updated': datetime.now().isoformat(),
                'total_entries': len(all_results),
                'timeframes': list(set(data.get('timeframe') for data in all_results.values() if isinstance(data, dict))),
                'symbols': list(set(data.get('symbol') for data in all_results.values() if isinstance(data, dict)))
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Ошибка обновления метаданных: {e}")
    
    def get_cache_info(self) -> Dict:
        """
        Получение информации о кэше.
        
        Returns:
            Словарь с информацией о кэше
        """
        try:
            all_results = self.load_all_results()
            best_results = self.get_best_results()
            
            info = {
                'cache_dir': str(self.cache_dir),
                'total_entries': len(all_results),
                'best_results_count': len(best_results),
                'available_timeframes': list(best_results.keys()),
                'last_updated': None
            }
            
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    info['last_updated'] = metadata.get('last_updated')
            
            return info
            
        except Exception as e:
            self.logger.error(f"Ошибка получения информации о кэше: {e}")
            return {} 
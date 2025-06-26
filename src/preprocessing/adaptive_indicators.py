"""
Адаптивный выбор технических индикаторов.

Этот модуль автоматически выбирает лучшие индикаторы и их параметры
на основе прибыльности торговых сигналов.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Импортируем кэш индикаторов
from .indicator_cache import IndicatorCache


class AdaptiveIndicatorSelector:
    """
    Адаптивный селектор индикаторов.
    
    Автоматически выбирает лучшие индикаторы и их параметры
    на основе прибыльности торговых сигналов.
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация адаптивного селектора.
        
        Args:
            config: Конфигурация
        """
        self.config = config or {}
        self.logger = logger.bind(name="AdaptiveIndicatorSelector")
        
        # Инициализируем кэш
        cache_dir = self.config.get('cache_dir', 'models/indicator_cache')
        self.cache = IndicatorCache(cache_dir)
        
        # Настройки горизонта предсказаний
        self.prediction_horizons = {
            '5m': 12,   # 12 свечей для 5-минутного таймфрейма
            '15m': 10,  # 10 свечей для 15-минутного таймфрейма
            '1h': 8,    # 8 свечей для часового таймфрейма
            '4h': 6,    # 6 свечей для 4-часового таймфрейма
            '1d': 5     # 5 свечей для дневного таймфрейма
        }
        
        # Библиотека индикаторов с различными параметрами
        self.indicator_library = {
            'rsi': {
                'function': self._calculate_rsi,
                'params': [7, 10, 14, 21, 30]
            },
            'macd': {
                'function': self._calculate_macd,
                'params': [
                    (8, 21, 5), (10, 23, 7), (12, 26, 9), (15, 30, 12)
                ]
            },
            'bollinger': {
                'function': self._calculate_bollinger_bands,
                'params': [
                    (10, 1.5), (15, 2.0), (20, 2.0), (25, 2.5)
                ]
            },
            'sma': {
                'function': self._calculate_sma,
                'params': [5, 10, 15, 20, 30, 50, 100, 200]
            },
            'ema': {
                'function': self._calculate_ema,
                'params': [5, 10, 15, 20, 30, 50, 100]
            },
            'stochastic': {
                'function': self._calculate_stochastic,
                'params': [7, 14, 21]
            },
            'williams_r': {
                'function': self._calculate_williams_r,
                'params': [7, 14, 21]
            },
            'cci': {
                'function': self._calculate_cci,
                'params': [10, 20, 30]
            },
            'adx': {
                'function': self._calculate_adx,
                'params': [10, 14, 20]
            },
            'atr': {
                'function': self._calculate_atr,
                'params': [7, 14, 21]
            },
            'obv': {
                'function': self._calculate_obv,
                'params': [None]  # OBV не требует периода
            },
            'vwap': {
                'function': self._calculate_vwap,
                'params': [None]  # VWAP не требует периода
            },
            # НОВЫЕ ИНДИКАТОРЫ
            'momentum': {
                'function': self._calculate_momentum,
                'params': [10, 14, 20]
            },
            'ichimoku': {
                'function': self._calculate_ichimoku,
                'params': [None]  # Использует стандартные параметры
            },
            'hull_ma': {
                'function': self._calculate_hull_ma,
                'params': [5, 10, 20, 50]
            },
            'awesome_oscillator': {
                'function': self._calculate_awesome_oscillator,
                'params': [None]  # Использует стандартные параметры
            },
            # ДОПОЛНИТЕЛЬНЫЕ ИНДИКАТОРЫ
            'ultimate_oscillator': {
                'function': self._calculate_ultimate_oscillator,
                'params': [None]  # Использует стандартные параметры
            },
            'bulls_bears_power': {
                'function': self._calculate_bulls_bears_power,
                'params': [None]  # Использует стандартные параметры
            },
            'fast_stochastic_rsi': {
                'function': self._calculate_fast_stochastic_rsi,
                'params': [None]  # Использует стандартные параметры
            }
        }
        
        # Лучшие индикаторы (будут найдены в процессе оптимизации)
        self.best_indicators = {}
        self.optimization_results = {}
    
    def optimize_indicators(self, data: pd.DataFrame, timeframe: str = '5m', 
                          symbol: str = "BTC_USDT", force_recalculate: bool = False) -> Dict:
        """
        Оптимизация индикаторов на основе прибыльности.
        
        Args:
            data: Исторические данные OHLCV
            timeframe: Таймфрейм для определения горизонта предсказаний
            symbol: Торговая пара
            force_recalculate: Принудительный пересчет (игнорировать кэш)
            
        Returns:
            Словарь с лучшими индикаторами и их параметрами
        """
        try:
            self.logger.info(f"🎯 Начинаем оптимизацию индикаторов для {timeframe}")
            
            # Проверяем кэш, если не принудительный пересчет
            if not force_recalculate:
                cached_results = self.cache.load_optimization_results(timeframe, symbol)
                if cached_results:
                    self.logger.info(f"✅ Найдены сохраненные результаты для {symbol}_{timeframe}")
                    self.logger.info(f"✅ Используем сохраненные результаты для {symbol}_{timeframe}")
                    self.best_indicators[timeframe] = cached_results.get('best_combination', [])
                    self.optimization_results[timeframe] = cached_results
                    return cached_results
            
            # Если кэш пуст или принудительный пересчет, выполняем оптимизацию
            self.logger.info(f"🔄 Выполняем новую оптимизацию для {symbol}_{timeframe}")
            
            # Определяем горизонт предсказаний
            prediction_horizon = self.prediction_horizons.get(timeframe, 12)
            self.logger.info(f"Горизонт предсказаний: {prediction_horizon} свечей")
            
            # Создаем целевую переменную
            target_data = self._create_target_variable(data, prediction_horizon)
            
            # Генерируем все возможные комбинации индикаторов
            indicator_combinations = self._generate_indicator_combinations()
            
            best_score = 0
            best_combination = None
            results = []
            
            # Тестируем каждую комбинацию
            for i, combination in enumerate(indicator_combinations):
                if i % 10 == 0:
                    self.logger.info(f"Тестируем комбинацию {i+1}/{len(indicator_combinations)}")
                
                try:
                    # Добавляем индикаторы
                    test_data = self._add_indicators(data.copy(), combination)
                    
                    if test_data is None or test_data.empty:
                        continue
                    
                    # Оцениваем комбинацию
                    score = self._evaluate_indicator_combination(test_data, target_data)
                    
                    results.append({
                        'combination': combination,
                        'score': score,
                        'indicator_count': len(combination)
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_combination = combination
                        
                except Exception as e:
                    self.logger.debug(f"Ошибка тестирования комбинации: {e}")
                    continue
            
            # Сохраняем лучшую комбинацию
            if best_combination:
                self.best_indicators[timeframe] = best_combination
                self.optimization_results[timeframe] = {
                    'best_combination': best_combination,
                    'best_score': best_score,
                    'prediction_horizon': prediction_horizon,
                    'all_results': results
                }
                
                # Сохраняем результаты в кэш
                self.cache.save_optimization_results(timeframe, self.optimization_results[timeframe], symbol)
                
                self.logger.info(f"✅ Оптимизация завершена. Лучший score: {best_score:.4f}")
                self.logger.info(f"Лучшие индикаторы: {best_combination}")
                
                return self.optimization_results[timeframe]
            else:
                self.logger.error("❌ Не удалось найти подходящие индикаторы")
                return {}
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка оптимизации индикаторов: {e}")
            return {}
    
    def get_cached_results_info(self) -> Dict:
        """
        Получение информации о сохраненных результатах.
        
        Returns:
            Словарь с информацией о кэше
        """
        return self.cache.get_cache_info()
    
    def get_best_cached_results(self) -> Dict:
        """
        Получение лучших сохраненных результатов.
        
        Returns:
            Словарь с лучшими результатами
        """
        return self.cache.get_best_results()
    
    def clear_cache(self) -> bool:
        """
        Очистка кэша результатов.
        
        Returns:
            True если очистка успешна
        """
        return self.cache.clear_cache()
    
    def _create_target_variable(self, data: pd.DataFrame, horizon: int) -> pd.Series:
        """
        Создание целевой переменной для заданного горизонта.
        
        Args:
            data: Исторические данные
            horizon: Горизонт предсказаний в свечах
            
        Returns:
            Целевая переменная (0=падение, 1=боковик, 2=рост)
        """
        try:
            # Рассчитываем будущие изменения цены
            future_returns = data['close'].shift(-horizon) / data['close'] - 1
            
            # НОВОЕ: Динамические множители волатильности
            volatility_multipliers = self._calculate_adaptive_volatility_multipliers(data)
            
            # Определяем пороги на основе адаптивной волатильности
            volatility = data['close'].pct_change().rolling(20).std()
            fall_threshold = -volatility * volatility_multipliers['fall']
            rise_threshold = volatility * volatility_multipliers['rise']
            
            # Создаем целевую переменную
            target = pd.Series(1, index=data.index)  # По умолчанию боковик
            
            # Падение
            target[future_returns < fall_threshold] = 0
            
            # Рост
            target[future_returns > rise_threshold] = 2
            
            # Удаляем NaN значения
            target = target.dropna()
            
            return target
            
        except Exception as e:
            self.logger.error(f"Ошибка создания целевой переменной: {e}")
            return pd.Series()
    
    def _calculate_adaptive_volatility_multipliers(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Расчет адаптивных множителей волатильности на основе рыночных условий.
        
        Args:
            data: Исторические данные
            
        Returns:
            Словарь с множителями для падения и роста
        """
        try:
            # Базовые множители
            base_fall_multiplier = 1.5
            base_rise_multiplier = 1.5
            
            # Рассчитываем различные метрики рыночных условий
            returns = data['close'].pct_change()
            
            # 1. Текущая волатильность относительно исторической
            current_vol = returns.rolling(20).std().reindex(data.index).fillna(returns.std())
            historical_vol = returns.rolling(100).std().reindex(data.index).fillna(returns.std())
            vol_ratio = (current_vol / historical_vol).reindex(data.index).fillna(1.0)
            
            # 2. Тренд рынка
            short_ma = data['close'].rolling(20).mean().reindex(data.index).fillna(method='bfill')
            long_ma = data['close'].rolling(50).mean().reindex(data.index).fillna(method='bfill')
            trend_strength = ((short_ma - long_ma) / long_ma).reindex(data.index).fillna(0)
            
            # 3. Асимметрия движений (бычий/медвежий рынок)
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                pos_vol = positive_returns.rolling(20).std().reindex(data.index).fillna(returns.std())
                neg_vol = negative_returns.rolling(20).std().reindex(data.index).fillna(returns.std())
                asymmetry_ratio = (pos_vol / neg_vol).reindex(data.index).fillna(1.0)
            else:
                asymmetry_ratio = pd.Series(1.0, index=data.index)
            
            # 4. Объемная активность
            volume_ma = data['volume'].rolling(20).mean().reindex(data.index).fillna(method='bfill')
            volume_ratio = (data['volume'] / volume_ma).reindex(data.index).fillna(1.0)
            
            # 5. Волатильность волатильности (изменчивость рынка)
            vol_of_vol = current_vol.rolling(20).std().reindex(data.index).fillna(current_vol.std())
            vol_of_vol_normalized = (vol_of_vol / current_vol).reindex(data.index).fillna(0.3)
            
            # Заполняем NaN значения в других метриках
            vol_ratio = vol_ratio.fillna(1.0)
            trend_strength = trend_strength.fillna(0.0)
            
            # НОВОЕ: Адаптивные множители на основе рыночных условий
            fall_multiplier = pd.Series(base_fall_multiplier, index=data.index)
            rise_multiplier = pd.Series(base_rise_multiplier, index=data.index)
            
            # Корректировка на основе волатильности
            vol_adjustment = np.where(vol_ratio > 1.2, 0.8,  # Высокая волатильность - снижаем пороги
                            np.where(vol_ratio < 0.8, 1.3,   # Низкая волатильность - повышаем пороги
                            1.0))  # Нормальная волатильность
            
            # Корректировка на основе тренда
            trend_adjustment = np.where(trend_strength > 0.02, 1.1,  # Сильный восходящий тренд
                              np.where(trend_strength < -0.02, 0.9,  # Сильный нисходящий тренд
                              1.0))  # Боковой тренд
            
            # Корректировка на основе асимметрии
            asymmetry_adjustment = np.where(asymmetry_ratio > 1.1, 1.1,  # Бычий рынок
                                  np.where(asymmetry_ratio < 0.9, 0.9,   # Медвежий рынок
                                  1.0))  # Сбалансированный рынок
            
            # Корректировка на основе объема
            volume_adjustment = np.where(volume_ratio > 1.5, 0.9,  # Высокий объем - снижаем пороги
                               np.where(volume_ratio < 0.7, 1.2,   # Низкий объем - повышаем пороги
                               1.0))  # Нормальный объем
            
            # Корректировка на основе изменчивости волатильности
            vol_of_vol_adjustment = np.where(vol_of_vol_normalized > 0.5, 0.8,  # Высокая изменчивость
                                   np.where(vol_of_vol_normalized < 0.2, 1.2,   # Низкая изменчивость
                                   1.0))  # Нормальная изменчивость
            
            # Применяем все корректировки
            fall_multiplier = fall_multiplier * vol_adjustment * trend_adjustment * asymmetry_adjustment * volume_adjustment * vol_of_vol_adjustment
            rise_multiplier = rise_multiplier * vol_adjustment * trend_adjustment * asymmetry_adjustment * volume_adjustment * vol_of_vol_adjustment
            
            # Ограничиваем множители разумными пределами
            fall_multiplier = fall_multiplier.clip(0.5, 3.0)
            rise_multiplier = rise_multiplier.clip(0.5, 3.0)
            
            # Логируем статистику множителей
            avg_fall_mult = fall_multiplier.mean()
            avg_rise_mult = rise_multiplier.mean()
            
            self.logger.info(f"Адаптивные множители волатильности:")
            self.logger.info(f"  Падение: {avg_fall_mult:.2f} (диапазон: {fall_multiplier.min():.2f}-{fall_multiplier.max():.2f})")
            self.logger.info(f"  Рост: {avg_rise_mult:.2f} (диапазон: {rise_multiplier.min():.2f}-{rise_multiplier.max():.2f})")
            
            return {
                'fall': fall_multiplier,
                'rise': rise_multiplier
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета адаптивных множителей: {e}")
            # Возвращаем базовые множители в случае ошибки
            return {
                'fall': pd.Series(1.5, index=data.index),
                'rise': pd.Series(1.5, index=data.index)
            }
    
    def _generate_indicator_combinations(self) -> List[List]:
        """
        Генерация комбинаций индикаторов для тестирования.
        
        Returns:
            Список комбинаций индикаторов
        """
        combinations = []
        
        # Базовые индикаторы (обязательные)
        base_indicators = ['rsi', 'macd', 'bollinger', 'sma', 'ema']
        
        # Дополнительные индикаторы
        additional_indicators = ['stochastic', 'williams_r', 'cci', 'adx', 'atr', 'obv', 'vwap']
        
        # Генерируем комбинации разного размера
        for size in range(3, 8):  # От 3 до 7 индикаторов
            # Обязательно включаем базовые индикаторы
            for base_combo in itertools.combinations(base_indicators, min(size-2, len(base_indicators))):
                remaining_slots = size - len(base_combo)
                
                if remaining_slots > 0:
                    for additional_combo in itertools.combinations(additional_indicators, remaining_slots):
                        combination = list(base_combo) + list(additional_combo)
                        combinations.append(combination)
                else:
                    combinations.append(list(base_combo))
        
        return combinations[:50]  # Ограничиваем количество для скорости
    
    def _add_indicators(self, data: pd.DataFrame, combination: List) -> pd.DataFrame:
        """
        Добавление индикаторов из комбинации.
        
        Args:
            data: Данные
            combination: Комбинация индикаторов
            
        Returns:
            DataFrame с добавленными индикаторами
        """
        try:
            for indicator_name in combination:
                if indicator_name in self.indicator_library:
                    indicator_info = self.indicator_library[indicator_name]
                    params = indicator_info['params']
                    
                    # Берем первый набор параметров для тестирования
                    if params and params[0] is not None:
                        if isinstance(params[0], tuple):
                            data = indicator_info['function'](data, *params[0])
                        else:
                            data = indicator_info['function'](data, params[0])
                    else:
                        data = indicator_info['function'](data)
            
            # Удаляем NaN значения
            data = data.dropna()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления индикаторов: {e}")
            return pd.DataFrame()
    
    def _evaluate_indicator_combination(self, data: pd.DataFrame, target: pd.Series) -> float:
        """
        Оценка комбинации индикаторов.
        
        Args:
            data: Данные с индикаторами
            target: Целевая переменная
            
        Returns:
            Оценка комбинации (0-1)
        """
        try:
            # Выбираем только числовые колонки (индикаторы)
            feature_columns = data.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in feature_columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            if len(feature_columns) < 3:
                return 0.0
            
            # Приводим данные к одному индексу
            common_index = data.index.intersection(target.index)
            if len(common_index) < 100:
                return 0.0
            
            X = data.loc[common_index, feature_columns]
            y = target.loc[common_index]
            
            # Простая оценка на основе RandomForest
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Дополнительная оценка на основе важности признаков
            feature_importance = np.mean(model.feature_importances_)
            
            # Комбинированная оценка
            score = 0.7 * accuracy + 0.3 * feature_importance
            
            return score
            
        except Exception as e:
            self.logger.debug(f"Ошибка оценки комбинации: {e}")
            return 0.0
    
    def get_best_indicators(self, timeframe: str = '5m') -> Dict:
        """
        Получение лучших индикаторов для таймфрейма.
        
        Args:
            timeframe: Таймфрейм
            
        Returns:
            Словарь с лучшими индикаторами
        """
        return self.best_indicators.get(timeframe, {})
    
    def add_best_indicators(self, data: pd.DataFrame, timeframe: str = '5m') -> pd.DataFrame:
        """
        Добавление лучших индикаторов к данным.
        
        Args:
            data: Данные OHLCV
            timeframe: Таймфрейм
            
        Returns:
            DataFrame с лучшими индикаторами
        """
        try:
            if timeframe not in self.best_indicators:
                self.logger.warning(f"Нет оптимизированных индикаторов для {timeframe}, используем базовые")
                return self._add_basic_indicators(data)
            
            best_combo = self.best_indicators[timeframe]
            return self._add_indicators(data, best_combo)
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления лучших индикаторов: {e}")
            return self._add_basic_indicators(data)
    
    def _add_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление базовых индикаторов.
        
        Args:
            data: Данные OHLCV
            
        Returns:
            DataFrame с базовыми индикаторами
        """
        try:
            # RSI
            data = self._calculate_rsi(data, 14)
            
            # MACD
            data = self._calculate_macd(data, 12, 26, 9)
            
            # Bollinger Bands
            data = self._calculate_bollinger_bands(data, 20, 2.0)
            
            # SMA
            data = self._calculate_sma(data, 20)
            data = self._calculate_sma(data, 50)
            
            # EMA
            data = self._calculate_ema(data, 20)
            data = self._calculate_ema(data, 50)
            
            return data.dropna()
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления базовых индикаторов: {e}")
            return data
    
    # Методы расчета индикаторов
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Расчет RSI."""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            df[f'rsi_{period}'] = rsi
            return df
        except:
            return df
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Расчет MACD."""
        try:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            df['macd_line'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_histogram'] = histogram
            return df
        except:
            return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Расчет Bollinger Bands."""
        try:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            bandwidth = (upper - lower) / sma
            position = (df['close'] - lower) / (upper - lower)
            
            df[f'bb_upper_{period}'] = upper
            df[f'bb_middle_{period}'] = sma
            df[f'bb_lower_{period}'] = lower
            df[f'bb_bandwidth_{period}'] = bandwidth
            df[f'bb_position_{period}'] = position
            return df
        except:
            return df
    
    def _calculate_sma(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Расчет SMA."""
        try:
            sma = df['close'].rolling(window=period).mean()
            df[f'sma_{period}'] = sma
            df[f'price_to_sma_{period}'] = (df['close'] - sma) / sma
            return df
        except:
            return df
    
    def _calculate_ema(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Расчет EMA."""
        try:
            ema = df['close'].ewm(span=period).mean()
            df[f'ema_{period}'] = ema
            df[f'price_to_ema_{period}'] = (df['close'] - ema) / ema
            return df
        except:
            return df
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Расчет Stochastic Oscillator."""
        try:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(window=3).mean()
            
            df[f'stoch_k_{period}'] = k_percent
            df[f'stoch_d_{period}'] = d_percent
            return df
        except:
            return df
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Расчет Williams %R."""
        try:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            williams_r = -100 * ((high_max - df['close']) / (high_max - low_min))
            df[f'williams_r_{period}'] = williams_r
            return df
        except:
            return df
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Расчет CCI (Commodity Channel Index)."""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad)
            df[f'cci_{period}'] = cci
            return df
        except:
            return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Расчет ADX (Average Directional Index)."""
        try:
            high_diff = df['high'].diff()
            low_diff = df['low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            tr = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    np.abs(df['high'] - df['close'].shift(1)),
                    np.abs(df['low'] - df['close'].shift(1))
                )
            )
            
            plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / pd.Series(tr).rolling(window=period).mean()
            minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / pd.Series(tr).rolling(window=period).mean()
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = pd.Series(dx).rolling(window=period).mean()
            
            df[f'adx_{period}'] = adx
            df[f'plus_di_{period}'] = plus_di
            df[f'minus_di_{period}'] = minus_di
            return df
        except:
            return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Расчет ATR (Average True Range)."""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()
            
            df[f'atr_{period}'] = atr
            return df
        except:
            return df
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет OBV (On-Balance Volume)."""
        try:
            obv = pd.Series(index=df.index, dtype=float)
            obv.iloc[0] = df['volume'].iloc[0]
            
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            df['obv'] = obv
            return df
        except:
            return df
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет VWAP."""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap'] = vwap
            return df
        except:
            return df
    
    # НОВЫЕ МЕТОДЫ РАСЧЕТА ИНДИКАТОРОВ
    
    def _calculate_momentum(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Расчет Momentum (Моментум)."""
        try:
            momentum = df['close'] - df['close'].shift(period)
            df[f'momentum_{period}'] = momentum
            return df
        except:
            return df
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет Ichimoku Cloud."""
        try:
            # Стандартные параметры Ichimoku
            tenkan_period = 9
            kijun_period = 26
            senkou_span_b_period = 52
            displacement = 26
            
            # Tenkan-sen (Conversion Line)
            tenkan_high = df['high'].rolling(window=tenkan_period).max()
            tenkan_low = df['low'].rolling(window=tenkan_period).min()
            tenkan_sen = (tenkan_high + tenkan_low) / 2
            
            # Kijun-sen (Base Line)
            kijun_high = df['high'].rolling(window=kijun_period).max()
            kijun_low = df['low'].rolling(window=kijun_period).min()
            kijun_sen = (kijun_high + kijun_low) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
            
            # Senkou Span B (Leading Span B)
            senkou_span_b_high = df['high'].rolling(window=senkou_span_b_period).max()
            senkou_span_b_low = df['low'].rolling(window=senkou_span_b_period).min()
            senkou_span_b = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(displacement)
            
            # Chikou Span (Lagging Span)
            chikou_span = df['close'].shift(-displacement)
            
            df['ichimoku_tenkan'] = tenkan_sen
            df['ichimoku_kijun'] = kijun_sen
            df['ichimoku_senkou_a'] = senkou_span_a
            df['ichimoku_senkou_b'] = senkou_span_b
            df['ichimoku_chikou'] = chikou_span
            
            return df
        except:
            return df
    
    def _calculate_hull_ma(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Расчет Hull Moving Average."""
        try:
            # Hull MA = WMA(2*WMA(n/2) - WMA(n))
            half_period = period // 2
            sqrt_period = int(np.sqrt(period))
            
            # WMA(n/2)
            wma_half = df['close'].rolling(window=half_period).apply(
                lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
            )
            
            # WMA(n)
            wma_full = df['close'].rolling(window=period).apply(
                lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
            )
            
            # 2*WMA(n/2) - WMA(n)
            raw_hull = 2 * wma_half - wma_full
            
            # WMA(sqrt(n)) от результата
            hull_ma = raw_hull.rolling(window=sqrt_period).apply(
                lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
            )
            
            df[f'hull_ma_{period}'] = hull_ma
            return df
        except:
            return df
    
    def _calculate_awesome_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет Awesome Oscillator (Чудесный осциллятор Билла Вильямса)."""
        try:
            # Медианная цена
            median_price = (df['high'] + df['low']) / 2
            
            # 5-периодная SMA медианной цены
            ao_fast = median_price.rolling(window=5).mean()
            
            # 34-периодная SMA медианной цены
            ao_slow = median_price.rolling(window=34).mean()
            
            # Awesome Oscillator = AO_fast - AO_slow
            awesome_oscillator = ao_fast - ao_slow
            
            df['awesome_oscillator'] = awesome_oscillator
            return df
        except:
            return df
    
    def _calculate_ultimate_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет Ultimate Oscillator."""
        try:
            # Параметры Ultimate Oscillator
            period1 = 7
            period2 = 14
            period3 = 28
            weight1 = 4.0
            weight2 = 2.0
            weight3 = 1.0
            
            # Расчет True Range
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            
            # Расчет Buying Pressure (BP) и True Range (TR)
            bp = df['close'] - np.minimum(df['low'], df['close'].shift(1))
            tr = true_range
            
            # Скользящие средние для разных периодов
            avg_bp1 = bp.rolling(window=period1).mean()
            avg_tr1 = tr.rolling(window=period1).mean()
            avg_bp2 = bp.rolling(window=period2).mean()
            avg_tr2 = tr.rolling(window=period2).mean()
            avg_bp3 = bp.rolling(window=period3).mean()
            avg_tr3 = tr.rolling(window=period3).mean()
            
            # Расчет Ultimate Oscillator
            uo1 = 100 * (avg_bp1 / avg_tr1)
            uo2 = 100 * (avg_bp2 / avg_tr2)
            uo3 = 100 * (avg_bp3 / avg_tr3)
            
            ultimate_oscillator = (weight1 * uo1 + weight2 * uo2 + weight3 * uo3) / (weight1 + weight2 + weight3)
            
            df['ultimate_oscillator'] = ultimate_oscillator
            return df
        except:
            return df
    
    def _calculate_bulls_bears_power(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет Bulls Power & Bears Power."""
        try:
            # Параметры
            period = 13
            
            # Медианная цена
            median_price = (df['high'] + df['low']) / 2
            
            # Bulls Power = High - EMA(median_price)
            ema_median = median_price.ewm(span=period).mean()
            bulls_power = df['high'] - ema_median
            
            # Bears Power = Low - EMA(median_price)
            bears_power = df['low'] - ema_median
            
            df['bulls_power'] = bulls_power
            df['bears_power'] = bears_power
            
            return df
        except:
            return df
    
    def _calculate_fast_stochastic_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет Fast Stochastic RSI."""
        try:
            # Сначала рассчитываем RSI
            rsi_period = 14
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Затем применяем Stochastic к RSI
            stoch_period = 14
            rsi_high = rsi.rolling(window=stoch_period).max()
            rsi_low = rsi.rolling(window=stoch_period).min()
            
            # Fast Stochastic RSI
            fast_stoch_rsi = 100 * (rsi - rsi_low) / (rsi_high - rsi_low)
            
            # Сглаживание (K% и D%)
            k_period = 3
            d_period = 3
            
            k_percent = fast_stoch_rsi.rolling(window=k_period).mean()
            d_percent = k_percent.rolling(window=d_period).mean()
            
            df['fast_stoch_rsi_k'] = k_percent
            df['fast_stoch_rsi_d'] = d_percent
            
            return df
        except:
            return df 
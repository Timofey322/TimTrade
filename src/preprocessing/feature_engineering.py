"""
Модуль для инженерной обработки признаков.

Этот модуль предоставляет функциональность для:
- Создания технических индикаторов
- Обработки множественных таймфреймов
- Балансировки данных
- Создания целевой переменной
- Подготовки данных для ML моделей
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Импорт модулей
from .indicators import TechnicalIndicators
from .target_creation import TargetCreator
from .data_balancing import DataBalancer
from .multi_timeframe import MultiTimeframeProcessor
from .adaptive_indicators import AdaptiveIndicatorSelector

# Импорт новых продвинутых модулей
try:
    from .advanced_features import AdvancedFeatureEngine
    from ..data_collection.sentiment_collector import SentimentCollector
except ImportError:
    # Fallback для случаев запуска не из пакета
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from preprocessing.advanced_features import AdvancedFeatureEngine
        from data_collection.sentiment_collector import SentimentCollector
    except ImportError:
        AdvancedFeatureEngine = None
        SentimentCollector = None

class FeatureEngineer:
    """
    Класс для инженерной обработки признаков.
    
    Поддерживает:
    - Создание технических индикаторов
    - Обработку множественных таймфреймов
    - Балансировку данных
    - Создание целевой переменной
    - Подготовку данных для ML
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация инженера признаков.
        
        Args:
            config: Конфигурация обработки признаков
        """
        self.config = config or {}
        self.logger = logger.bind(name="FeatureEngineer")
        
        # Инициализация компонентов
        self.indicators_config = self.config.get('indicators', {})
        self.target_config = self.config.get('target_creation', {})
        self.balancing_config = self.config.get('data_balancing', {})
        self.multi_tf_config = self.config.get('multi_timeframe', {})
        
        # Создание компонентов
        self.indicators = TechnicalIndicators(self.indicators_config)
        self.target_creator = TargetCreator(self.target_config)
        self.balancer = DataBalancer(self.balancing_config)
        self.multi_tf_processor = MultiTimeframeProcessor(self.multi_tf_config)
        
        # НОВОЕ: Инициализация адаптивного селектора индикаторов
        self.adaptive_selector = AdaptiveIndicatorSelector(self.config.get('adaptive_indicators', {}))
        
        # НОВОЕ: Инициализация продвинутых модулей
        advanced_config = self.config.get('advanced_features', {})
        
        if AdvancedFeatureEngine and advanced_config.get('enabled', False):
            self.advanced_engine = AdvancedFeatureEngine(advanced_config)
        else:
            self.advanced_engine = None
        
        # Инициализация simplified sentiment collector (только Bybit)
        sentiment_config = advanced_config.get('api_keys', {}) if advanced_config else {}
        
        if SentimentCollector and advanced_config.get('enabled', False):
            self.sentiment_collector = SentimentCollector(sentiment_config)
        else:
            self.sentiment_collector = None
        
        # Настройки горизонта предсказаний
        self.prediction_horizons = {
            '5m': 12,   # 12 свечей для 5-минутного таймфрейма
            '15m': 10,  # 10 свечей для 15-минутного таймфрейма
            '1h': 8,    # 8 свечей для часового таймфрейма
            '4h': 6,    # 6 свечей для 4-часового таймфрейма
            '1d': 5     # 5 свечей для дневного таймфрейма
        }
        
        # Статистика обработки
        self.processing_stats = {
            'total_features': 0,
            'timeframes_processed': 0,
            'balance_ratio_before': 0,
            'balance_ratio_after': 0,
            'target_distribution': {},
            'feature_importance': {},
            'processing_time': 0
        }
        
        # Флаг для использования кэшированных индикаторов
        self._use_cached_indicators = True  # По умолчанию используем кэш
        
        self.logger.info("FeatureEngineer инициализирован")
    
    def process_single_timeframe(self, df: pd.DataFrame, symbol: str, 
                                timeframe: str = '5m', use_adaptive: bool = True) -> Optional[pd.DataFrame]:
        """
        Обработка данных одного таймфрейма.
        
        Args:
            df: DataFrame с данными
            symbol: Торговая пара
            timeframe: Таймфрейм данных
            use_adaptive: Использовать адаптивный выбор индикаторов
        
        Returns:
            DataFrame с признаками или None при ошибке
        """
        try:
            self.logger.info(f"Начинаем инжиниринг признаков для {symbol} на {timeframe}")
            
            # Определяем горизонт предсказаний
            prediction_horizon = self.prediction_horizons.get(timeframe, 12)
            self.logger.info(f"Горизонт предсказаний: {prediction_horizon} свечей")
            
            # Создаем копию данных
            processed_df = df.copy()
            
            # НОВОЕ: Адаптивный выбор индикаторов
            if use_adaptive and self.config.get('adaptive_indicators', {}).get('enabled', True):
                self.logger.info("🎯 Используем адаптивный выбор индикаторов")
                
                # Определяем, нужно ли принудительно пересчитывать
                force_recalculate = not self._use_cached_indicators
                
                # Оптимизируем индикаторы на основе прибыльности
                optimization_results = self.adaptive_selector.optimize_indicators(
                    processed_df, timeframe, force_recalculate=force_recalculate
                )
                
                if optimization_results and 'best_combination' in optimization_results:
                    self.logger.info(f"✅ Найдены лучшие индикаторы: {optimization_results['best_combination']}")
                    processed_df = self.adaptive_selector.add_best_indicators(processed_df, timeframe)
                else:
                    self.logger.warning("⚠️ Адаптивная оптимизация не удалась, используем базовые индикаторы")
                    processed_df = self.indicators.add_all_indicators(processed_df)
            else:
                # Используем стандартные индикаторы
                self.logger.info("📊 Используем стандартные индикаторы")
                processed_df = self.indicators.add_all_indicators(processed_df)
            
            # НОВОЕ: Создание целевой переменной с правильным горизонтом
            processed_df = self._create_target_with_horizon(processed_df, prediction_horizon)
            
            # НОВОЕ: Дополнительные признаки на основе горизонта
            processed_df = self._add_horizon_based_features(processed_df, prediction_horizon)
            
            # НОВОЕ: Добавляем simplified sentiment features (только Bybit)
            if self.sentiment_collector is not None:
                processed_df = self._add_simplified_sentiment_features(processed_df)
            
            # Удаляем строки с NaN значениями
            initial_rows = len(processed_df)
            processed_df = processed_df.dropna()
            removed_rows = initial_rows - len(processed_df)
            
            if removed_rows > 0:
                self.logger.warning(f"Удалено {removed_rows} строк с NaN значениями")
            
            # Проверка на достаточное количество данных
            if len(processed_df) < 100:
                self.logger.error(f"Недостаточно данных после обработки: {len(processed_df)} строк")
                return None
            
            # Обновление статистики
            self.processing_stats['total_features'] = len(processed_df.columns)
            self.processing_stats['balance_ratio_before'] = self._calculate_balance_ratio(processed_df)
            
            self.logger.info(f"Инжиниринг признаков завершен. Размер данных: {processed_df.shape}")
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Ошибка инжиниринга признаков для {symbol}: {e}")
            return None
    
    def _create_target_with_horizon(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Создание целевой переменной с заданным горизонтом предсказаний.
        
        Args:
            df: DataFrame с данными
            horizon: Горизонт предсказаний в свечах
            
        Returns:
            DataFrame с целевой переменной
        """
        try:
            # Рассчитываем будущие изменения цены
            future_returns = df['close'].shift(-horizon) / df['close'] - 1
            
            # НОВОЕ: Динамические множители волатильности
            volatility_multipliers = self._calculate_adaptive_volatility_multipliers(df)
            
            # Определяем пороги на основе адаптивной волатильности
            volatility = df['close'].pct_change().rolling(20).std()
            fall_threshold = -volatility * volatility_multipliers['fall']
            rise_threshold = volatility * volatility_multipliers['rise']
            
            # Создаем целевую переменную
            target = pd.Series(1, index=df.index)  # По умолчанию боковик
            
            # Падение
            target[future_returns < fall_threshold] = 0
            
            # Рост
            target[future_returns > rise_threshold] = 2
            
            # Добавляем целевую переменную
            df['target'] = target
            
            # НОВОЕ: Добавляем информацию о горизонте
            df['prediction_horizon'] = horizon
            
            self.logger.info(f"Создана целевая переменная с горизонтом {horizon} свечей")
            self.logger.info(f"Распределение классов: {target.value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка создания целевой переменной: {e}")
            return df
    
    def _calculate_adaptive_volatility_multipliers(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
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
            current_vol = returns.rolling(20).std()
            historical_vol = returns.rolling(100).std()
            vol_ratio = current_vol / historical_vol
            
            # 2. Тренд рынка
            short_ma = data['close'].rolling(20).mean()
            long_ma = data['close'].rolling(50).mean()
            trend_strength = (short_ma - long_ma) / long_ma
            
            # 3. Асимметрия движений (бычий/медвежий рынок)
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                pos_vol = positive_returns.rolling(20).std()
                neg_vol = negative_returns.rolling(20).std()
                asymmetry_ratio = pos_vol / neg_vol
                # Заполняем NaN значения и приводим к общему индексу
                asymmetry_ratio = asymmetry_ratio.reindex(data.index).fillna(1.0)
            else:
                asymmetry_ratio = pd.Series(1.0, index=data.index)
            
            # 4. Объемная активность
            volume_ma = data['volume'].rolling(20).mean()
            volume_ratio = data['volume'] / volume_ma
            volume_ratio = volume_ratio.fillna(1.0)
            
            # 5. Волатильность волатильности (изменчивость рынка)
            vol_of_vol = current_vol.rolling(20).std()
            vol_of_vol_normalized = vol_of_vol / current_vol
            vol_of_vol_normalized = vol_of_vol_normalized.fillna(0.3)  # Среднее значение
            
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
    
    def _add_horizon_based_features(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Добавление признаков на основе горизонта предсказаний.
        
        Args:
            df: DataFrame с данными
            horizon: Горизонт предсказаний
            
        Returns:
            DataFrame с дополнительными признаками
        """
        try:
            # НОВОЕ: Признаки тренда на горизонте предсказаний
            df[f'trend_{horizon}'] = df['close'].rolling(horizon).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else (-1 if x.iloc[-1] < x.iloc[0] else 0)
            )
            
            # НОВОЕ: Волатильность на горизонте
            df[f'volatility_{horizon}'] = df['close'].pct_change().rolling(horizon).std()
            
            # НОВОЕ: Максимальное и минимальное значение на горизонте
            df[f'max_price_{horizon}'] = df['close'].rolling(horizon).max()
            df[f'min_price_{horizon}'] = df['close'].rolling(horizon).min()
            
            # НОВОЕ: Расстояние до максимума и минимума
            df[f'distance_to_max_{horizon}'] = (df[f'max_price_{horizon}'] - df['close']) / df['close']
            df[f'distance_to_min_{horizon}'] = (df['close'] - df[f'min_price_{horizon}']) / df['close']
            
            # НОВОЕ: Моментум на горизонте
            df[f'momentum_{horizon}'] = df['close'] / df['close'].shift(horizon) - 1
            
            # НОВОЕ: Объемный профиль на горизонте
            df[f'volume_profile_{horizon}'] = df['volume'].rolling(horizon).mean()
            df[f'volume_ratio_{horizon}'] = df['volume'] / df[f'volume_profile_{horizon}']
            
            self.logger.info(f"Добавлены признаки на основе горизонта {horizon}")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления признаков горизонта: {e}")
            return df
    
    def process_multi_timeframe(self, data_dict: Dict[str, pd.DataFrame], symbol: str) -> Optional[pd.DataFrame]:
        """
        Обработка данных множественных таймфреймов.
        
        Args:
            data_dict: Словарь с данными для каждого таймфрейма
            symbol: Торговая пара
        
        Returns:
            DataFrame с объединенными признаками или None при ошибке
        """
        try:
            self.logger.info(f"Обрабатываем множественные таймфреймы для {symbol}")
            
            # Валидация данных
            if not self.multi_tf_processor.validate_timeframe_data(data_dict):
                self.logger.error(f"Валидация данных множественных таймфреймов не пройдена для {symbol}")
                return None
            
            # Обработка каждого таймфрейма отдельно
            processed_data = {}
            
            for timeframe, df in data_dict.items():
                self.logger.info(f"Обрабатываем таймфрейм {timeframe}")
                
                # Создание технических индикаторов для данного таймфрейма
                processed_df = df.copy()
                processed_df = self.process_single_timeframe(processed_df, symbol, timeframe)
                
                if processed_df is not None:
                    processed_data[timeframe] = processed_df
                    self.logger.info(f"Таймфрейм {timeframe}: {len(processed_df)} строк, {len(processed_df.columns)} колонок")
                else:
                    self.logger.warning(f"Ошибка обработки таймфрейма {timeframe}")
            
            if not processed_data:
                self.logger.error("Нет данных после обработки всех таймфреймов")
                return None
            
            # Объединение данных множественных таймфреймов
            combined_df = self.multi_tf_processor.process_multi_timeframe_data(processed_data, symbol)
            
            if combined_df is None or combined_df.empty:
                self.logger.error("Ошибка объединения данных множественных таймфреймов")
                return None
            
            # Удаление строк с NaN значениями
            initial_rows = len(combined_df)
            combined_df = combined_df.dropna()
            final_rows = len(combined_df)
            
            if final_rows < initial_rows:
                self.logger.warning(f"Удалено {initial_rows - final_rows} строк с NaN значениями")
            
            # Проверка на достаточное количество данных
            if len(combined_df) < 100:
                self.logger.error(f"Недостаточно данных после обработки: {len(combined_df)} строк")
                return None
            
            # Обновление статистики
            self.processing_stats['total_features'] = len(combined_df.columns)
            self.processing_stats['timeframes_processed'] = len(processed_data)
            self.processing_stats['balance_ratio_before'] = self._calculate_balance_ratio(combined_df)
            
            self.logger.info(f"Обработка множественных таймфреймов завершена: {len(combined_df)} строк, {len(combined_df.columns)} колонок")
            self.logger.info(f"Колонки в итоговом DataFrame: {list(combined_df.columns)}")
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки множественных таймфреймов для {symbol}: {e}")
            return None
    
    def balance_data(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """
        Балансировка данных.
        
        Args:
            df: DataFrame с признаками и целевой переменной
            symbol: Торговая пара
        
        Returns:
            Сбалансированный DataFrame или None при ошибке
        """
        try:
            if df is None or df.empty:
                self.logger.error(f"Пустые данные для балансировки {symbol}")
                return None
            
            # Проверка наличия целевой переменной
            if 'target' not in df.columns:
                self.logger.error(f"Отсутствует целевая переменная для {symbol}")
                return None
            
            self.logger.info(f"Балансируем данные для {symbol}")
            
            # Разделение на признаки и целевую переменную
            X = df.drop('target', axis=1)
            y = df['target']
            
            # Балансировка данных
            X_balanced, y_balanced = self.balancer.balance_data(X, y, method='auto')
            
            if X_balanced is None or y_balanced is None:
                self.logger.error(f"Ошибка балансировки данных для {symbol}")
                return None
            
            # Объединение обратно в DataFrame
            balanced_df = X_balanced.copy()
            balanced_df['target'] = y_balanced
            
            # Обновление статистики
            self.processing_stats['balance_ratio_after'] = self._calculate_balance_ratio(balanced_df)
            
            self.logger.info(f"Балансировка завершена: {len(balanced_df)} строк")
            self.logger.info(f"Соотношение классов до: {self.processing_stats['balance_ratio_before']:.3f}")
            self.logger.info(f"Соотношение классов после: {self.processing_stats['balance_ratio_after']:.3f}")
            
            return balanced_df
            
        except Exception as e:
            self.logger.error(f"Ошибка балансировки данных для {symbol}: {e}")
            return None
    
    def prepare_for_training(self, df: pd.DataFrame, symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Подготовка данных для обучения модели.
        
        Args:
            df: DataFrame с признаками и целевой переменной
            symbol: Торговая пара
        
        Returns:
            Tuple с признаками и целевой переменной или (None, None) при ошибке
        """
        try:
            if df is None or df.empty:
                self.logger.error(f"Пустые данные для подготовки к обучению {symbol}")
                return None, None
            
            # Проверка наличия целевой переменной
            if 'target' not in df.columns:
                self.logger.error(f"Отсутствует целевая переменная для {symbol}")
                return None, None
            
            # Разделение на признаки и целевую переменную
            X = df.drop('target', axis=1)
            y = df['target']
            
            # Удаление колонок с константными значениями
            constant_columns = [col for col in X.columns if X[col].nunique() <= 1]
            if constant_columns:
                self.logger.warning(f"Удалены константные колонки: {constant_columns}")
                X = X.drop(constant_columns, axis=1)
            
            # Удаление колонок с высоким процентом NaN значений
            nan_threshold = 0.5
            nan_columns = [col for col in X.columns if X[col].isnull().sum() / len(X) > nan_threshold]
            if nan_columns:
                self.logger.warning(f"Удалены колонки с высоким процентом NaN: {nan_columns}")
                X = X.drop(nan_columns, axis=1)
            
            # Заполнение оставшихся NaN значений
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Проверка на достаточное количество признаков
            if len(X.columns) < 5:
                self.logger.error(f"Недостаточно признаков после очистки: {len(X.columns)}")
                return None, None
            
            # Обновление статистики
            self.processing_stats['target_distribution'] = y.value_counts().to_dict()
            
            self.logger.info(f"Подготовка завершена: {len(X)} строк, {len(X.columns)} признаков")
            self.logger.info(f"Распределение целевой переменной: {dict(y.value_counts())}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Ошибка подготовки данных для обучения {symbol}: {e}")
            return None, None
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """
        Получение важности признаков от модели.
        
        Args:
            model: Обученная модель с feature_importances_
            feature_names: Список названий признаков
        
        Returns:
            Словарь с важностью признаков
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                
                # Сортировка по важности
                sorted_importance = dict(sorted(importance_dict.items(), 
                                               key=lambda x: x[1], reverse=True))
                
                self.processing_stats['feature_importance'] = sorted_importance
                
                self.logger.info("Важность признаков получена")
                return sorted_importance
            else:
                self.logger.warning("Модель не поддерживает feature_importances_")
                return {}
                
        except Exception as e:
            self.logger.error(f"Ошибка получения важности признаков: {e}")
            return {}
    
    def _calculate_balance_ratio(self, df: pd.DataFrame) -> float:
        """
        Расчет соотношения классов.
        
        Args:
            df: DataFrame с целевой переменной
        
        Returns:
            Соотношение классов
        """
        try:
            # Ищем целевую переменную (может быть target, target_dynamic, target_3class)
            target_columns = ['target', 'target_dynamic', 'target_3class']
            target_col = None
            
            for col in target_columns:
                if col in df.columns:
                    target_col = col
                    break
            
            if target_col is None:
                return 0.0
            
            target_counts = df[target_col].value_counts()
            if len(target_counts) < 2:
                return 0.0
            
            min_count = target_counts.min()
            max_count = target_counts.max()
            
            return min_count / max_count if max_count > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def get_processing_statistics(self) -> Dict:
        """
        Получение статистики обработки.
        
        Returns:
            Словарь со статистикой
        """
        stats = self.processing_stats.copy()
        
        # Добавление информации о конфигурации
        stats['indicators_enabled'] = len(self.indicators_config) > 0
        stats['balancing_enabled'] = self.balancing_config.get('enabled', False)
        stats['multi_timeframe_enabled'] = self.multi_tf_config.get('enabled', False)
        
        return stats
    
    def reset_statistics(self):
        """Сброс статистики обработки."""
        self.processing_stats = {
            'total_features': 0,
            'timeframes_processed': 0,
            'balance_ratio_before': 0,
            'balance_ratio_after': 0,
            'target_distribution': {},
            'feature_importance': {},
            'processing_time': 0
        }
        self.logger.info("Статистика обработки сброшена")

    def add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет rolling статистики для улучшения признаков.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с добавленными rolling статистиками
        """
        try:
            self.logger.info("Добавление rolling статистик")
            
            # Rolling статистики для цены
            price_columns = ['close', 'high', 'low', 'open']
            for col in price_columns:
                if col in df.columns:
                    # Rolling mean с разными периодами
                    df[f'rolling_mean_5_{col}'] = df[col].rolling(window=5).mean()
                    df[f'rolling_mean_10_{col}'] = df[col].rolling(window=10).mean()
                    df[f'rolling_mean_20_{col}'] = df[col].rolling(window=20).mean()
                    
                    # Rolling max/min
                    df[f'rolling_max_50_{col}'] = df[col].rolling(window=50).max()
                    df[f'rolling_min_50_{col}'] = df[col].rolling(window=50).min()
                    
                    # Rolling std
                    df[f'rolling_std_20_{col}'] = df[col].rolling(window=20).std()
                    
                    # Rolling quantiles
                    df[f'rolling_q75_20_{col}'] = df[col].rolling(window=20).quantile(0.75)
                    df[f'rolling_q25_20_{col}'] = df[col].rolling(window=20).quantile(0.25)
            
            # Rolling статистики для объема
            if 'volume' in df.columns:
                df['rolling_mean_5_volume'] = df['volume'].rolling(window=5).mean()
                df['rolling_max_50_volume'] = df['volume'].rolling(window=50).max()
                df['rolling_std_20_volume'] = df['volume'].rolling(window=20).std()
                df['volume_ratio_5'] = df['volume'] / df['rolling_mean_5_volume']
                df['volume_ratio_50'] = df['volume'] / df['rolling_max_50_volume']
            
            # Rolling статистики для технических индикаторов
            indicator_columns = [col for col in df.columns if any(indicator in col.lower() 
                                                               for indicator in ['rsi', 'macd', 'sma', 'ema', 'bb'])]
            
            for col in indicator_columns:
                if col in df.columns:
                    df[f'rolling_mean_5_{col}'] = df[col].rolling(window=5).mean()
                    df[f'rolling_std_20_{col}'] = df[col].rolling(window=20).std()
                    df[f'rolling_max_50_{col}'] = df[col].rolling(window=50).max()
                    df[f'rolling_min_50_{col}'] = df[col].rolling(window=50).min()
            
            # Ценовые отношения с rolling статистиками
            if 'close' in df.columns:
                df['price_vs_rolling_mean_5'] = df['close'] / df['rolling_mean_5_close']
                df['price_vs_rolling_mean_20'] = df['close'] / df['rolling_mean_20_close']
                df['price_vs_rolling_max_50'] = df['close'] / df['rolling_max_50_close']
                df['price_vs_rolling_min_50'] = df['close'] / df['rolling_min_50_close']
            
            self.logger.info(f"Добавлено {len([col for col in df.columns if 'rolling_' in col])} rolling признаков")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка при добавлении rolling статистик: {e}")
            return df

    def select_top_features(self, df: pd.DataFrame, target_col: str = 'target', 
                           n_features: int = 30, importance_threshold: float = 0.005) -> pd.DataFrame:
        """
        Отбирает только топ-N признаков с важностью выше порога.
        
        Args:
            df: DataFrame с данными
            target_col: Название целевой переменной
            n_features: Количество топ признаков для отбора
            importance_threshold: Минимальный порог важности признака
            
        Returns:
            DataFrame только с отобранными признаками
        """
        try:
            self.logger.info(f"Отбор топ-{n_features} признаков с порогом важности {importance_threshold}")
            
            # Подготовка данных
            X = df.drop([target_col], axis=1, errors='ignore')
            y = df[target_col]
            
            # Удаляем колонки с NaN
            X = X.dropna(axis=1)
            
            # Используем mutual information для оценки важности
            from sklearn.feature_selection import mutual_info_classif
            
            # Вычисляем важность признаков
            importance_scores = mutual_info_classif(X, y, random_state=42)
            feature_importance = dict(zip(X.columns, importance_scores))
            
            # Сортируем по важности
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Фильтруем по порогу важности
            filtered_features = [(name, score) for name, score in sorted_features if score >= importance_threshold]
            
            # Берем топ-N признаков
            top_features = filtered_features[:n_features]
            
            # Логируем результаты
            self.logger.info(f"Найдено {len(filtered_features)} признаков с важностью >= {importance_threshold}")
            self.logger.info(f"Отобрано топ-{len(top_features)} признаков:")
            
            for i, (feature, importance) in enumerate(top_features, 1):
                self.logger.info(f"{i:2d}. {feature}: {importance:.6f}")
            
            # Создаем новый DataFrame только с отобранными признаками
            selected_columns = [target_col] + [feature for feature, _ in top_features]
            selected_df = df[selected_columns].copy()
            
            self.logger.info(f"Итоговый размер данных: {selected_df.shape}")
            return selected_df
            
        except Exception as e:
            self.logger.error(f"Ошибка при отборе признаков: {e}")
            return df 

    def process_features(self, data: pd.DataFrame, symbol: str, timeframe: str,
                         prediction_horizon: int = None, adaptive_indicators_enabled: bool = True,
                         fixed_indicators: List[str] = None) -> pd.DataFrame:
        """
        Универсальный метод для обработки признаков с гибкими настройками.
        
        Args:
            data: DataFrame с данными
            symbol: Торговая пара
            timeframe: Таймфрейм данных
            prediction_horizon: Горизонт предсказаний (если None, берется из конфигурации)
            adaptive_indicators_enabled: Использовать ли адаптивный выбор индикаторов
            fixed_indicators: Список фиксированных индикаторов для использования
        
        Returns:
            DataFrame с обработанными признаками
        """
        try:
            self.logger.info(f"Начинаем инжиниринг признаков для {symbol} на {timeframe}")
            
            # Определяем горизонт предсказаний
            if prediction_horizon is None:
                prediction_horizon = self.prediction_horizons.get(timeframe, 12)
            self.logger.info(f"Горизонт предсказаний: {prediction_horizon} свечей")
            
            # Создаем копию данных
            processed_df = data.copy()
            
            # Выбираем способ создания индикаторов
            if adaptive_indicators_enabled and self.config.get('adaptive_indicators', {}).get('enabled', True):
                self.logger.info("🎯 Используем адаптивный выбор индикаторов")
                
                # Оптимизируем индикаторы на основе прибыльности
                optimization_results = self.adaptive_selector.optimize_indicators(
                    processed_df, timeframe, force_recalculate=not self._use_cached_indicators
                )
                
                if optimization_results and 'best_combination' in optimization_results:
                    self.logger.info(f"✅ Найдены лучшие индикаторы: {optimization_results['best_combination']}")
                    processed_df = self.adaptive_selector.add_best_indicators(processed_df, timeframe)
                else:
                    self.logger.warning("⚠️ Адаптивная оптимизация не удалась, используем базовые индикаторы")
                    processed_df = self.indicators.add_all_indicators(processed_df)
                    
            elif fixed_indicators:
                self.logger.info(f"✅ Используем фиксированные индикаторы: {fixed_indicators}")
                # Добавляем только указанные индикаторы
                for indicator in fixed_indicators:
                    if indicator == 'rsi':
                        processed_df = self.indicators.add_rsi(processed_df)
                    elif indicator == 'macd':
                        processed_df = self.indicators.add_macd(processed_df)
                    elif indicator == 'obv':
                        processed_df = self.indicators.add_obv(processed_df)
                    elif indicator == 'vwap':
                        processed_df = self.indicators.add_vwap(processed_df)
                    elif indicator == 'atr':
                        processed_df = self.indicators.add_atr(processed_df)
                    elif indicator == 'williams_r':
                        processed_df = self.indicators.add_williams_r(processed_df)
                    elif indicator == 'cci':
                        processed_df = self.indicators.add_cci(processed_df)
                    elif indicator == 'mfi':
                        processed_df = self.indicators.add_advanced_volume_indicators(processed_df)
                    elif indicator == 'sma':
                        processed_df = self.indicators.add_sma(processed_df, 20)
                        processed_df = self.indicators.add_sma(processed_df, 50)
                    elif indicator == 'ema':
                        processed_df = self.indicators.add_ema(processed_df, 12)
                        processed_df = self.indicators.add_ema(processed_df, 26)
                    elif indicator == 'bollinger_bands':
                        processed_df = self.indicators.add_bollinger_bands(processed_df)
                    else:
                        self.logger.warning(f"Индикатор {indicator} не поддерживается")
                        
            else:
                # Используем стандартные индикаторы
                self.logger.info("📊 Используем стандартные индикаторы")
                processed_df = self.indicators.add_all_indicators(processed_df)
            
            # Создание целевой переменной с правильным горизонтом
            processed_df = self._create_target_with_horizon(processed_df, prediction_horizon)
            
            # Добавление признаков на основе горизонта
            processed_df = self._add_horizon_based_features(processed_df, prediction_horizon)
            
            # Удаляем строки с NaN значениями
            initial_rows = len(processed_df)
            processed_df = processed_df.dropna()
            removed_rows = initial_rows - len(processed_df)
            
            if removed_rows > 0:
                self.logger.warning(f"Удалено {removed_rows} строк с NaN значениями")
            
            # Проверка на достаточное количество данных
            if len(processed_df) < 100:
                self.logger.error(f"Недостаточно данных после обработки: {len(processed_df)} строк")
                return processed_df
            
            # Обновление статистики
            self.processing_stats['total_features'] = len(processed_df.columns)
            self.processing_stats['balance_ratio_before'] = self._calculate_balance_ratio(processed_df)
            
            self.logger.info(f"Инжиниринг признаков завершен. Размер данных: {processed_df.shape}")
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Ошибка инжиниринга признаков для {symbol}: {e}")
            return data 

    def _add_simplified_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет упрощенные sentiment фичи (только Bybit Opportunities).
        
        Args:
            df: DataFrame с данными
        
        Returns:
            DataFrame с добавленными sentiment фичами
        """
        try:
            self.logger.info("Добавление упрощенных sentiment фичей (Bybit Opportunities)")
            
            # Получаем Bybit market opportunities
            bybit_data = self.sentiment_collector.get_bybit_opportunities()
            
            if bybit_data:
                # Базовые sentiment фичи
                df['bybit_market_sentiment'] = bybit_data.get('market_sentiment', 0.5)
                
                # Количество горячих секторов
                hot_sectors = bybit_data.get('hot_sectors', [])
                df['bybit_hot_sectors_count'] = len(hot_sectors)
                
                # Анализ трендинговых монет
                trending_coins = bybit_data.get('trending_coins', [])
                if trending_coins:
                    positive_trending = sum(1 for coin in trending_coins if coin.get('sentiment', 0) > 0)
                    df['bybit_positive_trending_ratio'] = positive_trending / len(trending_coins)
                else:
                    df['bybit_positive_trending_ratio'] = 0.5
                
                # Gainers vs Losers ratio
                gainers_losers = bybit_data.get('gainers_losers', {})
                gainers_count = len(gainers_losers.get('gainers', []))
                losers_count = len(gainers_losers.get('losers', []))
                
                if gainers_count + losers_count > 0:
                    df['bybit_gainers_ratio'] = gainers_count / (gainers_count + losers_count)
                else:
                    df['bybit_gainers_ratio'] = 0.5
                
                # Composite sentiment score (simplified)
                df['sentiment_composite_score'] = (
                    df['bybit_market_sentiment'] * 0.4 +
                    df['bybit_positive_trending_ratio'] * 0.3 +
                    df['bybit_gainers_ratio'] * 0.3
                )
                
                # Market regime classification (simplified)
                df['market_regime_bullish'] = (df['sentiment_composite_score'] > 0.6).astype(int)
                df['market_regime_bearish'] = (df['sentiment_composite_score'] < 0.4).astype(int)
                df['market_regime_neutral'] = (
                    (df['sentiment_composite_score'] >= 0.4) & 
                    (df['sentiment_composite_score'] <= 0.6)
                ).astype(int)
                
                self.logger.info("✅ Добавлены simplified sentiment фичи:")
                self.logger.info(f"  • Market Sentiment: {df['bybit_market_sentiment'].iloc[-1]:.3f}")
                self.logger.info(f"  • Hot Sectors: {df['bybit_hot_sectors_count'].iloc[-1]}")
                self.logger.info(f"  • Positive Trending Ratio: {df['bybit_positive_trending_ratio'].iloc[-1]:.3f}")
                self.logger.info(f"  • Gainers Ratio: {df['bybit_gainers_ratio'].iloc[-1]:.3f}")
                self.logger.info(f"  • Composite Score: {df['sentiment_composite_score'].iloc[-1]:.3f}")
                
                # Определяем текущий режим
                if df['market_regime_bullish'].iloc[-1]:
                    regime = "🐂 БЫЧИЙ"
                elif df['market_regime_bearish'].iloc[-1]:
                    regime = "🐻 МЕДВЕЖИЙ"
                else:
                    regime = "😐 НЕЙТРАЛЬНЫЙ"
                
                self.logger.info(f"  • Market Regime: {regime}")
                
            else:
                self.logger.warning("Не удалось получить Bybit данные, добавляем нейтральные значения")
                # Добавляем нейтральные значения при ошибке
                df['bybit_market_sentiment'] = 0.5
                df['bybit_hot_sectors_count'] = 3
                df['bybit_positive_trending_ratio'] = 0.5
                df['bybit_gainers_ratio'] = 0.5
                df['sentiment_composite_score'] = 0.5
                df['market_regime_bullish'] = 0
                df['market_regime_bearish'] = 0
                df['market_regime_neutral'] = 1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления sentiment фичей: {e}")
            # В случае ошибки добавляем нейтральные значения
            df['bybit_market_sentiment'] = 0.5
            df['bybit_hot_sectors_count'] = 3
            df['bybit_positive_trending_ratio'] = 0.5
            df['bybit_gainers_ratio'] = 0.5
            df['sentiment_composite_score'] = 0.5
            df['market_regime_bullish'] = 0
            df['market_regime_bearish'] = 0
            df['market_regime_neutral'] = 1
            return df
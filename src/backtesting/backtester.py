"""
Модуль для бэктестинга торговых стратегий.

Этот модуль предоставляет функциональность для:
- Загрузки обученных моделей
- Генерации торговых сигналов
- Симуляции торговли
- Расчета метрик производительности
- Визуализации результатов
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Импорты из нашей системы
from src.preprocessing.technical_indicators import indicators
from src.preprocessing.target_creation import TargetCreator
from src.trading.risk_manager import AdaptiveRiskManager
from src.data_collection.news_analyzer import NewsAnalyzer


class Backtester:
    """
    Класс для бэктестинга ML торговой модели.
    
    Поддерживает:
    - Бэктестинг на исторических данных
    - Анализ торговых сигналов
    - Расчет метрик производительности
    - Визуализацию результатов
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация бэктестера.
        
        Args:
            config: Конфигурация бэктестинга
        """
        self.config = config or {}
        self.logger = logger.bind(name="Backtester")
        
        # Настройки по умолчанию
        self.default_config = {
            'initial_capital': 10000,
            'commission': 0.001,  # 0.1%
            'min_confidence': 0.7,
            'trend_filter': True,
            'volatility_filter': True,
            'max_positions': 3,
            'adaptive_risk': True,  # Используем адаптивное управление рисками
            'news_sentiment_weight': 0.3,  # Вес новостного сигнала
            'position_size': 0.1,
            'stop_loss': 0.02,
            'take_profit': 0.04
        }
        
        # Обновление конфигурации
        if config:
            self.default_config.update(config)
        
        # Инициализация компонентов
        self.risk_manager = AdaptiveRiskManager(self.config.get('risk_management', {}))
        self.news_analyzer = NewsAnalyzer(self.config.get('news_analysis', {}))
        self.target_creator = TargetCreator()
        
        # Состояние бэктестинга
        self.reset_state()
        
        # Создание директории для результатов
        self.results_dir = Path("backtest_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def reset_state(self):
        """Сброс состояния бэктестинга."""
        self.capital = self.default_config['initial_capital']
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.current_step = 0
        
        # Статистика
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.total_loss = 0
    
    def load_model(self, model_path: str) -> bool:
        """
        Загрузка обученной модели.
        
        Args:
            model_path: Путь к файлу модели
            
        Returns:
            True если модель загружена успешно
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            # Проверяем, что это словарь с данными модели
            if isinstance(model_data, dict) and 'class_model' in model_data:
                # Создаем новый экземпляр ImprovedXGBoostModel
                from src.ml_models.improved_xgboost_model import ImprovedXGBoostModel
                self.model = ImprovedXGBoostModel()
                
                # Восстанавливаем состояние модели
                self.model.class_model = model_data.get('class_model')
                self.model.reg_model = model_data.get('reg_model')
                self.model.ensemble_class_models = model_data.get('ensemble_class_models', [])
                self.model.ensemble_reg_models = model_data.get('ensemble_reg_models', [])
                self.model.feature_selector = model_data.get('feature_selector')
                self.model.selected_features = model_data.get('selected_features', [])
                self.model.training_results = model_data.get('training_results', {})
                self.model.config = model_data.get('config', {})
                self.model.default_params = model_data.get('default_params', {})
                self.model.ensemble_config = model_data.get('ensemble_config', {})
                self.model.is_trained = model_data.get('is_trained', False)
                
                # Проверяем, что модель действительно обучена
                if not self.model.is_trained:
                    self.logger.warning("⚠️ Модель помечена как необученная")
                    return False
                    
            else:
                # Если это уже объект модели
                self.model = model_data
            
            self.logger.info(f"Модель загружена из {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка признаков для бэктестинга.
        
        Args:
            data: DataFrame с OHLCV данными
            
        Returns:
            DataFrame с подготовленными признаками
        """
        try:
            df = data.copy()
            
            # Получаем список ожидаемых признаков из модели
            expected_features = []
            
            # Сначала пытаемся получить признаки из загруженной модели
            if hasattr(self, 'model') and self.model is not None:
                if hasattr(self.model, 'selected_features') and self.model.selected_features is not None:
                    expected_features = self.model.selected_features
                    self.logger.info(f"Получены признаки из модели: {len(expected_features)} признаков")
                else:
                    self.logger.warning("Модель не содержит selected_features")
            
            # Если не удалось получить из модели, пытаемся из файла
            if not expected_features:
                try:
                    # Пытаемся загрузить из selected_features.json
                    features_file = Path("models/selected_features.json")
                    if features_file.exists():
                        with open(features_file, 'r') as f:
                            expected_features = json.load(f)
                        self.logger.info(f"Получены признаки из selected_features.json: {len(expected_features)} признаков")
                    else:
                        self.logger.error("Не найдены файлы моделей")
                        return pd.DataFrame()
                except Exception as e:
                    self.logger.error(f"Не удалось загрузить список признаков: {e}")
                    return pd.DataFrame()
            
            if not expected_features:
                self.logger.error("Список признаков пуст")
                return pd.DataFrame()
            
            self.logger.info(f"Используем список признаков из модели: {len(expected_features)} признаков")
            self.logger.debug(f"Первые 10 признаков: {expected_features[:10]}")
            
            # Генерируем все необходимые признаки
            # Базовые OHLCV признаки уже есть
            
            # Технические индикаторы
            if 'rsi_14' not in df.columns:
                df['rsi_14'] = indicators.calculate_rsi(df, 14)
            
            if 'macd_line' not in df.columns:
                macd_data = indicators.calculate_macd(df)
                df['macd_line'] = macd_data['macd']
                df['macd_signal'] = macd_data['signal']
                df['macd_histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            if 'bb_upper_20' not in df.columns:
                bb_data = indicators.calculate_bollinger_bands(df, 20)
                df['bb_upper_20'] = bb_data['upper']
                df['bb_middle_20'] = bb_data['middle']
                df['bb_lower_20'] = bb_data['lower']
                df['bb_bandwidth_20'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
                df['bb_position_20'] = (df['close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
            
            # SMA
            for period in [10, 20, 50, 200]:
                if f'sma_{period}' not in df.columns:
                    df[f'sma_{period}'] = indicators.calculate_sma(df, period)
            
            # EMA
            for period in [12, 26, 50]:
                if f'ema_{period}' not in df.columns:
                    df[f'ema_{period}'] = indicators.calculate_ema(df, period)
            
            # Stochastic
            if 'stoch_k' not in df.columns:
                stoch_data = indicators.calculate_stochastic(df, 14)
                df['stoch_k'] = stoch_data['k']
                df['stoch_d'] = stoch_data['d']
            
            # Williams %R
            if 'williams_r' not in df.columns:
                df['williams_r'] = indicators.calculate_williams_r(df, 14)
            
            # CCI
            if 'cci' not in df.columns:
                df['cci'] = indicators.calculate_cci(df, 20)
            
            # ADX
            if 'adx' not in df.columns:
                adx_data = indicators.calculate_adx(df, 14)
                df['adx'] = adx_data['adx']
                df['plus_di'] = adx_data['plus_di']
                df['minus_di'] = adx_data['minus_di']
            
            # ATR
            if 'atr' not in df.columns:
                df['atr'] = indicators.calculate_atr(df, 14)
            
            # OBV
            if 'obv' not in df.columns:
                df['obv'] = indicators.calculate_obv(df)
            
            # VWAP
            if 'vwap' not in df.columns:
                df['vwap'] = indicators.calculate_vwap(df)
            
            # Momentum
            if 'momentum' not in df.columns:
                df['momentum'] = indicators.calculate_momentum(df, 10)
            
            # Ichimoku
            if 'tenkan_sen' not in df.columns:
                ichimoku_data = indicators.calculate_ichimoku(df)
                df['tenkan_sen'] = ichimoku_data['tenkan_sen']
                df['kijun_sen'] = ichimoku_data['kijun_sen']
                df['senkou_span_a'] = ichimoku_data['senkou_span_a']
                df['senkou_span_b'] = ichimoku_data['senkou_span_b']
                df['chikou_span'] = ichimoku_data['chikou_span']
            
            # Hull MA
            if 'hull_ma' not in df.columns:
                df['hull_ma'] = indicators.calculate_hull_ma(df, 20)
            
            # Awesome Oscillator
            if 'awesome_oscillator' not in df.columns:
                df['awesome_oscillator'] = indicators.calculate_awesome_oscillator(df)
            
            # Ultimate Oscillator
            if 'ultimate_oscillator' not in df.columns:
                df['ultimate_oscillator'] = indicators.calculate_ultimate_oscillator(df)
            
            # Bulls/Bears Power
            if 'bulls_power' not in df.columns:
                bb_power_data = indicators.calculate_bulls_bears_power(df)
                df['bulls_power'] = bb_power_data['bulls_power']
                df['bears_power'] = bb_power_data['bears_power']
            
            # Fast Stochastic RSI
            if 'fast_stoch_rsi' not in df.columns:
                df['fast_stoch_rsi'] = indicators.calculate_fast_stochastic_rsi(df)
            
            # Дополнительные признаки
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Волатильность
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            # Тренд
            df['trend'] = np.where(df['close'] > df['sma_50'], 1, -1)
            
            # Дополнительные признаки, которые могут отсутствовать
            # Price to SMA ratios
            if 'price_to_sma_20' not in df.columns:
                df['price_to_sma_20'] = df['close'] / df['sma_20']
            if 'price_to_sma_50' not in df.columns:
                df['price_to_sma_50'] = df['close'] / df['sma_50']
            
            # Price to EMA ratios
            if 'price_to_ema_12' not in df.columns:
                df['price_to_ema_12'] = df['close'] / df['ema_12']
            if 'price_to_ema_26' not in df.columns:
                df['price_to_ema_26'] = df['close'] / df['ema_26']
            
            # Volatility periods
            if 'volatility_20' not in df.columns:
                df['volatility_20'] = df['price_change'].rolling(window=20).std()
            if 'volatility_50' not in df.columns:
                df['volatility_50'] = df['price_change'].rolling(window=50).std()
            
            # Volume features
            if 'relative_volume' not in df.columns:
                df['relative_volume'] = df['volume'] / df['volume'].rolling(window=20).mean()
            if 'volume_price_sma' not in df.columns:
                df['volume_price_sma'] = df['volume'] * df['close']
            if 'volume_ratio' not in df.columns:
                df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=10).mean()
            
            # Trend features
            if 'trend_12' not in df.columns:
                df['trend_12'] = np.where(df['close'] > df['close'].shift(12), 1, -1)
            
            # Price range features
            if 'max_price_12' not in df.columns:
                df['max_price_12'] = df['high'].rolling(window=12).max()
            if 'min_price_12' not in df.columns:
                df['min_price_12'] = df['low'].rolling(window=12).min()
            if 'distance_to_max_12' not in df.columns:
                df['distance_to_max_12'] = (df['max_price_12'] - df['close']) / df['close']
            if 'distance_to_min_12' not in df.columns:
                df['distance_to_min_12'] = (df['close'] - df['min_price_12']) / df['close']
            
            # Volume profile
            if 'volume_ratio_12' not in df.columns:
                df['volume_ratio_12'] = df['volume'] / df['volume'].rolling(window=12).mean()
            
            # Мультитаймфреймовые признаки (15m)
            # Создаем простые версии мультитаймфреймовых признаков
            if '15m_15m_open' not in df.columns:
                df['15m_15m_open'] = df['open'].rolling(window=3).mean()  # Примерно 15m для 5m данных
            if '15m_15m_high' not in df.columns:
                df['15m_15m_high'] = df['high'].rolling(window=3).max()
            if '15m_15m_low' not in df.columns:
                df['15m_15m_low'] = df['low'].rolling(window=3).min()
            if '15m_15m_close' not in df.columns:
                df['15m_15m_close'] = df['close'].rolling(window=3).mean()
            if '15m_15m_volume' not in df.columns:
                df['15m_15m_volume'] = df['volume'].rolling(window=3).mean()
            
            # 15m технические индикаторы
            if '15m_15m_rsi_7' not in df.columns:
                df['15m_15m_rsi_7'] = indicators.calculate_rsi(df, 7)
            if '15m_15m_obv' not in df.columns:
                df['15m_15m_obv'] = indicators.calculate_obv(df)
            if '15m_15m_vwap' not in df.columns:
                df['15m_15m_vwap'] = indicators.calculate_vwap(df)
            
            # 15m целевые и прогнозные признаки
            if '15m_15m_target_dynamic' not in df.columns:
                # Создаем простую целевую переменную
                future_returns = df['close'].shift(-3) / df['close'] - 1
                df['15m_15m_target_dynamic'] = np.where(future_returns > 0.01, 2, 
                                                      np.where(future_returns < -0.01, 0, 1))
            if '15m_15m_prediction_horizon' not in df.columns:
                df['15m_15m_prediction_horizon'] = 3  # Горизонт 3 свечи
            
            # 15m трендовые признаки
            if '15m_15m_trend_10' not in df.columns:
                df['15m_15m_trend_10'] = np.where(df['close'] > df['close'].shift(10), 1, -1)
            if '15m_15m_volatility_10' not in df.columns:
                df['15m_15m_volatility_10'] = df['price_change'].rolling(window=10).std()
            if '15m_15m_max_price_10' not in df.columns:
                df['15m_15m_max_price_10'] = df['high'].rolling(window=10).max()
            if '15m_15m_min_price_10' not in df.columns:
                df['15m_15m_min_price_10'] = df['low'].rolling(window=10).min()
            if '15m_15m_distance_to_max_10' not in df.columns:
                df['15m_15m_distance_to_max_10'] = (df['15m_15m_max_price_10'] - df['close']) / df['close']
            if '15m_15m_distance_to_min_10' not in df.columns:
                df['15m_15m_distance_to_min_10'] = (df['close'] - df['15m_15m_min_price_10']) / df['close']
            if '15m_15m_momentum_10' not in df.columns:
                df['15m_15m_momentum_10'] = df['close'] / df['close'].shift(10) - 1
            if '15m_15m_volume_profile_10' not in df.columns:
                df['15m_15m_volume_profile_10'] = df['volume'].rolling(window=10).mean()
            if '15m_15m_volume_ratio_10' not in df.columns:
                df['15m_15m_volume_ratio_10'] = df['volume'] / df['volume'].rolling(window=10).mean()
            
            # Удаляем NaN значения
            initial_rows = len(df)
            
            # Проверяем, сколько NaN в каждой колонке
            nan_counts = df.isnull().sum()
            if nan_counts.sum() > 0:
                self.logger.warning(f"Найдены NaN значения по колонкам: {nan_counts[nan_counts > 0].to_dict()}")
                
                # Заполняем NaN значения вместо удаления строк
                # Для числовых колонок используем forward fill, затем backward fill, затем 0
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                # Для нечисловых колонок используем 0
                non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
                df[non_numeric_columns] = df[non_numeric_columns].fillna(0)
                
                self.logger.info("NaN значения заполнены вместо удаления строк")
            
            # Проверяем, остались ли NaN после заполнения
            remaining_nans = df.isnull().sum().sum()
            if remaining_nans > 0:
                self.logger.warning(f"Остались {remaining_nans} NaN значений после заполнения")
                # Удаляем только строки, где все значения NaN
                df = df.dropna(how='all')
                self.logger.info(f"Удалены строки, где все значения NaN. Осталось строк: {len(df)}")
            
            if len(df) == 0:
                self.logger.error("Все строки удалены после обработки NaN")
                return pd.DataFrame()
            
            # Выравниваем признаки с ожидаемыми
            df = self._align_features(df, expected_features)
            
            # Финальная проверка
            if df.empty:
                self.logger.error("DataFrame пуст после подготовки признаков")
                return df
            
            self.logger.info(f"Подготовка признаков завершена. Размер: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка подготовки признаков: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def _align_features(self, df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
        """
        Выравнивание признаков с ожидаемыми признаками модели.
        
        Args:
            df: DataFrame с признаками
            expected_features: Список ожидаемых признаков
            
        Returns:
            DataFrame с выровненными признаками
        """
        try:
            self.logger.debug(f"Начинаем выравнивание признаков. Доступно колонок: {len(df.columns)}")
            self.logger.debug(f"Ожидается признаков: {len(expected_features)}")
            
            # Проверяем, какие признаки есть в данных
            available_features = [col for col in expected_features if col in df.columns]
            missing_features = [col for col in expected_features if col not in df.columns]
            
            self.logger.info(f"Доступно признаков: {len(available_features)}/{len(expected_features)}")
            
            if missing_features:
                self.logger.warning(f"Отсутствуют признаки: {missing_features}")
                # Добавляем отсутствующие признаки с нулями
                for feature in missing_features:
                    df[feature] = 0.0
                self.logger.info(f"Добавлено {len(missing_features)} отсутствующих признаков с нулевыми значениями")
            
            # Убираем лишние признаки
            extra_features = [f for f in df.columns if f not in expected_features and f not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            if extra_features:
                self.logger.warning(f"Лишние признаки: {extra_features}")
            
            # Проверяем на NaN и Inf
            nan_check = df[expected_features].isnull().sum()
            if nan_check.sum() > 0:
                self.logger.warning(f"Найдены NaN значения: {nan_check[nan_check > 0].to_dict()}")
                # Заполняем NaN нулями
                df[expected_features] = df[expected_features].fillna(0)
                self.logger.info("NaN значения заполнены нулями")
            
            # Проверяем на Inf
            inf_check = np.isinf(df[expected_features].values).sum()
            if inf_check > 0:
                self.logger.warning(f"Найдены Inf значения: {inf_check}")
                # Заменяем Inf на большие числа
                df[expected_features] = df[expected_features].replace([np.inf, -np.inf], [1e6, -1e6])
                self.logger.info("Inf значения заменены на 1e6/-1e6")
            
            # Приводим к нужному порядку
            try:
                aligned_df = df[expected_features]
                self.logger.info(f"Признаки приведены к порядку модели: {len(expected_features)} признаков")
                self.logger.debug(f"Итоговый размер: {aligned_df.shape}")
                return aligned_df
            except KeyError as e:
                self.logger.error(f"Ошибка при выборе признаков: {e}")
                self.logger.error(f"Ожидаемые признаки: {expected_features}")
                self.logger.error(f"Доступные колонки: {list(df.columns)}")
                raise
            
        except Exception as e:
            self.logger.error(f"Ошибка выравнивания признаков: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация торговых сигналов на основе dual-head модели (классификация + регрессия).
        """
        try:
            self.logger.info(f"Начинаем генерацию сигналов. Размер входных данных: {data.shape}")
            
            # Подготавливаем признаки
            result_df = self.prepare_features(data)
            
            if result_df.empty:
                self.logger.error("Не удалось подготовить признаки")
                return pd.DataFrame()
            
            self.logger.info(f"Признаки подготовлены. Размер: {result_df.shape}")
            
            # Проверяем модель
            if not hasattr(self, 'model') or self.model is None:
                self.logger.error("Модель не загружена")
                return pd.DataFrame()
            
            # Получаем предсказания и вероятности
            try:
                if hasattr(self.model, 'predict_improved_dual'):
                    self.logger.info("Используем predict_improved_dual")
                    class_pred, class_proba, reg_pred = self.model.predict_improved_dual(result_df)
                elif hasattr(self.model, 'predict_dual'):
                    self.logger.info("Используем predict_dual")
                    class_pred, class_proba, reg_pred = self.model.predict_dual(result_df)
                elif hasattr(self.model, 'predict'):
                    self.logger.info("Используем predict и predict_proba")
                    class_pred = self.model.predict(result_df)
                    class_proba = self.model.predict_proba(result_df)
                    reg_pred = np.zeros(len(class_pred))
                else:
                    self.logger.error("Модель не поддерживает предсказания")
                    return pd.DataFrame()
                
                self.logger.info(f"Предсказания получены. Размер: {len(class_pred)}")
                
            except Exception as e:
                self.logger.error(f"Ошибка при получении предсказаний: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return pd.DataFrame()
            
            # Создаем DataFrame с сигналами - используем исходные данные для OHLCV
            # Проверяем, есть ли timestamp в индексе или в колонках
            if 'timestamp' in data.columns:
                signals_df = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            elif data.index.name == 'timestamp':
                # Если timestamp в индексе, создаем колонку из индекса
                signals_df = data[['open', 'high', 'low', 'close', 'volume']].copy()
                signals_df['timestamp'] = data.index
            else:
                # Если нет timestamp, создаем его из индекса
                signals_df = data[['open', 'high', 'low', 'close', 'volume']].copy()
                signals_df['timestamp'] = data.index
            
            # Убеждаемся, что длины совпадают
            min_length = min(len(signals_df), len(class_pred))
            signals_df = signals_df.iloc[:min_length]
            class_pred = class_pred[:min_length]
            class_proba = class_proba[:min_length]
            
            # Безопасная обработка reg_pred
            if reg_pred is not None and len(reg_pred) > 0:
                reg_pred = reg_pred[:min_length]
            else:
                reg_pred = np.zeros(min_length)
            
            signals_df['prediction'] = class_pred
            signals_df['confidence'] = np.max(class_proba, axis=1)
            signals_df['predicted_return'] = reg_pred
            
            # Генерируем сигналы только по уверенности модели
            buy_condition = (
                (signals_df['prediction'] == 2) &  # Rise prediction
                (signals_df['confidence'] >= 0.5)
            )
            
            sell_condition = (
                (signals_df['prediction'] == 0) &  # Fall prediction
                (signals_df['confidence'] >= 0.5)
            )
            
            self.logger.info(f"Сигналов только по уверенности (buy): {buy_condition.sum()}, (sell): {sell_condition.sum()}")
            signals_df.loc[buy_condition, 'signal'] = 1
            signals_df.loc[sell_condition, 'signal'] = -1
            
            # НОВОЕ: Ограничение частоты сигналов
            signals_df = self._limit_signal_frequency(signals_df)
            
            # Статистика сигналов
            buy_signals = len(signals_df[signals_df['signal'] == 1])
            sell_signals = len(signals_df[signals_df['signal'] == -1])
            total_signals = buy_signals + sell_signals
            
            self.logger.info(f"Сгенерировано сигналов: {total_signals} (buy: {buy_signals}, sell: {sell_signals})")
            
            return signals_df
            
        except Exception as e:
            self.logger.error(f"Ошибка при предсказании: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def _apply_advanced_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применение продвинутых фильтров для сигналов.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с примененными фильтрами
        """
        try:
            # 1. Фильтр тренда
            df = self._add_trend_filters(df)
            
            # 2. Фильтр волатильности
            df = self._add_volatility_filters(df)
            
            # 3. Фильтр объема
            df = self._add_volume_filters(df)
            
            # 4. НОВОЕ: Фильтр момента
            df = self._add_momentum_filters(df)
            
            # 5. НОВОЕ: Фильтр времени (избегаем торговли в определенные часы)
            df = self._add_time_filters(df)
            
            # 6. НОВОЕ: Фильтр спреда
            df = self._add_spread_filters(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка применения фильтров: {e}")
            return df
    
    def _add_trend_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление фильтров по тренду.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с фильтрами тренда
        """
        try:
            # Простой фильтр по тренду на основе SMA
            if 'sma_20' in df.columns and 'close' in df.columns:
                # Восходящий тренд: цена выше SMA20
                df['trend_filter'] = np.where(df['close'] > df['sma_20'], 1, -1)
            else:
                # Если нет SMA, используем простой фильтр по цене
                df['price_change'] = df['close'].pct_change()
                df['trend_filter'] = np.where(df['price_change'] > 0, 1, -1)
            
            # Добавляем фильтр волатильности
            if 'volatility_20' in df.columns:
                # Торгуем только при нормальной волатильности
                volatility_threshold = df['volatility_20'].quantile(0.8)  # 80-й перцентиль
                df['volatility_filter'] = df['volatility_20'] < volatility_threshold
            else:
                df['volatility_filter'] = True
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления фильтров тренда: {e}")
            df['trend_filter'] = 1  # По умолчанию разрешаем все сделки
            df['volatility_filter'] = True
            return df
    
    def _add_volatility_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление фильтров волатильности.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с фильтрами волатильности
        """
        try:
            # Добавляем фильтр волатильности
            if 'volatility_20' in df.columns:
                # Торгуем только при нормальной волатильности
                volatility_threshold = df['volatility_20'].quantile(0.8)  # 80-й перцентиль
                df['volatility_filter'] = df['volatility_20'] < volatility_threshold
            else:
                df['volatility_filter'] = True
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления фильтров волатильности: {e}")
            df['volatility_filter'] = True
            return df
    
    def _add_volume_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление фильтров по объему.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с фильтрами объема
        """
        try:
            # Фильтр объема - торгуем только при достаточном объеме
            if 'volume' in df.columns:
                # Рассчитываем скользящее среднее объема
                volume_sma = df['volume'].rolling(window=20).mean()
                
                # Минимальное соотношение объема (из конфигурации или по умолчанию)
                min_volume_ratio = self.default_config.get('min_volume_ratio', 1.2)
                
                # Текущий объем должен быть выше среднего в min_volume_ratio раз
                df['volume_filter'] = df['volume'] > (volume_sma * min_volume_ratio)
                
                # Заполняем NaN значения (первые 20 строк)
                df['volume_filter'] = df['volume_filter'].fillna(False)
            else:
                df['volume_filter'] = True
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления фильтров объема: {e}")
            df['volume_filter'] = True
            return df
    
    def _add_momentum_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление фильтров момента.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с фильтрами момента
        """
        try:
            # Рассчитываем RSI
            if 'rsi_14' not in df.columns:
                df['rsi_14'] = indicators.calculate_rsi(df, 14)
            
            # Рассчитываем MACD
            if 'macd_line' not in df.columns:
                df['macd_line'] = indicators.calculate_macd(df)['macd']
            
            # Рассчитываем момент цены
            df['price_momentum'] = df['close'].pct_change(5)
            
            # Фильтр момента для покупки
            buy_momentum = (
                (df['rsi_14'] > 30) & (df['rsi_14'] < 70) &  # RSI не перекуплен/перепродан
                (df['macd_line'] > 0) &  # MACD положительный
                (df['price_momentum'] > 0)  # Цена растет
            )
            
            # Фильтр момента для продажи
            sell_momentum = (
                (df['rsi_14'] > 30) & (df['rsi_14'] < 70) &  # RSI не перекуплен/перепродан
                (df['macd_line'] < 0) &  # MACD отрицательный
                (df['price_momentum'] < 0)  # Цена падает
            )
            
            # Общий фильтр момента
            df['momentum_filter'] = buy_momentum | sell_momentum
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета фильтров момента: {e}")
            df['momentum_filter'] = True
            return df
    
    def _add_time_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление временных фильтров.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с временными фильтрами
        """
        try:
            # Конвертируем timestamp в datetime
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['datetime'].dt.hour
            
            # Избегаем торговли в часы низкой ликвидности (2-6 утра UTC)
            # и в часы высокой волатильности (14-16 UTC - время выхода новостей)
            low_liquidity_hours = [2, 3, 4, 5, 6]
            high_volatility_hours = [14, 15, 16]
            
            df['time_filter'] = ~(
                df['hour'].isin(low_liquidity_hours) | 
                df['hour'].isin(high_volatility_hours)
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета временных фильтров: {e}")
            df['time_filter'] = True
            return df
    
    def _add_spread_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление фильтров спреда.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с фильтрами спреда
        """
        try:
            # Рассчитываем спред (разница между high и low)
            df['spread'] = (df['high'] - df['low']) / df['close']
            
            # Рассчитываем средний спред
            avg_spread = df['spread'].rolling(window=20).mean()
            
            # Фильтр: спред не должен быть слишком большим (>2% от цены)
            df['spread_filter'] = df['spread'] < 0.02
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета фильтров спреда: {e}")
            df['spread_filter'] = True
            return df
    
    def _limit_signal_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ограничение частоты сигналов.
        
        Args:
            df: DataFrame с сигналами
            
        Returns:
            DataFrame с ограниченными сигналами
        """
        try:
            # Минимальный интервал между сигналами (в барах)
            min_interval = 10
            
            # Находим индексы сигналов
            signal_indices = df[df['signal'] != 0].index
            
            self.logger.info(f"Найдено {len(signal_indices)} сигналов до фильтрации частоты")
            
            # Фильтруем слишком частые сигналы
            filtered_signals = []
            last_signal_idx = -min_interval
            
            for i, idx in enumerate(signal_indices):
                # Используем позицию в массиве вместо арифметики с timestamp
                if i == 0 or (i - last_signal_idx) >= min_interval:
                    filtered_signals.append(idx)
                    last_signal_idx = i
            
            self.logger.info(f"После фильтрации частоты осталось {len(filtered_signals)} сигналов")
            
            # Сохраняем исходные значения сигналов для отфильтрованных индексов
            original_signals = df.loc[filtered_signals, 'signal'].copy()
            
            # Сбрасываем все сигналы
            df['signal'] = 0
            
            # Устанавливаем только отфильтрованные сигналы с их исходными значениями
            df.loc[filtered_signals, 'signal'] = original_signals
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка ограничения частоты сигналов: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return df
    
    def execute_trade(self, signal: int, price: float, timestamp: datetime, 
                     confidence: float) -> Dict:
        """
        Выполнение торговой сделки.
        
        Args:
            signal: Торговый сигнал (1 = buy, -1 = sell, 0 = hold)
            price: Цена исполнения
            timestamp: Время сделки
            confidence: Уверенность модели
            
        Returns:
            Словарь с информацией о сделке
        """
        commission = self.default_config['commission']
        slippage = self.default_config['slippage']
        
        if signal == 1:  # Покупка
            # Рассчитываем размер позиции
            position_value = self.capital * self.default_config['position_size']
            shares = position_value / price
            
            # Учитываем комиссии и проскальзывание
            total_cost = position_value * (1 + commission + slippage)
            
            if total_cost <= self.capital:
                self.capital -= total_cost
                
                trade = {
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': price,
                    'shares': shares,
                    'value': position_value,
                    'commission': position_value * commission,
                    'slippage': position_value * slippage,
                    'confidence': confidence,
                    'capital_after': self.capital
                }
                
                self.positions.append(trade)
                self.trades.append(trade)
                
                return trade
            else:
                self.logger.warning(f"Недостаточно средств для покупки: {total_cost:.2f}")
                return {}
                
        elif signal == -1:  # Продажа
            if self.positions:
                # Закрываем последнюю позицию
                position = self.positions.pop()
                
                # Рассчитываем доход
                current_value = position['shares'] * price
                total_revenue = current_value * (1 - commission - slippage)
                
                self.capital += total_revenue
                
                trade = {
                    'timestamp': timestamp,
                    'action': 'sell',
                    'price': price,
                    'shares': position['shares'],
                    'value': current_value,
                    'commission': current_value * commission,
                    'slippage': current_value * slippage,
                    'confidence': confidence,
                    'capital_after': self.capital,
                    'pnl': total_revenue - position['value'],
                    'pnl_pct': (total_revenue - position['value']) / position['value']
                }
                
                self.trades.append(trade)
                
                return trade
            else:
                self.logger.warning("Нет открытых позиций для продажи")
                return {}
        
        return {}
    
    def run_backtest(self, data: pd.DataFrame, model_path: str = None) -> Dict:
        """
        Запуск бэктестинга.
        
        Args:
            data: Исторические данные OHLCV
            model_path: Путь к модели (если не загружена)
            
        Returns:
            Словарь с результатами бэктестинга
        """
        try:
            self.logger.info("Запуск бэктестинга")
            
            # Загружаем модель если нужно
            if model_path and not hasattr(self, 'model'):
                if not self.load_model(model_path):
                    return {}
            
            # Генерируем сигналы
            signals_df = self.generate_signals(data)
            
            if signals_df.empty:
                self.logger.error("Не удалось сгенерировать сигналы")
                return {}
            
            # Сбрасываем состояние
            self.reset_state()
            
            # Проходим по данным
            for idx, row in signals_df.iterrows():
                timestamp = row.get('timestamp', idx)
                price = row['close']
                signal = row['signal']
                confidence = row['confidence']
                
                # Выполняем сделку
                if signal != 0:
                    trade = self.execute_trade(signal, price, timestamp, confidence)
                    if trade:
                        self.logger.debug(f"Сделка: {trade['action']} по цене {price:.2f}")
                
                # Записываем состояние капитала
                current_equity = self.capital
                if self.positions:
                    # Добавляем стоимость открытых позиций
                    for pos in self.positions:
                        current_equity += pos['shares'] * price
                
                self.equity_curve.append({
                    'timestamp': timestamp,
                    'equity': current_equity,
                    'capital': self.capital,
                    'positions_value': current_equity - self.capital
                })
            
            # Закрываем все открытые позиции в конце
            if self.positions:
                last_price = signals_df.iloc[-1]['close']
                for position in self.positions:
                    trade = self.execute_trade(-1, last_price, signals_df.index[-1], 1.0)
            
            # Рассчитываем метрики производительности
            self.calculate_performance_metrics()
            
            self.logger.info("Бэктестинг завершен")
            return self.performance_metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка бэктестинга: {e}")
            return {}
    
    def calculate_performance_metrics(self):
        """Расчет метрик производительности."""
        try:
            if not self.equity_curve:
                return {}
            
            # Создаем DataFrame с кривой доходности
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['returns'] = equity_df['equity'].pct_change()
            
            # Базовые метрики
            initial_equity = self.default_config['initial_capital']
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity - initial_equity) / initial_equity
            
            # Годовые метрики (предполагаем 252 торговых дня)
            days = len(equity_df)
            annual_return = (1 + total_return) ** (252 / days) - 1
            
            # Волатильность
            daily_returns = equity_df['returns'].dropna()
            volatility = daily_returns.std() * np.sqrt(252)
            
            # Коэффициент Шарпа (безрисковая ставка = 0)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Максимальная просадка
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Статистика сделок
            winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
            
            win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            # Profit Factor
            if losing_trades and avg_loss != 0:
                profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades)))
            else:
                profit_factor = float('inf')
            
            # Сохраняем метрики
            self.performance_metrics = {
                'initial_capital': initial_equity,
                'final_capital': final_equity,
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'equity_curve': self.equity_curve,
                'trades': self.trades
            }
            
            return self.performance_metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета метрик: {e}")
            return {}
    
    def plot_results(self, save_path: str = None):
        """
        Визуализация результатов бэктестинга.
        
        Args:
            save_path: Путь для сохранения графиков
        """
        try:
            if not self.performance_metrics:
                self.logger.warning("Нет данных для визуализации")
                return
            
            # Создаем графики
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Результаты бэктестинга', fontsize=16)
            
            # Кривая доходности
            equity_df = pd.DataFrame(self.equity_curve)
            axes[0, 0].plot(equity_df['timestamp'], equity_df['equity'])
            axes[0, 0].set_title('Кривая доходности')
            axes[0, 0].set_ylabel('Капитал')
            axes[0, 0].grid(True)
            
            # Распределение доходности
            returns = pd.DataFrame(self.equity_curve)['equity'].pct_change().dropna()
            axes[0, 1].hist(returns, bins=50, alpha=0.7)
            axes[0, 1].set_title('Распределение доходности')
            axes[0, 1].set_xlabel('Доходность')
            axes[0, 1].set_ylabel('Частота')
            axes[0, 1].grid(True)
            
            # Просадка
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            axes[1, 0].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
            axes[1, 0].set_title('Просадка')
            axes[1, 0].set_ylabel('Просадка')
            axes[1, 0].grid(True)
            
            # Статистика сделок
            if self.trades:
                trade_pnls = [t.get('pnl', 0) for t in self.trades if 'pnl' in t]
                axes[1, 1].bar(range(len(trade_pnls)), trade_pnls, 
                              color=['green' if pnl > 0 else 'red' for pnl in trade_pnls])
                axes[1, 1].set_title('P&L по сделкам')
                axes[1, 1].set_xlabel('Номер сделки')
                axes[1, 1].set_ylabel('P&L')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Графики сохранены в {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Ошибка визуализации: {e}")
    
    def save_results(self, filename: str = None):
        """
        Сохранение результатов бэктестинга.
        
        Args:
            filename: Имя файла для сохранения
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"backtest_results_{timestamp}.json"
            
            # Убираем дублирование папки backtest_results в пути
            if filename.startswith('backtest_results/'):
                filepath = Path(filename)
            else:
                filepath = self.results_dir / filename
            
            # Подготавливаем данные для сохранения
            results_data = {
                'config': self.default_config,
                'performance_metrics': self.performance_metrics,
                'trades_summary': [
                    {
                        'timestamp': str(trade['timestamp']),
                        'action': trade['action'],
                        'price': trade['price'],
                        'pnl': trade.get('pnl', 0),
                        'confidence': trade.get('confidence', 0)
                    }
                    for trade in self.trades
                ]
            }
            
            # Функция для преобразования Timestamp объектов
            def convert_timestamps(obj):
                if isinstance(obj, dict):
                    return {key: convert_timestamps(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_timestamps(item) for item in obj]
                elif hasattr(obj, 'isoformat'):  # Timestamp объекты
                    return obj.isoformat()
                elif isinstance(obj, np.integer):  # numpy int64
                    return int(obj)
                elif isinstance(obj, np.floating):  # numpy float64
                    return float(obj)
                else:
                    return obj
            
            # Преобразуем все Timestamp объекты
            results_data = convert_timestamps(results_data)
            
            # Сохраняем в JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Результаты сохранены в {filepath}")
            
            # Выводим основные метрики
            self.print_summary()
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения результатов: {e}")
    
    def print_summary(self):
        """Вывод сводки результатов."""
        if not self.performance_metrics:
            return
        
        metrics = self.performance_metrics
        
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТЫ БЭКТЕСТИНГА")
        print("="*50)
        print(f"Начальный капитал: ${metrics['initial_capital']:,.2f}")
        print(f"Конечный капитал: ${metrics['final_capital']:,.2f}")
        print(f"Общая доходность: {metrics['total_return']:.2%}")
        print(f"Годовая доходность: {metrics['annual_return']:.2%}")
        print(f"Волатильность: {metrics['volatility']:.2%}")
        print(f"Коэффициент Шарпа: {metrics['sharpe_ratio']:.2f}")
        print(f"Максимальная просадка: {metrics['max_drawdown']:.2%}")
        print(f"Всего сделок: {metrics['total_trades']}")
        print(f"Прибыльных сделок: {metrics['winning_trades']}")
        print(f"Убыточных сделок: {metrics['losing_trades']}")
        print(f"Процент прибыльных: {metrics['win_rate']:.2%}")
        print(f"Средняя прибыль: ${metrics['avg_win']:.2f}")
        print(f"Средний убыток: ${metrics['avg_loss']:.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print("="*50)
    
    def run_backtest_with_model(self, data: pd.DataFrame, model, symbol: str) -> Dict:
        """
        Запуск бэктеста с моделью в памяти.
        
        Args:
            data: Данные для бэктеста
            model: Обученная модель
            symbol: Торговая пара
            
        Returns:
            Результаты бэктеста
        """
        try:
            self.logger.info(f"Запуск бэктеста для {symbol}")
            
            # Устанавливаем модель
            self.model = model
            
            # Проверяем, что модель обучена
            if hasattr(self.model, 'is_trained'):
                if not self.model.is_trained:
                    self.logger.error("Модель не обучена")
                    return {}
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'is_trained'):
                if not self.model.model.is_trained:
                    self.logger.error("Модель не обучена")
                    return {}
            
            self.logger.info("Модель проверена и готова к использованию")
            
            # Сбрасываем состояние
            self.reset_state()
            
            # Подготавливаем данные
            if 'target' in data.columns or 'target_dynamic' in data.columns:
                # Данные уже предобработаны - убираем целевые переменные
                processed_data = data.copy()
                target_columns = ['target', 'target_dynamic']
                for col in target_columns:
                    if col in processed_data.columns:
                        processed_data = processed_data.drop(columns=[col])
            else:
                # Нужно добавить технические индикаторы
                processed_data = self.prepare_features(data)
            
            if processed_data.empty:
                self.logger.error("Нет данных для бэктеста")
                return {}
            
            # Генерируем сигналы
            signals_data = self.generate_signals(processed_data)
            
            if signals_data.empty:
                self.logger.error("Не удалось сгенерировать сигналы")
                return {}
            
            # Запускаем симуляцию торговли
            self._simulate_trading(signals_data)
            
            # Рассчитываем метрики
            metrics = self.calculate_performance_metrics()
            
            self.logger.info("Бэктест завершен успешно")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка бэктеста: {e}")
            return {}
    
    def _simulate_trading(self, signals_data: pd.DataFrame):
        """
        Симуляция торговли с учётом предсказанного target_return (TP/SL) и горизонта удержания.
        """
        try:
            self.logger.info("Начинаем симуляцию торговли (TP/SL + max_horizon)")
            max_horizon = 3  # Горизонт удержания позиции (свечи)
            open_positions = []
            for idx, row in signals_data.iterrows():
                timestamp = row.get('timestamp', idx)
                price = row['close']
                signal = row['signal']
                confidence = row['confidence']
                predicted_return = row.get('predicted_return', 0)
                # Открытие позиции
                if signal != 0:
                    pos = {
                        'open_idx': idx,
                        'open_time': timestamp,
                        'open_price': price,
                        'direction': signal,
                        'confidence': confidence,
                        'predicted_return': predicted_return,
                        'horizon': max_horizon,
                        'bars_held': 0,
                        'tp': price * (1 + abs(predicted_return) if signal == 1 else 1 - abs(predicted_return)),
                        'sl': price * (1 - abs(predicted_return) if signal == 1 else 1 + abs(predicted_return)),
                    }
                    open_positions.append(pos)
                # Проверка открытых позиций
                closed_positions = []
                for pos in open_positions:
                    pos['bars_held'] += 1
                    # Проверка TP/SL
                    if pos['direction'] == 1:
                        if price >= pos['tp'] or price <= pos['sl'] or pos['bars_held'] >= pos['horizon']:
                            self.execute_trade(1, pos['open_price'], pos['open_time'], pos['confidence'])
                            self.execute_trade(-1, price, timestamp, pos['confidence'])
                            closed_positions.append(pos)
                    elif pos['direction'] == -1:
                        if price <= pos['tp'] or price >= pos['sl'] or pos['bars_held'] >= pos['horizon']:
                            self.execute_trade(-1, pos['open_price'], pos['open_time'], pos['confidence'])
                            self.execute_trade(1, price, timestamp, pos['confidence'])
                            closed_positions.append(pos)
                open_positions = [p for p in open_positions if p not in closed_positions]
                # Записываем состояние капитала
                current_equity = self.capital
                if self.positions:
                    # Добавляем стоимость открытых позиций
                    for pos in self.positions:
                        current_equity += pos['shares'] * price
                
                self.equity_curve.append({
                    'timestamp': timestamp,
                    'equity': current_equity,
                    'capital': self.capital,
                    'positions_value': current_equity - self.capital
                })
            
            # Закрываем все открытые позиции в конце
            if self.positions:
                last_price = signals_data.iloc[-1]['close']
                for position in self.positions:
                    trade = self.execute_trade(-1, last_price, signals_data.index[-1], 1.0)
            
            self.logger.info(f"Симуляция завершена. Всего сделок: {len(self.trades)}")
            
        except Exception as e:
            self.logger.error(f"Ошибка симуляции торговли: {e}")


def run_backtest_example():
    """Пример использования бэктестера."""
    # Конфигурация
    config = {
        'initial_capital': 10000,
        'commission': 0.001,
        'slippage': 0.0005,
        'position_size': 0.1,
        'min_confidence': 0.6
    }
    
    # Создаем бэктестер
    backtester = Backtester(config)
    
    # Путь к модели (замените на реальный путь)
    model_path = "models/xgboost_BTC_USDT.pkl"
    
    # Загружаем данные (замените на реальные данные)
    # data = pd.read_csv("data/btc_data.csv")
    
    # Запускаем бэктестинг
    # results = backtester.run_backtest(data, model_path)
    
    # Визуализируем результаты
    # backtester.plot_results()
    
    # Сохраняем результаты
    # backtester.save_results()
    
    print("Бэктестер готов к использованию!")


if __name__ == "__main__":
    run_backtest_example() 
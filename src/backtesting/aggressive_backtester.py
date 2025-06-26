"""
Агрессивный бэктестер с ослабленными фильтрами для максимальной прибыли.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import indicators
from src.backtesting.adaptive_filter_strategy import AdaptiveFilterStrategy
import warnings
warnings.filterwarnings('ignore')


class AggressiveBacktester:
    """
    Агрессивный бэктестер с ослабленными фильтрами для максимальной прибыли.
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация агрессивного бэктестера.
        
        Args:
            config: Конфигурация бэктестера
        """
        self.config = config or {}
        self.logger = logger.bind(name="AggressiveBacktester")
        
        # Базовые настройки
        self.default_config = {
            'initial_capital': 10000,  # Начальный капитал $10,000
            'commission': 0.001,       # Комиссия 0.1%
            'slippage': 0.0005,        # Проскальзывание 0.05%
            'max_positions': 5,        # Максимум 5 позиций
            'position_size': 0.15,     # 15% от капитала на позицию
            'min_gap_between_signals': 2,  # Минимум 2 свечи между сигналами
            'max_capital': 1000000,    # Максимальный капитал $1M (защита от экспоненциального роста)
        }
        
        # Состояние бэктестера
        self.capital = self.default_config['initial_capital']
        self.initial_capital = self.default_config['initial_capital']
        self.max_capital = self.default_config['max_capital']
        self.position_count = 0
        self.max_positions = self.default_config['max_positions']
        self.position_size = self.default_config['position_size']
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        self.positions = []  # Список открытых позиций
        
        # Адаптивная стратегия
        self.adaptive_strategy = AdaptiveFilterStrategy(self.config)
        
        # Модель
        self.model = None
        
        # Директории
        self.results_dir = Path("backtest_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def reset_state(self):
        """Сброс состояния бэктестера."""
        self.capital = self.default_config['initial_capital']
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        self.position_count = 0
        
        # АГРЕССИВНЫЕ настройки
        self.max_positions = self.default_config['max_positions']
        self.position_size = self.default_config['position_size']
        self.min_confidence = self.default_config['min_confidence']
    
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
                
            # Проверяем новый формат AdvancedEnsembleModel
            if isinstance(model_data, dict) and 'models' in model_data:
                from src.ml_models.advanced_xgboost_model import AdvancedEnsembleModel
                self.model = AdvancedEnsembleModel()
                
                # Восстанавливаем состояние модели
                self.model.models = model_data.get('models', {})
                self.model.training_results = model_data.get('training_results', {})
                self.model.config = model_data.get('config', {})
                self.model.is_trained = model_data.get('is_trained', False)
                
                self.logger.info(f"Модель AdvancedEnsembleModel загружена: {model_path}")
                return True
                
            # Проверяем старый формат ImprovedXGBoostModel
            elif isinstance(model_data, dict) and 'class_model' in model_data:
                # Создаем новый экземпляр ImprovedXGBoostModel
                from src.ml_models.advanced_xgboost_model import ImprovedXGBoostModel
                self.model = ImprovedXGBoostModel()
                
                # Восстанавливаем состояние модели
                self.model.class_model = model_data.get('class_model')
                self.model.reg_model = model_data.get('reg_model')
                self.model.ensemble_class_models = model_data.get('ensemble_class_models', [])
                self.model.ensemble_reg_models = model_data.get('ensemble_reg_models', [])
                self.model.selected_features = model_data.get('selected_features')
                self.model.training_results = model_data.get('training_results', {})
                self.model.config = model_data.get('config', {})
                self.model.is_trained = model_data.get('is_trained', False)
                
                self.logger.info(f"Модель ImprovedXGBoostModel загружена (save_model формат): {model_path}")
                return True
                
            # Проверяем, что это объект ImprovedXGBoostModel (save формат)
            elif hasattr(model_data, 'class_model') and hasattr(model_data, 'predict_improved_dual'):
                self.model = model_data
                # Восстановить логгер после десериализации
                from loguru import logger as global_logger
                self.model.logger = global_logger.bind(name="ImprovedXGBoostModel")
                self.logger.info(f"Модель загружена (save формат): {model_path}")
                return True
                
            else:
                self.logger.error("Неверный формат файла модели")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка признаков для модели.
        
        Args:
            data: Исходные данные
            
        Returns:
            DataFrame с признаками
        """
        try:
            df = data.copy()
            
            # Если модель уже загружена, используем только те признаки, которые были при обучении
            if self.model and hasattr(self.model, 'training_results') and self.model.training_results:
                # Получаем список признаков из модели
                if 'feature_importance' in self.model.training_results:
                    expected_features = list(self.model.training_results['feature_importance'].keys())
                    self.logger.info(f"Используем признаки из обученной модели: {len(expected_features)} признаков")
                    
                    # Проверяем, какие признаки есть в данных
                    available_features = [col for col in expected_features if col in df.columns]
                    missing_features = [col for col in expected_features if col not in df.columns]
                    
                    if missing_features:
                        self.logger.warning(f"Отсутствуют признаки: {missing_features}")
                        # Добавляем недостающие признаки как нули
                        for feature in missing_features:
                            df[feature] = 0
                    
                    # Выбираем только нужные признаки в правильном порядке
                    result_df = df[expected_features].copy()
                    
                    # Заполняем оставшиеся NaN значения
                    result_df = result_df.fillna(0)
                    
                    return result_df
            
            # Если модель не загружена или нет информации о признаках, используем базовые
            self.logger.info("Используем базовые признаки")
            
            # Базовые признаки
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Волатильность
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            # Создаем SMA если их нет
            if 'sma_20' not in df.columns:
                df['sma_20'] = df['close'].rolling(window=20).mean()
            if 'sma_50' not in df.columns:
                df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Тренд
            df['trend'] = np.where(df['close'] > df['sma_50'], 1, -1)
            
            # Дополнительные признаки, которые могут отсутствовать
            # Price to SMA ratios
            if 'price_to_sma_20' not in df.columns:
                df['price_to_sma_20'] = df['close'] / df['sma_20']
            if 'price_to_sma_50' not in df.columns:
                df['price_to_sma_50'] = df['close'] / df['sma_50']
            
            # Volume indicators
            if 'relative_volume' not in df.columns:
                df['relative_volume'] = df['volume'] / df['volume'].rolling(window=20).mean()
            if 'volume_price_sma' not in df.columns:
                df['volume_price_sma'] = df['volume'] * df['close']
            
            # Momentum indicators
            if 'momentum_5' not in df.columns:
                df['momentum_5'] = df['close'].pct_change(5)
            if 'momentum_10' not in df.columns:
                df['momentum_10'] = df['close'].pct_change(10)
            
            # Volatility indicators
            if 'volatility_10' not in df.columns:
                df['volatility_10'] = df['price_change'].rolling(window=10).std()
            if 'volatility_50' not in df.columns:
                df['volatility_50'] = df['price_change'].rolling(window=50).std()
            
            # ATR
            if 'atr_50' not in df.columns:
                try:
                    df['atr_50'] = indicators.calculate_atr(df, 50)
                except:
                    df['atr_50'] = 0
            
            # Заполняем NaN значения
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка подготовки признаков: {e}")
            return data
    
    def _align_features(self, df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
        """
        Выравнивание признаков с ожидаемыми.
        
        Args:
            df: DataFrame с признаками
            expected_features: Список ожидаемых признаков
            
        Returns:
            DataFrame с выровненными признаками
        """
        try:
            # Проверяем, какие признаки отсутствуют
            missing_features = set(expected_features) - set(df.columns)
            if missing_features:
                self.logger.warning(f"Отсутствуют признаки: {missing_features}")
                
                # Создаем недостающие признаки с нулевыми значениями
                for feature in missing_features:
                    df[feature] = 0
            
            # Выбираем только нужные признаки в правильном порядке
            result_df = df[expected_features].copy()
            
            # Заполняем оставшиеся NaN значения
            result_df = result_df.fillna(0)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Ошибка выравнивания признаков: {e}")
            return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация торговых сигналов с АДАПТИВНЫМИ фильтрами на основе рыночных условий.
        
        Args:
            data: DataFrame с данными
            
        Returns:
            DataFrame с сигналами
        """
        try:
            if self.model is None:
                self.logger.error("Модель не загружена")
                return pd.DataFrame()
            
            # Получаем адаптивные фильтры на основе текущих рыночных условий
            adaptive_filters = self.adaptive_strategy.calculate_adaptive_filters(data)
            
            # Получаем рекомендации по стратегии
            recommendations = self.adaptive_strategy.get_strategy_recommendations(data)
            market_regime = self.adaptive_strategy.get_market_regime(data)
            
            self.logger.info(f"🎯 Режим рынка: {market_regime}")
            self.logger.info(f"📊 Рекомендации: {recommendations}")
            
            # Подготавливаем признаки
            result_df = self.prepare_features(data)
            
            # Выравниваем признаки с моделью
            if self.model.selected_features:
                result_df = self._align_features(result_df, self.model.selected_features)
            
            # Получаем предсказания и вероятности
            try:
                if hasattr(self.model, 'predict_improved_dual'):
                    self.logger.info("Используем predict_improved_dual")
                    class_pred, class_proba, reg_pred = self.model.predict_improved_dual(result_df)
                elif hasattr(self.model, 'predict_dual'):
                    self.logger.info("Используем predict_dual")
                    class_pred, class_proba, reg_pred = self.model.predict_dual(result_df)
                elif hasattr(self.model, 'predict') and hasattr(self.model, 'predict_proba'):
                    self.logger.info("Используем predict и predict_proba (AdvancedEnsembleModel)")
                    class_pred = self.model.predict(result_df)
                    class_proba = self.model.predict_proba(result_df)
                    reg_pred = np.zeros(len(class_pred))
                elif hasattr(self.model, 'predict'):
                    self.logger.info("Используем только predict")
                    class_pred = self.model.predict(result_df)
                    # Создаем фиктивные вероятности для совместимости
                    class_proba = np.zeros((len(class_pred), 3))
                    for i, pred in enumerate(class_pred):
                        class_proba[i, int(pred)] = 1.0
                    reg_pred = np.zeros(len(class_pred))
                else:
                    self.logger.error("Модель не поддерживает предсказания")
                    return pd.DataFrame()
                    
            except Exception as e:
                self.logger.error(f"Ошибка получения предсказаний: {e}")
                return pd.DataFrame()
            
            # Создаем DataFrame с сигналами
            signals_df = data.copy()
            
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
            
            # АДАПТИВНЫЕ условия для сигналов на основе рыночных условий
            adaptive_min_confidence = adaptive_filters['min_confidence']
            
            buy_condition = (
                (signals_df['prediction'] == 2) &  # Long signal
                (signals_df['confidence'] > adaptive_min_confidence)  # Адаптивный порог
            )
            
            sell_condition = (
                (signals_df['prediction'] == 0) &  # Short signal
                (signals_df['confidence'] > adaptive_min_confidence)  # Адаптивный порог
            )
            
            self.logger.info(f"Адаптивный порог уверенности: {adaptive_min_confidence:.3f}")
            self.logger.info(f"Сигналов по адаптивной уверенности (buy): {buy_condition.sum()}, (sell): {sell_condition.sum()}")
            
            signals_df.loc[buy_condition, 'signal'] = 1
            signals_df.loc[sell_condition, 'signal'] = -1
            
            # АДАПТИВНОЕ ограничение частоты сигналов
            adaptive_min_gap = adaptive_filters['min_gap_between_signals']
            signals_df = self._limit_signal_frequency_adaptive(signals_df, adaptive_min_gap)
            
            # Статистика сигналов
            buy_signals = len(signals_df[signals_df['signal'] == 1])
            sell_signals = len(signals_df[signals_df['signal'] == -1])
            total_signals = buy_signals + sell_signals
            
            self.logger.info(f"🎯 Сгенерировано адаптивных сигналов: {total_signals} (buy: {buy_signals}, sell: {sell_signals})")
            self.logger.info(f"📈 Адаптивные параметры: min_gap={adaptive_min_gap}, max_positions={adaptive_filters['max_positions']}, position_size={adaptive_filters['position_size']:.3f}")
            
            return signals_df
            
        except Exception as e:
            self.logger.error(f"Ошибка при предсказании: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def _limit_signal_frequency_aggressive(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        АГРЕССИВНОЕ ограничение частоты сигналов (ослаблено).
        
        Args:
            df: DataFrame с сигналами
            
        Returns:
            DataFrame с ограниченными сигналами
        """
        try:
            # Инициализируем столбец signal если его нет
            if 'signal' not in df.columns:
                df['signal'] = 0
            
            # Находим индексы сигналов
            buy_signals = df[df['signal'] == 1].index
            sell_signals = df[df['signal'] == -1].index
            
            # АГРЕССИВНЫЕ настройки - разрешаем больше сигналов
            min_gap = 3  # Снижено с 10 до 3
            
            # Фильтруем buy сигналы
            filtered_buy = []
            last_buy = -min_gap
            
            for idx in buy_signals:
                if idx - last_buy >= min_gap:
                    filtered_buy.append(idx)
                    last_buy = idx
            
            # Фильтруем sell сигналы
            filtered_sell = []
            last_sell = -min_gap
            
            for idx in sell_signals:
                if idx - last_sell >= min_gap:
                    filtered_sell.append(idx)
                    last_sell = idx
            
            # Сбрасываем все сигналы
            df['signal'] = 0
            
            # Устанавливаем отфильтрованные сигналы
            df.loc[filtered_buy, 'signal'] = 1
            df.loc[filtered_sell, 'signal'] = -1
            
            self.logger.info(f"После фильтрации: buy: {len(filtered_buy)}, sell: {len(filtered_sell)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка ограничения частоты сигналов: {e}")
            return df
    
    def _limit_signal_frequency_adaptive(self, df: pd.DataFrame, min_gap: int) -> pd.DataFrame:
        """
        АДАПТИВНОЕ ограничение частоты сигналов на основе рыночных условий.
        
        Args:
            df: DataFrame с сигналами
            min_gap: Минимальный промежуток между сигналами
            
        Returns:
            DataFrame с ограниченными сигналами
        """
        try:
            # Инициализируем столбец signal если его нет
            if 'signal' not in df.columns:
                df['signal'] = 0
            
            # Сбрасываем индекс для работы с числовыми индексами
            df_reset = df.reset_index(drop=True)
            
            # Находим индексы сигналов
            buy_signals = df_reset[df_reset['signal'] == 1].index.tolist()
            sell_signals = df_reset[df_reset['signal'] == -1].index.tolist()
            
            # Уменьшаем минимальный промежуток для более частых сделок
            actual_min_gap = max(1, min_gap // 2)  # Уменьшаем в 2 раза, но минимум 1
            self.logger.info(f"Адаптивный минимальный промежуток: {actual_min_gap} свечей (было {min_gap})")
            
            # Фильтруем buy сигналы
            filtered_buy = []
            last_buy = -actual_min_gap
            
            for idx in buy_signals:
                if idx - last_buy >= actual_min_gap:
                    filtered_buy.append(idx)
                    last_buy = idx
            
            # Фильтруем sell сигналы
            filtered_sell = []
            last_sell = -actual_min_gap
            
            for idx in sell_signals:
                if idx - last_sell >= actual_min_gap:
                    filtered_sell.append(idx)
                    last_sell = idx
            
            # Сбрасываем все сигналы
            df_reset['signal'] = 0
            
            # Устанавливаем отфильтрованные сигналы
            df_reset.loc[filtered_buy, 'signal'] = 1
            df_reset.loc[filtered_sell, 'signal'] = -1
            
            # Восстанавливаем оригинальный индекс
            df['signal'] = df_reset['signal'].values
            
            self.logger.info(f"После адаптивной фильтрации: buy: {len(filtered_buy)}, sell: {len(filtered_sell)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка адаптивного ограничения частоты сигналов: {e}")
            return df
    
    def execute_trade(self, signal: int, price: float, timestamp: datetime, 
                     confidence: float, prediction_horizon: int = 12) -> Dict:
        """
        Выполнение торговой операции с АДАПТИВНЫМИ настройками и логированием причин отказа.
        Args:
            signal: Сигнал (1 - buy, -1 - sell)
            price: Цена исполнения
            timestamp: Время исполнения
            confidence: Уверенность в сигнале
            prediction_horizon: Временной горизонт для автозакрытия (по умолчанию 12)
        Returns:
            Словарь с информацией о сделке
        """
        try:
            commission = self.default_config['commission']
            slippage = self.default_config['slippage']
            
            # Более агрессивные стоп-лосс и тейк-профит
            stop_loss = price * 0.98  # 2% стоп-лосс
            take_profit = price * 1.05  # 5% тейк-профит
            
            base_position_size = self.position_size
            # Более агрессивный множитель уверенности
            confidence_multiplier = 1.0 + (confidence - 0.4) * 2.5  # 0.5-2.0x
            position_size = base_position_size * confidence_multiplier
            position_size = min(position_size, 0.3)  # Максимум 30% капитала на позицию
            
            if signal == 1:  # Buy
                # Проверяем, можем ли открыть новую позицию
                if self.position_count < self.max_positions:
                    # ИСПРАВЛЕНИЕ: Ограничиваем размер позиции абсолютным значением
                    max_position_value = 10000  # Максимум $10,000 на позицию
                    position_value = min(self.capital * position_size, max_position_value)
                    
                    shares = position_value / price
                    execution_price = price * (1 + slippage)
                    actual_shares = shares * (1 - commission)
                    
                    position = {
                        'action': 'BUY',
                        'type': 'long',
                        'entry_price': price,
                        'price': price,
                        'shares': shares,
                        'timestamp': timestamp,
                        'confidence': confidence,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'bars_held': 0,  # НОВОЕ: счетчик баров
                        'max_horizon': prediction_horizon  # НОВОЕ: максимальный горизонт
                    }
                    
                    # Добавляем позицию в список открытых позиций
                    if self.positions is None:
                        self.positions = []
                    self.positions.append(position)
                    
                    self.position_count += 1
                    
                    trade_info = {
                        'action': 'BUY',
                        'price': execution_price,
                        'shares': actual_shares,
                        'timestamp': timestamp,
                        'confidence': confidence,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'max_horizon': prediction_horizon
                    }
                    self.trades.append(trade_info)
                    self.logger.info(f"Открыта длинная позиция: {trade_info}")
                    return trade_info
                else:
                    reason = "Достигнут лимит позиций"
                    self.logger.info(f"Отказ от открытия позиции: {reason}")
                    return {'reason': reason}
                    
            elif signal == -1:  # Sell
                # Закрываем все открытые длинные позиции
                if self.positions and len(self.positions) > 0:
                    total_pnl = 0
                    closed_positions = []
                    
                    for position in self.positions:
                        if position['type'] == 'long':
                            exit_value = position['shares'] * price * (1 - commission)
                            entry_value = position['shares'] * position['entry_price'] * (1 + commission)
                            pnl = exit_value - entry_value
                            total_pnl += pnl
                            closed_positions.append(position)
                    
                    # Удаляем закрытые позиции
                    for pos in closed_positions:
                        self.positions.remove(pos)
                        self.position_count -= 1
                    
                    self.capital += total_pnl
                    
                    # Ограничиваем максимальный капитал
                    if self.capital > self.max_capital:
                        self.capital = self.max_capital
                        self.logger.warning(f"Капитал ограничен до максимума: ${self.max_capital:,.2f}")
                    
                    trade_info = {
                        'action': 'CLOSE_LONG',
                        'price': price,
                        'pnl': total_pnl,
                        'timestamp': timestamp,
                        'confidence': confidence,
                        'positions_closed': len(closed_positions),
                        'bars_held': closed_positions[0].get('bars_held', 0) if closed_positions else 0
                    }
                    self.trades.append(trade_info)
                    self.logger.info(f"Закрыто {len(closed_positions)} длинных позиций: {trade_info}")
                    return trade_info
                else:
                    reason = "Нет открытых длинных позиций для закрытия"
                    self.logger.info(f"Отказ от закрытия позиции: {reason}")
                    return {'reason': reason}
            else:
                reason = "Неизвестный сигнал"
                self.logger.info(f"Отказ от сделки: {reason}")
                return {'reason': reason}
                
        except Exception as e:
            self.logger.error(f"Ошибка выполнения сделки: {e}")
            return {}
    
    def run_backtest(self, data: pd.DataFrame, model_path: str = None) -> Dict:
        """
        Запуск агрессивного бэктеста.
        
        Args:
            data: DataFrame с данными
            model_path: Путь к модели
            
        Returns:
            Словарь с результатами
        """
        try:
            self.logger.info("Запуск агрессивного бэктеста...")
            
            # Загружаем модель
            if model_path and not self.load_model(model_path):
                return {}
            
            # Генерируем сигналы
            signals_data = self.generate_signals(data)
            
            if signals_data.empty:
                self.logger.error("Не удалось сгенерировать сигналы")
                return {}
            
            # Симулируем торговлю
            self._simulate_trading_aggressive(signals_data)
            
            # Рассчитываем метрики
            metrics = self.calculate_performance_metrics()
            
            # Сохраняем результаты
            self.save_results()
            
            self.logger.info("Агрессивный бэктест завершен")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка бэктеста: {e}")
            return {}
    
    def _simulate_trading_aggressive(self, signals_data: pd.DataFrame):
        """
        АДАПТИВНАЯ симуляция торговли с динамическими параметрами.
        
        Args:
            signals_data: DataFrame с сигналами
        """
        try:
            self.equity_curve = []
            initial_capital = self.capital
            
            # Получаем адаптивные фильтры для текущих рыночных условий
            adaptive_filters = self.adaptive_strategy.calculate_adaptive_filters(signals_data)
            
            # Обновляем параметры бэктестера на основе адаптивных фильтров
            self.max_positions = adaptive_filters['max_positions']
            self.position_size = adaptive_filters['position_size']
            
            # Счетчики для ограничения логирования
            log_counter = 0
            max_logs_per_1000 = 50  # Максимум 50 логов на каждые 1000 сигналов
            
            for i, (idx, row) in enumerate(signals_data.iterrows()):
                # Используем индекс как timestamp, если столбец timestamp отсутствует
                timestamp = row.get('timestamp', idx)
                
                # Обновляем equity curve
                current_equity = self.capital
                
                # Если есть открытая позиция, рассчитываем unrealized PnL
                if self.current_position and 'type' in self.current_position:
                    if self.current_position['type'] == 'long':
                        unrealized_pnl = (row['close'] - self.current_position['entry_price']) * self.current_position['shares']
                    else:  # short
                        unrealized_pnl = (self.current_position['entry_price'] - row['close']) * self.current_position['shares']
                    current_equity += unrealized_pnl
                
                self.equity_curve.append({
                    'timestamp': timestamp,
                    'equity': current_equity,
                    'price': row['close']
                })
                
                # Ограничиваем логирование - выводим только каждые N сообщений
                should_log = (log_counter % max(1, len(signals_data) // max_logs_per_1000)) == 0
                
                # НОВОЕ: Проверяем автозакрытие по временному горизонту для всех позиций
                positions_to_close = []
                for pos in self.positions:
                    # Увеличиваем счетчик баров для каждой позиции
                    if 'bars_held' not in pos:
                        pos['bars_held'] = 0
                    pos['bars_held'] += 1
                    
                    # Получаем максимальный горизонт для этой позиции
                    max_horizon = pos.get('max_horizon', 12)  # По умолчанию 12 свечей
                    
                    # Проверяем, не превышен ли временной горизонт
                    if pos['bars_held'] >= max_horizon:
                        positions_to_close.append((pos, 'horizon'))
                        if should_log:
                            self.logger.info(f"Автозакрытие по горизонту: позиция открыта {pos['bars_held']} баров, горизонт {max_horizon}")
                
                # Закрываем позиции по временному горизонту
                for pos, reason in positions_to_close:
                    pnl = (row['close'] - pos['entry_price']) * pos['shares']
                    self.capital += pos['shares'] * row['close']
                    
                    close_info = {
                        'action': 'CLOSE_HORIZON',
                        'price': row['close'],
                        'pnl': pnl,
                        'timestamp': timestamp,
                        'bars_held': pos['bars_held'],
                        'max_horizon': pos.get('max_horizon', 12),
                        'reason': reason
                    }
                    self.trades.append(close_info)
                    
                    if should_log:
                        self.logger.info(f"Закрыта позиция по горизонту: {close_info}")
                    
                    # Удаляем позицию из списка
                    self.positions.remove(pos)
                    self.position_count -= 1
                
                # Обрабатываем сигналы
                if row['signal'] == 1:  # Buy signal
                    if len(self.positions) < self.max_positions:
                        # Открываем длинную позицию
                        shares = (self.capital * self.position_size) / row['close']
                        stop_loss = row['close'] * 0.985  # 1.5% stop loss
                        take_profit = row['close'] * 1.03  # 3% take profit
                        
                        # Получаем горизонт предсказания из данных
                        prediction_horizon = row.get('prediction_horizon', 12)  # По умолчанию 12
                        
                        position = {
                            'action': 'BUY',
                            'type': 'long',
                            'entry_price': row['close'],
                            'price': row['close'],
                            'shares': shares,
                            'timestamp': timestamp,
                            'confidence': row.get('confidence', 0.5),
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'bars_held': 0,  # НОВОЕ: счетчик баров
                            'max_horizon': prediction_horizon  # НОВОЕ: максимальный горизонт
                        }
                        
                        self.positions.append(position)
                        self.current_position = position
                        self.capital -= shares * row['close']
                        
                        # Добавляем сделку в список
                        trade_info = {
                            'action': 'BUY',
                            'price': row['close'],
                            'shares': shares,
                            'timestamp': timestamp,
                            'confidence': row.get('confidence', 0.5),
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'max_horizon': prediction_horizon
                        }
                        self.trades.append(trade_info)
                        
                        if should_log:
                            self.logger.info(f"Открыта длинная позиция: {position}")
                    
                    elif should_log:
                        self.logger.info("Отказ от открытия позиции: Достигнут лимит позиций")
                
                elif row['signal'] == -1:  # Sell signal
                    if self.current_position and 'type' in self.current_position and self.current_position['type'] == 'long':
                        # Закрываем длинную позицию
                        pnl = (row['close'] - self.current_position['entry_price']) * self.current_position['shares']
                        self.capital += self.current_position['shares'] * row['close']
                        
                        close_info = {
                            'action': 'CLOSE_LONG',
                            'price': row['close'],
                            'pnl': pnl,
                            'timestamp': timestamp,
                            'confidence': row.get('confidence', 0.5),
                            'bars_held': self.current_position.get('bars_held', 0)
                        }
                        
                        # Добавляем сделку в список
                        self.trades.append(close_info)
                        
                        if should_log:
                            self.logger.info(f"Закрыта длинная позиция: {close_info}")
                        
                        # Удаляем позицию из списка
                        if self.current_position in self.positions:
                            self.positions.remove(self.current_position)
                            self.position_count -= 1
                        
                        self.current_position = None
                    
                    elif should_log:
                        self.logger.info("Отказ от закрытия позиции: Нет открытой длинной позиции для закрытия")
                
                # Проверяем stop loss и take profit для всех позиций
                positions_to_close_sl_tp = []
                for pos in self.positions:
                    if row['close'] <= pos['stop_loss'] or row['close'] >= pos['take_profit']:
                        positions_to_close_sl_tp.append(pos)
                
                for pos in positions_to_close_sl_tp:
                    # Закрываем позицию по stop loss или take profit
                    pnl = (row['close'] - pos['entry_price']) * pos['shares']
                    self.capital += pos['shares'] * row['close']
                    
                    # Добавляем сделку в список
                    sl_tp_info = {
                        'action': 'CLOSE_SL_TP',
                        'price': row['close'],
                        'pnl': pnl,
                        'timestamp': timestamp,
                        'bars_held': pos.get('bars_held', 0),
                        'reason': 'stop_loss' if row['close'] <= pos['stop_loss'] else 'take_profit'
                    }
                    self.trades.append(sl_tp_info)
                    
                    if should_log:
                        self.logger.info(f"Закрыта позиция по SL/TP: pnl={pnl:.2f}")
                    
                    # Удаляем позицию из списка
                    self.positions.remove(pos)
                    self.position_count -= 1
                    
                    # Если это была текущая позиция, сбрасываем её
                    if self.current_position == pos:
                        self.current_position = None
                
                # Обновляем адаптивные фильтры каждые 1000 сигналов
                if i % 1000 == 0 and i > 0:
                    adaptive_filters = self.adaptive_strategy.calculate_adaptive_filters(signals_data.iloc[max(0, i-1000):i+1])
                    self.max_positions = adaptive_filters['max_positions']
                    self.position_size = adaptive_filters['position_size']
                
                log_counter += 1
                
                # Показываем прогресс каждые 5000 сигналов
                if i % 5000 == 0 and i > 0:
                    self.logger.info(f"Обработано {i}/{len(signals_data)} сигналов ({i/len(signals_data)*100:.1f}%)")
            
            # Закрываем все открытые позиции в конце
            if self.positions:
                final_price = signals_data.iloc[-1]['close']
                for position in self.positions:
                    pnl = (final_price - position['entry_price']) * position['shares']
                    self.capital += position['shares'] * final_price
                    
                    final_close_info = {
                        'action': 'CLOSE_FINAL',
                        'price': final_price,
                        'pnl': pnl,
                        'timestamp': signals_data.index[-1],
                        'bars_held': position.get('bars_held', 0),
                        'reason': 'end_of_backtest'
                    }
                    self.trades.append(final_close_info)
                    
                self.logger.info(f"Закрыто {len(self.positions)} финальных позиций")
                self.positions = []
                self.position_count = 0
                self.current_position = None
            
            self.logger.info(f"Симуляция торговли завершена. Итоговый капитал: {self.capital:.2f}")
            
        except Exception as e:
            self.logger.error(f"Ошибка в симуляции торговли: {e}")
            raise
    
    def calculate_performance_metrics(self):
        """
        Расчет метрик производительности.
        
        Returns:
            Словарь с метриками
        """
        try:
            if not self.equity_curve:
                return {}
            
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['returns'] = equity_df['equity'].pct_change()
            
            # Базовые метрики
            total_return = (self.capital - self.default_config['initial_capital']) / self.default_config['initial_capital']
            
            # Подсчитываем сделки с PnL (исключаем открытие позиций)
            trades_with_pnl = [t for t in self.trades if 'pnl' in t and t['action'] in ['CLOSE_LONG', 'CLOSE_SL_TP', 'CLOSE_HORIZON', 'CLOSE_FINAL']]
            total_trades = len(trades_with_pnl)
            
            if total_trades > 0:
                winning_trades = len([t for t in trades_with_pnl if t['pnl'] > 0])
                win_rate = winning_trades / total_trades
                
                pnls = [t['pnl'] for t in trades_with_pnl]
                avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
                avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
                
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                
                # Статистика по типам закрытия
                horizon_closes = [t for t in trades_with_pnl if t['action'] == 'CLOSE_HORIZON']
                sl_tp_closes = [t for t in trades_with_pnl if t['action'] == 'CLOSE_SL_TP']
                signal_closes = [t for t in trades_with_pnl if t['action'] == 'CLOSE_LONG']
                final_closes = [t for t in trades_with_pnl if t['action'] == 'CLOSE_FINAL']
                
                horizon_count = len(horizon_closes)
                sl_tp_count = len(sl_tp_closes)
                signal_count = len(signal_closes)
                final_count = len(final_closes)
                
                # Среднее время удержания позиций
                avg_bars_held = np.mean([t.get('bars_held', 0) for t in trades_with_pnl if 'bars_held' in t])
                
            else:
                win_rate = avg_win = avg_loss = profit_factor = 0
                horizon_count = sl_tp_count = signal_count = final_count = 0
                avg_bars_held = 0
            
            # Волатильность и Sharpe ratio
            if len(equity_df) > 1:
                volatility = equity_df['returns'].std() * np.sqrt(252 * 288)  # Годовая волатильность
                sharpe_ratio = (total_return / volatility) if volatility > 0 else 0
            else:
                volatility = sharpe_ratio = 0
            
            # Максимальная просадка
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
            max_drawdown = equity_df['drawdown'].min()
            
            # Общее количество открытых позиций
            total_positions_opened = len([t for t in self.trades if t['action'] == 'BUY'])
            
            metrics = {
                'total_return': total_return,
                'total_trades': total_trades,
                'total_positions_opened': total_positions_opened,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_capital': self.capital,
                'initial_capital': self.default_config['initial_capital'],
                'avg_bars_held': avg_bars_held,
                'horizon_closes': horizon_count,
                'sl_tp_closes': sl_tp_count,
                'signal_closes': signal_count,
                'final_closes': final_count
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета метрик: {e}")
            return {}
    
    def save_results(self, filename: str = None):
        """
        Сохранение результатов бэктеста.
        
        Args:
            filename: Имя файла для сохранения
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"aggressive_backtest_results_{timestamp}.json"
            
            filepath = self.results_dir / filename
            
            # Подготавливаем данные для сохранения
            results = {
                'config': self.default_config,
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'metrics': self.calculate_performance_metrics(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Сохраняем в JSON
            import json
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Результаты сохранены: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения результатов: {e}")
    
    def create_backtest_plots(self, save_path: str = None):
        """
        Создание графиков результатов бэктеста.
        
        Args:
            save_path: Путь для сохранения графиков
        """
        try:
            if not self.equity_curve:
                self.logger.warning("Нет данных для создания графиков")
                return
            
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Настройка стиля
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Создаем фигуру с подграфиками
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Результаты Агрессивного Бэктеста', fontsize=16, fontweight='bold')
            
            # Подготавливаем данные
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            equity_df = equity_df.set_index('timestamp')
            
            # 1. Кривая доходности
            axes[0, 0].plot(equity_df.index, equity_df['equity'], linewidth=2, color='blue')
            axes[0, 0].set_title('Кривая Доходности', fontweight='bold')
            axes[0, 0].set_ylabel('Капитал ($)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Добавляем начальный капитал
            initial_capital = self.default_config['initial_capital']
            axes[0, 0].axhline(y=initial_capital, color='red', linestyle='--', alpha=0.7, label=f'Начальный капитал: ${initial_capital:,.0f}')
            axes[0, 0].legend()
            
            # 2. Цена актива
            axes[0, 1].plot(equity_df.index, equity_df['price'], linewidth=1, color='green', alpha=0.8)
            axes[0, 1].set_title('Цена Актива', fontweight='bold')
            axes[0, 1].set_ylabel('Цена ($)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Просадка
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
            
            axes[1, 0].fill_between(equity_df.index, equity_df['drawdown'], 0, alpha=0.3, color='red')
            axes[1, 0].plot(equity_df.index, equity_df['drawdown'], color='red', linewidth=1)
            axes[1, 0].set_title('Просадка', fontweight='bold')
            axes[1, 0].set_ylabel('Просадка (%)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Статистика сделок
            if self.trades:
                # Подготавливаем данные о сделках
                trades_df = pd.DataFrame(self.trades)
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                
                # Считаем PnL по сделкам
                pnl_trades = [t for t in self.trades if 'pnl' in t]
                if pnl_trades:
                    pnls = [t['pnl'] for t in pnl_trades]
                    winning_trades = [p for p in pnls if p > 0]
                    losing_trades = [p for p in pnls if p < 0]
                    
                    # Гистограмма PnL
                    axes[1, 1].hist(pnls, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                    axes[1, 1].set_title('Распределение PnL по Сделкам', fontweight='bold')
                    axes[1, 1].set_xlabel('PnL ($)')
                    axes[1, 1].set_ylabel('Количество сделок')
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # Добавляем статистику
                    win_rate = len(winning_trades) / len(pnls) * 100 if pnls else 0
                    avg_win = np.mean(winning_trades) if winning_trades else 0
                    avg_loss = np.mean(losing_trades) if losing_trades else 0
                    
                    stats_text = f'Всего сделок: {len(pnls)}\nВинрейт: {win_rate:.1f}%\nСр. прибыль: ${avg_win:.2f}\nСр. убыток: ${avg_loss:.2f}'
                    axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes, 
                                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                else:
                    axes[1, 1].text(0.5, 0.5, 'Нет завершенных сделок', ha='center', va='center', 
                                  transform=axes[1, 1].transAxes, fontsize=12)
                    axes[1, 1].set_title('Статистика Сделок', fontweight='bold')
            else:
                axes[1, 1].text(0.5, 0.5, 'Нет сделок', ha='center', va='center', 
                              transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].set_title('Статистика Сделок', fontweight='bold')
            
            # Настройка макета
            plt.tight_layout()
            
            # Сохранение или показ
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Графики сохранены: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Ошибка создания графиков: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def print_summary(self):
        """
        Вывод сводки результатов бэктеста.
        """
        try:
            metrics = self.calculate_performance_metrics()
            
            if not metrics:
                self.logger.warning("Нет метрик для отображения")
                return
            
            print("\n" + "="*80)
            print("РЕЗУЛЬТАТЫ АГРЕССИВНОГО БЭКТЕСТА")
            print("="*80)
            
            # Основные метрики
            print(f"📊 ОБЩАЯ СТАТИСТИКА:")
            print(f"   Начальный капитал: ${metrics['initial_capital']:,.2f}")
            print(f"   Конечный капитал:  ${metrics['final_capital']:,.2f}")
            print(f"   Общая доходность:  {metrics['total_return']*100:.2f}%")
            print(f"   Максимальная просадка: {metrics['max_drawdown']*100:.2f}%")
            print(f"   Волатильность: {metrics['volatility']*100:.2f}%")
            print(f"   Коэффициент Шарпа: {metrics['sharpe_ratio']:.3f}")
            
            print(f"\n📈 ТОРГОВАЯ СТАТИСТИКА:")
            print(f"   Всего открыто позиций: {metrics['total_positions_opened']}")
            print(f"   Всего закрыто сделок: {metrics['total_trades']}")
            print(f"   Винрейт: {metrics['win_rate']*100:.1f}%")
            print(f"   Средний выигрыш: ${metrics['avg_win']:.2f}")
            print(f"   Средний проигрыш: ${metrics['avg_loss']:.2f}")
            print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"   Среднее время удержания: {metrics['avg_bars_held']:.1f} баров")
            
            print(f"\n🔍 АНАЛИЗ ТИПОВ ЗАКРЫТИЯ:")
            print(f"   По временному горизонту: {metrics['horizon_closes']} ({metrics['horizon_closes']/max(1, metrics['total_trades'])*100:.1f}%)")
            print(f"   По стоп-лосс/тейк-профит: {metrics['sl_tp_closes']} ({metrics['sl_tp_closes']/max(1, metrics['total_trades'])*100:.1f}%)")
            print(f"   По сигналу: {metrics['signal_closes']} ({metrics['signal_closes']/max(1, metrics['total_trades'])*100:.1f}%)")
            print(f"   Финальное закрытие: {metrics['final_closes']} ({metrics['final_closes']/max(1, metrics['total_trades'])*100:.1f}%)")
            
            # Анализ эффективности временных горизонтов
            if metrics['horizon_closes'] > 0:
                horizon_trades = [t for t in self.trades if t['action'] == 'CLOSE_HORIZON' and 'pnl' in t]
                horizon_pnls = [t['pnl'] for t in horizon_trades]
                horizon_win_rate = len([p for p in horizon_pnls if p > 0]) / len(horizon_pnls) * 100
                avg_horizon_pnl = np.mean(horizon_pnls)
                
                print(f"\n⏰ ЭФФЕКТИВНОСТЬ ВРЕМЕННЫХ ГОРИЗОНТОВ:")
                print(f"   Винрейт горизонтов: {horizon_win_rate:.1f}%")
                print(f"   Средний PnL горизонтов: ${avg_horizon_pnl:.2f}")
                print(f"   Средний горизонт: {np.mean([t.get('max_horizon', 12) for t in horizon_trades]):.1f} баров")
            
            print("\n" + "="*80)
            
        except Exception as e:
            self.logger.error(f"Ошибка вывода сводки: {e}")


def run_aggressive_backtest_example():
    """Пример запуска агрессивного бэктеста."""
    try:
        # Загружаем конфигурацию
        import yaml
        with open('config/aggressive_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Создаем бэктестер
        backtester = AggressiveBacktester(config)
        
        # Загружаем данные (пример)
        # data = load_data()  # Здесь должна быть загрузка данных
        
        # Запускаем бэктест
        # results = backtester.run_backtest(data, "models/aggressive_xgboost_BTC_USDT_latest.pkl")
        
        # Выводим результаты
        # backtester.print_summary()
        
        print("Агрессивный бэктестер готов к использованию!")
        
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    run_aggressive_backtest_example() 
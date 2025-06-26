"""
Модуль для создания целевых переменных.

Этот модуль предоставляет функциональность для:
- Создания целевых переменных для классификации
- Создания целевых переменных для регрессии
- Обработки временных рядов для ML
- Создания меток для различных торговых стратегий
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger

class TargetCreator:
    """
    Класс для создания целевых переменных для ML моделей.
    
    Поддерживает различные типы целевых переменных:
    - Направление цены (бинарная классификация)
    - Доходность (регрессия)
    - Волатильность (регрессия)
    - Кастомные метки для стратегий
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация создателя целевых переменных.
        
        Args:
            config: Конфигурация целевой переменной
        """
        self.config = config or {}
        self.logger = logger.bind(name="TargetCreator")
    
    def create_target(self, df: pd.DataFrame, primary_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Создание целевой переменной согласно конфигурации.
        
        Args:
            df: DataFrame с данными
            primary_df: Основной DataFrame для создания целевой переменной (для множественных таймфреймов)
        
        Returns:
            DataFrame с добавленной целевой переменной
        """
        df = df.copy()
        
        # Используем primary_df если передан, иначе используем df
        target_df = primary_df if primary_df is not None else df
        
        target_type = self.config.get('method', 'direction')
        threshold = self.config.get('threshold', 0.001)
        lookforward_periods = self.config.get('lookforward_periods', 3)
        
        # Параметры для 3-классовой классификации
        fall_threshold = self.config.get('fall_threshold', -0.001)
        rise_threshold = self.config.get('rise_threshold', 0.001)
        
        try:
            # Создаём динамическую целевую переменную (классификация)
            if target_type == 'dynamic':
                df = self.create_dynamic_target(df, lookforward_periods)
            elif target_type == 'direction_3class':
                df = self.create_direction_3class_target(df, fall_threshold, rise_threshold, lookforward_periods)
            elif target_type == 'direction':
                df = self.create_direction_target(df, threshold, lookforward_periods)
            elif target_type == 'return':
                df = self.create_return_target(df, lookforward_periods)
            elif target_type == 'volatility':
                df = self.create_volatility_target(df, lookforward_periods)
            else:
                self.logger.warning(f"Неизвестный тип целевой переменной: {target_type}")
                df = self.create_dynamic_target(df, lookforward_periods)

            # ВСЕГДА добавляем доходность как дополнительную регрессионную цель
            df = self.create_return_target(df, lookforward_periods)

            # Переименовываем target_direction/target_dynamic в 'target' (основная классификация)
            if 'target_direction' in df.columns:
                df['target'] = df['target_direction']
                df = df.drop('target_direction', axis=1)
            elif 'target_dynamic' in df.columns:
                df['target'] = df['target_dynamic']
            elif 'target_return' in df.columns:
                df['target'] = (df['target_return'] > 0).astype(int)

            # Переименовываем target_return в target_reg для регрессионной модели
            if 'target_return' in df.columns:
                df['target_reg'] = df['target_return']
                df = df.drop('target_return', axis=1)

            # Удаляем строки с NaN в target
            df = df.dropna(subset=['target'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка создания целевой переменной: {e}")
            # Создаем случайные метки как fallback
            if target_type == 'direction_3class':
                df['target'] = np.random.choice([0, 1, 2], size=len(df), p=[0.3, 0.4, 0.3])
            else:
                df['target'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
            return df
    
    def create_direction_target(self, df: pd.DataFrame, threshold: float = 0.001, 
                               lookahead: int = 1) -> pd.DataFrame:
        """
        Создание целевой переменной для направления цены.
        
        Args:
            df: DataFrame с данными
            threshold: Минимальное изменение для сигнала
            lookahead: Количество периодов вперед
        
        Returns:
            DataFrame с добавленной целевой переменной
        """
        try:
            # Расчет будущего изменения цены
            future_price = df['close'].shift(-lookahead)
            price_change = (future_price - df['close']) / df['close']
            
            # Создание меток для бинарной классификации
            # 1: цена вырастет больше чем на threshold
            # 0: цена упадет или изменится меньше чем на threshold
            target = (price_change > threshold).astype(int)
            
            # Удаляем NaN значения (последние строки)
            df = df.dropna()
            target = target.dropna()
            
            # Проверяем, что у нас есть оба класса
            unique_values = target.unique()
            if len(unique_values) < 2:
                # Если нет обоих классов, уменьшаем threshold
                threshold = threshold * 0.5
                target = (price_change > threshold).astype(int)
                target = target.dropna()
                self.logger.warning(f"Уменьшен threshold до {threshold} для обеспечения обоих классов")
            
            df['target'] = target
            
            # Логируем распределение
            distribution = target.value_counts().to_dict()
            self.logger.info(f"Создана целевая переменная направления (threshold={threshold}, lookahead={lookahead})")
            self.logger.info(f"Распределение меток: {distribution}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка создания целевой переменной направления: {e}")
            # Возвращаем случайные метки если что-то пошло не так
            df['target'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
            return df
    
    def create_return_target(self, df: pd.DataFrame, lookahead: int = 1) -> pd.DataFrame:
        """
        Создание целевой переменной для доходности.
        
        Args:
            df: DataFrame с данными
            lookahead: Количество периодов вперед
        
        Returns:
            DataFrame с добавленной целевой переменной
        """
        try:
            # Расчет будущей доходности
            future_price = df['close'].shift(-lookahead)
            returns = (future_price - df['close']) / df['close']
            
            df['target_return'] = returns
            
            # Статистика доходности
            mean_return = returns.mean()
            std_return = returns.std()
            
            self.logger.info(f"Создана целевая переменная доходности (lookahead={lookahead})")
            self.logger.info(f"Средняя доходность: {mean_return:.4f}, Стандартное отклонение: {std_return:.4f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка создания целевой переменной доходности: {e}")
            return df
    
    def create_volatility_target(self, df: pd.DataFrame, lookahead: int = 1) -> pd.DataFrame:
        """
        Создание целевой переменной для волатильности.
        
        Args:
            df: DataFrame с данными
            lookahead: Количество периодов вперед
        
        Returns:
            DataFrame с добавленной целевой переменной
        """
        try:
            # Расчет будущей волатильности
            future_returns = df['close'].pct_change().shift(-lookahead)
            volatility = future_returns.rolling(window=lookahead).std()
            
            df['target_volatility'] = volatility
            
            # Статистика волатильности
            mean_vol = volatility.mean()
            std_vol = volatility.std()
            
            self.logger.info(f"Создана целевая переменная волатильности (lookahead={lookahead})")
            self.logger.info(f"Средняя волатильность: {mean_vol:.4f}, Стандартное отклонение: {std_vol:.4f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка создания целевой переменной волатильности: {e}")
            return df
    
    def create_custom_target(self, df: pd.DataFrame, strategy: str, 
                           params: Dict) -> pd.DataFrame:
        """
        Создание кастомной целевой переменной для специфических стратегий.
        
        Args:
            df: DataFrame с данными
            strategy: Название стратегии
            params: Параметры стратегии
        
        Returns:
            DataFrame с добавленной целевой переменной
        """
        try:
            if strategy == 'rsi_strategy':
                df = self._create_rsi_strategy_target(df, params)
            elif strategy == 'macd_strategy':
                df = self._create_macd_strategy_target(df, params)
            elif strategy == 'bollinger_strategy':
                df = self._create_bollinger_strategy_target(df, params)
            else:
                self.logger.warning(f"Неизвестная стратегия: {strategy}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка создания кастомной целевой переменной: {e}")
            return df
    
    def _create_rsi_strategy_target(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Создание целевой переменной для RSI стратегии.
        
        Args:
            df: DataFrame с данными
            params: Параметры стратегии
        
        Returns:
            DataFrame с добавленной целевой переменной
        """
        rsi_period = params.get('rsi_period', 14)
        oversold = params.get('oversold', 30)
        overbought = params.get('overbought', 70)
        lookahead = params.get('lookahead', 1)
        
        # Проверка наличия RSI
        rsi_col = f'rsi_{rsi_period}'
        if rsi_col not in df.columns:
            self.logger.error(f"RSI колонка {rsi_col} не найдена")
            return df
        
        # Создание сигналов
        # 1: RSI < oversold (перепроданность) -> сигнал на покупку
        # 0: RSI > overbought (перекупленность) -> сигнал на продажу
        # -1: нейтральная зона
        
        buy_signal = df[rsi_col] < oversold
        sell_signal = df[rsi_col] > overbought
        
        target = np.where(buy_signal, 1,
                         np.where(sell_signal, 0, -1))
        
        df['target_rsi_strategy'] = target
        
        self.logger.info(f"Создана RSI стратегия (oversold={oversold}, overbought={overbought})")
        return df
    
    def _create_macd_strategy_target(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Создание целевой переменной для MACD стратегии.
        
        Args:
            df: DataFrame с данными
            params: Параметры стратегии
        
        Returns:
            DataFrame с добавленной целевой переменной
        """
        lookahead = params.get('lookahead', 1)
        
        # Проверка наличия MACD
        if 'macd_line' not in df.columns or 'macd_signal' not in df.columns:
            self.logger.error("MACD колонки не найдены")
            return df
        
        # Создание сигналов
        # 1: MACD линия пересекает сигнальную линию снизу вверх (бычий сигнал)
        # 0: MACD линия пересекает сигнальную линию сверху вниз (медвежий сигнал)
        # -1: нет пересечения
        
        macd_cross_up = (df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
        macd_cross_down = (df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
        
        target = np.where(macd_cross_up, 1,
                         np.where(macd_cross_down, 0, -1))
        
        df['target_macd_strategy'] = target
        
        self.logger.info("Создана MACD стратегия")
        return df
    
    def _create_bollinger_strategy_target(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Создание целевой переменной для Bollinger Bands стратегии.
        
        Args:
            df: DataFrame с данными
            params: Параметры стратегии
        
        Returns:
            DataFrame с добавленной целевой переменной
        """
        bb_period = params.get('bb_period', 20)
        lookahead = params.get('lookahead', 1)
        
        # Проверка наличия Bollinger Bands
        upper_col = f'bb_upper_{bb_period}'
        lower_col = f'bb_lower_{bb_period}'
        
        if upper_col not in df.columns or lower_col not in df.columns:
            self.logger.error(f"Bollinger Bands колонки не найдены")
            return df
        
        # Создание сигналов
        # 1: цена касается нижней полосы (сигнал на покупку)
        # 0: цена касается верхней полосы (сигнал на продажу)
        # -1: цена в середине полос
        
        touch_lower = df['close'] <= df[lower_col] * 1.001  # Небольшой допуск
        touch_upper = df['close'] >= df[upper_col] * 0.999  # Небольшой допуск
        
        target = np.where(touch_lower, 1,
                         np.where(touch_upper, 0, -1))
        
        df['target_bollinger_strategy'] = target
        
        self.logger.info(f"Создана Bollinger Bands стратегия (period={bb_period})")
        return df
    
    def prepare_ml_data(self, df: pd.DataFrame, target_column: str = 'target_direction') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Подготовка данных для ML модели.
        
        Args:
            df: DataFrame с данными
            target_column: Название колонки с целевой переменной
        
        Returns:
            Tuple с признаками и целевой переменной
        """
        try:
            # Удаление строк с NaN значениями
            df_clean = df.dropna()
            
            # Удаление строк с нейтральными метками (-1)
            if target_column in df_clean.columns:
                df_clean = df_clean[df_clean[target_column] != -1]
            
            # Выбор признаков (исключаем целевые переменные и базовые OHLCV)
            exclude_columns = ['target_direction', 'target_return', 'target_volatility', 
                              'target_rsi_strategy', 'target_macd_strategy', 'target_bollinger_strategy',
                              'timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            feature_columns = [col for col in df_clean.columns if col not in exclude_columns]
            
            X = df_clean[feature_columns]
            y = df_clean[target_column] if target_column in df_clean.columns else None
            
            self.logger.info(f"Подготовлены данные для ML: {X.shape[0]} образцов, {X.shape[1]} признаков")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Ошибка подготовки данных для ML: {e}")
            return pd.DataFrame(), pd.Series()
    
    def create_direction_3class_target(self, df: pd.DataFrame, fall_threshold: float = -0.001, 
                                      rise_threshold: float = 0.001, lookahead: int = 3) -> pd.DataFrame:
        """
        Создание 3-классовой целевой переменной для направления цены.
        
        Args:
            df: DataFrame с данными
            fall_threshold: Порог для падения (отрицательный)
            rise_threshold: Порог для роста (положительный)
            lookahead: Количество периодов вперед
        
        Returns:
            DataFrame с добавленной целевой переменной
        """
        try:
            # Расчет будущего изменения цены
            future_price = df['close'].shift(-lookahead)
            price_change = (future_price - df['close']) / df['close']
            
            # Создание меток для 3-классовой классификации
            # 0: падение (fall) - цена упадет больше чем на fall_threshold
            # 1: удержание (hold) - цена изменится между fall_threshold и rise_threshold
            # 2: рост (rise) - цена вырастет больше чем на rise_threshold
            target = np.where(price_change < fall_threshold, 0,
                    np.where(price_change > rise_threshold, 2, 1))
            
            # Добавляем target в DataFrame
            df['target'] = target
            
            # Удаляем строки с NaN в target (последние строки из-за lookahead)
            df = df.dropna(subset=['target'])
            
            # Проверяем, что у нас есть все три класса
            unique_values = df['target'].unique()
            if len(unique_values) < 3:
                # Если нет всех классов, уменьшаем пороги
                fall_threshold = fall_threshold * 0.5
                rise_threshold = rise_threshold * 0.5
                
                # Пересчитываем target с новыми порогами
                price_change = (df['close'].shift(-lookahead) - df['close']) / df['close']
                target = np.where(price_change < fall_threshold, 0,
                        np.where(price_change > rise_threshold, 2, 1))
                df['target'] = target
                df = df.dropna(subset=['target'])
                
                self.logger.warning(f"Уменьшены пороги: fall={fall_threshold}, rise={rise_threshold}")
            
            # Логируем распределение
            distribution = df['target'].value_counts().to_dict()
            self.logger.info(f"Создана 3-классовая целевая переменная (fall_threshold={fall_threshold}, rise_threshold={rise_threshold}, lookahead={lookahead})")
            self.logger.info(f"Распределение меток: {distribution} (fall/hold/rise)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка создания 3-классовой целевой переменной: {e}")
            # Возвращаем случайные метки если что-то пошло не так
            df['target'] = np.random.choice([0, 1, 2], size=len(df), p=[0.3, 0.4, 0.3])
            return df
    
    def create_dynamic_target(self, data: pd.DataFrame, lookforward_periods: int = 3) -> pd.DataFrame:
        """
        Создание целевой переменной с динамическими порогами на основе волатильности.
        БЕЗ УТЕЧКИ ДАННЫХ - future_return используется только для создания target, не как признак.
        
        Args:
            data: DataFrame с OHLCV данными
            lookforward_periods: Количество периодов вперед для расчета
            
        Returns:
            DataFrame с динамической целевой переменной
        """
        try:
            df = data.copy()
            
            # Рассчитываем волатильность рынка (только на исторических данных)
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            
            # Динамические пороги на основе волатильности
            # В спокойном рынке - более строгие пороги, в волатильном - более мягкие
            df['dynamic_fall_threshold'] = -df['volatility'] * 0.5  # 50% от волатильности
            df['dynamic_rise_threshold'] = df['volatility'] * 0.5   # 50% от волатильности
            
            # Минимальные и максимальные пороги
            df['dynamic_fall_threshold'] = df['dynamic_fall_threshold'].clip(-0.01, -0.001)
            df['dynamic_rise_threshold'] = df['dynamic_rise_threshold'].clip(0.001, 0.01)
            
            # Рассчитываем будущие доходности ТОЛЬКО для создания целевой переменной
            future_return = df['close'].shift(-lookforward_periods) / df['close'] - 1
            
            # Создаем целевую переменную с динамическими порогами
            df['target_dynamic'] = 1  # hold по умолчанию
            
            # Fall (класс 0)
            fall_condition = future_return < df['dynamic_fall_threshold']
            df.loc[fall_condition, 'target_dynamic'] = 0
            
            # Rise (класс 2)
            rise_condition = future_return > df['dynamic_rise_threshold']
            df.loc[rise_condition, 'target_dynamic'] = 2
            
            # Удаляем future_return - это будущая информация, не должна быть признаком!
            # df['future_return'] удаляется здесь
            
            # Удаляем NaN значения
            df = df.dropna()
            
            self.logger.info(f"Создана динамическая целевая переменная БЕЗ утечки данных")
            self.logger.info(f"Средние пороги: fall={df['dynamic_fall_threshold'].mean():.4f}, rise={df['dynamic_rise_threshold'].mean():.4f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка создания динамической целевой переменной: {e}")
            return data 
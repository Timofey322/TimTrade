"""
Модуль для расчета технических индикаторов.

Этот модуль предоставляет функциональность для:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (SMA, EMA)
- Волатильность и другие индикаторы
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger

class TechnicalIndicators:
    """
    Класс для расчета технических индикаторов.
    
    Поддерживает основные технические индикаторы:
    - RSI, MACD, Bollinger Bands
    - Moving Averages (SMA, EMA)
    - Волатильность и моменты
    - Объемные индикаторы
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация калькулятора индикаторов.
        
        Args:
            config: Конфигурация индикаторов
        """
        self.config = config or {}
        self.logger = logger.bind(name="TechnicalIndicators")
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление всех индикаторов согласно конфигурации.
        
        Args:
            df: DataFrame с OHLCV данными
        
        Returns:
            DataFrame с добавленными индикаторами
        """
        df = df.copy()
        
        # RSI
        if self.config.get('rsi', {}).get('enabled', False):
            period = self.config['rsi'].get('period', 14)
            df = self.add_rsi(df, period)
        
        # MACD
        if self.config.get('macd', {}).get('enabled', False):
            fast = self.config['macd'].get('fast_period', 12)
            slow = self.config['macd'].get('slow_period', 26)
            signal = self.config['macd'].get('signal_period', 9)
            df = self.add_macd(df, fast, slow, signal)
        
        # Bollinger Bands
        if self.config.get('bollinger_bands', {}).get('enabled', False):
            period = self.config['bollinger_bands'].get('period', 20)
            std_dev = self.config['bollinger_bands'].get('std_dev', 2)
            df = self.add_bollinger_bands(df, period, std_dev)
        
        # SMA
        if self.config.get('sma', {}).get('enabled', False):
            periods = self.config['sma'].get('periods', [10, 20, 50, 200])
            for period in periods:
                df = self.add_sma(df, period)
        
        # EMA
        if self.config.get('ema', {}).get('enabled', False):
            periods = self.config['ema'].get('periods', [12, 26])
            for period in periods:
                df = self.add_ema(df, period)
        
        # Волатильность
        if self.config.get('volatility', {}).get('enabled', False):
            periods = self.config['volatility'].get('periods', [10, 20, 50])
            for period in periods:
                df = self.add_volatility(df, period)
        
        # Дополнительные индикаторы
        df = self.add_price_features(df)
        df = self.add_volume_features(df)
        
        return df
    
    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Добавление RSI (Relative Strength Index).
        
        Args:
            df: DataFrame с данными
            period: Период для расчета
        
        Returns:
            DataFrame с добавленным RSI
        """
        try:
            # Расчет изменения цены
            delta = df['close'].diff()
            
            # Разделение на положительные и отрицательные изменения
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Расчет RSI
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            df[f'rsi_{period}'] = rsi
            
            self.logger.debug(f"Добавлен RSI с периодом {period}")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета RSI: {e}")
            return df
    
    def add_macd(self, df: pd.DataFrame, fast_period: int = 12, 
                 slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Добавление MACD (Moving Average Convergence Divergence).
        
        Args:
            df: DataFrame с данными
            fast_period: Быстрый период EMA
            slow_period: Медленный период EMA
            signal_period: Период сигнальной линии
        
        Returns:
            DataFrame с добавленным MACD
        """
        try:
            # Расчет быстрой и медленной EMA
            ema_fast = df['close'].ewm(span=fast_period).mean()
            ema_slow = df['close'].ewm(span=slow_period).mean()
            
            # MACD линия
            macd_line = ema_fast - ema_slow
            
            # Сигнальная линия
            signal_line = macd_line.ewm(span=signal_period).mean()
            
            # Гистограмма MACD
            macd_histogram = macd_line - signal_line
            
            df[f'macd_line'] = macd_line
            df[f'macd_signal'] = signal_line
            df[f'macd_histogram'] = macd_histogram
            
            self.logger.debug(f"Добавлен MACD ({fast_period}, {slow_period}, {signal_period})")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета MACD: {e}")
            return df
    
    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, 
                           std_dev: float = 2) -> pd.DataFrame:
        """
        Добавление Bollinger Bands.
        
        Args:
            df: DataFrame с данными
            period: Период для SMA
            std_dev: Количество стандартных отклонений
        
        Returns:
            DataFrame с добавленными Bollinger Bands
        """
        try:
            # Средняя линия (SMA)
            middle_band = df['close'].rolling(window=period).mean()
            
            # Стандартное отклонение
            std = df['close'].rolling(window=period).std()
            
            # Верхняя и нижняя полосы
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            # Ширина полос
            bandwidth = (upper_band - lower_band) / middle_band
            
            # Позиция цены относительно полос
            bb_position = (df['close'] - lower_band) / (upper_band - lower_band)
            
            df[f'bb_upper_{period}'] = upper_band
            df[f'bb_middle_{period}'] = middle_band
            df[f'bb_lower_{period}'] = lower_band
            df[f'bb_bandwidth_{period}'] = bandwidth
            df[f'bb_position_{period}'] = bb_position
            
            self.logger.debug(f"Добавлены Bollinger Bands с периодом {period}")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета Bollinger Bands: {e}")
            return df
    
    def add_sma(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Добавление Simple Moving Average.
        
        Args:
            df: DataFrame с данными
            period: Период для SMA
        
        Returns:
            DataFrame с добавленным SMA
        """
        try:
            sma = df['close'].rolling(window=period).mean()
            df[f'sma_{period}'] = sma
            
            # Расстояние от цены до SMA
            df[f'price_to_sma_{period}'] = (df['close'] - sma) / sma
            
            self.logger.debug(f"Добавлен SMA с периодом {period}")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета SMA: {e}")
            return df
    
    def add_ema(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Добавление Exponential Moving Average.
        
        Args:
            df: DataFrame с данными
            period: Период для EMA
        
        Returns:
            DataFrame с добавленным EMA
        """
        try:
            ema = df['close'].ewm(span=period).mean()
            df[f'ema_{period}'] = ema
            
            # Расстояние от цены до EMA
            df[f'price_to_ema_{period}'] = (df['close'] - ema) / ema
            
            self.logger.debug(f"Добавлен EMA с периодом {period}")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета EMA: {e}")
            return df
    
    def add_volatility(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Добавление индикаторов волатильности.
        
        Args:
            df: DataFrame с данными
            period: Период для расчета
        
        Returns:
            DataFrame с добавленными индикаторами волатильности
        """
        try:
            # Логарифмические доходности
            log_returns = np.log(df['close'] / df['close'].shift(1))
            
            # Историческая волатильность
            volatility = log_returns.rolling(window=period).std() * np.sqrt(period)
            
            # True Range
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            df[f'volatility_{period}'] = volatility
            df[f'atr_{period}'] = atr
            
            self.logger.debug(f"Добавлена волатильность с периодом {period}")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета волатильности: {e}")
            return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление ценовых признаков.
        
        Args:
            df: DataFrame с данными
        
        Returns:
            DataFrame с добавленными ценовыми признаками
        """
        try:
            # Доходности
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Ценовые уровни
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Тело свечи
            df['body_size'] = abs(df['close'] - df['open'])
            df['body_ratio'] = df['body_size'] / (df['high'] - df['low'])
            
            # Тени свечи
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            
            # Моментум
            df['momentum_1'] = df['close'] / df['close'].shift(1) - 1
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            
            self.logger.debug("Добавлены ценовые признаки")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления ценовых признаков: {e}")
            return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление объемных признаков.
        
        Args:
            df: DataFrame с данными
        
        Returns:
            DataFrame с добавленными объемными признаками
        """
        try:
            # Средний объем
            df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            
            # Относительный объем
            df['relative_volume'] = df['volume'] / df['volume_sma_20']
            
            # Объемная цена
            df['volume_price'] = df['volume'] * df['close']
            df['volume_price_sma'] = df['volume_price'].rolling(window=20).mean()
            
            # Индикатор объема
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
            
            self.logger.debug("Добавлены объемные признаки")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления объемных признаков: {e}")
            return df
    
    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление On-Balance Volume (OBV).
        
        Args:
            df: DataFrame с данными
        
        Returns:
            DataFrame с добавленным OBV
        """
        try:
            # Расчет изменения цены
            price_change = df['close'].diff()
            
            # Инициализация OBV
            obv = pd.Series(index=df.index, dtype=float)
            obv.iloc[0] = df['volume'].iloc[0]
            
            # Расчет OBV
            for i in range(1, len(df)):
                if price_change.iloc[i] > 0:
                    obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
                elif price_change.iloc[i] < 0:
                    obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            df['obv'] = obv
            
            # Дополнительные OBV индикаторы
            df['obv_sma_10'] = obv.rolling(window=10).mean()
            df['obv_ratio'] = obv / obv.rolling(window=20).mean()
            
            self.logger.debug("Добавлен OBV (On-Balance Volume)")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета OBV: {e}")
            return df
    
    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление Volume Weighted Average Price (VWAP).
        
        Args:
            df: DataFrame с данными
        
        Returns:
            DataFrame с добавленным VWAP
        """
        try:
            # Типичная цена
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Объемно-взвешенная цена
            volume_price = typical_price * df['volume']
            
            # VWAP
            vwap = volume_price.cumsum() / df['volume'].cumsum()
            
            # Также добавим скользящий VWAP для различных периодов
            for period in [20, 50]:
                vwap_rolling = (
                    volume_price.rolling(window=period).sum() / 
                    df['volume'].rolling(window=period).sum()
                )
                df[f'vwap_{period}'] = vwap_rolling
                
                # Отклонение цены от VWAP
                df[f'price_to_vwap_{period}'] = (df['close'] - vwap_rolling) / vwap_rolling
            
            df['vwap'] = vwap
            df['price_to_vwap'] = (df['close'] - vwap) / vwap
            
            self.logger.debug("Добавлен VWAP (Volume Weighted Average Price)")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета VWAP: {e}")
            return df
    
    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Добавление Average True Range (ATR) - улучшенная версия.
        
        Args:
            df: DataFrame с данными
            period: Период для расчета ATR
        
        Returns:
            DataFrame с добавленным ATR
        """
        try:
            # True Range компоненты
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            # True Range
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            # ATR (среднее значение True Range)
            atr = true_range.rolling(window=period).mean()
            
            # Дополнительные ATR индикаторы
            df[f'atr_{period}'] = atr
            df[f'atr_percentage_{period}'] = atr / df['close'] * 100
            df[f'atr_ratio_{period}'] = atr / atr.rolling(window=period*2).mean()
            
            # ATR в качестве фильтра волатильности
            df[f'atr_high_volatility_{period}'] = (atr > atr.rolling(window=period*2).mean()).astype(int)
            
            self.logger.debug(f"Добавлен ATR с периодом {period}")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета ATR: {e}")
            return df
    
    def add_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Добавление Williams %R индикатора.
        
        Args:
            df: DataFrame с данными
            period: Период для расчета
        
        Returns:
            DataFrame с добавленным Williams %R
        """
        try:
            # Максимум и минимум за период
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            
            # Williams %R
            williams_r = (highest_high - df['close']) / (highest_high - lowest_low) * -100
            
            df[f'williams_r_{period}'] = williams_r
            
            # Дополнительные сигналы Williams %R
            df[f'williams_r_oversold_{period}'] = (williams_r < -80).astype(int)
            df[f'williams_r_overbought_{period}'] = (williams_r > -20).astype(int)
            df[f'williams_r_momentum_{period}'] = williams_r.diff()
            
            self.logger.debug(f"Добавлен Williams %R с периодом {period}")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета Williams %R: {e}")
            return df
    
    def add_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Добавление Commodity Channel Index (CCI).
        
        Args:
            df: DataFrame с данными
            period: Период для расчета
        
        Returns:
            DataFrame с добавленным CCI
        """
        try:
            # Типичная цена
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Скользящее среднее типичной цены
            sma_tp = typical_price.rolling(window=period).mean()
            
            # Среднее абсолютное отклонение
            mad = typical_price.rolling(window=period).apply(
                lambda x: np.abs(x - x.mean()).mean()
            )
            
            # CCI
            cci = (typical_price - sma_tp) / (0.015 * mad)
            
            df[f'cci_{period}'] = cci
            
            # Дополнительные CCI сигналы
            df[f'cci_overbought_{period}'] = (cci > 100).astype(int)
            df[f'cci_oversold_{period}'] = (cci < -100).astype(int)
            df[f'cci_momentum_{period}'] = cci.diff()
            df[f'cci_zero_cross_{period}'] = ((cci > 0) & (cci.shift() <= 0)).astype(int)
            
            self.logger.debug(f"Добавлен CCI с периодом {period}")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета CCI: {e}")
            return df
    
    def add_advanced_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление продвинутых объемных индикаторов.
        
        Args:
            df: DataFrame с данными
        
        Returns:
            DataFrame с продвинутыми объемными индикаторами
        """
        try:
            # Money Flow Index (MFI)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            # Positive и Negative Money Flow
            positive_flow = money_flow.where(df['close'] > df['close'].shift(), 0)
            negative_flow = money_flow.where(df['close'] < df['close'].shift(), 0)
            
            # MFI
            positive_mf = positive_flow.rolling(window=14).sum()
            negative_mf = negative_flow.rolling(window=14).sum()
            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            
            df['mfi_14'] = mfi
            df['mfi_overbought'] = (mfi > 80).astype(int)
            df['mfi_oversold'] = (mfi < 20).astype(int)
            
            # Chaikin Money Flow (CMF)
            mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            mf_volume = mf_multiplier * df['volume']
            cmf = mf_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            
            df['cmf_20'] = cmf
            df['cmf_bullish'] = (cmf > 0).astype(int)
            
            self.logger.debug("Добавлены продвинутые объемные индикаторы")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета продвинутых объемных индикаторов: {e}")
            return df 
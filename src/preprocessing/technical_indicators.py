"""
Технические индикаторы для торговой системы.

Этот модуль содержит все функции расчета технических индикаторов,
которые используются как в feature engineering, так и в бэктестинге.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger


class TechnicalIndicators:
    """
    Класс для расчета технических индикаторов.
    
    Содержит все функции расчета индикаторов, которые могут быть
    использованы в различных частях системы.
    """
    
    def __init__(self):
        """Инициализация класса технических индикаторов."""
        self.logger = logger.bind(name="TechnicalIndicators")
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Расчет RSI (Relative Strength Index).
        
        Args:
            df: DataFrame с данными OHLCV
            period: Период для расчета RSI
            
        Returns:
            Series с значениями RSI
        """
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            self.logger.error(f"Ошибка расчета RSI: {e}")
            return pd.Series(index=df.index)
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Расчет MACD (Moving Average Convergence Divergence).
        
        Args:
            df: DataFrame с данными OHLCV
            fast: Быстрый период EMA
            slow: Медленный период EMA
            signal: Период сигнальной линии
            
        Returns:
            Словарь с MACD, сигнальной линией и гистограммой
        """
        try:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
        except Exception as e:
            self.logger.error(f"Ошибка расчета MACD: {e}")
            return {
                'macd': pd.Series(index=df.index),
                'signal': pd.Series(index=df.index),
                'histogram': pd.Series(index=df.index)
            }
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """
        Расчет полос Боллинджера.
        
        Args:
            df: DataFrame с данными OHLCV
            period: Период для скользящего среднего
            std_dev: Количество стандартных отклонений
            
        Returns:
            Словарь с верхней, средней и нижней полосами
        """
        try:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return {
                'upper': upper_band,
                'middle': sma,
                'lower': lower_band
            }
        except Exception as e:
            self.logger.error(f"Ошибка расчета полос Боллинджера: {e}")
            return {
                'upper': pd.Series(index=df.index),
                'middle': pd.Series(index=df.index),
                'lower': pd.Series(index=df.index)
            }
    
    def calculate_sma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Расчет простого скользящего среднего (SMA).
        
        Args:
            df: DataFrame с данными OHLCV
            period: Период для расчета
            
        Returns:
            Series с значениями SMA
        """
        try:
            return df['close'].rolling(window=period).mean()
        except Exception as e:
            self.logger.error(f"Ошибка расчета SMA: {e}")
            return pd.Series(index=df.index)
    
    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Расчет экспоненциального скользящего среднего (EMA).
        
        Args:
            df: DataFrame с данными OHLCV
            period: Период для расчета
            
        Returns:
            Series с значениями EMA
        """
        try:
            return df['close'].ewm(span=period).mean()
        except Exception as e:
            self.logger.error(f"Ошибка расчета EMA: {e}")
            return pd.Series(index=df.index)
    
    def calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """
        Расчет стохастического осциллятора.
        
        Args:
            df: DataFrame с данными OHLCV
            period: Период для расчета
            
        Returns:
            Словарь с %K и %D
        """
        try:
            lowest_low = df['low'].rolling(window=period).min()
            highest_high = df['high'].rolling(window=period).max()
            k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=3).mean()
            
            return {
                'k': k_percent,
                'd': d_percent
            }
        except Exception as e:
            self.logger.error(f"Ошибка расчета стохастика: {e}")
            return {
                'k': pd.Series(index=df.index),
                'd': pd.Series(index=df.index)
            }
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Расчет индикатора Williams %R.
        
        Args:
            df: DataFrame с данными OHLCV
            period: Период для расчета
            
        Returns:
            Series с значениями Williams %R
        """
        try:
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
            return williams_r
        except Exception as e:
            self.logger.error(f"Ошибка расчета Williams %R: {e}")
            return pd.Series(index=df.index)
    
    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Расчет CCI (Commodity Channel Index).
        
        Args:
            df: DataFrame с данными OHLCV
            period: Период для расчета
            
        Returns:
            Series с значениями CCI
        """
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            return cci
        except Exception as e:
            self.logger.error(f"Ошибка расчета CCI: {e}")
            return pd.Series(index=df.index)
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """
        Расчет ADX (Average Directional Index).
        
        Args:
            df: DataFrame с данными OHLCV
            period: Период для расчета
            
        Returns:
            Словарь с ADX, +DI и -DI
        """
        try:
            # True Range
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            # Directional Movement
            up_move = df['high'] - df['high'].shift()
            down_move = df['low'].shift() - df['low']
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return {
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di
            }
        except Exception as e:
            self.logger.error(f"Ошибка расчета ADX: {e}")
            return {
                'adx': pd.Series(index=df.index),
                'plus_di': pd.Series(index=df.index),
                'minus_di': pd.Series(index=df.index)
            }
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Расчет ATR (Average True Range).
        
        Args:
            df: DataFrame с данными OHLCV
            period: Период для расчета
            
        Returns:
            Series с значениями ATR
        """
        try:
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr
        except Exception as e:
            self.logger.error(f"Ошибка расчета ATR: {e}")
            return pd.Series(index=df.index)
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """
        Расчет OBV (On-Balance Volume).
        
        Args:
            df: DataFrame с данными OHLCV
            
        Returns:
            Series с значениями OBV
        """
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
            
            return obv
        except Exception as e:
            self.logger.error(f"Ошибка расчета OBV: {e}")
            return pd.Series(index=df.index)
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Расчет VWAP (Volume Weighted Average Price).
        
        Args:
            df: DataFrame с данными OHLCV
            
        Returns:
            Series с значениями VWAP
        """
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            return vwap
        except Exception as e:
            self.logger.error(f"Ошибка расчета VWAP: {e}")
            return pd.Series(index=df.index)
    
    def calculate_momentum(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """
        Расчет индикатора Momentum.
        
        Args:
            df: DataFrame с данными OHLCV
            period: Период для расчета
            
        Returns:
            Series с значениями Momentum
        """
        try:
            return df['close'] - df['close'].shift(period)
        except Exception as e:
            self.logger.error(f"Ошибка расчета Momentum: {e}")
            return pd.Series(index=df.index)
    
    def calculate_ichimoku(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Расчет индикатора Ichimoku Cloud.
        
        Args:
            df: DataFrame с данными OHLCV
            
        Returns:
            Словарь с компонентами Ichimoku
        """
        try:
            # Tenkan-sen (Conversion Line)
            period9_high = df['high'].rolling(window=9).max()
            period9_low = df['low'].rolling(window=9).min()
            tenkan_sen = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line)
            period26_high = df['high'].rolling(window=26).max()
            period26_low = df['low'].rolling(window=26).min()
            kijun_sen = (period26_high + period26_low) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Senkou Span B (Leading Span B)
            period52_high = df['high'].rolling(window=52).max()
            period52_low = df['low'].rolling(window=52).min()
            senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
            
            # Chikou Span (Lagging Span)
            chikou_span = df['close'].shift(-26)
            
            return {
                'tenkan_sen': tenkan_sen,
                'kijun_sen': kijun_sen,
                'senkou_span_a': senkou_span_a,
                'senkou_span_b': senkou_span_b,
                'chikou_span': chikou_span
            }
        except Exception as e:
            self.logger.error(f"Ошибка расчета Ichimoku: {e}")
            return {
                'tenkan_sen': pd.Series(index=df.index),
                'kijun_sen': pd.Series(index=df.index),
                'senkou_span_a': pd.Series(index=df.index),
                'senkou_span_b': pd.Series(index=df.index),
                'chikou_span': pd.Series(index=df.index)
            }
    
    def calculate_hull_ma(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Расчет Hull Moving Average.
        
        Args:
            df: DataFrame с данными OHLCV
            period: Период для расчета
            
        Returns:
            Series с значениями Hull MA
        """
        try:
            wma_half = df['close'].rolling(window=period//2).mean()
            wma_full = df['close'].rolling(window=period).mean()
            raw_hma = 2 * wma_half - wma_full
            hull_ma = raw_hma.rolling(window=int(np.sqrt(period))).mean()
            return hull_ma
        except Exception as e:
            self.logger.error(f"Ошибка расчета Hull MA: {e}")
            return pd.Series(index=df.index)
    
    def calculate_awesome_oscillator(self, df: pd.DataFrame) -> pd.Series:
        """
        Расчет Awesome Oscillator.
        
        Args:
            df: DataFrame с данными OHLCV
            
        Returns:
            Series с значениями Awesome Oscillator
        """
        try:
            median_price = (df['high'] + df['low']) / 2
            ao = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()
            return ao
        except Exception as e:
            self.logger.error(f"Ошибка расчета Awesome Oscillator: {e}")
            return pd.Series(index=df.index)
    
    def calculate_ultimate_oscillator(self, df: pd.DataFrame) -> pd.Series:
        """
        Расчет Ultimate Oscillator.
        
        Args:
            df: DataFrame с данными OHLCV
            
        Returns:
            Series с значениями Ultimate Oscillator
        """
        try:
            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            ], axis=1).max(axis=1)
            
            bp = df['close'] - pd.concat([df['low'], df['close'].shift()], axis=1).min(axis=1)
            
            avg7 = bp.rolling(window=7).sum() / tr.rolling(window=7).sum()
            avg14 = bp.rolling(window=14).sum() / tr.rolling(window=14).sum()
            avg28 = bp.rolling(window=28).sum() / tr.rolling(window=28).sum()
            
            uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / (4 + 2 + 1)
            return uo
        except Exception as e:
            self.logger.error(f"Ошибка расчета Ultimate Oscillator: {e}")
            return pd.Series(index=df.index)
    
    def calculate_bulls_bears_power(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Расчет Bulls and Bears Power.
        
        Args:
            df: DataFrame с данными OHLCV
            
        Returns:
            Словарь с Bulls Power и Bears Power
        """
        try:
            ema = df['close'].ewm(span=13).mean()
            bulls_power = df['high'] - ema
            bears_power = df['low'] - ema
            
            return {
                'bulls_power': bulls_power,
                'bears_power': bears_power
            }
        except Exception as e:
            self.logger.error(f"Ошибка расчета Bulls/Bears Power: {e}")
            return {
                'bulls_power': pd.Series(index=df.index),
                'bears_power': pd.Series(index=df.index)
            }
    
    def calculate_fast_stochastic_rsi(self, df: pd.DataFrame) -> pd.Series:
        """
        Расчет Fast Stochastic RSI.
        
        Args:
            df: DataFrame с данными OHLCV
            
        Returns:
            Series с значениями Fast Stochastic RSI
        """
        try:
            rsi = self.calculate_rsi(df, period=14)
            rsi_min = rsi.rolling(window=14).min()
            rsi_max = rsi.rolling(window=14).max()
            fast_stoch_rsi = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)
            return fast_stoch_rsi
        except Exception as e:
            self.logger.error(f"Ошибка расчета Fast Stochastic RSI: {e}")
            return pd.Series(index=df.index)


# Создаем глобальный экземпляр для использования в других модулях
indicators = TechnicalIndicators() 
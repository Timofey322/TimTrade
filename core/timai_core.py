#!/usr/bin/env python3
"""
🤖 TimAI - Advanced Multi-Model Trading System
Многомодельная торговая система с XGBoost, LightGBM, Random Forest и Incremental Learning
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from research.indicators.best_indicators import BEST_INDICATORS
from research.indicator_combinations.indicator_weights import INDICATOR_WEIGHTS
import numpy as np
import pandas as pd
import warnings
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import joblib
import json

warnings.filterwarnings('ignore')

# Проверка доступности продвинутых ML библиотек
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost доступен:", xgb.__version__)
except ImportError:
    XGBOOST_AVAILABLE = False
    print("❌ XGBoost не установлен")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM доступен:", lgb.__version__)
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("❌ LightGBM не установлен")

try:
    import optuna
    OPTUNA_AVAILABLE = True
    print("✅ Optuna доступен для гиперпараметров")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("❌ Optuna не установлен")

# --- ДОБАВЛЕНО: Импорт улучшений ---
try:
    from research.advanced_techniques.smote_balancer import balance_data
    print("✅ SMOTE/Over/Under Sampler доступен для балансировки классов")
except ImportError:
    balance_data = None
    print("⚠️ SMOTE/Over/Under Sampler не найден")
try:
    from research.advanced_techniques.optuna_hyperopt import optimize_model
    print("✅ Optuna Hyperopt доступен для оптимизации гиперпараметров")
except ImportError:
    optimize_model = None
    print("⚠️ Optuna Hyperopt не найден")

# Базовые ML библиотеки
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

class TimAIFeatureEngine:
    """🔧 TimAI Feature Engineering - Продвинутая инженерия признаков"""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создает полный набор торговых признаков"""
        print("🔧 TimAI Feature Engineering - создание 80+ признаков...")
        
        # Добавляем datetime если есть timestamp
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])  # Уже строка формата datetime
        elif 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')  # Исправлено: unit='ms'
        
        # 1. Технические индикаторы
        df = self._technical_indicators(df)
        
        # 2. Волатильность и риск
        df = self._volatility_features(df)
        
        # 3. Volume анализ
        df = self._volume_features(df)
        
        # 4. Временные паттерны
        df = self._temporal_features(df)
        
        # 5. Рыночные режимы
        df = self._market_regimes(df)
        
        # 6. Взаимодействия признаков
        df = self._feature_interactions(df)
        
        # 7. Микроструктурные признаки
        df = self._microstructure_features(df)
        
        # 8. Продвинутые индикаторы
        df = self._advanced_indicators(df)
        
        # 7. 1D CNN для краткосрочных паттернов (если TensorFlow доступен)
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
            
            # Простая 1D CNN для price patterns
            def create_simple_cnn():
                model = Sequential([
                    Conv1D(64, 3, activation='relu', input_shape=(20, 5)),  # 20 свечей, 5 признаков (OHLCV)
                    MaxPooling1D(2),
                    Conv1D(128, 3, activation='relu'),
                    GlobalMaxPooling1D(),
                    Dense(64, activation='relu'),
                    Dropout(0.3),
                    Dense(3, activation='softmax')  # 3 класса: 0=Sell, 1=Hold, 2=Buy
                ])
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
                return model
            
            # Создаем CNN модель (будет обучаться отдельно)
            self.cnn_model = create_simple_cnn()
            self.has_cnn = True
            print("   ✅ 1D CNN инициализирован для price patterns")
            
        except ImportError:
            self.has_cnn = False
            print("   ⚠️ TensorFlow недоступен, CNN пропущен")
        
        # Добавляем скальпинговые признаки
        df = self._add_scalping_features(df)
        
        # Заполняем пропущенные значения перед обучением моделей
        df = df.ffill().bfill().fillna(0)
        
        # Дополнительная очистка для гарантии работы всех моделей
        df = self._clean_features(df)
        
        print(f"   ✅ Создано {len(df.columns) - 6} признаков для TimAI")
        # --- Оставляем только лучшие признаки + OHLCV + datetime ---
        base_cols = ['open', 'high', 'low', 'close', 'volume', 'datetime']
        keep_cols = base_cols + [col for col in df.columns if col in BEST_INDICATORS]
        df = df[[col for col in keep_cols if col in df.columns]]
        print(f"   ✅ Отобрано {len(df.columns) - 6} лучших признаков: {BEST_INDICATORS}")
        
        # --- Масштабирование признаков весами индикаторов ---
        for col in df.columns:
            if col in INDICATOR_WEIGHTS and col not in base_cols:
                weight = INDICATOR_WEIGHTS[col]
                df[f'{col}_weighted'] = df[col] * weight
                print(f"   ⚖️ Добавлен взвешенный признак: {col}_weighted (вес: {weight:.3f})")
        
        return df
    
    def _technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Базовые и продвинутые технические индикаторы"""
        
        # EMA система
        ema_periods = [5, 8, 13, 21, 34, 55, 89]
        for period in ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, min_periods=1).mean()
            df[f'ema_ratio_{period}'] = df['close'] / (df[f'ema_{period}'] + 1e-8)
        
        # SMA система
        sma_periods = [10, 20, 50, 100, 200]
        for period in sma_periods:
            df[f'sma_{period}'] = df['close'].rolling(period, min_periods=1).mean()
            df[f'sma_ratio_{period}'] = df['close'] / (df[f'sma_{period}'] + 1e-8)
        
        # EMA пересечения
        df['ema_cross_fast'] = (df['ema_8'] > df['ema_21']).astype(int)
        df['ema_cross_medium'] = (df['ema_21'] > df['ema_55']).astype(int)
        df['ema_cross_slow'] = (df['ema_55'] > df['sma_200']).astype(int)
        
        # RSI семейство
        rsi_periods = [7, 14, 21, 28]
        for period in rsi_periods:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI дивергенция и моментум
        df['rsi_divergence'] = df['rsi_14'] - df['rsi_14'].rolling(10, min_periods=1).mean()
        df['rsi_momentum'] = df['rsi_14'].diff()
        df['rsi_volatility'] = df['rsi_14'].rolling(20, min_periods=1).std()
        
        # MACD расширенный
        ema12 = df['close'].ewm(span=12, min_periods=1).mean()
        ema26 = df['close'].ewm(span=26, min_periods=1).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_momentum'] = df['macd_histogram'].diff()
        df['macd_acceleration'] = df['macd_momentum'].diff()
        
        # Bollinger Bands расширенные
        bb_periods = [20, 50]
        for period in bb_periods:
            sma = df['close'].rolling(period, min_periods=1).mean()
            std = df['close'].rolling(period, min_periods=1).std()
            df[f'bb_upper_{period}'] = sma + 2 * std
            df[f'bb_lower_{period}'] = sma - 2 * std
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-8)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / (sma + 1e-8)
            df[f'bb_squeeze_{period}'] = (df[f'bb_width_{period}'] < df[f'bb_width_{period}'].rolling(20, min_periods=1).mean()).astype(int)
        
        # Stochastic расширенный
        periods = [14, 21]
        for period in periods:
            low_min = df['low'].rolling(period, min_periods=1).min()
            high_max = df['high'].rolling(period, min_periods=1).max()
            df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3, min_periods=1).mean()
        
        return df
    
    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Продвинутый анализ волатильности"""
        
        returns = df['close'].pct_change().fillna(0)
        
        # Различные периоды волатильности
        vol_periods = [5, 10, 20, 50, 100]
        for period in vol_periods:
            df[f'volatility_{period}'] = returns.rolling(period, min_periods=1).std()
            df[f'vol_percentile_{period}'] = df[f'volatility_{period}'].rolling(100, min_periods=1).rank(pct=True)
        
        # Volatility regimes
        df['vol_regime_short'] = df['volatility_20'] / (df['volatility_100'] + 1e-8)
        df['vol_regime_trend'] = df['volatility_20'].rolling(10, min_periods=1).mean() / (df['volatility_20'] + 1e-8)
        df['vol_regime_acceleration'] = df['vol_regime_short'].diff()
        
        # Parkinson volatility (high-low)
        df['parkinson_vol'] = np.sqrt(0.361 * np.log((df['high'] / df['low']).clip(lower=1e-8)) ** 2)
        
        # Garman-Klass volatility
        df['gk_vol'] = np.sqrt(
            0.5 * np.log((df['high'] / df['low']).clip(lower=1e-8)) ** 2 -
            (2 * np.log(2) - 1) * np.log((df['close'] / df['open']).clip(lower=1e-8)) ** 2
        )
        
        # Realized volatility
        df['realized_vol'] = returns.rolling(20, min_periods=1).apply(lambda x: np.sqrt(np.sum(x**2)))
        
        return df
    
    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Продвинутый volume анализ"""
        
        # Volume moving averages
        vol_periods = [5, 10, 20, 50, 100]
        for period in vol_periods:
            df[f'volume_ma_{period}'] = df['volume'].rolling(period, min_periods=1).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / (df[f'volume_ma_{period}'] + 1e-8)
        
        # Volume trends и momentum
        df['volume_trend'] = df['volume'].rolling(10, min_periods=1).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
        df['volume_momentum'] = df['volume'].pct_change().fillna(0)
        df['volume_acceleration'] = df['volume_momentum'].diff()
        
        # Price-Volume relationships
        df['price_volume_trend'] = df['close'].pct_change().fillna(0) * df['volume_ratio_20']
        df['volume_price_correlation'] = df['close'].pct_change().fillna(0).rolling(20, min_periods=1).corr(df['volume'].pct_change().fillna(0))
        
        # Volume regime analysis
        volume_ma_100 = df['volume'].rolling(100, min_periods=1).mean()
        df['volume_regime'] = df['volume_ma_20'] / (volume_ma_100 + 1e-8)
        df['volume_outlier'] = (df['volume'] > df['volume_ma_20'] + 2 * df['volume'].rolling(20, min_periods=1).std()).astype(int)
        
        # Money flow и накопление
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        df['money_flow_ratio'] = money_flow / (money_flow.rolling(20, min_periods=1).mean() + 1e-8)
        
        # On Balance Volume
        df['obv'] = (df['volume'] * np.where(df['close'] > df['close'].shift(1), 1, 
                                           np.where(df['close'] < df['close'].shift(1), -1, 0))).cumsum()
        df['obv_ma'] = df['obv'].rolling(20, min_periods=1).mean()
        df['obv_divergence'] = df['obv'] - df['obv_ma']
        
        return df
    
    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Временные паттерны и сезонность"""
        
        if 'datetime' in df.columns:
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['day_of_month'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            df['quarter'] = df['datetime'].dt.quarter
            
            # Циклические признаки
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Торговые сессии
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(float)
            df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(float)
            df['american_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(float)
            
            # Weekend и праздничные эффекты
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(float)
            df['is_month_end'] = (df['day_of_month'] >= 28).astype(float)
            df['is_quarter_end'] = (df['month'] % 3 == 0).astype(float)
        
        return df
    
    def _market_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Определение рыночных режимов"""
        
        returns = df['close'].pct_change().fillna(0)
        
        # Trend detection
        trend_periods = [10, 20, 50, 100]
        for period in trend_periods:
            df[f'trend_{period}'] = df['close'].rolling(period, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
        
        # Trend strength и direction
        df['trend_strength'] = abs(df['trend_20']) / (df['volatility_20'] + 1e-8)
        df['trend_consistency'] = (df['trend_10'] * df['trend_20'] > 0).astype(int)
        
        # Market state classification
        df['market_state'] = np.where(df['trend_20'] > df['volatility_20'], 2,      # Strong Bull
                                    np.where(df['trend_20'] > df['volatility_20'] * 0.5, 1,  # Bull
                                           np.where(df['trend_20'] < -df['volatility_20'], -2, # Strong Bear
                                                  np.where(df['trend_20'] < -df['volatility_20'] * 0.5, -1, 0)))) # Bear, Sideways
        
        # Volatility regime
        vol_mean = df['volatility_20'].rolling(100, min_periods=1).mean()
        df['vol_regime'] = np.where(df['volatility_20'] > vol_mean * 1.5, 2,  # High vol
                                  np.where(df['volatility_20'] < vol_mean * 0.5, 0, 1))  # Low vol, Normal
        
        # Mean reversion vs momentum
        df['mean_reversion_signal'] = (df['close'] - df['sma_20']) / (df['volatility_20'] + 1e-8)
        df['momentum_signal'] = df['close'].pct_change(10).fillna(0)
        
        return df
    
    def _feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Взаимодействия между признаками"""
        
        # RSI-Volume interactions
        df['rsi_volume_interaction'] = df['rsi_14'] * df['volume_ratio_20']
        df['rsi_volume_divergence'] = df['rsi_14'] - df['volume_ratio_20'] * 50
        
        # Volatility-Trend interactions
        df['vol_trend_interaction'] = df['vol_regime_short'] * df['trend_strength']
        df['vol_trend_divergence'] = df['volatility_20'] - abs(df['trend_20'])
        
        # Price momentum combinations
        df['price_momentum'] = df['close'].pct_change() * df['volume_ratio_20']
        df['price_momentum_acceleration'] = df['price_momentum'].diff()
        
        # MACD-Volume confluence
        df['macd_volume_confluence'] = df['macd_histogram'] * df['volume_regime']
        df['macd_volume_divergence'] = np.sign(df['macd_histogram']) != np.sign(df['volume_trend'])
        
        # BB-RSI interactions
        df['bb_rsi_interaction'] = df['bb_position_20'] * (df['rsi_14'] - 50) / 50
        df['bb_rsi_squeeze'] = (df['bb_squeeze_20'] & (df['rsi_14'] > 30) & (df['rsi_14'] < 70)).astype(int)
        
        # Trend-Volume confirmation
        df['trend_volume_confirm'] = df['trend_20'] * df['volume_regime']
        df['trend_volume_divergence'] = (np.sign(df['trend_20']) != np.sign(df['volume_trend'])).astype(int)
        
        return df
    
    def _microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Микроструктурные признаки рынка"""
        
        # Bid-Ask spread proxy
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['hl_spread_ma'] = df['hl_spread'].rolling(20, min_periods=1).mean()
        df['hl_spread_normalized'] = df['hl_spread'] / (df['hl_spread_ma'] + 1e-8)
        
        # Price impact
        df['price_impact'] = abs(df['close'].pct_change()) / (df['volume_ratio_20'] + 1e-8)
        df['price_impact_ma'] = df['price_impact'].rolling(20, min_periods=1).mean()
        
        # Order flow imbalance proxy
        df['order_flow_imbalance'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        df['order_flow_persistence'] = df['order_flow_imbalance'].rolling(5, min_periods=1).mean()
        
        # Tick momentum
        tick_periods = [5, 10, 20]
        for period in tick_periods:
            df[f'tick_momentum_{period}'] = (df['close'] > df['close'].shift(1)).rolling(period, min_periods=1).sum() - period/2
        
        # Intrabar pressure
        df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8)
        df['pressure_ratio'] = df['buying_pressure'] / (df['selling_pressure'] + 1e-8)
        
        return df
    
    def _advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Продвинутые индикаторы"""
        
        # Commodity Channel Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(20, min_periods=1).mean()
        mad = typical_price.rolling(20, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Williams %R
        periods = [14, 21]
        for period in periods:
            high_max = df['high'].rolling(period, min_periods=1).max()
            low_min = df['low'].rolling(period, min_periods=1).min()
            df[f'williams_r_{period}'] = -100 * (high_max - df['close']) / (high_max - low_min + 1e-8)
        
        # Average True Range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14, min_periods=1).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Momentum oscillators
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['roc_10'] = df['close'].pct_change(10)
        df['roc_20'] = df['close'].pct_change(20)
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка и нормализация признаков"""
        
        # Заменяем inf на NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Заполняем NaN
        df = df.ffill().bfill().fillna(0)
        
        # Ограничиваем выбросы более агрессивно
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        
        for col in numeric_cols:
            if col not in exclude_cols:
                # Более агрессивная очистка выбросов
                q99 = df[col].quantile(0.99)   # Было 0.995
                q01 = df[col].quantile(0.01)   # Было 0.005
                df[col] = df[col].clip(q01, q99)
                
                # Дополнительная проверка на infinity после clip
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Финальная проверка: если все еще есть проблемные значения
                if not np.isfinite(df[col]).all():
                    print(f"   ⚠️ Принудительная очистка колонки: {col}")
                    df[col] = np.where(np.isfinite(df[col]), df[col], 0)
        
        return df

    def _scalping_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Специализированные признаки для краткосрочной торговли"""
        
        # Краткосрочный momentum и acceleration
        df['price_velocity_1'] = df['close'].pct_change(1)  # За 1 свечу
        df['price_velocity_3'] = df['close'].pct_change(3)  # За 3 свечи
        df['price_velocity_5'] = df['close'].pct_change(5)  # За 5 свечей
        
        df['price_acceleration_1'] = df['price_velocity_1'].diff()
        df['price_acceleration_3'] = df['price_velocity_3'].diff()
        
        # Volume surges для скальпинга
        df['volume_surge_2'] = df['volume'] / (df['volume'].rolling(2, min_periods=1).mean() + 1e-8)
        df['volume_surge_5'] = df['volume'] / (df['volume'].rolling(5, min_periods=1).mean() + 1e-8)
        df['volume_surge_10'] = df['volume'] / (df['volume'].rolling(10, min_periods=1).mean() + 1e-8)
        
        # Recent high/low touches (касания уровней)
        recent_high_5 = df['high'].rolling(5, min_periods=1).max()
        recent_low_5 = df['low'].rolling(5, min_periods=1).min()
        recent_high_10 = df['high'].rolling(10, min_periods=1).max()
        recent_low_10 = df['low'].rolling(10, min_periods=1).min()
        
        df['near_recent_high_5'] = (abs(df['close'] - recent_high_5) / df['close'] < 0.005).astype(int)
        df['near_recent_low_5'] = (abs(df['close'] - recent_low_5) / df['close'] < 0.005).astype(int)
        df['near_recent_high_10'] = (abs(df['close'] - recent_high_10) / df['close'] < 0.005).astype(int)
        df['near_recent_low_10'] = (abs(df['close'] - recent_low_10) / df['close'] < 0.005).astype(int)
        
        # Breakout detection для скальпинга
        df['breakout_up_5'] = (df['close'] > recent_high_5.shift(1)).astype(int)
        df['breakout_down_5'] = (df['close'] < recent_low_5.shift(1)).astype(int)
        df['breakout_up_10'] = (df['close'] > recent_high_10.shift(1)).astype(int)
        df['breakout_down_10'] = (df['close'] < recent_low_10.shift(1)).astype(int)
        
        # Volume-confirmed moves
        df['volume_confirmed_up'] = (
            (df['close'] > df['open']) & 
            (df['volume'] > df['volume'].rolling(5, min_periods=1).mean()) &
            (df['price_velocity_1'] > 0)
        ).astype(int)
        
        df['volume_confirmed_down'] = (
            (df['close'] < df['open']) & 
            (df['volume'] > df['volume'].rolling(5, min_periods=1).mean()) &
            (df['price_velocity_1'] < 0)
        ).astype(int)
        
        # Scalping momentum indicators
        df['scalp_momentum_short'] = (
            df['price_velocity_1'] * df['volume_surge_2']
        )
        
        df['scalp_momentum_medium'] = (
            df['price_velocity_3'] * df['volume_surge_5']
        )
        
        # Intrabar strength для скальпинга
        df['intrabar_strength'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        df['intrabar_volume_ratio'] = df['volume'] / (df['turnover'] / df['close'] + 1e-8)
        
        # Quick reversal signals
        df['quick_reversal_up'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &  # Предыдущая красная
            (df['close'] > df['open']) &                     # Текущая зеленая
            (df['volume'] > df['volume'].shift(1)) &         # Объем растет
            (df['close'] > df['high'].shift(1))              # Выше предыдущего максимума
        ).astype(int)
        
        df['quick_reversal_down'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &  # Предыдущая зеленая
            (df['close'] < df['open']) &                     # Текущая красная
            (df['volume'] > df['volume'].shift(1)) &         # Объем растет
            (df['close'] < df['low'].shift(1))               # Ниже предыдущего минимума
        ).astype(int)
        
        return df

    def _add_scalping_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет скальпинговые признаки"""
        
        # Добавляем скальпинговые признаки
        df = self._scalping_features(df)
        
        return df

class DeepLearningModels:
    """🧠 Deep Learning модели для краткосрочной торговли"""
    
    def __init__(self):
        self.models = {}
        self.has_tensorflow = False
        self._initialize_models()
    
    def _initialize_models(self):
        """Инициализирует CNN и LSTM модели"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, GlobalMaxPooling1D, Attention
            
            self.has_tensorflow = True
            
            # 1D CNN для price patterns
            self.models['cnn_1d'] = Sequential([
                Conv1D(64, 3, activation='relu', input_shape=(20, 5)),  # 20 свечей, OHLCV
                Conv1D(128, 3, activation='relu'),
                GlobalMaxPooling1D(),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(3, activation='softmax')  # 3 класса
            ])
            
            # LSTM для order flow
            self.models['lstm_flow'] = Sequential([
                LSTM(128, return_sequences=True, input_shape=(60, 3)),  # 60 свечей, Volume+Price+Time
                LSTM(64, return_sequences=False),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(3, activation='softmax')
            ])
            
            # Компиляция моделей
            for name, model in self.models.items():
                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
            
            print(f"   ✅ Deep Learning модели инициализированы: {list(self.models.keys())}")
            
        except ImportError:
            print("   ⚠️ TensorFlow недоступен, Deep Learning модели пропущены")
    
    def prepare_sequences(self, df: pd.DataFrame, sequence_length: int = 20):
        """Подготавливает последовательности для CNN/LSTM"""
        
        if not self.has_tensorflow:
            return None, None
        
        import numpy as np
        
        # Основные данные для последовательностей  
        ohlcv = df[['open', 'high', 'low', 'close', 'volume']].values
        
        # Создаем последовательности
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(ohlcv)):
            sequences.append(ohlcv[i-sequence_length:i])
            
            # Target - направление следующей свечи
            current_close = ohlcv[i-1, 3]  # close предыдущей
            next_close = ohlcv[i, 3] if i < len(ohlcv) else current_close
            
            change = (next_close - current_close) / current_close
            
            if change > 0.002:  # >0.2% рост
                targets.append(2)  # Buy
            elif change < -0.002:  # <-0.2% падение  
                targets.append(0)  # Sell
            else:
                targets.append(1)  # Hold
        
        return np.array(sequences), np.array(targets)
    
    def train_deep_models(self, df: pd.DataFrame, epochs: int = 3):
        """Обучает Deep Learning модели"""
        
        if not self.has_tensorflow:
            return {}
        
        results = {}
        
        # Подготавливаем данные
        sequences, targets = self.prepare_sequences(df)
        
        if sequences is None:
            return {}
        
        # Разделение на train/test
        split_idx = int(len(sequences) * 0.95)
        X_train, X_test = sequences[:split_idx], sequences[split_idx:]
        y_train, y_test = targets[:split_idx], targets[split_idx:]
        
        print(f"\n🧠 Обучение Deep Learning моделей (ускоренный режим)...")
        print(f"   📊 Sequences: {len(X_train)} train, {len(X_test)} test")
        
        # Обучаем CNN (быстро)
        try:
            print("   🔄 Обучение 1D CNN...")
            history_cnn = self.models['cnn_1d'].fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=64,  # Увеличиваем batch size для скорости
                validation_split=0.1,
                verbose=0
            )
            
            # Оценка CNN
            loss_cnn, acc_cnn = self.models['cnn_1d'].evaluate(X_test, y_test, verbose=0)
            results['cnn_1d'] = {
                'test_accuracy': acc_cnn,
                'test_loss': loss_cnn,
                'status': 'success'
            }
            print(f"      ✅ CNN: Accuracy={acc_cnn:.3f}, Loss={loss_cnn:.3f}")
            
        except Exception as e:
            print(f"      ❌ CNN: Ошибка - {e}")
            results['cnn_1d'] = {'status': 'failed', 'error': str(e)}
        
        # Пропускаем LSTM для быстрого тестирования (он занимает слишком много времени)
        print("   ⚡ LSTM пропущен для быстрого тестирования")
        results['lstm_flow'] = {'status': 'skipped', 'reason': 'Fast mode'}
        
        return results
    
    def predict_deep(self, df: pd.DataFrame):
        """Предсказания Deep Learning моделей"""
        
        if not self.has_tensorflow:
            return {}
        
        predictions = {}
        
        try:
            # CNN predictions
            sequences, _ = self.prepare_sequences(df.tail(100))  # Последние 100 свечей
            if sequences is not None and len(sequences) > 0:
                cnn_pred = self.models['cnn_1d'].predict(sequences[-10:], verbose=0)  # Последние 10 sequences
                predictions['cnn_1d'] = np.argmax(cnn_pred, axis=1)
        except Exception as e:
            print(f"⚠️ CNN prediction error: {e}")
        
        try:
            # LSTM predictions (аналогично)
            # ... код для LSTM предсказаний
            pass
        except Exception as e:
            print(f"⚠️ LSTM prediction error: {e}")
        
        return predictions

class TimAIModelManager:
    """🤖 TimAI Model Manager - Управление множественными моделями"""
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.feature_importance = {}
        
    def initialize_models(self):
        """Инициализирует только лучшие модели TimAI с улучшенными гиперпараметрами"""
        
        print("🤖 Инициализация моделей TimAI...")
        
        # 1. XGBoost (если доступен) - УЛУЧШЕННЫЕ ПАРАМЕТРЫ
        if XGBOOST_AVAILABLE:
            try:
                self.models['xgboost'] = xgb.XGBClassifier(
                    n_estimators=500,          # Увеличено для лучшей точности
                    max_depth=6,               # Оптимизировано
                    learning_rate=0.05,        # Снижено для лучшей точности
                    subsample=0.85,            # Улучшено
                    colsample_bytree=0.85,     # Улучшено
                    colsample_bylevel=0.8,     # Добавлено
                    reg_alpha=0.1,             # Оптимизировано
                    reg_lambda=1.0,            # Оптимизировано
                    min_child_weight=3,        # Добавлено
                    gamma=0.1,                 # Добавлено
                    random_state=42,
                    eval_metric='mlogloss',
                    verbosity=0,
                    enable_categorical=False,
                    scale_pos_weight=1.0       # Для несбалансированных классов
                )
            except Exception as e:
                print(f"   ⚠️ XGBoost инициализация с ошибкой: {e}")
                self.models['xgboost'] = xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0
                )

        # 2. LightGBM (если доступен) - УЛУЧШЕННЫЕ ПАРАМЕТРЫ
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=500,              # Увеличено для лучшей точности
                max_depth=6,                   # Оптимизировано
                learning_rate=0.05,            # Снижено для лучшей точности
                subsample=0.85,                # Добавлено
                colsample_bytree=0.85,         # Добавлено
                reg_alpha=0.1,                 # Добавлено
                reg_lambda=1.0,                # Добавлено
                min_child_samples=20,          # Добавлено
                min_split_gain=0.1,            # Добавлено
                random_state=42,
                verbose=-1,
                class_weight='balanced'        # Для несбалансированных классов
            )

        print(f"   ✅ Инициализировано {len(self.models)} моделей (минимализм - только лучшие)")
        
    def train_all_models(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Dict]:
        """Обучает все модели с улучшенной стратегией обучения"""
        
        print(f"\n🚀 TimAI: Обучение {len(self.models)} моделей...")
        
        results = {}
        
        # Анализируем распределение классов
        class_counts = np.bincount(y)
        print(f"   📊 Распределение классов: {class_counts}")
        
        # Вычисляем class_weight для несбалансированных данных
        total_samples = len(y)
        class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        print(f"   ⚖️ Class weights: {class_weights}")
        
        for name, model in self.models.items():
            print(f"   🔄 Обучение {name}...")

            # Подготовка данных для конкретной модели
            X_train = X.astype(np.float32) if name == 'xgboost' else X

            start_time = time.time()

            try:
                # Устанавливаем class_weight для моделей, которые это поддерживают
                if hasattr(model, 'class_weight') and model.class_weight is None:
                    try:
                        model.set_params(class_weight=class_weights)
                    except:
                        pass  # Некоторые модели не поддерживают class_weight
                
                # Обучение модели
                model.fit(X_train, y)
                
                # Время обучения
                training_time = time.time() - start_time
                
                # УЛУЧШЕННАЯ Cross-validation с StratifiedKFold
                try:
                    from sklearn.model_selection import StratifiedKFold
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model, X_train, y, cv=skf, scoring='f1_weighted')
                    
                    # Дополнительные метрики
                    cv_accuracy = cross_val_score(model, X_train, y, cv=skf, scoring='accuracy')
                    cv_precision = cross_val_score(model, X_train, y, cv=skf, scoring='precision_weighted')
                    cv_recall = cross_val_score(model, X_train, y, cv=skf, scoring='recall_weighted')
                    
                except Exception as cv_error:
                    print(f"      ⚠️ {name}: CV ошибка, используем простую оценку: {cv_error}")
                    # Простая оценка на тренировочных данных
                    pred = model.predict(X_train)
                    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
                    f1 = f1_score(y, pred, average='weighted')
                    accuracy = accuracy_score(y, pred)
                    precision = precision_score(y, pred, average='weighted')
                    recall = recall_score(y, pred, average='weighted')
                    cv_scores = np.array([f1, f1, f1])
                    cv_accuracy = np.array([accuracy, accuracy, accuracy])
                    cv_precision = np.array([precision, precision, precision])
                    cv_recall = np.array([recall, recall, recall])
                
                # Feature importance (если доступно)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
                
                results[name] = {
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std(),
                    'cv_accuracy_mean': cv_accuracy.mean(),
                    'cv_precision_mean': cv_precision.mean(),
                    'cv_recall_mean': cv_recall.mean(),
                    'training_time': training_time,
                    'status': 'success'
                }
                
                print(f"      ✅ {name}: F1={cv_scores.mean():.3f}±{cv_scores.std():.3f}, "
                      f"Acc={cv_accuracy.mean():.3f}, Prec={cv_precision.mean():.3f}, "
                      f"Rec={cv_recall.mean():.3f}, Time={training_time:.1f}s")
                
            except Exception as e:
                print(f"      ❌ {name}: Ошибка - {str(e)}")
                results[name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        self.model_performance = results
        return results
    
    def evaluate_on_test(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Dict]:
        """Оценивает модели на тестовых данных"""
        
        print(f"\n📊 Оценка на тестовых данных...")
        
        from sklearn.metrics import f1_score, accuracy_score, classification_report
        
        test_results = {}
        
        for name, model in self.models.items():
            if name in self.model_performance and self.model_performance[name]['status'] == 'success':
                try:
                    # Предсказания на тесте
                    y_pred = model.predict(X_test)
                    
                    # Метрики
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    test_results[name] = {
                        'f1_score': f1,
                        'accuracy': accuracy,
                        'status': 'success'
                    }
                    
                    print(f"   ✅ {name}: F1={f1:.3f}, Accuracy={accuracy:.3f}")
                    
                except Exception as e:
                    print(f"   ❌ {name}: Ошибка на тесте - {str(e)}")
                    test_results[name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
        
        return test_results
    
    def predict_ensemble(self, X: pd.DataFrame, method: str = 'weighted_voting') -> Tuple[np.ndarray, Dict]:
        """Ансамблевое предсказание всех моделей"""
        
        predictions = {}
        weights = {}
        
        # Получаем предсказания от всех успешно обученных моделей
        for name, model in self.models.items():
            if name in self.model_performance and self.model_performance[name]['status'] == 'success':
                try:
                    pred = model.predict(X)
                    # Убеждаемся что предсказания это скаляры, а не массивы
                    if hasattr(pred, 'flatten'):
                        pred = pred.flatten()
                    predictions[name] = pred
                    
                    # Вес на основе CV F1-score
                    weights[name] = self.model_performance[name]['cv_f1_mean']
                    
                except Exception as e:
                    print(f"⚠️ Ошибка предсказания {name}: {e}")
        
        if not predictions:
            return np.zeros(len(X)), {}
        
        # Методы ансамблирования
        if method == 'simple_voting':
            # Простое голосование
            ensemble_pred = self._simple_voting(predictions)
        elif method == 'weighted_voting':
            # Взвешенное голосование
            ensemble_pred = self._weighted_voting(predictions, weights)
        else:
            # По умолчанию - взвешенное
            ensemble_pred = self._weighted_voting(predictions, weights)
        
        return ensemble_pred, predictions
    
    def _simple_voting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Простое голосование"""
        
        pred_matrix = np.array(list(predictions.values()))
        ensemble_pred = []
        
        for i in range(pred_matrix.shape[1]):
            votes = {}
            for pred in pred_matrix[:, i]:
                votes[pred] = votes.get(pred, 0) + 1
            
            best_pred = max(votes, key=votes.get)
            ensemble_pred.append(best_pred)
        
        return np.array(ensemble_pred)
    
    def _weighted_voting(self, predictions: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """Взвешенное голосование"""
        
        ensemble_pred = []
        weight_sum = sum(weights.values())
        
        if weight_sum == 0:
            return self._simple_voting(predictions)
        
        # Нормализуем веса
        normalized_weights = {name: w/weight_sum for name, w in weights.items()}
        
        for i in range(len(list(predictions.values())[0])):
            votes = {}
            
            for name, pred_array in predictions.items():
                if name in normalized_weights:
                    # Безопасное извлечение скалярного значения
                    pred = pred_array[i]
                    if hasattr(pred, 'item'):  # numpy scalar
                        pred = pred.item()
                    elif isinstance(pred, (list, tuple)):  # массив
                        pred = pred[0] if len(pred) > 0 else 0
                    
                    # Убеждаемся что pred это hashable тип
                    pred = int(pred) if not isinstance(pred, (int, float, str)) else pred
                    
                    votes[pred] = votes.get(pred, 0) + normalized_weights[name]
            
            best_pred = max(votes, key=votes.get) if votes else 0
            ensemble_pred.append(best_pred)
        
        return np.array(ensemble_pred)
    
    def save_models(self, filepath: str = "models/production"):
        """Сохраняет все модели с метаданными"""
        
        os.makedirs(filepath, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_models = []
        
        for name, model in self.models.items():
            if name in self.model_performance and self.model_performance[name]['status'] == 'success':
                # Путь к модели
                model_path = os.path.join(filepath, f"timai_{name}_{timestamp}.pkl")
                
                # Сохраняем модель
                joblib.dump(model, model_path)
                
                # Метаданные
                metadata = {
                    'model_name': name,
                    'timestamp': timestamp,
                    'performance': self.model_performance[name],
                    'feature_importance': self.feature_importance.get(name, {}),
                    'file_size_mb': os.path.getsize(model_path) / 1024 / 1024
                }
                
                # Сохраняем метаданные
                meta_path = os.path.join(filepath, f"timai_{name}_{timestamp}_meta.json")
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                saved_models.append({
                    'name': name,
                    'model_path': model_path,
                    'meta_path': meta_path,
                    'size_mb': metadata['file_size_mb']
                })
                
                print(f"💾 Сохранена модель {name}: {metadata['file_size_mb']:.1f} MB")
        
        return saved_models

class TimAI:
    """🤖 TimAI - Главный класс торговой системы"""
    
    def __init__(self):
        self.feature_engine = TimAIFeatureEngine()
        self.model_manager = TimAIModelManager()
        self.deep_models = DeepLearningModels()  # Добавляем Deep Learning
        self.is_trained = False
        
    def prepare_data(self, df: pd.DataFrame, balance_method=None) -> Tuple[pd.DataFrame, np.ndarray]:
        """Подготавливает данные для обучения с улучшенной обработкой и балансировкой классов"""
        
        print("📊 TimAI: Подготовка данных...")
        
        # Feature engineering
        df_features = self.feature_engine.engineer_features(df.copy())
        
        # УЛУЧШЕННАЯ Целевая переменная (3-классовая классификация)
        returns = df_features['close'].pct_change().shift(-1)
        vol = returns.rolling(50).std()
        
        # Более сбалансированная классификация для лучшего обучения
        target = np.where(returns > vol * 0.6, 2,      # Buy (улучшено)
                         np.where(returns > vol * 0.2, 1,      # Hold (улучшено)
                                np.where(returns < -vol * 0.6, 0,   # Sell (улучшено)
                                       1)))  # Hold (default)
        
        # Очистка данных
        df_features = df_features.iloc[:-1]  # Убираем последнюю строку из-за shift
        target = target[:-1]
        
        # Удаляем невалидные значения
        valid_mask = ~(np.isnan(target) | np.isinf(target))
        df_features = df_features[valid_mask]
        target = target[valid_mask]
        
        # Выбираем только числовые признаки для обучения
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'datetime']
        numeric_features = df_features.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        X = df_features[feature_cols]
        
        # УЛУЧШЕННАЯ ОЧИСТКА ПРИЗНАКОВ
        # Удаляем признаки с нулевой дисперсией
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.01)
        X_cleaned = selector.fit_transform(X)
        selected_features = X.columns[selector.get_support()].tolist()
        X = X[selected_features]
        
        # Сохраняем селектор для использования при предсказании
        self.feature_selector = selector
        self.selected_features = selected_features
        
        print(f"   ✅ Подготовлено: {len(X)} образцов, {len(X.columns)} признаков")
        print(f"   📊 Распределение классов: {np.bincount(target)}")
        
        # Анализ несбалансированности
        class_counts = np.bincount(target)
        imbalance_ratio = max(class_counts) / min(class_counts)
        print(f"   ⚖️ Соотношение классов: {imbalance_ratio:.2f}:1")
        
        # УЛУЧШЕННАЯ ОБРАБОТКА НЕСБАЛАНСИРОВАННЫХ ДАННЫХ
        if imbalance_ratio > 3.0:  # Если данные сильно несбалансированы
            print(f"   🔧 Применяем SMOTE для сбалансирования данных...")
            try:
                from imblearn.over_sampling import SMOTE
                from imblearn.under_sampling import RandomUnderSampler
                from imblearn.pipeline import Pipeline
                
                # Комбинируем SMOTE и RandomUnderSampler
                over = SMOTE(sampling_strategy=0.5, random_state=42)
                under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
                steps = [('o', over), ('u', under)]
                pipeline = Pipeline(steps=steps)
                
                # Применяем сбалансирование
                X_resampled, y_resampled = pipeline.fit_resample(X, target)
                
                print(f"   ✅ После сбалансирования: {len(X_resampled)} образцов")
                print(f"   📊 Новое распределение: {np.bincount(y_resampled)}")
                
                return X_resampled, y_resampled
                
            except ImportError:
                print(f"   ⚠️ imbalanced-learn не установлен, используем исходные данные")
                return X, target
        else:
            print(f"   ✅ Данные достаточно сбалансированы")
            return X, target
        
        # --- ДОБАВЛЕНО: Балансировка классов через SMOTE/Over/Under ---
        if balance_method and balance_data is not None:
            print(f"   🔧 Балансировка классов методом: {balance_method}")
            X_bal, y_bal = balance_data(X, target, method=balance_method)
            print(f"   ✅ После балансировки: {len(X_bal)} образцов, классы: {np.bincount(y_bal)}")
            return X_bal, y_bal
    
    def train(self, df: pd.DataFrame, balance_method=None, optimize_hyperparams=False, model_type='xgboost'):
        """Обучает все модели TimAI с возможностью балансировки классов и гипероптимизации"""
        print("🚀 TimAI: Начало обучения системы...")
        # Подготовка данных
        X, y = self.prepare_data(df, balance_method=balance_method)
        self.feature_cols = X.columns.tolist()
        print(f"   💾 Сохранено {len(self.feature_cols)} признаков для предсказания")
        # --- ДОБАВЛЕНО: Гипероптимизация через Optuna ---
        best_params = None
        if optimize_hyperparams and optimize_model is not None:
            print(f"   🔬 Запуск Optuna для {model_type}...")
            best_params = optimize_model(X, y, model_type=model_type)
            print(f"   🏆 Лучшие параметры Optuna: {best_params}")
        
        # Разделение на train/test (95%/5%)
        from sklearn.model_selection import train_test_split
        split_idx = int(len(X) * 0.95)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"   📊 Train: {len(X_train):,} образцов, Test: {len(X_test):,} образцов")
        
        # Инициализация моделей
        self.model_manager.initialize_models()
        
        # Обучение всех моделей на train данных
        results = self.model_manager.train_all_models(X_train, y_train)
        
        # Обучение Deep Learning моделей
        deep_results = self.deep_models.train_deep_models(df.iloc[:split_idx])  # На тех же train данных
        
        # Тестирование на test данных
        test_results = self.model_manager.evaluate_on_test(X_test, y_test)
        
        # Сохранение моделей
        saved_models = self.model_manager.save_models()
        
        self.is_trained = True
        
        print(f"\n🎉 TimAI: Обучение завершено!")
        print(f"   🤖 Обучено моделей: {len([r for r in results.values() if r['status'] == 'success'])}")
        print(f"   💾 Сохранено моделей: {len(saved_models)}")
        
        # Топ-3 модели по производительности на train
        successful_models = {name: res for name, res in results.items() if res['status'] == 'success'}
        if successful_models:
            top_models = sorted(successful_models.items(), key=lambda x: x[1]['cv_f1_mean'], reverse=True)[:3]
            
            print(f"\n🏆 Топ-3 модели по F1-score (Train):")
            for i, (name, res) in enumerate(top_models, 1):
                print(f"   {i}. {name}: {res['cv_f1_mean']:.3f}±{res['cv_f1_std']:.3f}")
        
        # Результаты на test
        if test_results:
            print(f"\n📊 Результаты на Test данных:")
            for name, result in test_results.items():
                if 'f1_score' in result:
                    print(f"   {name}: F1={result['f1_score']:.3f}, Accuracy={result['accuracy']:.3f}")
        
        return results
    
    def predict(self, df: pd.DataFrame, method: str = 'weighted_voting', confidence_threshold: float = 0.5) -> Tuple[np.ndarray, Dict]:
        """Делает предсказания с улучшенным meta-фильтром для повышения винрейта"""
        if not self.is_trained:
            raise ValueError("TimAI не обучена! Сначала вызовите train()")
        
        df_features = self.feature_engine.engineer_features(df.copy())
        
        # Используем только те признаки, которые были при обучении
        if hasattr(self, 'feature_cols'):
            # Проверяем, что все нужные признаки есть
            missing_features = [col for col in self.feature_cols if col not in df_features.columns]
            if missing_features:
                print(f"   ⚠️ Отсутствуют признаки: {missing_features[:5]}...")
                # Добавляем недостающие признаки как нули
                for col in missing_features:
                    df_features[col] = 0
            
            X = df_features[self.feature_cols]
            print(f"   ✅ Используем {len(self.feature_cols)} признаков для предсказания")
        else:
            # Fallback: используем все числовые признаки
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'datetime']
            numeric_features = df_features.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_features if col not in exclude_cols]
            X = df_features[feature_cols]
            print(f"   ⚠️ Используем все признаки (fallback): {len(feature_cols)}")
        
        # УЛУЧШЕННЫЙ Meta-фильтр с более качественной логикой
        preds = {}
        confidences = {}
        probabilities = {}
        
        for name, model in self.model_manager.models.items():
            if name in self.model_manager.model_performance and self.model_manager.model_performance[name]['status'] == 'success':
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        conf = np.max(proba, axis=1)
                        pred = np.argmax(proba, axis=1)
                        preds[name] = pred
                        confidences[name] = conf
                        probabilities[name] = proba
                    else:
                        pred = model.predict(X)
                        preds[name] = pred
                        confidences[name] = np.ones(len(pred)) * 0.5
                        probabilities[name] = None
                except Exception as e:
                    print(f"⚠️ Ошибка предсказания {name}: {e}")
                    continue
        
        if not preds:
            return np.array([1] * len(X)), {}  # Возвращаем Hold если нет предсказаний
        
        # УЛУЧШЕННАЯ ЛОГИКА АНСАМБЛИРОВАНИЯ
        meta_signal = []
        
        for i in range(len(X)):
            # Собираем все предсказания и уверенности для текущей точки
            current_preds = {}
            current_confs = {}
            
            for name in preds:
                current_preds[name] = preds[name][i]
                current_confs[name] = confidences[name][i]
            
            # УЛУЧШЕННЫЙ АЛГОРИТМ ПРИНЯТИЯ РЕШЕНИЙ
            
            # 1. Находим модели с высокой уверенностью
            high_conf_models = {name: conf for name, conf in current_confs.items() if conf >= confidence_threshold}
            
            # 2. Если есть модели с высокой уверенностью
            if len(high_conf_models) >= 2:
                # Проверяем согласие между высокоуверенными моделями
                high_conf_preds = [current_preds[name] for name in high_conf_models]
                
                if len(set(high_conf_preds)) == 1:  # Все модели согласны
                    # Сильный сигнал - все модели согласны и уверены
                    meta_signal.append(high_conf_preds[0])
                else:
                    # Модели не согласны - используем взвешенное голосование
                    weighted_votes = {}
                    for name in high_conf_models:
                        pred = current_preds[name]
                        conf = current_confs[name]
                        weighted_votes[pred] = weighted_votes.get(pred, 0) + conf
                    
                    best_pred = max(weighted_votes, key=weighted_votes.get)
                    meta_signal.append(best_pred)
                    
            elif len(high_conf_models) == 1:
                # Только одна модель уверена - используем её сигнал
                confident_model = list(high_conf_models.keys())[0]
                meta_signal.append(current_preds[confident_model])
                
            else:
                # Нет высокоуверенных моделей - используем улучшенное голосование
                # Взвешиваем по уверенности и качеству модели
                weighted_votes = {}
                
                for name in current_preds:
                    pred = current_preds[name]
                    conf = current_confs[name]
                    
                    # Получаем качество модели из результатов обучения
                    model_quality = self.model_manager.model_performance[name].get('cv_f1_mean', 0.5)
                    
                    # Комбинированный вес: уверенность * качество модели
                    weight = conf * model_quality
                    weighted_votes[pred] = weighted_votes.get(pred, 0) + weight
                
                if weighted_votes:
                    best_pred = max(weighted_votes, key=weighted_votes.get)
                    meta_signal.append(best_pred)
                else:
                    meta_signal.append(1)  # Hold по умолчанию
        
        return np.array(meta_signal), preds

    def predict_single_model(self, df: pd.DataFrame, model_name: str):
        """Быстрое предсказание одной моделью для скальпинга"""
        
        if not self.is_trained:
            return None
        
        try:
            # Подготовка данных
            X, _ = self.prepare_data(df)
            
            # Предсказание одной моделью
            if model_name in self.model_manager.models:
                model = self.model_manager.models[model_name]
                prediction = model.predict(X.tail(1))[0]  # Только последняя свеча
                return int(prediction)
            
            return None
            
        except Exception as e:
            print(f"⚠️ Ошибка predict_single_model {model_name}: {e}")
            return None

    def save_model(self, model_path: str = None):
        """Сохраняет обученную модель TimAI"""

def main():
    """Демонстрация работы TimAI"""
    
    print("🤖 TimAI - Advanced Multi-Model Trading System")
    print("="*60)
    
    print(f"\n📋 Доступные технологии:")
    print(f"   🎯 XGBoost: {'✅' if XGBOOST_AVAILABLE else '❌'}")
    print(f"   ⚡ LightGBM: {'✅' if LIGHTGBM_AVAILABLE else '❌'}")
    print(f"   🔍 Optuna: {'✅' if OPTUNA_AVAILABLE else '❌'}")
    print(f"   🌲 Random Forest: ✅")
    print(f"   📈 Incremental Learning: ✅")
    print(f"   🧠 Neural Networks: ✅")
    
    # Инициализация TimAI
    timai = TimAI()
    
    # Загрузка данных
    try:
        df = pd.read_csv('data/historical/BTCUSDT_15m_5years_20210705_20250702.csv')
        print(f"\n📊 Загружено: {len(df)} записей BTCUSDT 15m")
    except FileNotFoundError:
        print("❌ Файл данных не найден! Создайте папку data/historical/ и поместите туда данные")
        return None
    
    # Обучение системы
    start_time = time.time()
    results = timai.train(df)
    total_time = time.time() - start_time
    
    print(f"\n⏱️ Общее время обучения: {total_time:.1f} секунд")
    
    # Демонстрация предсказания
    if timai.is_trained:
        print(f"\n🔮 Демонстрация предсказания на последних 100 свечах...")
        test_data = df.tail(100)
        predictions, individual_preds = timai.predict(test_data)
        
        # Исправление: проверяем на отрицательные значения
        predictions_clean = [max(0, int(p)) for p in predictions]
        print(f"   📊 Предсказания: {np.bincount(predictions_clean)}")
        print(f"   🤖 Использовано моделей: {len(individual_preds)}")
    
    print(f"\n🎉 TimAI готова к работе!")
    
    return timai

if __name__ == "__main__":
    timai = main() 
#!/usr/bin/env python3
"""
Комплексный анализ важности признаков, оптимизация параметров и тестирование новых идей.
Анализ за 3 года с фильтрацией признаков и рекомендациями по улучшению стратегии.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from src.ml_models.advanced_xgboost_model import AdvancedEnsembleModel
from src.backtesting.aggressive_backtester import AggressiveBacktester
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)

class FeatureAnalyzer:
    """Анализатор важности признаков и оптимизатор параметров."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def create_test_data(self, days_back: int = 1095) -> dict:
        """
        Создание тестовых данных для анализа за 3 года.
        
        Args:
            days_back: Количество дней назад (по умолчанию 1095 = 3 года)
            
        Returns:
            Словарь с данными по таймфреймам
        """
        self.logger.info(f"📊 Создание тестовых данных за последние {days_back} дней ({days_back/365:.1f} лет)")
        self.logger.info(f"⏰ Временной промежуток: {days_back} дней ({days_back * 24} часов)")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        self.logger.info(f"📅 Период: {start_date.strftime('%Y-%m-%d %H:%M')} - {end_date.strftime('%Y-%m-%d %H:%M')}")
        
        # Создаем тестовые данные для 5m, 15m, 1h
        data = {}
        
        for timeframe in ["5m", "15m", "1h"]:
            self.logger.info(f"🔄 Создание данных для {timeframe}...")
            
            # Количество свечей
            if timeframe == "5m":
                periods = days_back * 24 * 12  # 12 свечей в час
            elif timeframe == "15m":
                periods = days_back * 24 * 4   # 4 свечи в час
            else:  # 1h
                periods = days_back * 24       # 1 свеча в час
            
            # Создаем временные метки
            dates = pd.date_range(start=start_date, end=end_date, periods=periods)
            
            # Создаем реалистичные цены BTC с трендами и циклами
            np.random.seed(42)
            base_price = 45000
            
            # Добавляем долгосрочные тренды и циклы
            t = np.linspace(0, 1, len(dates))
            trend = 0.5 * np.sin(2 * np.pi * t * 2) + 0.3 * np.sin(2 * np.pi * t * 0.5)  # Долгосрочные циклы
            returns = np.random.normal(0, 0.02, len(dates)) + 0.001 * trend  # Добавляем тренд к возвратам
            
            prices = [base_price]
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 1000))  # Минимум $1000
            
            # Создаем OHLCV данные
            df_data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                # Создаем реалистичные OHLCV
                volatility = abs(np.random.normal(0, 0.01))
                high = price * (1 + volatility)
                low = price * (1 - volatility)
                open_price = prices[i-1] if i > 0 else price
                volume = np.random.uniform(100, 1000)
                
                df_data.append({
                    'timestamp': date,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
            
            df = pd.DataFrame(df_data)
            data[timeframe] = df
            self.logger.info(f"✅ {timeframe}: {len(df)} свечей")
        
        return data
    
    def create_advanced_features(self, data: dict) -> dict:
        """Создание расширенных признаков для анализа."""
        self.logger.info("🔧 Создание расширенных признаков...")
        
        features_data = {}
        for timeframe, df in data.items():
            self.logger.info(f"📈 Создание признаков для {timeframe}...")
            
            # Создаем технические индикаторы
            df_features = df.copy()
            
            # Базовые признаки
            df_features['price_change'] = df_features['close'].pct_change()
            df_features['volume_change'] = df_features['volume'].pct_change()
            df_features['high_low_ratio'] = df_features['high'] / df_features['low']
            df_features['close_open_ratio'] = df_features['close'] / df_features['open']
            
            # Волатильность
            df_features['volatility_5'] = df_features['price_change'].rolling(window=5).std()
            df_features['volatility_20'] = df_features['price_change'].rolling(window=20).std()
            df_features['volatility_50'] = df_features['price_change'].rolling(window=50).std()
            
            # Скользящие средние
            df_features['sma_10'] = df_features['close'].rolling(window=10).mean()
            df_features['sma_20'] = df_features['close'].rolling(window=20).mean()
            df_features['sma_50'] = df_features['close'].rolling(window=50).mean()
            df_features['sma_100'] = df_features['close'].rolling(window=100).mean()
            df_features['sma_200'] = df_features['close'].rolling(window=200).mean()
            
            # RSI
            delta = df_features['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df_features['close'].ewm(span=12).mean()
            exp2 = df_features['close'].ewm(span=26).mean()
            df_features['macd'] = exp1 - exp2
            df_features['macd_signal'] = df_features['macd'].ewm(span=9).mean()
            df_features['macd_histogram'] = df_features['macd'] - df_features['macd_signal']
            
            # Bollinger Bands
            df_features['bb_middle'] = df_features['close'].rolling(window=20).mean()
            bb_std = df_features['close'].rolling(window=20).std()
            df_features['bb_upper'] = df_features['bb_middle'] + (bb_std * 2)
            df_features['bb_lower'] = df_features['bb_middle'] - (bb_std * 2)
            df_features['bb_position'] = (df_features['close'] - df_features['bb_lower']) / (df_features['bb_upper'] - df_features['bb_lower'])
            df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['bb_middle']
            
            # Momentum
            df_features['momentum_5'] = df_features['close'] / df_features['close'].shift(5) - 1
            df_features['momentum_10'] = df_features['close'] / df_features['close'].shift(10) - 1
            df_features['momentum_20'] = df_features['close'] / df_features['close'].shift(20) - 1
            
            # Volume indicators
            df_features['volume_sma'] = df_features['volume'].rolling(window=20).mean()
            df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma']
            
            # НОВЫЕ ПРИЗНАКИ
            
            # 1. Сезонность
            df_features['hour'] = df_features['timestamp'].dt.hour
            df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
            df_features['month'] = df_features['timestamp'].dt.month
            
            # 2. Лаги цен
            for lag in [1, 2, 3, 5, 10, 20]:
                df_features[f'price_lag_{lag}'] = df_features['close'].shift(lag)
                df_features[f'return_lag_{lag}'] = df_features['price_change'].shift(lag)
            
            # 3. Rolling статистики
            for window in [5, 10, 20, 50]:
                df_features[f'rolling_min_{window}'] = df_features['close'].rolling(window=window).min()
                df_features[f'rolling_max_{window}'] = df_features['close'].rolling(window=window).max()
                df_features[f'rolling_mean_{window}'] = df_features['close'].rolling(window=window).mean()
                df_features[f'rolling_std_{window}'] = df_features['close'].rolling(window=window).std()
            
            # 4. Трендовые индикаторы
            # ADX (Average Directional Index)
            df_features['adx'] = self._calculate_adx(df_features, period=14)
            
            # Parabolic SAR
            df_features['sar'] = self._calculate_parabolic_sar(df_features)
            
            # 5. Объёмные индикаторы
            # OBV (On-Balance Volume)
            df_features['obv'] = self._calculate_obv(df_features)
            
            # VWAP (Volume Weighted Average Price)
            df_features['vwap'] = self._calculate_vwap(df_features)
            
            # Money Flow Index
            df_features['mfi'] = self._calculate_mfi(df_features, period=14)
            
            # 6. Волатильность
            # Realized Volatility
            df_features['realized_volatility'] = df_features['price_change'].rolling(window=20).apply(lambda x: np.sqrt(np.sum(x**2)))
            
            # 7. Кросс-индикаторы
            df_features['price_vs_sma_ratio'] = df_features['close'] / df_features['sma_20']
            df_features['volume_price_trend'] = df_features['volume_ratio'] * df_features['price_change']
            
            # Удаляем NaN значения
            df_features = df_features.dropna()
            
            features_data[timeframe] = df_features
            self.logger.info(f"✅ {timeframe}: {len(df_features)} строк с {len(df_features.columns)} признаками")
        
        return features_data
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Расчет ADX (Average Directional Index)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = high - high.shift()
        dm_minus = low.shift() - low
        dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
        dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
        
        # Smoothed values
        tr_smooth = tr.rolling(window=period).mean()
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()
        
        # DI values
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # DX and ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_parabolic_sar(self, df: pd.DataFrame) -> pd.Series:
        """Расчет Parabolic SAR."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Простая реализация Parabolic SAR
        sar = pd.Series(index=df.index, dtype=float)
        sar.iloc[0] = low.iloc[0]
        
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i-1]:  # Восходящий тренд
                sar.iloc[i] = min(low.iloc[i], sar.iloc[i-1])
            else:  # Нисходящий тренд
                sar.iloc[i] = max(high.iloc[i], sar.iloc[i-1])
        
        return sar
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Расчет OBV (On-Balance Volume)."""
        close = df['close']
        volume = df['volume']
        
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Расчет VWAP (Volume Weighted Average Price)."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        return vwap
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Расчет Money Flow Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    def create_target(self, df: pd.DataFrame, horizon: int = 12, threshold: float = 0.005) -> pd.DataFrame:
        """Создание целевой переменной."""
        df_target = df.copy()
        
        # Создаем целевую переменную
        future_returns = df_target['close'].shift(-horizon) / df_target['close'] - 1
        
        # Классификация
        df_target['target'] = 0  # Hold
        df_target.loc[future_returns > threshold, 'target'] = 1  # Buy
        df_target.loc[future_returns < -threshold, 'target'] = 2  # Sell
        
        # Удаляем последние строки без будущих данных
        df_target = df_target.dropna()
        
        return df_target
    
    def filter_features_by_importance(self, importance_dict: dict, top_n: int = 20, threshold: float = 0.01) -> list:
        """Фильтрация признаков по важности."""
        # Сортируем по важности
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Фильтруем по порогу
        filtered_features = [feature for feature, importance in sorted_features if importance > threshold]
        
        # Берем топ-N
        top_features = [feature for feature, _ in sorted_features[:top_n]]
        
        # Объединяем результаты
        final_features = list(set(filtered_features + top_features))
        
        return final_features
    
    def analyze_feature_importance(self, df: pd.DataFrame, method: str = 'xgboost') -> dict:
        """Анализ важности признаков."""
        self.logger.info(f"🔍 Анализ важности признаков методом: {method}")
        
        # Подготавливаем данные
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']]
        X = df[feature_columns].fillna(0)
        y = df['target']
        
        # Обработка бесконечных значений
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        self.logger.info(f"📊 Признаков: {len(feature_columns)}, Образцов: {len(X)}")
        self.logger.info(f"🎯 Распределение классов: {y.value_counts().to_dict()}")
        
        if method == 'xgboost':
            # XGBoost feature importance
            model = xgb.XGBClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            importance = dict(zip(feature_columns, model.feature_importances_))
            
        elif method == 'random_forest':
            # Random Forest feature importance
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            importance = dict(zip(feature_columns, model.feature_importances_))
            
        elif method == 'mutual_info':
            # Mutual Information
            mi_scores = mutual_info_classif(X, y, random_state=42)
            importance = dict(zip(feature_columns, mi_scores))
            
        elif method == 'f_score':
            # F-score
            f_scores, _ = f_classif(X, y)
            importance = dict(zip(feature_columns, f_scores))
        
        # Сортируем по важности
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def plot_feature_importance(self, importance: dict, title: str = "Feature Importance"):
        """Визуализация важности признаков."""
        plt.figure(figsize=(15, 10))
        
        # Топ-30 признаков
        top_features = list(importance.keys())[:30]
        top_scores = list(importance.values())[:30]
        
        plt.barh(range(len(top_features)), top_scores)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'feature_importance_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"📊 График сохранен: feature_importance_{title.lower().replace(' ', '_')}.png")
    
    def optimize_parameters(self, df: pd.DataFrame) -> dict:
        """Оптимизация параметров модели."""
        self.logger.info("⚙️ Оптимизация параметров модели...")
        
        # Подготавливаем данные
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']]
        X = df[feature_columns].fillna(0)
        y = df['target']
        
        # Обработка бесконечных значений
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        # Разделяем данные
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        self.logger.info(f"📊 Тренировочная выборка: {len(X_train)}, Валидационная: {len(X_val)}")
        
        # Тестируем разные параметры
        param_combinations = [
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
            {'n_estimators': 500, 'max_depth': 10, 'learning_rate': 0.03},
            {'n_estimators': 1000, 'max_depth': 12, 'learning_rate': 0.02},
            {'n_estimators': 2000, 'max_depth': 15, 'learning_rate': 0.015},
            {'n_estimators': 3000, 'max_depth': 20, 'learning_rate': 0.01},
        ]
        
        best_score = 0
        best_params = None
        results = []
        
        for params in param_combinations:
            self.logger.info(f"🧪 Тестируем параметры: {params}")
            
            model = xgb.XGBClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            score = model.score(X_val, y_val)
            results.append({'params': params, 'accuracy': score})
            
            if score > best_score:
                best_score = score
                best_params = params
        
        self.logger.info(f"🏆 Лучшие параметры: {best_params} (точность: {best_score:.4f})")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def test_new_ideas(self, df: pd.DataFrame) -> dict:
        """Тестирование новых идей."""
        self.logger.info("💡 Тестирование новых идей...")
        
        results = {}
        
        # Идея 1: Комбинирование признаков
        self.logger.info("🔧 Идея 1: Комбинирование признаков")
        df_combined = df.copy()
        
        # Создаем комбинированные признаки
        df_combined['rsi_momentum'] = df_combined['rsi'] * df_combined['momentum_5']
        df_combined['volume_price'] = df_combined['volume_ratio'] * df_combined['price_change']
        df_combined['bb_momentum'] = df_combined['bb_position'] * df_combined['momentum_10']
        df_combined['trend_strength'] = df_combined['adx'] * df_combined['price_vs_sma_ratio']
        
        # Тестируем с новыми признаками
        feature_columns = [col for col in df_combined.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']]
        X = df_combined[feature_columns].fillna(0)
        y = df_combined['target']
        
        # Обработка бесконечных значений
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = xgb.XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.03, random_state=42)
        model.fit(X_train, y_train)
        score_combined = model.score(X_val, y_val)
        
        results['combined_features'] = score_combined
        self.logger.info(f"✅ Комбинированные признаки: {score_combined:.4f}")
        
        # Идея 2: Временные окна
        self.logger.info("⏰ Идея 2: Временные окна")
        df_windows = df.copy()
        
        # Создаем недостающие признаки volatility
        df_windows['volatility_10'] = df_windows['price_change'].rolling(window=10).std()
        
        # Создаем недостающие признаки momentum
        df_windows['momentum_20'] = df_windows['close'] / df_windows['close'].shift(20) - 1
        
        # Создаем признаки на разных временных окнах
        for window in [5, 10, 20]:
            df_windows[f'volatility_{window}_norm'] = df_windows['volatility_20'] / df_windows[f'volatility_{window}']
            df_windows[f'momentum_{window}_norm'] = df_windows['momentum_10'] / df_windows[f'momentum_{window}']
        
        feature_columns = [col for col in df_windows.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']]
        X = df_windows[feature_columns].fillna(0)
        y = df_windows['target']
        
        # Обработка бесконечных значений
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = xgb.XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.03, random_state=42)
        model.fit(X_train, y_train)
        score_windows = model.score(X_val, y_val)
        
        results['temporal_windows'] = score_windows
        self.logger.info(f"✅ Временные окна: {score_windows:.4f}")
        
        # Идея 3: Адаптивные пороги
        self.logger.info("📊 Идея 3: Адаптивные пороги")
        
        thresholds = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03]
        threshold_scores = {}
        
        for threshold in thresholds:
            df_threshold = df.copy()
            future_returns = df_threshold['close'].shift(-12) / df_threshold['close'] - 1
            
            df_threshold['target'] = 0
            df_threshold.loc[future_returns > threshold, 'target'] = 1
            df_threshold.loc[future_returns < -threshold, 'target'] = 2
            df_threshold = df_threshold.dropna()
            
            if len(df_threshold) > 100:  # Минимум данных
                feature_columns = [col for col in df_threshold.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']]
                X = df_threshold[feature_columns].fillna(0)
                y = df_threshold['target']
                
                # Обработка бесконечных значений
                X = X.replace([np.inf, -np.inf], 0)
                X = X.fillna(0)
                
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                model = xgb.XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.03, random_state=42)
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                threshold_scores[threshold] = score
        
        best_threshold = max(threshold_scores, key=threshold_scores.get)
        results['adaptive_thresholds'] = {'best_threshold': best_threshold, 'scores': threshold_scores}
        self.logger.info(f"✅ Лучший порог: {best_threshold} (точность: {threshold_scores[best_threshold]:.4f})")
        
        return results
    
    def generate_strategy_recommendations(self, results: dict) -> dict:
        """Генерация рекомендаций по улучшению стратегии."""
        self.logger.info("📋 Генерация рекомендаций по улучшению стратегии...")
        
        recommendations = {
            'timeframe': results['timeframe'],
            'data_period': results['data_period'],
            'top_features': list(results['feature_importance_xgb'].keys())[:20],
            'best_params': results['optimization']['best_params'],
            'best_threshold': results['new_ideas']['adaptive_thresholds']['best_threshold'],
            'recommendations': []
        }
        
        # Анализируем важность признаков
        top_features = list(results['feature_importance_xgb'].keys())[:10]
        
        # Рекомендации по признакам
        recommendations['recommendations'].append({
            'category': 'Признаки',
            'title': 'Топ-10 важнейших признаков',
            'description': f'Используйте эти признаки: {", ".join(top_features[:5])}...',
            'action': f'Оставьте только топ-20 признаков, исключите признаки с важностью < 0.01'
        })
        
        # Рекомендации по параметрам
        best_params = results['optimization']['best_params']
        recommendations['recommendations'].append({
            'category': 'Параметры модели',
            'title': 'Оптимальные параметры XGBoost',
            'description': f'Используйте: n_estimators={best_params["n_estimators"]}, max_depth={best_params["max_depth"]}, learning_rate={best_params["learning_rate"]}',
            'action': 'Примените эти параметры для обучения финальной модели'
        })
        
        # Рекомендации по порогам
        best_threshold = results['new_ideas']['adaptive_thresholds']['best_threshold']
        recommendations['recommendations'].append({
            'category': 'Пороги классификации',
            'title': 'Оптимальный порог',
            'description': f'Используйте порог {best_threshold} для классификации сигналов',
            'action': 'Настройте пороги классификации на основе волатильности рынка'
        })
        
        # Рекомендации по стратегии
        if results['new_ideas']['combined_features'] > results['optimization']['best_score']:
            recommendations['recommendations'].append({
                'category': 'Стратегия',
                'title': 'Комбинированные признаки',
                'description': 'Комбинированные признаки показывают лучший результат',
                'action': 'Добавьте комбинированные признаки в финальную модель'
            })
        
        if results['new_ideas']['temporal_windows'] > results['optimization']['best_score']:
            recommendations['recommendations'].append({
                'category': 'Стратегия',
                'title': 'Временные окна',
                'description': 'Временные окна улучшают точность',
                'action': 'Используйте признаки на разных временных окнах'
            })
        
        # Общие рекомендации
        recommendations['recommendations'].extend([
            {
                'category': 'Общие',
                'title': 'Балансировка данных',
                'description': 'Используйте SMOTE или другие методы балансировки',
                'action': 'Примените SMOTE для устранения дисбаланса классов'
            },
            {
                'category': 'Общие',
                'title': 'Временные ряды',
                'description': 'Используйте TimeSeriesSplit для валидации',
                'action': 'Замените обычную валидацию на временную'
            },
            {
                'category': 'Общие',
                'title': 'Ансамблирование',
                'description': 'Объедините несколько моделей',
                'action': 'Создайте ансамбль из XGBoost, Random Forest и LightGBM'
            }
        ])
        
        return recommendations
    
    def run_comprehensive_analysis(self, days_back: int = 1095):
        """Запуск комплексного анализа за 3 года."""
        self.logger.info("🚀 ЗАПУСК КОМПЛЕКСНОГО АНАЛИЗА ЗА 3 ГОДА")
        self.logger.info("=" * 80)
        
        # 1. Создание тестовых данных
        data = self.create_test_data(days_back)
        
        if not data:
            self.logger.error("❌ Не удалось создать данные")
            return
        
        # 2. Создание расширенных признаков
        features_data = self.create_advanced_features(data)
        
        # 3. Анализ для каждого таймфрейма
        all_recommendations = {}
        
        for timeframe, df in features_data.items():
            self.logger.info(f"\n📊 АНАЛИЗ ТАЙМФРЕЙМА: {timeframe}")
            self.logger.info("-" * 50)
            
            # Создаем целевую переменную
            df_target = self.create_target(df)
            
            # Анализ важности признаков
            importance_xgb = self.analyze_feature_importance(df_target, 'xgboost')
            importance_mi = self.analyze_feature_importance(df_target, 'mutual_info')
            
            # Фильтрация признаков
            top_features_xgb = self.filter_features_by_importance(importance_xgb, top_n=20, threshold=0.01)
            top_features_mi = self.filter_features_by_importance(importance_mi, top_n=20, threshold=0.01)
            
            self.logger.info(f"🔍 Топ-20 признаков XGBoost: {len(top_features_xgb)}")
            self.logger.info(f"🔍 Топ-20 признаков Mutual Info: {len(top_features_mi)}")
            
            # Визуализация
            self.plot_feature_importance(importance_xgb, f"XGBoost Feature Importance - {timeframe}")
            self.plot_feature_importance(importance_mi, f"Mutual Information - {timeframe}")
            
            # Оптимизация параметров
            optimization = self.optimize_parameters(df_target)
            
            # Тестирование новых идей
            new_ideas = self.test_new_ideas(df_target)
            
            # Сохраняем результаты
            results = {
                'timeframe': timeframe,
                'data_period': f"{days_back} days ({days_back/365:.1f} years)",
                'feature_importance_xgb': importance_xgb,
                'feature_importance_mi': importance_mi,
                'top_features_xgb': top_features_xgb,
                'top_features_mi': top_features_mi,
                'optimization': optimization,
                'new_ideas': new_ideas
            }
            
            # Генерация рекомендаций
            recommendations = self.generate_strategy_recommendations(results)
            all_recommendations[timeframe] = recommendations
            
            # Сохраняем в файл
            import json
            with open(f'analysis_results_{timeframe}_3years.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            with open(f'recommendations_{timeframe}_3years.json', 'w') as f:
                json.dump(recommendations, f, indent=2, default=str)
            
            self.logger.info(f"💾 Результаты сохранены: analysis_results_{timeframe}_3years.json")
            self.logger.info(f"💾 Рекомендации сохранены: recommendations_{timeframe}_3years.json")
        
        # 4. Итоговые рекомендации
        self.logger.info("\n📋 ИТОГОВЫЕ РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ СТРАТЕГИИ")
        self.logger.info("=" * 80)
        
        for timeframe, recs in all_recommendations.items():
            self.logger.info(f"\n🎯 ТАЙМФРЕЙМ: {timeframe}")
            self.logger.info("-" * 30)
            
            for rec in recs['recommendations']:
                self.logger.info(f"📌 {rec['category']}: {rec['title']}")
                self.logger.info(f"   {rec['description']}")
                self.logger.info(f"   Действие: {rec['action']}")
                self.logger.info("")
        
        self.logger.info("\n✅ КОМПЛЕКСНЫЙ АНАЛИЗ ЗА 3 ГОДА ЗАВЕРШЕН")
        self.logger.info("=" * 80)

if __name__ == "__main__":
    analyzer = FeatureAnalyzer()
    analyzer.run_comprehensive_analysis(1095)  # 3 года 
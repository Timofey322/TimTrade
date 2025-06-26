#!/usr/bin/env python3
"""
Модуль для создания продвинутых фичей на основе on-chain метрик и sentiment анализа
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sys
import os
from loguru import logger

# Добавляем путь к корневой директории
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_collection.onchain_collector import OnChainCollector
from src.data_collection.sentiment_collector import SentimentCollector

class AdvancedFeatureEngine:
    """
    Класс для создания продвинутых фичей с использованием:
    - On-chain метрик (whale movements, exchange flows, network activity)
    - Sentiment анализа (Fear & Greed, social sentiment, news sentiment)
    - Cross-correlation анализа между ценой и fundamentals
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация движка продвинутых фичей.
        
        Args:
            config: Конфигурация с API ключами и параметрами
        """
        self.config = config or {}
        self.logger = logger.bind(name="AdvancedFeatureEngine")
        
        # Инициализируем коллекторы данных
        self.onchain_collector = OnChainCollector(config)
        self.sentiment_collector = SentimentCollector(config)
        
        # Параметры
        self.lookback_periods = self.config.get('lookback_periods', [7, 14, 30])
        self.correlation_windows = self.config.get('correlation_windows', [14, 30, 60])
        
    def create_onchain_features(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """
        Создает фичи на основе on-chain метрик.
        
        Args:
            symbol: Символ криптовалюты
            days: Количество дней для анализа
        
        Returns:
            DataFrame с on-chain фичами
        """
        try:
            self.logger.info(f"Создание on-chain фичей для {symbol}")
            
            # Получаем все on-chain данные
            onchain_data = self.onchain_collector.get_comprehensive_onchain_data(symbol, days)
            
            if not onchain_data:
                return pd.DataFrame()
            
            features_list = []
            
            # Обрабатываем каждый тип on-chain данных
            for data_type, df in onchain_data.items():
                if df.empty:
                    continue
                
                # Создаем фичи для конкретного типа данных
                type_features = self._create_onchain_type_features(df, data_type)
                features_list.append(type_features)
            
            # Объединяем все фичи
            if features_list:
                combined_features = features_list[0]
                for features in features_list[1:]:
                    combined_features = combined_features.merge(
                        features, on='timestamp', how='outer'
                    )
                
                # Добавляем composite on-chain score
                combined_features['onchain_composite_score'] = self._calculate_onchain_composite_score(
                    onchain_data
                )
                
                # Заполняем пропущенные значения
                combined_features = combined_features.fillna(method='ffill').fillna(0)
                
                return combined_features
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Ошибка создания on-chain фичей: {e}")
            return pd.DataFrame()
    
    def create_sentiment_features(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """
        Создает фичи на основе sentiment анализа.
        
        Args:
            symbol: Символ криптовалюты
            days: Количество дней для анализа
        
        Returns:
            DataFrame с sentiment фичами
        """
        try:
            self.logger.info(f"Создание sentiment фичей для {symbol}")
            
            # Получаем различные типы sentiment данных
            fear_greed_df = self.sentiment_collector.get_fear_greed_index(days)
            social_df = self.sentiment_collector.get_social_sentiment(symbol, days)
            news_df = self.sentiment_collector.get_news_sentiment(symbol, days)
            bybit_data = self.sentiment_collector.get_bybit_opportunities()
            funding_df = self.sentiment_collector.get_funding_rates_sentiment()
            
            features_list = []
            
            # Fear & Greed фичи
            if not fear_greed_df.empty:
                fg_features = self._create_fear_greed_features(fear_greed_df)
                features_list.append(fg_features)
            
            # Social sentiment фичи
            if not social_df.empty:
                social_features = self._create_social_features(social_df)
                features_list.append(social_features)
            
            # News sentiment фичи
            if not news_df.empty:
                news_features = self._create_news_features(news_df)
                features_list.append(news_features)
            
            # Funding rates фичи
            if not funding_df.empty:
                funding_features = self._create_funding_features(funding_df, days)
                features_list.append(funding_features)
            
            # Объединяем фичи
            if features_list:
                combined_features = features_list[0]
                for features in features_list[1:]:
                    combined_features = combined_features.merge(
                        features, on='timestamp', how='outer'
                    )
                
                # Добавляем composite sentiment score
                combined_features['sentiment_composite_score'] = self.sentiment_collector.calculate_composite_sentiment_score(
                    fear_greed_df, social_df, news_df, bybit_data, funding_df
                )
                
                # Добавляем Bybit opportunities фичи
                combined_features = self._add_bybit_features(combined_features, bybit_data)
                
                # Заполняем пропущенные значения
                combined_features = combined_features.fillna(method='ffill').fillna(0.5)
                
                return combined_features
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Ошибка создания sentiment фичей: {e}")
            return pd.DataFrame()
    
    def create_cross_correlation_features(self, price_df: pd.DataFrame, 
                                        onchain_df: pd.DataFrame,
                                        sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает фичи на основе кросс-корреляции между ценой и fundamentals.
        
        Args:
            price_df: DataFrame с ценовыми данными
            onchain_df: DataFrame с on-chain фичами
            sentiment_df: DataFrame с sentiment фичами
        
        Returns:
            DataFrame с correlation фичами
        """
        try:
            self.logger.info("Создание cross-correlation фичей")
            
            correlation_features = []
            
            # Объединяем все данные по timestamp
            merged_df = price_df.copy()
            
            if not onchain_df.empty:
                merged_df = merged_df.merge(onchain_df, on='timestamp', how='left')
            
            if not sentiment_df.empty:
                merged_df = merged_df.merge(sentiment_df, on='timestamp', how='left')
            
            # Заполняем пропуски
            merged_df = merged_df.fillna(method='ffill').fillna(0)
            
            # Вычисляем корреляции для разных окон
            for window in self.correlation_windows:
                window_features = self._calculate_correlation_features(merged_df, window)
                correlation_features.append(window_features)
            
            # Объединяем correlation фичи
            if correlation_features:
                final_corr_df = correlation_features[0]
                for corr_df in correlation_features[1:]:
                    final_corr_df = final_corr_df.merge(corr_df, on='timestamp', how='outer')
                
                return final_corr_df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Ошибка создания correlation фичей: {e}")
            return pd.DataFrame()
    
    def create_comprehensive_features(self, symbol: str, price_df: pd.DataFrame, 
                                    days: int = 60) -> pd.DataFrame:
        """
        Создает полный набор продвинутых фичей.
        
        Args:
            symbol: Символ криптовалюты
            price_df: DataFrame с ценовыми данными
            days: Количество дней для анализа
        
        Returns:
            DataFrame с полным набором продвинутых фичей
        """
        try:
            self.logger.info(f"Создание полного набора продвинутых фичей для {symbol}")
            
            # Создаем различные типы фичей
            onchain_features = self.create_onchain_features(symbol, days)
            sentiment_features = self.create_sentiment_features(symbol, days)
            
            # Начинаем с ценовых данных
            comprehensive_df = price_df.copy()
            
            # Добавляем on-chain фичи
            if not onchain_features.empty:
                comprehensive_df = comprehensive_df.merge(
                    onchain_features, on='timestamp', how='left'
                )
            
            # Добавляем sentiment фичи
            if not sentiment_features.empty:
                comprehensive_df = comprehensive_df.merge(
                    sentiment_features, on='timestamp', how='left'
                )
            
            # Создаем correlation фичи
            correlation_features = self.create_cross_correlation_features(
                price_df, onchain_features, sentiment_features
            )
            
            if not correlation_features.empty:
                comprehensive_df = comprehensive_df.merge(
                    correlation_features, on='timestamp', how='left'
                )
            
            # Добавляем advanced composite features
            comprehensive_df = self._add_advanced_composite_features(comprehensive_df)
            
            # Заполняем пропуски
            comprehensive_df = comprehensive_df.fillna(method='ffill').fillna(0)
            
            self.logger.info(f"Создано {comprehensive_df.shape[1]} продвинутых фичей")
            
            return comprehensive_df
            
        except Exception as e:
            self.logger.error(f"Ошибка создания полного набора фичей: {e}")
            return price_df.copy()
    
    # Вспомогательные методы
    
    def _create_onchain_type_features(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Создает фичи для конкретного типа on-chain данных."""
        try:
            features = df[['timestamp']].copy()
            
            # Получаем числовые колонки
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in df.columns:
                    col_data = df[col]
                    prefix = f"{data_type}_{col}"
                    
                    # Базовые фичи
                    features[f"{prefix}_value"] = col_data
                    
                    # Rolling статистики
                    for period in self.lookback_periods:
                        features[f"{prefix}_ma_{period}"] = col_data.rolling(period).mean()
                        features[f"{prefix}_std_{period}"] = col_data.rolling(period).std()
                        features[f"{prefix}_min_{period}"] = col_data.rolling(period).min()
                        features[f"{prefix}_max_{period}"] = col_data.rolling(period).max()
                    
                    # Momentum фичи
                    features[f"{prefix}_roc_7"] = col_data.pct_change(7)
                    features[f"{prefix}_roc_14"] = col_data.pct_change(14)
                    
                    # Z-score
                    features[f"{prefix}_zscore"] = (col_data - col_data.rolling(30).mean()) / col_data.rolling(30).std()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Ошибка создания {data_type} фичей: {e}")
            return pd.DataFrame({'timestamp': df['timestamp']})
    
    def _create_fear_greed_features(self, fg_df: pd.DataFrame) -> pd.DataFrame:
        """Создает фичи на основе Fear & Greed Index."""
        try:
            features = fg_df[['timestamp']].copy()
            
            # Базовые фичи
            features['fear_greed_value'] = fg_df['fear_greed_normalized']
            features['fear_greed_sentiment'] = fg_df['sentiment_numeric']
            
            # Rolling статистики
            for period in self.lookback_periods:
                features[f'fear_greed_ma_{period}'] = fg_df['fear_greed_normalized'].rolling(period).mean()
                features[f'fear_greed_volatility_{period}'] = fg_df['fear_greed_normalized'].rolling(period).std()
            
            # Экстремальные значения
            features['fear_greed_extreme_fear'] = (fg_df['value'] <= 25).astype(int)
            features['fear_greed_extreme_greed'] = (fg_df['value'] >= 75).astype(int)
            
            # Momentum
            features['fear_greed_momentum'] = fg_df['fear_greed_normalized'].diff(7)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Ошибка создания Fear & Greed фичей: {e}")
            return pd.DataFrame({'timestamp': fg_df['timestamp']})
    
    def _create_social_features(self, social_df: pd.DataFrame) -> pd.DataFrame:
        """Создает фичи на основе social sentiment."""
        try:
            features = social_df[['timestamp']].copy()
            
            # Базовые фичи
            features['social_sentiment'] = social_df['social_sentiment']
            features['social_mention_volume'] = social_df['mention_volume']
            features['social_sentiment_volatility'] = social_df['sentiment_volatility']
            
            # Rolling статистики
            for period in [3, 7, 14]:
                features[f'social_sentiment_ma_{period}'] = social_df['social_sentiment'].rolling(period).mean()
                features[f'social_volume_ma_{period}'] = social_df['mention_volume'].rolling(period).mean()
            
            # Аномалии в упоминаниях
            features['social_volume_spike'] = social_df['high_mention_day']
            
            return features
            
        except Exception as e:
            self.logger.error(f"Ошибка создания social фичей: {e}")
            return pd.DataFrame({'timestamp': social_df['timestamp']})
    
    def _create_news_features(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Создает фичи на основе news sentiment."""
        try:
            features = news_df[['timestamp']].copy()
            
            # Базовые фичи
            features['news_sentiment'] = news_df['avg_sentiment']
            features['news_count'] = news_df['news_count']
            features['news_sentiment_std'] = news_df['sentiment_std']
            
            # Rolling статистики
            for period in [3, 7]:
                features[f'news_sentiment_ma_{period}'] = news_df['avg_sentiment'].rolling(period).mean()
                features[f'news_count_ma_{period}'] = news_df['news_count'].rolling(period).mean()
            
            # Экстремальные новости
            features['news_very_positive'] = (news_df['avg_sentiment'] > 0.5).astype(int)
            features['news_very_negative'] = (news_df['avg_sentiment'] < -0.5).astype(int)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Ошибка создания news фичей: {e}")
            return pd.DataFrame({'timestamp': news_df['timestamp']})
    
    def _create_funding_features(self, funding_df: pd.DataFrame, days: int) -> pd.DataFrame:
        """Создает фичи на основе funding rates."""
        try:
            # Создаем daily timestamps для соответствия другим фичам
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            # Расширяем funding data на все дни (т.к. funding rates обновляются реже)
            features = pd.DataFrame({'timestamp': dates})
            
            # Берем последние значения funding rates для всех дней
            if not funding_df.empty:
                latest_funding = funding_df.iloc[-1]
                
                features['funding_btc_sentiment'] = latest_funding.get('sentiment', 0)
                features['funding_btc_extremeness'] = latest_funding.get('extremeness', 0)
                
                # Категории funding sentiment
                features['funding_very_bullish'] = (latest_funding.get('sentiment', 0) > 0.5).astype(int)
                features['funding_very_bearish'] = (latest_funding.get('sentiment', 0) < -0.5).astype(int)
            else:
                features['funding_btc_sentiment'] = 0
                features['funding_btc_extremeness'] = 0
                features['funding_very_bullish'] = 0
                features['funding_very_bearish'] = 0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Ошибка создания funding фичей: {e}")
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            return pd.DataFrame({'timestamp': dates})
    
    def _add_bybit_features(self, features_df: pd.DataFrame, bybit_data: Dict) -> pd.DataFrame:
        """Добавляет фичи на основе Bybit opportunities."""
        try:
            if not bybit_data:
                return features_df
            
            # Общий market sentiment от Bybit
            features_df['bybit_market_sentiment'] = bybit_data.get('market_sentiment', 0.5)
            
            # Количество hot sectors
            hot_sectors = bybit_data.get('hot_sectors', [])
            features_df['bybit_hot_sectors_count'] = len(hot_sectors)
            
            # Анализ trending coins
            trending_coins = bybit_data.get('trending_coins', [])
            if trending_coins:
                positive_trending = sum(1 for coin in trending_coins if coin.get('sentiment', 0) > 0)
                features_df['bybit_positive_trending_ratio'] = positive_trending / len(trending_coins)
            else:
                features_df['bybit_positive_trending_ratio'] = 0.5
            
            # Gainers vs Losers ratio
            gainers_losers = bybit_data.get('gainers_losers', {})
            gainers_count = len(gainers_losers.get('gainers', []))
            losers_count = len(gainers_losers.get('losers', []))
            
            if gainers_count + losers_count > 0:
                features_df['bybit_gainers_ratio'] = gainers_count / (gainers_count + losers_count)
            else:
                features_df['bybit_gainers_ratio'] = 0.5
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления Bybit фичей: {e}")
            return features_df
    
    def _calculate_correlation_features(self, merged_df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Вычисляет correlation фичи для заданного окна."""
        try:
            features = merged_df[['timestamp']].copy()
            
            # Предполагаем, что в merged_df есть колонка 'close' для цены
            if 'close' not in merged_df.columns:
                return features
            
            price_col = 'close'
            
            # Correlation с on-chain метриками
            onchain_cols = [col for col in merged_df.columns if 'whale' in col or 'exchange' in col or 'network' in col]
            for col in onchain_cols[:5]:  # Ограничиваем количество для производительности
                if col in merged_df.columns:
                    correlation = merged_df[price_col].rolling(window).corr(merged_df[col])
                    features[f'corr_{col}_{window}d'] = correlation
            
            # Correlation с sentiment метриками
            sentiment_cols = [col for col in merged_df.columns if 'sentiment' in col or 'fear_greed' in col]
            for col in sentiment_cols[:5]:
                if col in merged_df.columns:
                    correlation = merged_df[price_col].rolling(window).corr(merged_df[col])
                    features[f'corr_{col}_{window}d'] = correlation
            
            return features
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета correlation фичей: {e}")
            return pd.DataFrame({'timestamp': merged_df['timestamp']})
    
    def _calculate_onchain_composite_score(self, onchain_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """Рассчитывает composite score для on-chain метрик."""
        try:
            # Получаем последние значения key метрик
            scores = []
            
            if 'whale_movements' in onchain_data and not onchain_data['whale_movements'].empty:
                whale_score = onchain_data['whale_movements']['whale_activity_spike'].iloc[-7:].mean()
                scores.append(whale_score * 0.3)
            
            if 'exchange_flows' in onchain_data and not onchain_data['exchange_flows'].empty:
                flow_score = onchain_data['exchange_flows']['bullish_flow'].iloc[-7:].mean()
                scores.append(flow_score * 0.4)
            
            if 'network_activity' in onchain_data and not onchain_data['network_activity'].empty:
                network_score = onchain_data['network_activity']['high_activity'].iloc[-7:].mean()
                scores.append(network_score * 0.2)
            
            if 'hodler_behavior' in onchain_data and not onchain_data['hodler_behavior'].empty:
                hodler_score = onchain_data['hodler_behavior']['accumulation_phase'].iloc[-7:].mean()
                scores.append(hodler_score * 0.1)
            
            composite_score = sum(scores) if scores else 0.5
            
            # Возвращаем как Series с одним значением для каждой строки
            return pd.Series([composite_score] * len(next(iter(onchain_data.values()))))
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета onchain composite score: {e}")
            return pd.Series([0.5])
    
    def _add_advanced_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет продвинутые composite фичи."""
        try:
            # Fundamental strength score (on-chain + sentiment)
            onchain_score = df.get('onchain_composite_score', 0.5)
            sentiment_score = df.get('sentiment_composite_score', 0.5)
            
            df['fundamental_strength'] = (onchain_score * 0.6 + sentiment_score * 0.4)
            
            # Market regime classification
            fear_greed = df.get('fear_greed_value', 0.5)
            social_sentiment = df.get('social_sentiment', 0.5)
            
            df['market_regime_bullish'] = ((fear_greed > 0.6) & (social_sentiment > 0.6)).astype(int)
            df['market_regime_bearish'] = ((fear_greed < 0.4) & (social_sentiment < 0.4)).astype(int)
            df['market_regime_neutral'] = (~df['market_regime_bullish'] & ~df['market_regime_bearish']).astype(int)
            
            # Divergence signals
            if 'corr_sentiment_composite_score_14d' in df.columns:
                df['price_sentiment_divergence'] = (df['corr_sentiment_composite_score_14d'] < 0.2).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления composite фичей: {e}")
            return df 
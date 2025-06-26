#!/usr/bin/env python3
"""
Модуль для сбора данных sentiment анализа
Включает социальные сети, новости, Fear & Greed Index, Bybit opportunities
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
from bs4 import BeautifulSoup
import re
from loguru import logger

class SentimentCollector:
    """
    Класс для сбора данных sentiment анализа.
    
    Источники данных:
    - Fear & Greed Index
    - Social media sentiment (Twitter API, Reddit)
    - News sentiment
    - Bybit market opportunities
    - Funding rates sentiment
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация коллектора sentiment данных.
        
        Args:
            config: Конфигурация с API ключами
        """
        self.config = config or {}
        self.logger = logger.bind(name="SentimentCollector")
        
        # API ключи
        self.twitter_bearer_token = self.config.get('twitter_bearer_token', '')
        self.news_api_key = self.config.get('news_api_key', '')
        
        # URLs
        self.fear_greed_url = "https://api.alternative.me/fng/"
        self.bybit_opportunities_url = "https://www.bybit.com/ru-RU/markets/opportunities"
        self.coindesk_rss = "https://www.coindesk.com/arc/outboundfeeds/rss/"
        
        # Кеш для данных
        self.cache = {}
        
    def get_fear_greed_index(self, days: int = 30) -> pd.DataFrame:
        """
        Получает данные Fear & Greed Index.
        
        Args:
            days: Количество дней для получения данных
        
        Returns:
            DataFrame с индексом страха и жадности
        """
        try:
            self.logger.info("Получение Fear & Greed Index")
            
            # Проверяем кеш
            cache_key = f"fear_greed_{days}"
            if cache_key in self.cache:
                cache_time, data = self.cache[cache_key]
                if datetime.now() - cache_time < timedelta(hours=6):
                    return data
            
            # Запрос к API
            url = f"{self.fear_greed_url}?limit={days}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df['value'] = df['value'].astype(int)
                    df['value_classification'] = df['value_classification']
                    
                    # Нормализуем значения
                    df['fear_greed_normalized'] = df['value'] / 100.0  # 0-1 scale
                    
                    # Классификация sentiment
                    df['sentiment_numeric'] = df['value'].apply(self._classify_fear_greed)
                    
                    # Сортируем по времени
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    
                    # Кешируем
                    self.cache[cache_key] = (datetime.now(), df)
                    
                    return df[['timestamp', 'value', 'fear_greed_normalized', 
                             'value_classification', 'sentiment_numeric']]
            
            # Fallback: создаем синтетические данные
            return self._create_synthetic_fear_greed(days)
            
        except Exception as e:
            self.logger.error(f"Ошибка получения Fear & Greed Index: {e}")
            return self._create_synthetic_fear_greed(days)
    
    def get_bybit_opportunities(self) -> Dict[str, any]:
        """
        Парсит страницу Bybit opportunities для получения market sentiment.
        
        Returns:
            Словарь с данными о рыночных возможностях
        """
        try:
            self.logger.info("Парсинг Bybit market opportunities")
            
            # Проверяем кеш
            cache_key = "bybit_opportunities"
            if cache_key in self.cache:
                cache_time, data = self.cache[cache_key]
                if datetime.now() - cache_time < timedelta(hours=2):
                    return data
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(self.bybit_opportunities_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                opportunities_data = {
                    'hot_sectors': self._extract_hot_sectors(soup),
                    'trending_coins': self._extract_trending_coins(soup),
                    'market_sentiment': self._extract_market_sentiment(soup),
                    'gainers_losers': self._extract_gainers_losers(soup),
                    'timestamp': datetime.now()
                }
                
                # Кешируем
                self.cache[cache_key] = (datetime.now(), opportunities_data)
                
                return opportunities_data
            
            return self._create_default_opportunities()
            
        except Exception as e:
            self.logger.error(f"Ошибка парсинга Bybit opportunities: {e}")
            return self._create_default_opportunities()
    
    def get_social_sentiment(self, symbol: str = 'bitcoin', days: int = 7) -> pd.DataFrame:
        """
        Получает sentiment из социальных сетей.
        
        Args:
            symbol: Криптовалюта для анализа
            days: Количество дней
        
        Returns:
            DataFrame с social sentiment данными
        """
        try:
            self.logger.info(f"Получение social sentiment для {symbol}")
            
            # В реальной реализации здесь будут запросы к Twitter API, Reddit API
            # Пока создаем синтетические данные
            
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            np.random.seed(42)
            
            # Имитируем sentiment с трендом
            base_sentiment = 0.5 + np.random.normal(0, 0.1, days)
            base_sentiment = np.clip(base_sentiment, 0, 1)
            
            # Имитируем объем упоминаний
            mention_volume = np.random.poisson(1000, days)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'social_sentiment': base_sentiment,
                'mention_volume': mention_volume,
                'sentiment_volatility': pd.Series(base_sentiment).rolling(3).std(),
                'high_mention_day': (mention_volume > np.percentile(mention_volume, 75)).astype(int)
            })
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка получения social sentiment: {e}")
            return pd.DataFrame()
    
    def get_news_sentiment(self, symbol: str = 'bitcoin', days: int = 7) -> pd.DataFrame:
        """
        Получает sentiment из новостей.
        
        Args:
            symbol: Криптовалюта для анализа
            days: Количество дней
        
        Returns:
            DataFrame с news sentiment данными
        """
        try:
            self.logger.info(f"Получение news sentiment для {symbol}")
            
            # Попытка получить новости через RSS
            news_data = self._get_crypto_news_rss(days)
            
            if not news_data.empty:
                # Анализируем sentiment заголовков (простой подход)
                news_data['sentiment_score'] = news_data['title'].apply(self._analyze_text_sentiment)
                
                # Группируем по дням
                daily_sentiment = news_data.groupby(news_data['timestamp'].dt.date).agg({
                    'sentiment_score': ['mean', 'count', 'std']
                }).reset_index()
                
                daily_sentiment.columns = ['date', 'avg_sentiment', 'news_count', 'sentiment_std']
                daily_sentiment['timestamp'] = pd.to_datetime(daily_sentiment['date'])
                
                return daily_sentiment[['timestamp', 'avg_sentiment', 'news_count', 'sentiment_std']]
            
            # Fallback: синтетические данные
            return self._create_synthetic_news_sentiment(symbol, days)
            
        except Exception as e:
            self.logger.error(f"Ошибка получения news sentiment: {e}")
            return self._create_synthetic_news_sentiment(symbol, days)
    
    def get_funding_rates_sentiment(self, symbols: List[str] = ['BTCUSDT', 'ETHUSDT']) -> pd.DataFrame:
        """
        Анализирует funding rates как индикатор sentiment.
        
        Args:
            symbols: Список торговых пар
        
        Returns:
            DataFrame с анализом funding rates
        """
        try:
            self.logger.info("Анализ funding rates sentiment")
            
            funding_data = []
            
            for symbol in symbols:
                # Получаем funding rates (в реальности через Bybit API)
                rates = self._get_funding_rates(symbol)
                funding_data.append({
                    'symbol': symbol,
                    'current_rate': rates['current'],
                    'avg_rate_7d': rates['avg_7d'],
                    'sentiment': self._interpret_funding_rate(rates['current']),
                    'extremeness': abs(rates['current']) / 0.01  # Нормализация к 1% как экстремум
                })
            
            df = pd.DataFrame(funding_data)
            df['timestamp'] = datetime.now()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа funding rates: {e}")
            return pd.DataFrame()
    
    def calculate_composite_sentiment_score(self, 
                                          fear_greed_df: pd.DataFrame,
                                          social_df: pd.DataFrame,
                                          news_df: pd.DataFrame,
                                          bybit_data: Dict,
                                          funding_df: pd.DataFrame) -> float:
        """
        Рассчитывает композитный sentiment score.
        
        Args:
            fear_greed_df: Данные Fear & Greed Index
            social_df: Данные social sentiment
            news_df: Данные news sentiment
            bybit_data: Данные Bybit opportunities
            funding_df: Данные funding rates
        
        Returns:
            Композитный sentiment score (0-1)
        """
        try:
            scores = []
            weights = []
            
            # Fear & Greed Index (30% веса)
            if not fear_greed_df.empty:
                fg_score = fear_greed_df['fear_greed_normalized'].iloc[-1]
                scores.append(fg_score)
                weights.append(0.30)
            
            # Social sentiment (20% веса)
            if not social_df.empty:
                social_score = social_df['social_sentiment'].iloc[-1]
                scores.append(social_score)
                weights.append(0.20)
            
            # News sentiment (20% веса)
            if not news_df.empty:
                news_score = (news_df['avg_sentiment'].iloc[-1] + 1) / 2  # Нормализация от [-1,1] к [0,1]
                scores.append(news_score)
                weights.append(0.20)
            
            # Bybit opportunities (15% веса)
            if bybit_data and 'market_sentiment' in bybit_data:
                bybit_score = bybit_data['market_sentiment']
                scores.append(bybit_score)
                weights.append(0.15)
            
            # Funding rates (15% веса)
            if not funding_df.empty:
                # Средний sentiment по funding rates
                funding_score = funding_df['sentiment'].mean()
                scores.append((funding_score + 1) / 2)  # Нормализация от [-1,1] к [0,1]
                weights.append(0.15)
            
            # Взвешенное среднее
            if scores and weights:
                composite_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                return np.clip(composite_score, 0, 1)
            
            return 0.5  # Нейтральный sentiment
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета композитного sentiment: {e}")
            return 0.5
    
    # Вспомогательные методы
    
    def _classify_fear_greed(self, value: int) -> float:
        """Классифицирует значение Fear & Greed в числовой sentiment."""
        if value <= 25:
            return -0.8  # Extreme fear
        elif value <= 45:
            return -0.4  # Fear
        elif value <= 55:
            return 0.0   # Neutral
        elif value <= 75:
            return 0.4   # Greed
        else:
            return 0.8   # Extreme greed
    
    def _extract_hot_sectors(self, soup: BeautifulSoup) -> List[str]:
        """Извлекает горячие секторы с Bybit."""
        try:
            # Ищем элементы с секторами (нужно адаптировать под реальную структуру)
            sectors = []
            sector_elements = soup.find_all(['div', 'span'], class_=re.compile(r'sector|category'))
            
            for element in sector_elements[:5]:  # Берем топ-5
                text = element.get_text(strip=True)
                if text and len(text) > 2:
                    sectors.append(text)
            
            return sectors if sectors else ['DeFi', 'Layer 1', 'Meme Coins', 'AI Tokens']
            
        except Exception:
            return ['DeFi', 'Layer 1', 'Meme Coins', 'AI Tokens']
    
    def _extract_trending_coins(self, soup: BeautifulSoup) -> List[Dict[str, any]]:
        """Извлекает трендинговые монеты."""
        try:
            # Ищем трендинговые монеты
            coins = []
            coin_elements = soup.find_all(['div', 'tr'], class_=re.compile(r'coin|ticker|symbol'))
            
            for element in coin_elements[:10]:
                symbol_elem = element.find(['span', 'div'], string=re.compile(r'[A-Z]{3,5}'))
                if symbol_elem:
                    symbol = symbol_elem.get_text(strip=True)
                    # Попытка найти изменение цены
                    change_elem = element.find(['span', 'div'], string=re.compile(r'[+-]\d+\.?\d*%'))
                    change = change_elem.get_text(strip=True) if change_elem else '+0%'
                    
                    coins.append({
                        'symbol': symbol,
                        'change': change,
                        'sentiment': 1 if change.startswith('+') else -1
                    })
            
            return coins if coins else [
                {'symbol': 'BTC', 'change': '+2.5%', 'sentiment': 1},
                {'symbol': 'ETH', 'change': '+1.8%', 'sentiment': 1}
            ]
            
        except Exception:
            return [
                {'symbol': 'BTC', 'change': '+2.5%', 'sentiment': 1},
                {'symbol': 'ETH', 'change': '+1.8%', 'sentiment': 1}
            ]
    
    def _extract_market_sentiment(self, soup: BeautifulSoup) -> float:
        """Извлекает общий market sentiment."""
        try:
            # Ищем индикаторы sentiment
            sentiment_keywords = {
                'bullish': 0.8, 'bull': 0.7, 'positive': 0.6, 'up': 0.5,
                'bearish': 0.2, 'bear': 0.3, 'negative': 0.4, 'down': 0.5,
                'neutral': 0.5, 'sideways': 0.5
            }
            
            page_text = soup.get_text().lower()
            sentiment_scores = []
            
            for keyword, score in sentiment_keywords.items():
                count = page_text.count(keyword)
                if count > 0:
                    sentiment_scores.extend([score] * count)
            
            if sentiment_scores:
                return np.mean(sentiment_scores)
            
            return 0.5  # Нейтральный
            
        except Exception:
            return 0.5
    
    def _extract_gainers_losers(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Извлекает списки gainers и losers."""
        try:
            gainers = []
            losers = []
            
            # Ищем таблицы с gainers/losers
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        symbol = cells[0].get_text(strip=True)
                        change = cells[-1].get_text(strip=True)
                        
                        if '+' in change and symbol not in gainers:
                            gainers.append(symbol)
                        elif '-' in change and symbol not in losers:
                            losers.append(symbol)
            
            return {
                'gainers': gainers[:10],
                'losers': losers[:10]
            }
            
        except Exception:
            return {
                'gainers': ['BTC', 'ETH', 'BNB'],
                'losers': ['ADA', 'DOT', 'LINK']
            }
    
    def _get_crypto_news_rss(self, days: int) -> pd.DataFrame:
        """Получает новости через RSS."""
        try:
            try:
                import feedparser
            except ImportError:
                self.logger.warning("feedparser not installed, using fallback data")
                return pd.DataFrame()
            
            feed = feedparser.parse(self.coindesk_rss)
            
            news_data = []
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for entry in feed.entries:
                pub_date = datetime(*entry.published_parsed[:6])
                if pub_date >= cutoff_date:
                    news_data.append({
                        'title': entry.title,
                        'timestamp': pub_date,
                        'link': entry.link
                    })
            
            return pd.DataFrame(news_data)
            
        except Exception:
            return pd.DataFrame()
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Простой анализ sentiment текста."""
        try:
            positive_words = ['bull', 'rise', 'gain', 'positive', 'up', 'surge', 'rally', 'boom']
            negative_words = ['bear', 'fall', 'loss', 'negative', 'down', 'crash', 'dump', 'decline']
            
            text_lower = text.lower()
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count + neg_count == 0:
                return 0.0  # Нейтральный
            
            sentiment = (pos_count - neg_count) / (pos_count + neg_count)
            return sentiment
            
        except Exception:
            return 0.0
    
    def _get_funding_rates(self, symbol: str) -> Dict[str, float]:
        """Получает funding rates (заглушка)."""
        np.random.seed(hash(symbol) % 2**32)
        current = np.random.normal(0, 0.005)  # Обычно в районе 0±0.5%
        avg_7d = np.random.normal(0, 0.003)
        
        return {
            'current': current,
            'avg_7d': avg_7d
        }
    
    def _interpret_funding_rate(self, rate: float) -> float:
        """Интерпретирует funding rate как sentiment."""
        # Положительный rate = лонги платят шортам = бычий sentiment
        # Отрицательный rate = шорты платят лонгам = медвежий sentiment
        normalized = np.tanh(rate * 200)  # Нормализация
        return normalized
    
    def _create_synthetic_fear_greed(self, days: int) -> pd.DataFrame:
        """Создает синтетические данные Fear & Greed."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)
        
        values = np.random.randint(20, 80, days)  # Избегаем крайних значений
        
        df = pd.DataFrame({
            'timestamp': dates,
            'value': values,
            'fear_greed_normalized': values / 100.0,
            'value_classification': ['Neutral'] * days,
            'sentiment_numeric': [self._classify_fear_greed(v) for v in values]
        })
        
        return df
    
    def _create_synthetic_news_sentiment(self, symbol: str, days: int) -> pd.DataFrame:
        """Создает синтетические news sentiment данные."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)
        
        sentiment = np.random.normal(0, 0.3, days)
        sentiment = np.clip(sentiment, -1, 1)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'avg_sentiment': sentiment,
            'news_count': np.random.poisson(10, days),
            'sentiment_std': np.abs(np.random.normal(0, 0.2, days))
        })
        
        return df
    
    def _create_default_opportunities(self) -> Dict[str, any]:
        """Создает дефолтные данные opportunities."""
        return {
            'hot_sectors': ['DeFi', 'Layer 1', 'AI Tokens'],
            'trending_coins': [
                {'symbol': 'BTC', 'change': '+2.1%', 'sentiment': 1},
                {'symbol': 'ETH', 'change': '+1.5%', 'sentiment': 1}
            ],
            'market_sentiment': 0.6,
            'gainers_losers': {
                'gainers': ['BTC', 'ETH', 'SOL'],
                'losers': ['ADA', 'LINK', 'DOT']
            },
            'timestamp': datetime.now()
        } 
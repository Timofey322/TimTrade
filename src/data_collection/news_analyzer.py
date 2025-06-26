"""
Модуль для анализа новостей и рыночных настроений.

Этот модуль предоставляет функциональность для:
- Сбора новостей с Bybit и других источников
- Анализа настроений рынка
- Интеграции новостных данных в торговую стратегию
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger
import time
import json
from bs4 import BeautifulSoup
import re


class NewsAnalyzer:
    """
    Класс для анализа новостей и рыночных настроений.
    
    Поддерживает:
    - Сбор новостей с Bybit
    - Анализ настроений
    - Интеграцию с торговой системой
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация анализатора новостей.
        
        Args:
            config: Конфигурация анализатора
        """
        self.config = config or {}
        self.logger = logger.bind(name="NewsAnalyzer")
        
        # Настройки по умолчанию
        self.default_config = {
            'bybit_url': 'https://www.bybit.com/ru-RU/markets/opportunities',
            'update_interval': 300,  # 5 минут
            'sentiment_keywords': {
                'positive': ['bullish', 'rally', 'surge', 'gain', 'up', 'positive', 'growth'],
                'negative': ['bearish', 'crash', 'drop', 'fall', 'down', 'negative', 'decline'],
                'neutral': ['stable', 'sideways', 'consolidation', 'range']
            },
            'crypto_keywords': ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency'],
            'max_news_age_hours': 24
        }
        
        # Обновление конфигурации
        if config:
            self.default_config.update(config)
        
        # Кэш новостей
        self.news_cache = []
        self.last_update = None
    
    def fetch_bybit_opportunities(self) -> List[Dict]:
        """
        Сбор данных с Bybit Opportunities.
        
        Returns:
            Список возможностей с Bybit
        """
        try:
            self.logger.info("Сбор данных с Bybit Opportunities")
            
            # Заголовки для имитации браузера
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Запрос к Bybit
            response = requests.get(self.default_config['bybit_url'], headers=headers, timeout=10)
            response.raise_for_status()
            
            # Парсинг HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            opportunities = []
            
            # Поиск элементов с возможностями (адаптируйте селекторы под реальную структуру)
            opportunity_elements = soup.find_all('div', class_=re.compile(r'opportunity|market|trading'))
            
            for element in opportunity_elements:
                try:
                    # Извлекаем информацию о возможности
                    opportunity = self._parse_opportunity_element(element)
                    if opportunity:
                        opportunities.append(opportunity)
                except Exception as e:
                    self.logger.debug(f"Ошибка парсинга элемента: {e}")
                    continue
            
            self.logger.info(f"Собрано {len(opportunities)} возможностей с Bybit")
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Ошибка сбора данных с Bybit: {e}")
            return []
    
    def _parse_opportunity_element(self, element) -> Optional[Dict]:
        """
        Парсинг элемента возможности.
        
        Args:
            element: HTML элемент
            
        Returns:
            Словарь с данными возможности или None
        """
        try:
            # Извлекаем текст
            text = element.get_text(strip=True)
            
            # Ищем ключевые слова
            opportunity = {
                'source': 'bybit',
                'timestamp': datetime.now().isoformat(),
                'text': text,
                'sentiment': self._analyze_sentiment(text),
                'crypto_related': self._is_crypto_related(text),
                'confidence': self._calculate_confidence(text)
            }
            
            # Извлекаем дополнительные данные если возможно
            title_elem = element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if title_elem:
                opportunity['title'] = title_elem.get_text(strip=True)
            
            # Ищем цены и проценты
            price_pattern = r'\$[\d,]+\.?\d*'
            percent_pattern = r'[\+\-]?\d+\.?\d*%'
            
            prices = re.findall(price_pattern, text)
            percents = re.findall(percent_pattern, text)
            
            if prices:
                opportunity['prices'] = prices
            if percents:
                opportunity['percents'] = percents
            
            return opportunity
            
        except Exception as e:
            self.logger.debug(f"Ошибка парсинга элемента: {e}")
            return None
    
    def _analyze_sentiment(self, text: str) -> str:
        """
        Анализ настроения текста.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Настроение: 'positive', 'negative', 'neutral'
        """
        try:
            text_lower = text.lower()
            
            positive_count = sum(1 for keyword in self.default_config['sentiment_keywords']['positive'] 
                               if keyword in text_lower)
            negative_count = sum(1 for keyword in self.default_config['sentiment_keywords']['negative'] 
                               if keyword in text_lower)
            neutral_count = sum(1 for keyword in self.default_config['sentiment_keywords']['neutral'] 
                              if keyword in text_lower)
            
            # Определяем настроение
            if positive_count > negative_count and positive_count > neutral_count:
                return 'positive'
            elif negative_count > positive_count and negative_count > neutral_count:
                return 'negative'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.debug(f"Ошибка анализа настроения: {e}")
            return 'neutral'
    
    def _is_crypto_related(self, text: str) -> bool:
        """
        Проверка, связан ли текст с криптовалютами.
        
        Args:
            text: Текст для проверки
            
        Returns:
            True если текст связан с криптовалютами
        """
        try:
            text_lower = text.lower()
            return any(keyword in text_lower for keyword in self.default_config['crypto_keywords'])
        except Exception:
            return False
    
    def _calculate_confidence(self, text: str) -> float:
        """
        Расчет уверенности в анализе.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Уверенность от 0 до 1
        """
        try:
            # Простая эвристика на основе длины текста и количества ключевых слов
            text_length = len(text)
            keyword_count = sum(1 for keyword_list in self.default_config['sentiment_keywords'].values() 
                              for keyword in keyword_list if keyword in text.lower())
            
            # Нормализуем уверенность
            confidence = min(keyword_count / max(text_length / 100, 1), 1.0)
            return round(confidence, 2)
            
        except Exception:
            return 0.5
    
    def get_market_sentiment(self, symbol: str = 'BTC') -> Dict:
        """
        Получение общего настроения рынка.
        
        Args:
            symbol: Символ для анализа
            
        Returns:
            Словарь с настроением рынка
        """
        try:
            # Обновляем кэш если нужно
            if (self.last_update is None or 
                datetime.now() - self.last_update > timedelta(seconds=self.default_config['update_interval'])):
                self._update_news_cache()
            
            # Фильтруем новости по символу и времени
            relevant_news = [
                news for news in self.news_cache
                if (symbol.lower() in news['text'].lower() or news['crypto_related']) and
                self._is_recent_news(news['timestamp'])
            ]
            
            if not relevant_news:
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.5,
                    'news_count': 0,
                    'last_update': self.last_update.isoformat() if self.last_update else None
                }
            
            # Анализируем настроения
            sentiments = [news['sentiment'] for news in relevant_news]
            confidences = [news['confidence'] for news in relevant_news]
            
            # Подсчитываем настроения
            sentiment_counts = {
                'positive': sentiments.count('positive'),
                'negative': sentiments.count('negative'),
                'neutral': sentiments.count('neutral')
            }
            
            # Определяем доминирующее настроение
            dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            
            # Средняя уверенность
            avg_confidence = np.mean(confidences)
            
            return {
                'sentiment': dominant_sentiment,
                'confidence': round(avg_confidence, 2),
                'news_count': len(relevant_news),
                'sentiment_distribution': sentiment_counts,
                'last_update': self.last_update.isoformat() if self.last_update else None
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка получения настроения рынка: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'news_count': 0,
                'last_update': None
            }
    
    def _update_news_cache(self):
        """Обновление кэша новостей."""
        try:
            self.logger.info("Обновление кэша новостей")
            
            # Собираем новые данные
            new_opportunities = self.fetch_bybit_opportunities()
            
            # Добавляем в кэш
            self.news_cache.extend(new_opportunities)
            
            # Очищаем старые новости
            cutoff_time = datetime.now() - timedelta(hours=self.default_config['max_news_age_hours'])
            self.news_cache = [
                news for news in self.news_cache
                if self._is_recent_news(news['timestamp'], cutoff_time)
            ]
            
            self.last_update = datetime.now()
            self.logger.info(f"Кэш обновлен. Всего новостей: {len(self.news_cache)}")
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления кэша новостей: {e}")
    
    def _is_recent_news(self, timestamp_str: str, cutoff_time: datetime = None) -> bool:
        """
        Проверка, является ли новость недавней.
        
        Args:
            timestamp_str: Временная метка новости
            cutoff_time: Время отсечения
            
        Returns:
            True если новость недавняя
        """
        try:
            if cutoff_time is None:
                cutoff_time = datetime.now() - timedelta(hours=self.default_config['max_news_age_hours'])
            
            news_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return news_time > cutoff_time
            
        except Exception:
            return False
    
    def get_sentiment_signal(self, symbol: str = 'BTC') -> Dict:
        """
        Получение торгового сигнала на основе настроений.
        
        Args:
            symbol: Символ для анализа
            
        Returns:
            Словарь с торговым сигналом
        """
        try:
            sentiment_data = self.get_market_sentiment(symbol)
            
            # Преобразуем настроение в торговый сигнал
            sentiment_signal = {
                'signal': 0,  # 0 = hold, 1 = buy, -1 = sell
                'confidence': sentiment_data['confidence'],
                'sentiment': sentiment_data['sentiment'],
                'news_count': sentiment_data['news_count']
            }
            
            # Логика сигналов на основе настроения
            if sentiment_data['sentiment'] == 'positive' and sentiment_data['confidence'] > 0.6:
                sentiment_signal['signal'] = 1  # Buy
            elif sentiment_data['sentiment'] == 'negative' and sentiment_data['confidence'] > 0.6:
                sentiment_signal['signal'] = -1  # Sell
            
            return sentiment_signal
            
        except Exception as e:
            self.logger.error(f"Ошибка получения сигнала настроения: {e}")
            return {
                'signal': 0,
                'confidence': 0.5,
                'sentiment': 'neutral',
                'news_count': 0
            } 
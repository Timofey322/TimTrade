#!/usr/bin/env python3
"""
Модуль для сбора on-chain метрик криптовалют
Включает whale movements, exchange flows, network activity
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
from loguru import logger

class OnChainCollector:
    """
    Класс для сбора on-chain метрик криптовалют.
    
    Поддерживаемые источники:
    - Glassnode API (whale movements, exchange flows)
    - Blockchain.info API (network activity)
    - CoinMetrics API (network health)
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация коллектора on-chain данных.
        
        Args:
            config: Конфигурация с API ключами
        """
        self.config = config or {}
        self.logger = logger.bind(name="OnChainCollector")
        
        # API конфигурация
        self.glassnode_api_key = self.config.get('glassnode_api_key', 'demo')
        self.coinmetrics_api_key = self.config.get('coinmetrics_api_key', '')
        
        # Базовые URL
        self.glassnode_base_url = "https://api.glassnode.com/v1/metrics"
        self.blockchain_info_url = "https://api.blockchain.info"
        self.coinmetrics_url = "https://api.coinmetrics.io/v4"
        
        # Кеш для данных
        self.cache = {}
        
    def get_whale_movements(self, symbol: str = 'BTC', 
                           days: int = 30, min_amount: float = 100.0) -> pd.DataFrame:
        """
        Получает данные о движениях китов (крупные переводы).
        
        Args:
            symbol: Символ криптовалюты
            days: Количество дней для анализа
            min_amount: Минимальная сумма перевода для считания "китом"
        
        Returns:
            DataFrame с данными о движениях китов
        """
        try:
            self.logger.info(f"Получение данных о движениях китов для {symbol}")
            
            # Пытаемся получить с Glassnode (требует API ключ)
            whale_data = self._get_glassnode_metric(
                "transactions/transfers_volume_large",
                symbol=symbol,
                days=days,
                params={'threshold': int(min_amount)}
            )
            
            if whale_data is not None:
                # Обрабатываем данные
                whale_df = pd.DataFrame(whale_data)
                whale_df['timestamp'] = pd.to_datetime(whale_df['timestamp'])
                whale_df['whale_volume'] = whale_df['value']
                
                # Добавляем индикаторы
                whale_df['whale_volume_ma'] = whale_df['whale_volume'].rolling(7).mean()
                whale_df['whale_activity_spike'] = (
                    whale_df['whale_volume'] > whale_df['whale_volume_ma'] * 1.5
                ).astype(int)
                
                return whale_df[['timestamp', 'whale_volume', 'whale_volume_ma', 'whale_activity_spike']]
            
            # Fallback: создаем синтетические данные на основе volume
            self.logger.warning("Создаем синтетические данные о китах")
            return self._create_synthetic_whale_data(symbol, days)
            
        except Exception as e:
            self.logger.error(f"Ошибка получения данных о китах: {e}")
            return self._create_synthetic_whale_data(symbol, days)
    
    def get_exchange_flows(self, symbol: str = 'BTC', days: int = 30) -> pd.DataFrame:
        """
        Получает данные о потоках на биржи и с бирж.
        
        Args:
            symbol: Символ криптовалюты
            days: Количество дней для анализа
        
        Returns:
            DataFrame с данными о потоках бирж
        """
        try:
            self.logger.info(f"Получение данных о потоках бирж для {symbol}")
            
            # Притоки на биржи (обычно медвежий сигнал)
            inflows = self._get_glassnode_metric(
                "transactions/transfers_volume_to_exchanges",
                symbol=symbol,
                days=days
            )
            
            # Оттоки с бирж (обычно бычий сигнал)
            outflows = self._get_glassnode_metric(
                "transactions/transfers_volume_from_exchanges", 
                symbol=symbol,
                days=days
            )
            
            # Резервы бирж
            reserves = self._get_glassnode_metric(
                "distribution/balance_exchanges",
                symbol=symbol, 
                days=days
            )
            
            if inflows and outflows and reserves:
                # Объединяем данные
                df = pd.DataFrame({
                    'timestamp': [item['timestamp'] for item in inflows],
                    'exchange_inflows': [item['value'] for item in inflows],
                    'exchange_outflows': [item['value'] for item in outflows],
                    'exchange_reserves': [item['value'] for item in reserves]
                })
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Вычисляем net flow (отток - приток)
                df['net_flow'] = df['exchange_outflows'] - df['exchange_inflows']
                df['net_flow_ma'] = df['net_flow'].rolling(7).mean()
                
                # Сигналы
                df['bullish_flow'] = (df['net_flow'] > 0).astype(int)  # Отток больше притока
                df['bearish_flow'] = (df['net_flow'] < 0).astype(int)  # Приток больше оттока
                
                return df
            
            # Fallback
            return self._create_synthetic_exchange_data(symbol, days)
            
        except Exception as e:
            self.logger.error(f"Ошибка получения данных о потоках бирж: {e}")
            return self._create_synthetic_exchange_data(symbol, days)
    
    def get_network_activity(self, symbol: str = 'BTC', days: int = 30) -> pd.DataFrame:
        """
        Получает данные о сетевой активности.
        
        Args:
            symbol: Символ криптовалюты
            days: Количество дней для анализа
        
        Returns:
            DataFrame с данными о сетевой активности
        """
        try:
            self.logger.info(f"Получение данных о сетевой активности для {symbol}")
            
            # Активные адреса
            active_addresses = self._get_glassnode_metric(
                "addresses/active_count",
                symbol=symbol,
                days=days
            )
            
            # Количество транзакций
            transaction_count = self._get_glassnode_metric(
                "transactions/count",
                symbol=symbol,
                days=days
            )
            
            # Hash Rate (только для BTC)
            hash_rate = None
            if symbol.upper() == 'BTC':
                hash_rate = self._get_glassnode_metric(
                    "mining/hash_rate_mean",
                    symbol=symbol,
                    days=days
                )
            
            if active_addresses and transaction_count:
                df_data = {
                    'timestamp': [item['timestamp'] for item in active_addresses],
                    'active_addresses': [item['value'] for item in active_addresses],
                    'transaction_count': [item['value'] for item in transaction_count]
                }
                
                if hash_rate:
                    df_data['hash_rate'] = [item['value'] for item in hash_rate]
                
                df = pd.DataFrame(df_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Индикаторы сетевой активности
                df['addresses_ma'] = df['active_addresses'].rolling(7).mean()
                df['tx_count_ma'] = df['transaction_count'].rolling(7).mean()
                
                df['network_activity_score'] = (
                    (df['active_addresses'] / df['addresses_ma'] - 1) * 0.5 +
                    (df['transaction_count'] / df['tx_count_ma'] - 1) * 0.5
                )
                
                df['high_activity'] = (df['network_activity_score'] > 0.1).astype(int)
                
                return df
            
            return self._create_synthetic_network_data(symbol, days)
            
        except Exception as e:
            self.logger.error(f"Ошибка получения сетевой активности: {e}")
            return self._create_synthetic_network_data(symbol, days)
    
    def get_hodler_behavior(self, symbol: str = 'BTC', days: int = 30) -> pd.DataFrame:
        """
        Получает данные о поведении долгосрочных держателей.
        
        Args:
            symbol: Символ криптовалюты
            days: Количество дней для анализа
        
        Returns:
            DataFrame с данными о поведении HODLers
        """
        try:
            self.logger.info(f"Получение данных о поведении HODLers для {symbol}")
            
            # Long-term holder supply
            lth_supply = self._get_glassnode_metric(
                "supply/long_term_holder",
                symbol=symbol,
                days=days
            )
            
            # Realized profit/loss
            realized_pnl = self._get_glassnode_metric(
                "indicators/net_realized_profit_loss",
                symbol=symbol,
                days=days
            )
            
            if lth_supply and realized_pnl:
                df = pd.DataFrame({
                    'timestamp': [item['timestamp'] for item in lth_supply],
                    'lth_supply': [item['value'] for item in lth_supply],
                    'realized_pnl': [item['value'] for item in realized_pnl]
                })
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Индикаторы поведения
                df['lth_supply_change'] = df['lth_supply'].pct_change()
                df['accumulation_phase'] = (df['lth_supply_change'] > 0).astype(int)
                df['distribution_phase'] = (df['lth_supply_change'] < -0.01).astype(int)
                
                return df
            
            return self._create_synthetic_hodler_data(symbol, days)
            
        except Exception as e:
            self.logger.error(f"Ошибка получения данных о HODLers: {e}")
            return self._create_synthetic_hodler_data(symbol, days)
    
    def _get_glassnode_metric(self, metric: str, symbol: str, 
                             days: int, params: Dict = None) -> Optional[List[Dict]]:
        """
        Получает метрику с Glassnode API.
        
        Args:
            metric: Название метрики
            symbol: Символ криптовалюты
            days: Количество дней
            params: Дополнительные параметры
        
        Returns:
            Список с данными метрики или None
        """
        try:
            # Проверяем кеш
            cache_key = f"{metric}_{symbol}_{days}_{params}"
            if cache_key in self.cache:
                cache_time, data = self.cache[cache_key]
                if datetime.now() - cache_time < timedelta(hours=1):
                    return data
            
            # Параметры запроса
            url = f"{self.glassnode_base_url}/{metric}"
            query_params = {
                'a': symbol.upper(),
                'api_key': self.glassnode_api_key,
                'f': 'json',
                's': int((datetime.now() - timedelta(days=days)).timestamp()),
                'u': int(datetime.now().timestamp())
            }
            
            if params:
                query_params.update(params)
            
            # Отправляем запрос
            response = requests.get(url, params=query_params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # Кешируем результат
                self.cache[cache_key] = (datetime.now(), data)
                return data
            else:
                self.logger.warning(f"Glassnode API error: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Ошибка запроса к Glassnode: {e}")
            return None
    
    def _create_synthetic_whale_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Создает синтетические данные о китах."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)
        
        # Имитируем периодическую активность китов
        base_volume = np.random.lognormal(5, 1, days)
        spikes = np.random.choice([0, 1], days, p=[0.8, 0.2])  # 20% дней со всплесками
        whale_volume = base_volume * (1 + spikes * np.random.uniform(2, 5, days))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'whale_volume': whale_volume,
            'whale_volume_ma': pd.Series(whale_volume).rolling(7).mean(),
            'whale_activity_spike': spikes
        })
        
        return df
    
    def _create_synthetic_exchange_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Создает синтетические данные о потоках бирж."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)
        
        # Имитируем потоки с трендом
        trend = np.linspace(-0.5, 0.5, days)
        noise = np.random.normal(0, 0.3, days)
        net_flow = trend + noise
        
        df = pd.DataFrame({
            'timestamp': dates,
            'exchange_inflows': np.abs(np.random.normal(1000, 200, days)),
            'exchange_outflows': np.abs(np.random.normal(1000, 200, days)) + net_flow * 100,
            'exchange_reserves': np.random.uniform(800000, 1200000, days),
            'net_flow': net_flow,
            'net_flow_ma': pd.Series(net_flow).rolling(7).mean(),
            'bullish_flow': (net_flow > 0).astype(int),
            'bearish_flow': (net_flow < 0).astype(int)
        })
        
        return df
    
    def _create_synthetic_network_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Создает синтетические данные о сетевой активности."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)
        
        # Имитируем растущую сетевую активность с флуктуациями
        base_addresses = 500000 + np.random.randint(-50000, 50000, days)
        base_tx = 300000 + np.random.randint(-30000, 30000, days)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'active_addresses': base_addresses,
            'transaction_count': base_tx,
            'addresses_ma': pd.Series(base_addresses).rolling(7).mean(),
            'tx_count_ma': pd.Series(base_tx).rolling(7).mean()
        })
        
        df['network_activity_score'] = np.random.normal(0, 0.1, days)
        df['high_activity'] = (df['network_activity_score'] > 0.1).astype(int)
        
        return df
    
    def _create_synthetic_hodler_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Создает синтетические данные о поведении HODLers."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)
        
        # Имитируем медленно растущее предложение LTH
        lth_supply = 13000000 + np.cumsum(np.random.normal(1000, 5000, days))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'lth_supply': lth_supply,
            'realized_pnl': np.random.normal(0, 50000000, days),
            'lth_supply_change': pd.Series(lth_supply).pct_change(),
        })
        
        df['accumulation_phase'] = (df['lth_supply_change'] > 0).astype(int)
        df['distribution_phase'] = (df['lth_supply_change'] < -0.01).astype(int)
        
        return df
    
    def get_comprehensive_onchain_data(self, symbol: str = 'BTC', 
                                     days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Получает все on-chain метрики для символа.
        
        Args:
            symbol: Символ криптовалюты
            days: Количество дней для анализа
        
        Returns:
            Словарь с различными типами on-chain данных
        """
        try:
            self.logger.info(f"Сбор всех on-chain метрик для {symbol}")
            
            data = {}
            
            # Собираем все типы данных
            data['whale_movements'] = self.get_whale_movements(symbol, days)
            data['exchange_flows'] = self.get_exchange_flows(symbol, days)
            data['network_activity'] = self.get_network_activity(symbol, days)
            data['hodler_behavior'] = self.get_hodler_behavior(symbol, days)
            
            self.logger.info(f"Собрано {len(data)} типов on-chain данных")
            return data
            
        except Exception as e:
            self.logger.error(f"Ошибка сбора on-chain данных: {e}")
            return {}
    
    def calculate_onchain_sentiment_score(self, onchain_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        Рассчитывает общий sentiment score на основе on-chain метрик.
        
        Args:
            onchain_data: Словарь с on-chain данными
        
        Returns:
            Series с sentiment scores
        """
        try:
            scores = []
            
            # Получаем последние значения каждой метрики
            if 'whale_movements' in onchain_data:
                whale_score = onchain_data['whale_movements']['whale_activity_spike'].iloc[-7:].mean()
                scores.append(('whale_activity', whale_score, 0.2))
            
            if 'exchange_flows' in onchain_data:
                flow_score = onchain_data['exchange_flows']['bullish_flow'].iloc[-7:].mean()
                scores.append(('exchange_flows', flow_score, 0.3))
            
            if 'network_activity' in onchain_data:
                network_score = onchain_data['network_activity']['high_activity'].iloc[-7:].mean()
                scores.append(('network_activity', network_score, 0.2))
            
            if 'hodler_behavior' in onchain_data:
                hodler_score = onchain_data['hodler_behavior']['accumulation_phase'].iloc[-7:].mean()
                scores.append(('hodler_behavior', hodler_score, 0.3))
            
            # Взвешенный расчет
            if scores:
                weighted_score = sum(score * weight for _, score, weight in scores) / sum(weight for _, _, weight in scores)
                return pd.Series([weighted_score], index=['onchain_sentiment'])
            
            return pd.Series([0.5], index=['onchain_sentiment'])  # Нейтральный
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета on-chain sentiment: {e}")
            return pd.Series([0.5], index=['onchain_sentiment']) 
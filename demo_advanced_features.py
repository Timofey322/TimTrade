#!/usr/bin/env python3
"""
Демонстрация продвинутых фичей: On-chain метрики и Sentiment анализ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yaml
import os
import sys
from loguru import logger

# Добавляем путь к исходникам
sys.path.append('src')

# Импортируем наши модули
from data_collection.onchain_collector import OnChainCollector
from data_collection.sentiment_collector import SentimentCollector  
from preprocessing.advanced_features import AdvancedFeatureEngine

def load_config():
    """Загружает конфигурацию."""
    try:
        with open('config/advanced_features_config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Конфиг не найден, используем дефолтные настройки")
        return {}

def demonstrate_onchain_metrics():
    """Демонстрирует работу с on-chain метриками."""
    print("\n" + "="*80)
    print("🐋 ДЕМОНСТРАЦИЯ ON-CHAIN МЕТРИК")
    print("="*80)
    
    config = load_config()
    collector = OnChainCollector(config.get('api_keys', {}))
    
    # Получаем данные для BTC
    print("\n📊 Получение данных о движениях китов...")
    whale_data = collector.get_whale_movements('BTC', days=30)
    print(f"Получено {len(whale_data)} записей о движениях китов")
    print(whale_data.head())
    
    print("\n💱 Получение данных о потоках бирж...")
    flows_data = collector.get_exchange_flows('BTC', days=30)
    print(f"Получено {len(flows_data)} записей о потоках бирж")
    print(flows_data.head())
    
    print("\n🌐 Получение данных о сетевой активности...")
    network_data = collector.get_network_activity('BTC', days=30)
    print(f"Получено {len(network_data)} записей о сетевой активности")
    print(network_data.head())
    
    print("\n💎 Получение данных о поведении HODLers...")
    hodler_data = collector.get_hodler_behavior('BTC', days=30)
    print(f"Получено {len(hodler_data)} записей о поведении HODLers")
    print(hodler_data.head())
    
    # Composite score
    all_data = {
        'whale_movements': whale_data,
        'exchange_flows': flows_data,
        'network_activity': network_data,
        'hodler_behavior': hodler_data
    }
    
    composite_score = collector.calculate_onchain_sentiment_score(all_data)
    print(f"\n🎯 Composite On-Chain Sentiment Score: {composite_score.iloc[0]:.3f}")
    
    return all_data

def demonstrate_sentiment_analysis():
    """Демонстрирует работу sentiment анализа."""
    print("\n" + "="*80)
    print("😱 ДЕМОНСТРАЦИЯ SENTIMENT АНАЛИЗА")
    print("="*80)
    
    config = load_config()
    collector = SentimentCollector(config.get('api_keys', {}))
    
    # Fear & Greed Index
    print("\n📈 Получение Fear & Greed Index...")
    fear_greed = collector.get_fear_greed_index(days=30)
    print(f"Получено {len(fear_greed)} записей Fear & Greed Index")
    if not fear_greed.empty:
        latest_fg = fear_greed.iloc[-1]
        print(f"Последнее значение: {latest_fg['value']} ({latest_fg['value_classification']})")
    
    # Bybit Opportunities
    print("\n🎯 Парсинг Bybit Market Opportunities...")
    bybit_data = collector.get_bybit_opportunities()
    print("Данные Bybit Opportunities:")
    for key, value in bybit_data.items():
        if key != 'timestamp':
            print(f"  {key}: {value}")
    
    # Social Sentiment
    print("\n📱 Получение Social Sentiment...")
    social_sentiment = collector.get_social_sentiment('bitcoin', days=7)
    print(f"Получено {len(social_sentiment)} записей social sentiment")
    if not social_sentiment.empty:
        avg_sentiment = social_sentiment['social_sentiment'].mean()
        print(f"Средний social sentiment: {avg_sentiment:.3f}")
    
    # News Sentiment
    print("\n📰 Получение News Sentiment...")
    news_sentiment = collector.get_news_sentiment('bitcoin', days=7)
    print(f"Получено {len(news_sentiment)} записей news sentiment")
    if not news_sentiment.empty:
        avg_news = news_sentiment['avg_sentiment'].mean()
        print(f"Средний news sentiment: {avg_news:.3f}")
    
    # Funding Rates
    print("\n💰 Анализ Funding Rates...")
    funding_data = collector.get_funding_rates_sentiment(['BTCUSDT', 'ETHUSDT'])
    print("Funding Rates Sentiment:")
    print(funding_data)
    
    # Composite Sentiment Score
    print("\n🎯 Расчет Composite Sentiment Score...")
    composite_score = collector.calculate_composite_sentiment_score(
        fear_greed, social_sentiment, news_sentiment, bybit_data, funding_data
    )
    print(f"Composite Sentiment Score: {composite_score:.3f}")
    
    return {
        'fear_greed': fear_greed,
        'bybit_data': bybit_data,
        'social_sentiment': social_sentiment,
        'news_sentiment': news_sentiment,
        'funding_data': funding_data,
        'composite_score': composite_score
    }

def demonstrate_advanced_features():
    """Демонстрирует создание продвинутых фичей."""
    print("\n" + "="*80)
    print("🚀 ДЕМОНСТРАЦИЯ ADVANCED FEATURE ENGINEERING")
    print("="*80)
    
    config = load_config()
    feature_engine = AdvancedFeatureEngine(config)
    
    # Создаем синтетические ценовые данные
    print("\n📊 Создание синтетических ценовых данных...")
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    np.random.seed(42)
    
    # Имитируем ценовые данные BTC
    base_price = 45000
    returns = np.random.normal(0.001, 0.03, 60)  # Средний рост 0.1% в день, волатильность 3%
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    price_df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(20000, 80000, 60)
    })
    
    print(f"Создано {len(price_df)} дней ценовых данных")
    print(f"Цена изменилась с ${price_df['close'].iloc[0]:,.0f} до ${price_df['close'].iloc[-1]:,.0f}")
    
    # Создаем on-chain фичи
    print("\n🐋 Создание On-Chain фичей...")
    onchain_features = feature_engine.create_onchain_features('BTC', days=60)
    print(f"Создано {onchain_features.shape[1]-1} on-chain фичей")
    
    # Создаем sentiment фичи
    print("\n😱 Создание Sentiment фичей...")
    sentiment_features = feature_engine.create_sentiment_features('BTC', days=60)
    print(f"Создано {sentiment_features.shape[1]-1} sentiment фичей")
    
    # Создаем полный набор продвинутых фичей
    print("\n🚀 Создание полного набора продвинутых фичей...")
    comprehensive_features = feature_engine.create_comprehensive_features('BTC', price_df, days=60)
    print(f"Итого создано {comprehensive_features.shape[1]} фичей")
    
    # Показываем примеры фичей
    print("\n📋 Примеры созданных фичей:")
    feature_columns = [col for col in comprehensive_features.columns if col != 'timestamp']
    
    # Группируем фичи по типам
    onchain_cols = [col for col in feature_columns if any(x in col.lower() for x in ['whale', 'exchange', 'network', 'hodler', 'onchain'])]
    sentiment_cols = [col for col in feature_columns if any(x in col.lower() for x in ['sentiment', 'fear', 'greed', 'social', 'news', 'bybit'])]
    correlation_cols = [col for col in feature_columns if 'corr_' in col]
    other_cols = [col for col in feature_columns if col not in onchain_cols + sentiment_cols + correlation_cols]
    
    print(f"\n🐋 On-Chain фичи ({len(onchain_cols)}):")
    for col in onchain_cols[:5]:
        latest_value = comprehensive_features[col].iloc[-1]
        print(f"  • {col}: {latest_value:.4f}")
    
    print(f"\n😱 Sentiment фичи ({len(sentiment_cols)}):")
    for col in sentiment_cols[:5]:
        latest_value = comprehensive_features[col].iloc[-1]
        print(f"  • {col}: {latest_value:.4f}")
    
    print(f"\n🔗 Correlation фичи ({len(correlation_cols)}):")
    for col in correlation_cols[:3]:
        latest_value = comprehensive_features[col].iloc[-1]
        print(f"  • {col}: {latest_value:.4f}")
    
    # Ключевые composite scores
    print(f"\n🎯 Ключевые индикаторы:")
    if 'onchain_composite_score' in comprehensive_features.columns:
        onchain_score = comprehensive_features['onchain_composite_score'].iloc[-1]
        print(f"  • On-Chain Composite Score: {onchain_score:.3f}")
    
    if 'sentiment_composite_score' in comprehensive_features.columns:
        sentiment_score = comprehensive_features['sentiment_composite_score'].iloc[-1]
        print(f"  • Sentiment Composite Score: {sentiment_score:.3f}")
    
    if 'fundamental_strength' in comprehensive_features.columns:
        fundamental_score = comprehensive_features['fundamental_strength'].iloc[-1]
        print(f"  • Fundamental Strength: {fundamental_score:.3f}")
    
    # Market regime
    if all(col in comprehensive_features.columns for col in ['market_regime_bullish', 'market_regime_bearish', 'market_regime_neutral']):
        regime_bullish = comprehensive_features['market_regime_bullish'].iloc[-1]
        regime_bearish = comprehensive_features['market_regime_bearish'].iloc[-1]
        regime_neutral = comprehensive_features['market_regime_neutral'].iloc[-1]
        
        if regime_bullish:
            regime = "🐂 БЫЧИЙ"
        elif regime_bearish:
            regime = "🐻 МЕДВЕЖИЙ"
        else:
            regime = "😐 НЕЙТРАЛЬНЫЙ"
            
        print(f"  • Market Regime: {regime}")
    
    return comprehensive_features

def create_visualization(comprehensive_features):
    """Создает визуализации для демонстрации."""
    print("\n" + "="*80)
    print("📊 СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("="*80)
    
    try:
        # Настройка стиля
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🚀 Advanced Features: On-Chain & Sentiment Analysis', fontsize=16, fontweight='bold')
        
        # График 1: Цена и On-Chain Score
        if 'close' in comprehensive_features.columns and 'onchain_composite_score' in comprehensive_features.columns:
            ax1 = axes[0, 0]
            ax1_twin = ax1.twinx()
            
            dates = comprehensive_features['timestamp']
            price = comprehensive_features['close']
            onchain = comprehensive_features['onchain_composite_score']
            
            line1 = ax1.plot(dates, price, 'b-', linewidth=2, label='BTC Price', alpha=0.8)
            line2 = ax1_twin.plot(dates, onchain, 'r-', linewidth=2, label='On-Chain Score', alpha=0.8)
            
            ax1.set_xlabel('Date')
            ax1.set_ylabel('BTC Price ($)', color='b')
            ax1_twin.set_ylabel('On-Chain Score', color='r')
            ax1.set_title('🐋 Price vs On-Chain Metrics')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1_twin.tick_params(axis='y', labelcolor='r')
            ax1.grid(True, alpha=0.3)
        
        # График 2: Sentiment Components
        ax2 = axes[0, 1]
        sentiment_cols = [col for col in comprehensive_features.columns if 'sentiment' in col.lower() and 'composite' not in col.lower()]
        
        if sentiment_cols:
            for i, col in enumerate(sentiment_cols[:3]):  # Первые 3 компонента
                values = comprehensive_features[col].rolling(7).mean()  # Сглаживание
                ax2.plot(comprehensive_features['timestamp'], values, 
                        linewidth=2, label=col.replace('_', ' ').title(), alpha=0.8)
            
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Sentiment Score')
            ax2.set_title('😱 Sentiment Components')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
        
        # График 3: Correlation Heatmap
        ax3 = axes[1, 0]
        
        # Выбираем ключевые фичи для корреляции
        key_features = []
        if 'close' in comprehensive_features.columns:
            key_features.append('close')
        
        feature_cols = comprehensive_features.columns.tolist()
        important_cols = [col for col in feature_cols if any(keyword in col.lower() 
                         for keyword in ['composite', 'whale', 'fear_greed', 'social', 'fundamental'])]
        
        key_features.extend(important_cols[:8])  # Ограничиваем количество
        
        if len(key_features) > 1:
            corr_data = comprehensive_features[key_features].corr()
            
            # Создаем маску для верхнего треугольника
            mask = np.triu(np.ones_like(corr_data, dtype=bool))
            
            im = ax3.imshow(corr_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Добавляем цифры
            for i in range(len(corr_data)):
                for j in range(len(corr_data)):
                    if not mask[i, j]:
                        text = ax3.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                                      ha="center", va="center", color="black", fontsize=8)
            
            ax3.set_xticks(range(len(key_features)))
            ax3.set_yticks(range(len(key_features)))
            ax3.set_xticklabels([col.replace('_', '\n') for col in key_features], 
                               rotation=45, ha='right', fontsize=8)
            ax3.set_yticklabels([col.replace('_', '\n') for col in key_features], fontsize=8)
            ax3.set_title('🔗 Feature Correlations')
            
            # Добавляем colorbar
            cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
            cbar.set_label('Correlation', fontsize=8)
        
        # График 4: Market Regime Distribution
        ax4 = axes[1, 1]
        
        regime_cols = [col for col in comprehensive_features.columns if 'market_regime' in col]
        if regime_cols:
            regime_counts = []
            regime_labels = []
            
            for col in regime_cols:
                count = comprehensive_features[col].sum()
                regime_counts.append(count)
                
                if 'bullish' in col:
                    regime_labels.append('🐂 Bullish')
                elif 'bearish' in col:
                    regime_labels.append('🐻 Bearish')
                else:
                    regime_labels.append('😐 Neutral')
            
            colors = ['green', 'red', 'gray']
            wedges, texts, autotexts = ax4.pie(regime_counts, labels=regime_labels, 
                                               colors=colors, autopct='%1.1f%%', 
                                               startangle=90)
            ax4.set_title('📊 Market Regime Distribution')
        
        plt.tight_layout()
        
        # Сохраняем график
        output_path = 'advanced_features_demo.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Визуализация сохранена: {output_path}")
        
    except Exception as e:
        print(f"❌ Ошибка создания визуализации: {e}")

def main():
    """Главная функция демонстрации."""
    print("🚀 " + "="*78)
    print("🚀 ДЕМОНСТРАЦИЯ ПРОДВИНУТЫХ ВОЗМОЖНОСТЕЙ ТОРГОВОЙ СИСТЕМЫ")
    print("🚀 " + "="*78)
    print("📈 On-Chain метрики: движения китов, потоки бирж, сетевая активность")
    print("😱 Sentiment анализ: Fear & Greed, социальные сети, новости, Bybit")
    print("🔗 Cross-correlation анализ между ценой и fundamentals")
    print("🎯 Composite scoring для принятия торговых решений")
    
    try:
        # Демонстрируем каждый компонент
        onchain_data = demonstrate_onchain_metrics()
        sentiment_data = demonstrate_sentiment_analysis()
        comprehensive_features = demonstrate_advanced_features()
        
        # Создаем визуализации
        create_visualization(comprehensive_features)
        
        # Итоговый отчет
        print("\n" + "="*80)
        print("📋 ИТОГОВЫЙ ОТЧЕТ")
        print("="*80)
        
        total_features = comprehensive_features.shape[1] - 1  # Исключаем timestamp
        
        onchain_features = len([col for col in comprehensive_features.columns 
                               if any(x in col.lower() for x in ['whale', 'exchange', 'network', 'hodler', 'onchain'])])
        
        sentiment_features = len([col for col in comprehensive_features.columns 
                                 if any(x in col.lower() for x in ['sentiment', 'fear', 'greed', 'social', 'news', 'bybit'])])
        
        correlation_features = len([col for col in comprehensive_features.columns if 'corr_' in col])
        
        print(f"📊 Всего создано фичей: {total_features}")
        print(f"  🐋 On-Chain фичи: {onchain_features}")
        print(f"  😱 Sentiment фичи: {sentiment_features}")
        print(f"  🔗 Correlation фичи: {correlation_features}")
        print(f"  🎯 Прочие фичи: {total_features - onchain_features - sentiment_features - correlation_features}")
        
        # Рекомендации по интеграции
        print("\n🚀 РЕКОМЕНДАЦИИ ПО ИНТЕГРАЦИИ:")
        print("1. 🔑 Получите API ключи для:")
        print("   • Glassnode (on-chain данные): https://glassnode.com/")
        print("   • Twitter Developer (social sentiment): https://developer.twitter.com/")
        print("   • News API (news sentiment): https://newsapi.org/")
        
        print("\n2. ⚙️  Обновите конфигурацию в config/advanced_features_config.yaml")
        
        print("\n3. 🎯 Добавьте в существующую систему:")
        print("   • Импортируйте AdvancedFeatureEngine в feature_engineering.py")
        print("   • Интегрируйте composite scores в confidence_scorer.py")
        print("   • Используйте market regime для risk management")
        
        print("\n4. 📈 Мониторинг и оптимизация:")
        print("   • Отслеживайте correlation breaks как сигналы изменения режима")
        print("   • Используйте sentiment extremes для контрарианских стратегий")
        print("   • Комбинируйте on-chain и sentiment для подтверждения сигналов")
        
        print("\n✅ Демонстрация завершена успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка в демонстрации: {e}")
        print(f"\n❌ Ошибка: {e}")

if __name__ == "__main__":
    main() 
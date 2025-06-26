#!/usr/bin/env python3
"""
Мониторинг Sentiment данных в реальном времени
"""

import sys
import time
from datetime import datetime
import yaml
sys.path.append('src')

from data_collection.sentiment_collector import SentimentCollector

def load_config():
    """Загружает конфигурацию."""
    try:
        with open('config/smart_adaptive_config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Ошибка загрузки конфига: {e}")
        return {}

def print_separator():
    """Печатает разделитель."""
    print("=" * 80)

def monitor_sentiment():
    """Основная функция мониторинга."""
    print("🚀 SENTIMENT MONITOR - Запуск мониторинга")
    print_separator()
    
    # Загружаем конфигурацию
    config = load_config()
    advanced_config = config.get('advanced_features', {})
    
    if not advanced_config.get('enabled', False):
        print("⚠️ Advanced features отключены в конфигурации")
        print("Включите их в config/smart_adaptive_config.yaml:")
        print("advanced_features:")
        print("  enabled: true")
        return
    
    # Инициализируем collector
    api_keys = advanced_config.get('api_keys', {})
    collector = SentimentCollector(api_keys)
    
    print("✅ Sentiment Collector инициализирован")
    print("📊 Начинаем мониторинг...")
    print_separator()
    
    # Настройки мониторинга
    update_interval = 300  # 5 минут
    iteration = 0
    
    try:
        while True:
            iteration += 1
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n📅 Обновление #{iteration} - {current_time}")
            print("-" * 60)
            
            try:
                # Получаем Bybit данные
                print("🌐 Получение Bybit Market Opportunities...")
                bybit_data = collector.get_bybit_opportunities()
                
                if bybit_data:
                    print("✅ Bybit данные получены:")
                    
                    # Основные метрики
                    market_sentiment = bybit_data.get('market_sentiment', 0.5)
                    hot_sectors = bybit_data.get('hot_sectors', [])
                    trending_coins = bybit_data.get('trending_coins', [])
                    gainers_losers = bybit_data.get('gainers_losers', {})
                    
                    print(f"   📈 Market Sentiment: {market_sentiment:.3f}")
                    print(f"   🔥 Hot Sectors: {len(hot_sectors)}")
                    print(f"   📊 Trending Coins: {len(trending_coins)}")
                    
                    # Gainers/Losers
                    gainers = gainers_losers.get('gainers', [])
                    losers = gainers_losers.get('losers', [])
                    print(f"   📈 Gainers: {len(gainers)}")
                    print(f"   📉 Losers: {len(losers)}")
                    
                    # Расчет композитного sentiment
                    if trending_coins:
                        positive_trending = sum(1 for coin in trending_coins if coin.get('sentiment', 0) > 0)
                        positive_ratio = positive_trending / len(trending_coins)
                    else:
                        positive_ratio = 0.5
                    
                    if len(gainers) + len(losers) > 0:
                        gainers_ratio = len(gainers) / (len(gainers) + len(losers))
                    else:
                        gainers_ratio = 0.5
                    
                    composite_score = (
                        market_sentiment * 0.4 +
                        positive_ratio * 0.3 +
                        gainers_ratio * 0.3
                    )
                    
                    print(f"\n🎯 КОМПОЗИТНЫЕ МЕТРИКИ:")
                    print(f"   Positive Trending Ratio: {positive_ratio:.3f}")
                    print(f"   Gainers Ratio: {gainers_ratio:.3f}")
                    print(f"   Composite Sentiment: {composite_score:.3f}")
                    
                    # Market Regime
                    if composite_score > 0.6:
                        regime = "🐂 БЫЧИЙ"
                        regime_desc = "Благоприятные условия для покупок"
                    elif composite_score < 0.4:
                        regime = "🐻 МЕДВЕЖИЙ"
                        regime_desc = "Осторожность, возможны распродажи"
                    else:
                        regime = "😐 НЕЙТРАЛЬНЫЙ"
                        regime_desc = "Смешанные сигналы, стандартная стратегия"
                    
                    print(f"\n🏛️ MARKET REGIME: {regime}")
                    print(f"   {regime_desc}")
                    
                    # Торговые рекомендации
                    print(f"\n💡 ТОРГОВЫЕ РЕКОМЕНДАЦИИ:")
                    if composite_score > 0.7:
                        print("   🚀 Высокий sentiment - снизить пороги покупок")
                        print("   📈 Рассмотреть увеличение размера позиций")
                        print("   ⏰ Хороший timing для входов")
                    elif composite_score < 0.3:
                        print("   🛡️ Низкий sentiment - повысить пороги покупок")
                        print("   📉 Рассмотреть уменьшение размера позиций") 
                        print("   ⚠️ Осторожность с новыми позициями")
                    else:
                        print("   ⚖️ Средний sentiment - стандартная стратегия")
                        print("   📊 Полагаться на технические индикаторы")
                    
                    # Алерты
                    alerts = []
                    if composite_score > 0.8:
                        alerts.append("🚨 ЭКСТРЕМАЛЬНАЯ ЖАДНОСТЬ - возможная коррекция")
                    elif composite_score < 0.2:
                        alerts.append("🚨 ЭКСТРЕМАЛЬНЫЙ СТРАХ - возможен отскок")
                    
                    if len(gainers) > len(losers) * 2:
                        alerts.append("⚡ Сильное доминирование растущих активов")
                    elif len(losers) > len(gainers) * 2:
                        alerts.append("⚡ Сильное доминирование падающих активов")
                    
                    if alerts:
                        print(f"\n🚨 АЛЕРТЫ:")
                        for alert in alerts:
                            print(f"   {alert}")
                    
                    # Детали по секторам (если есть)
                    if hot_sectors:
                        print(f"\n🔥 ГОРЯЧИЕ СЕКТОРА:")
                        for i, sector in enumerate(hot_sectors[:5], 1):
                            print(f"   {i}. {sector}")
                    
                    # Детали по трендинговым монетам (если есть)
                    if trending_coins:
                        print(f"\n📊 TRENDING COINS:")
                        for i, coin in enumerate(trending_coins[:5], 1):
                            coin_sentiment = coin.get('sentiment', 0)
                            sentiment_emoji = "📈" if coin_sentiment > 0 else "📉" if coin_sentiment < 0 else "➡️"
                            print(f"   {i}. {coin.get('name', 'Unknown')} {sentiment_emoji}")
                    
                else:
                    print("⚠️ Bybit данные недоступны")
                    print("   Используются дефолтные значения:")
                    print("   Market Sentiment: 0.500 (нейтральный)")
                    print("   Market Regime: 😐 НЕЙТРАЛЬНЫЙ")
                
            except Exception as e:
                print(f"❌ Ошибка получения данных: {e}")
                print("   Продолжаем мониторинг...")
            
            print(f"\n⏰ Следующее обновление через {update_interval//60} минут...")
            print("=" * 80)
            
            # Ждем до следующего обновления
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Мониторинг остановлен пользователем")
        print("👋 До свидания!")
    except Exception as e:
        print(f"\n❌ Критическая ошибка мониторинга: {e}")

if __name__ == "__main__":
    monitor_sentiment() 
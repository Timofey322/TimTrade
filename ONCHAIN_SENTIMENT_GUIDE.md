# 🚀 Руководство по On-Chain метрикам и Sentiment анализу

## 📋 Обзор

Этот модуль добавляет в вашу торговую систему продвинутые возможности анализа fundamentals:

### 🐋 **ON-CHAIN МЕТРИКИ**
- **Whale Movements** - отслеживание крупных переводов (>100 BTC)
- **Exchange Flows** - притоки/оттоки на биржи и с бирж
- **Network Activity** - активные адреса, количество транзакций, hash rate
- **HODLer Behavior** - поведение долгосрочных держателей

### 😱 **SENTIMENT АНАЛИЗ**
- **Fear & Greed Index** - индекс страха и жадности
- **Social Sentiment** - анализ настроений в социальных сетях
- **News Sentiment** - анализ новостей и медиа
- **Bybit Opportunities** - парсинг трендов с Bybit
- **Funding Rates** - анализ ставок финансирования

---

## 🔧 Быстрая настройка

### 1. Установка зависимостей

```bash
pip install feedparser beautifulsoup4
```

### 2. Получение API ключей

#### 🔑 **Glassnode API** (рекомендуется)
- Сайт: https://glassnode.com/
- План: Start ($39/месяц) или Professional ($149/месяц)
- Дает доступ к реальным on-chain данным

#### 🔑 **Twitter Developer API** (опционально)
- Сайт: https://developer.twitter.com/
- Бесплатный tier: 500,000 твитов/месяц
- Для анализа social sentiment

#### 🔑 **News API** (опционально)
- Сайт: https://newsapi.org/
- Бесплатный tier: 1,000 запросов/день
- Для анализа news sentiment

### 3. Обновление конфигурации

Отредактируйте `config/advanced_features_config.yaml`:

```yaml
api_keys:
  glassnode_api_key: "ваш_glassnode_ключ"
  twitter_bearer_token: "ваш_twitter_токен" 
  news_api_key: "ваш_news_api_ключ"
```

---

## 💡 Практические примеры использования

### Пример 1: Простой анализ sentiment

```python
from src.data_collection.sentiment_collector import SentimentCollector

# Инициализация
collector = SentimentCollector({'news_api_key': 'your_key'})

# Получение Fear & Greed Index
fear_greed = collector.get_fear_greed_index(days=30)
current_fg = fear_greed.iloc[-1]['value']

print(f"Текущий Fear & Greed: {current_fg}")

# Интерпретация
if current_fg < 25:
    print("🔥 ЭКСТРЕМАЛЬНЫЙ СТРАХ - возможна покупка")
elif current_fg > 75:
    print("🚨 ЭКСТРЕМАЛЬНАЯ ЖАДНОСТЬ - возможна продажа")
else:
    print("😐 Нейтральные настроения")
```

### Пример 2: Анализ whale movements

```python
from src.data_collection.onchain_collector import OnChainCollector

collector = OnChainCollector({'glassnode_api_key': 'your_key'})

# Получение данных о китах
whale_data = collector.get_whale_movements('BTC', days=30)

# Обнаружение всплесков активности
recent_spikes = whale_data['whale_activity_spike'].tail(7).sum()

if recent_spikes >= 3:
    print("🐋 ВЫСОКАЯ АКТИВНОСТЬ КИТОВ - возможны резкие движения")
elif recent_spikes == 0:
    print("💤 Киты спокойны - боковое движение")
```

### Пример 3: Анализ потоков бирж

```python
# Получение данных о потоках
flows = collector.get_exchange_flows('BTC', days=30)

# Анализ недавних потоков
recent_net_flow = flows['net_flow'].tail(7).mean()

if recent_net_flow > 0:
    print("💰 ОТТОК С БИРЖ - бычий сигнал")
else:
    print("📉 ПРИТОК НА БИРЖИ - медвежий сигнал")
```

### Пример 4: Интеграция в торговую стратегию

```python
from src.preprocessing.advanced_features import AdvancedFeatureEngine

# Создание продвинутых фичей
feature_engine = AdvancedFeatureEngine(config)
features = feature_engine.create_comprehensive_features('BTC', price_data)

# Получение ключевых scores
onchain_score = features['onchain_composite_score'].iloc[-1]
sentiment_score = features['sentiment_composite_score'].iloc[-1]
fundamental_strength = features['fundamental_strength'].iloc[-1]

# Торговая логика
if fundamental_strength > 0.7 and onchain_score > 0.6:
    print("🚀 СИЛЬНЫЕ FUNDAMENTALS - агрессивная покупка")
elif fundamental_strength < 0.3 and sentiment_score < 0.4:
    print("📉 СЛАБЫЕ FUNDAMENTALS - продажа/шорт")
```

---

## 🎯 Интеграция с существующей системой

### Шаг 1: Обновление feature engineering

Добавьте в `src/preprocessing/feature_engineering.py`:

```python
from .advanced_features import AdvancedFeatureEngine

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        # Добавляем advanced feature engine
        self.advanced_engine = AdvancedFeatureEngine(config.get('advanced_features', {}))
    
    def create_features(self, symbol, price_data):
        # Существующие фичи
        basic_features = self.create_basic_features(price_data)
        
        # Добавляем продвинутые фичи
        advanced_features = self.advanced_engine.create_comprehensive_features(
            symbol, price_data
        )
        
        # Объединяем
        all_features = basic_features.merge(advanced_features, on='timestamp')
        return all_features
```

### Шаг 2: Обновление confidence scorer

Добавьте в `src/preprocessing/confidence_scorer.py`:

```python
def calculate_advanced_confidence(self, features_row):
    """Расчет confidence с учетом fundamentals."""
    
    # Базовый confidence
    base_confidence = self.calculate_base_confidence(features_row)
    
    # Fundamental factors
    onchain_score = features_row.get('onchain_composite_score', 0.5)
    sentiment_score = features_row.get('sentiment_composite_score', 0.5)
    fundamental_strength = features_row.get('fundamental_strength', 0.5)
    
    # Модификаторы confidence
    if fundamental_strength > 0.7:
        confidence_multiplier = 1.2  # Увеличиваем на 20%
    elif fundamental_strength < 0.3:
        confidence_multiplier = 0.8  # Уменьшаем на 20%
    else:
        confidence_multiplier = 1.0
    
    # Финальный confidence
    final_confidence = base_confidence * confidence_multiplier
    
    return np.clip(final_confidence, 0, 1)
```

### Шаг 3: Обновление risk management

```python
def calculate_position_size_advanced(self, signal, confidence, features_row):
    """Расчет размера позиции с учетом fundamentals."""
    
    # Базовый размер
    base_size = self.calculate_base_position_size(signal, confidence)
    
    # Fundamental adjustment
    fundamental_strength = features_row.get('fundamental_strength', 0.5)
    market_regime = self.get_market_regime(features_row)
    
    if market_regime == 'bullish' and signal == 'buy':
        size_multiplier = 1.3  # Увеличиваем позицию в бычьем режиме
    elif market_regime == 'bearish' and signal == 'sell':
        size_multiplier = 1.3  # Увеличиваем шорт в медвежьем режиме
    elif market_regime == 'neutral':
        size_multiplier = 0.8  # Уменьшаем в нейтральном режиме
    else:
        size_multiplier = 1.0
    
    return base_size * size_multiplier

def get_market_regime(self, features_row):
    """Определение рыночного режима."""
    if features_row.get('market_regime_bullish', 0):
        return 'bullish'
    elif features_row.get('market_regime_bearish', 0):
        return 'bearish'
    else:
        return 'neutral'
```

---

## 📊 Интерпретация сигналов

### 🐋 On-Chain сигналы

| Метрика | Бычий сигнал | Медвежий сигнал |
|---------|--------------|-----------------|
| **Whale Movements** | Низкая активность (накопление) | Высокая активность (распродажа) |
| **Exchange Flows** | Отток с бирж > приток | Приток на биржи > отток |
| **Network Activity** | Рост активных адресов | Снижение активности |
| **HODLer Behavior** | Увеличение LTH supply | Распродажа долгосрочными |

### 😱 Sentiment сигналы

| Индикатор | Бычий сигнал | Медвежий сигнал |
|-----------|--------------|-----------------|
| **Fear & Greed** | Extreme Fear (0-25) | Extreme Greed (75-100) |
| **Social Sentiment** | Низкий hype при росте цены | Высокий hype при падении |
| **News Sentiment** | Негативные новости + рост | Позитивные новости + падение |
| **Funding Rates** | Отрицательные rates | Очень положительные rates |

### 🎯 Composite Scores

- **On-Chain Score > 0.7**: Сильные fundamentals
- **Sentiment Score < 0.3**: Возможен контрарианский сигнал
- **Fundamental Strength > 0.8**: Подтверждение тренда

---

## ⚠️ Важные замечания

### Ограничения demo версии
- Без API ключей система использует синтетические данные
- Реальные сигналы доступны только с платными API
- Bybit парсинг может быть нестабильным из-за изменений сайта

### Рекомендации по использованию
1. **Не полагайтесь только на один индикатор** - используйте комбинации
2. **Учитывайте временные рамки** - on-chain метрики имеют лаг
3. **Валидируйте сигналы** - сравнивайте с историческими данными
4. **Мониторьте качество данных** - API могут быть недоступны

### Оптимизация производительности
- Кешируйте API ответы (настроено в конфиге)
- Используйте параллельную обработку для multiple symbols
- Ограничивайте количество фичей в зависимости от вычислительных ресурсов

---

## 🚀 Продвинутые стратегии

### Стратегия 1: "Whale Tracker"
```python
def whale_tracker_strategy(features):
    """Следование за движениями китов."""
    
    whale_spike = features['whale_activity_spike']
    exchange_outflow = features['bullish_flow'] 
    price_momentum = features['price_roc_7d']
    
    # Покупка при оттоке китов + momentum
    if whale_spike and exchange_outflow and price_momentum > 0.02:
        return 'strong_buy'
    
    # Продажа при активности китов + приток на биржи
    elif whale_spike and not exchange_outflow:
        return 'sell'
    
    return 'hold'
```

### Стратегия 2: "Sentiment Contrarian"
```python
def sentiment_contrarian_strategy(features):
    """Контрарианская стратегия на основе sentiment."""
    
    fear_greed = features['fear_greed_value']
    social_sentiment = features['social_sentiment']
    funding_rates = features['funding_btc_sentiment']
    
    # Покупка при экстремальном страхе
    if fear_greed < 25 and social_sentiment < 0.3:
        return 'contrarian_buy'
    
    # Продажа при экстремальной жадности + высокий funding
    elif fear_greed > 75 and funding_rates > 0.5:
        return 'contrarian_sell'
    
    return 'hold'
```

### Стратегия 3: "Fundamental Confluence"
```python
def fundamental_confluence_strategy(features):
    """Стратегия слияния fundamentals."""
    
    onchain_score = features['onchain_composite_score']
    sentiment_score = features['sentiment_composite_score']
    fundamental_strength = features['fundamental_strength']
    market_regime = features['market_regime_bullish']
    
    # Максимальный confluence
    if (onchain_score > 0.7 and sentiment_score > 0.6 and 
        fundamental_strength > 0.8 and market_regime):
        return 'max_buy'
    
    # Обратный confluence
    elif (onchain_score < 0.3 and sentiment_score < 0.4 and 
          fundamental_strength < 0.2):
        return 'max_sell'
    
    return 'hold'
```

---

## 📈 Мониторинг и алерты

### Настройка алертов в коде:

```python
def setup_fundamental_alerts(features):
    """Настройка алертов для fundamental changes."""
    
    alerts = []
    
    # Whale activity alert
    if features['whale_activity_spike'].tail(3).sum() >= 2:
        alerts.append("🐋 HIGH WHALE ACTIVITY detected!")
    
    # Sentiment extreme alert
    fear_greed = features['fear_greed_value'].iloc[-1]
    if fear_greed < 20:
        alerts.append("😱 EXTREME FEAR - possible buy opportunity")
    elif fear_greed > 80:
        alerts.append("🤑 EXTREME GREED - possible sell signal")
    
    # Exchange flow alert
    net_flow_7d = features['net_flow'].tail(7).mean()
    if abs(net_flow_7d) > features['net_flow'].std() * 2:
        direction = "OUTFLOW" if net_flow_7d > 0 else "INFLOW"
        alerts.append(f"💱 SIGNIFICANT EXCHANGE {direction} detected")
    
    return alerts
```

---

## 🔮 Будущие улучшения

1. **Дополнительные on-chain метрики**:
   - MVRV ratio
   - Realized cap
   - Dormancy flow
   - Entity-adjusted metrics

2. **Улучшенный sentiment анализ**:
   - Reddit sentiment analysis
   - YouTube/TikTok mentions
   - Google Trends integration
   - Options flow sentiment

3. **Machine Learning интеграция**:
   - Предсказание sentiment changes
   - Anomaly detection в on-chain данных
   - Regime change prediction

4. **Real-time мониторинг**:
   - WebSocket для live data
   - Dashboard для мониторинга
   - Telegram/Discord боты для алертов

---

**🎯 Заключение**: Эти модули значительно расширяют аналитические возможности вашей торговой системы, добавляя fundamental analysis к техническому анализу. Правильное использование on-chain метрик и sentiment анализа может существенно улучшить качество торговых сигналов и risk management. 
# Инструкции по использованию системы торговли криптовалютами с ML

## Обзор системы

Эта система представляет собой комплексное решение для торговли криптовалютами с использованием машинного обучения. Система поддерживает:

- **Множественные таймфреймы**: 5m, 15m, 30m, 1h, 4h
- **Сбор данных**: Автоматический сбор OHLCV данных с бирж через CCXT
- **Предобработка**: Технические индикаторы, балансировка данных (SMOTE)
- **ML модели**: XGBoost с оптимизацией гиперпараметров
- **Кэширование**: Эффективное хранение и управление данными
- **Модульность**: Легко расширяемая архитектура

## Структура проекта

```
WW/
├── config/
│   └── config.yaml          # Конфигурация системы
├── data/
│   └── cache/               # Кэш данных
├── logs/                    # Логи системы
├── models/                  # Сохраненные модели
├── notebooks/               # Jupyter notebooks
├── src/
│   ├── data_collection/     # Сбор данных
│   ├── preprocessing/       # Предобработка
│   ├── ml_models/          # ML модели
│   ├── backtesting/        # Бэктестинг
│   └── trading/            # Торговая система
├── main.py                 # Основной скрипт
├── requirements.txt        # Зависимости
└── README.md              # Документация
```

## Установка и настройка

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка конфигурации

Отредактируйте файл `config/config.yaml`:

```yaml
# Настройки сбора данных
data_collection:
  exchange: "binance"
  symbols: ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
  
  # Множественные таймфреймы
  timeframes:
    - "5m"   # 5 минут
    - "15m"  # 15 минут
    - "30m"  # 30 минут
    - "1h"   # 1 час
    - "4h"   # 4 часа
  
  # Основной таймфрейм для обучения
  primary_timeframe: "1h"
  
  # Количество свечей для каждого таймфрейма
  limits:
    5m: 2000
    15m: 1500
    30m: 1200
    1h: 1000
    4h: 500
```

### 3. Создание необходимых директорий

```bash
mkdir -p data/cache logs models notebooks
```

## Использование

### Запуск основной демонстрации

```bash
python main.py
```

Этот скрипт демонстрирует полный пайплайн:
1. Сбор данных с множественных таймфреймов
2. Предобработка и инженерная обработка признаков
3. Балансировка данных
4. Обучение XGBoost модели
5. Оценка производительности
6. Сохранение модели

### Программное использование

```python
import yaml
from src.data_collection.collector import DataCollector
from src.preprocessing.feature_engineering import FeatureEngineer
from src.ml_models.xgboost_model import XGBoostModel

# Загрузка конфигурации
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Инициализация компонентов
collector = DataCollector(config['data_collection'])
feature_engineer = FeatureEngineer(config['preprocessing'])
model = XGBoostModel(config['ml_model'])

# Сбор данных для множественных таймфреймов
symbol_data = collector.fetch_multi_timeframe_data('BTC/USDT')

# Обработка данных
processed_df = feature_engineer.process_multi_timeframe(symbol_data, 'BTC/USDT')

# Балансировка данных
balanced_df = feature_engineer.balance_data(processed_df, 'BTC/USDT')

# Подготовка для обучения
X, y = feature_engineer.prepare_for_training(balanced_df, 'BTC/USDT')

# Обучение модели
model.train(X, y, 'BTC/USDT')
```

## Конфигурация

### Настройки множественных таймфреймов

```yaml
preprocessing:
  multi_timeframe:
    enabled: true
    features_per_timeframe:
      5m:
        - "rsi"
        - "macd"
        - "bollinger_bands"
        - "sma_20"
        - "ema_20"
        - "volatility"
      15m:
        - "rsi"
        - "macd"
        - "bollinger_bands"
        - "sma_20"
        - "ema_20"
        - "volatility"
      30m:
        - "rsi"
        - "macd"
        - "bollinger_bands"
        - "sma_20"
        - "ema_20"
        - "volatility"
      1h:
        - "rsi"
        - "macd"
        - "bollinger_bands"
        - "sma_20"
        - "ema_20"
        - "volatility"
        - "sma_50"
        - "ema_50"
      4h:
        - "rsi"
        - "macd"
        - "bollinger_bands"
        - "sma_20"
        - "ema_20"
        - "volatility"
        - "sma_50"
        - "ema_50"
        - "sma_200"
        - "ema_200"
    
    prefixes:
      5m: "tf5m_"
      15m: "tf15m_"
      30m: "tf30m_"
      1h: "tf1h_"
      4h: "tf4h_"
```

### Настройки балансировки данных

```yaml
preprocessing:
  data_balancing:
    enabled: true
    method: "auto"  # auto, smote, adasyn, borderline_smote, smote_enn, none
    sampling_strategy: "auto"
    k_neighbors: 5
    random_state: 42
```

### Настройки оптимизации гиперпараметров

```yaml
ml_model:
  hyperparameter_optimization:
    enabled: true
    method: "bayesian"  # bayesian, grid, random
    metric: "f1"  # f1, accuracy, precision, recall
    n_calls: 50
    checkpoint_enabled: true
    checkpoint_name: "xgboost_optimization"
    
    search_space:
      max_depth: [3, 10]
      learning_rate: [0.01, 0.3]
      n_estimators: [50, 300]
      subsample: [0.6, 1.0]
      colsample_bytree: [0.6, 1.0]
      reg_alpha: [0.1, 10.0]
      reg_lambda: [0.1, 10.0]
      min_child_weight: [3, 10]
```

## Модули системы

### 1. Сбор данных (`src/data_collection/`)

- **collector.py**: Сбор OHLCV данных с бирж
- **cache.py**: Кэширование данных на диск

**Возможности:**
- Поддержка множественных бирж через CCXT
- Множественные таймфреймы (5m, 15m, 30m, 1h, 4h)
- Валидация и очистка данных
- Кэширование с автоматическим истечением

### 2. Предобработка (`src/preprocessing/`)

- **indicators.py**: Технические индикаторы
- **target_creation.py**: Создание целевой переменной
- **data_balancing.py**: Балансировка данных (SMOTE)
- **feature_engineering.py**: Инженерная обработка признаков
- **multi_timeframe.py**: Обработка множественных таймфреймов

**Возможности:**
- 20+ технических индикаторов
- Автоматическая балансировка данных
- Обработка множественных таймфреймов
- Создание признаков с префиксами

### 3. ML модели (`src/ml_models/`)

- **xgboost_model.py**: XGBoost модель
- **hyperparameter_optimizer.py**: Оптимизация гиперпараметров

**Возможности:**
- XGBoost с настраиваемыми параметрами
- Bayesian оптимизация гиперпараметров
- Чекпоинты и восстановление
- Визуализация результатов

## Технические индикаторы

Система поддерживает следующие индикаторы:

### Трендовые
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- MACD (Moving Average Convergence Divergence)

### Осцилляторы
- RSI (Relative Strength Index)
- Bollinger Bands
- Stochastic Oscillator

### Волатильность
- ATR (Average True Range)
- Historical Volatility

### Объемные
- Volume SMA
- Volume EMA
- Volume Rate of Change

### Ценовые
- Price Rate of Change
- Price Momentum
- Price Acceleration

## Методы балансировки данных

### SMOTE (Synthetic Minority Over-sampling Technique)
- Создает синтетические образцы для миноритарного класса
- Сохраняет структуру данных
- Эффективен для несбалансированных наборов

### ADASYN (Adaptive Synthetic Sampling)
- Адаптивная версия SMOTE
- Фокусируется на сложных образцах
- Лучше для сильно несбалансированных данных

### BorderlineSMOTE
- Фокусируется на граничных образцах
- Создает более качественные синтетические образцы
- Подходит для сложных границ классов

### SMOTEENN (SMOTE + Edited Nearest Neighbors)
- Комбинация SMOTE и очистки
- Удаляет шумовые образцы
- Улучшает качество данных

## Оптимизация гиперпараметров

### Bayesian Optimization
- Использует гауссовские процессы
- Эффективный поиск оптимальных параметров
- Поддерживает чекпоинты

### Метрики оптимизации
- F1-score (по умолчанию)
- Accuracy
- Precision
- Recall

### Пространство поиска
- max_depth: [3, 10]
- learning_rate: [0.01, 0.3]
- n_estimators: [50, 300]
- subsample: [0.6, 1.0]
- colsample_bytree: [0.6, 1.0]
- reg_alpha: [0.1, 10.0]
- reg_lambda: [0.1, 10.0]
- min_child_weight: [3, 10]

## Логирование

Система использует структурированное логирование:

```python
from loguru import logger

# Настройка логирования
logger.add("logs/trading_system.log", rotation="1 day", retention="30 days")
logger.add(lambda msg: print(msg, end=''), level="INFO")
```

## Мониторинг и статистика

### Статистика сбора данных
- Количество запросов
- Процент успеха
- Общее количество свечей
- Время выполнения

### Статистика кэша
- Hit rate
- Размер кэша
- Количество сохранений/загрузок

### Статистика обработки
- Количество признаков
- Соотношение классов до/после балансировки
- Количество обработанных таймфреймов

### Статистика модели
- Параметры модели
- Метрики оценки
- Важность признаков

## Расширение системы

### Добавление новых индикаторов

```python
# В src/preprocessing/indicators.py
def add_custom_indicator(self, df):
    """Добавление пользовательского индикатора"""
    df['custom_indicator'] = # ваша логика
    return df
```

### Добавление новых методов балансировки

```python
# В src/preprocessing/data_balancing.py
def custom_balancing_method(self, X, y):
    """Пользовательский метод балансировки"""
    # ваша логика
    return X_balanced, y_balanced
```

### Добавление новых моделей

```python
# Создать новый файл в src/ml_models/
class CustomModel:
    def __init__(self, config):
        self.config = config
    
    def train(self, X, y):
        # логика обучения
        pass
    
    def predict(self, X):
        # логика предсказания
        pass
```

## Устранение неполадок

### Проблемы с подключением к бирже
1. Проверьте интернет-соединение
2. Убедитесь, что биржа доступна
3. Проверьте лимиты API

### Проблемы с памятью
1. Уменьшите количество свечей в конфигурации
2. Используйте меньше таймфреймов
3. Очистите кэш

### Проблемы с балансировкой
1. Проверьте распределение классов
2. Попробуйте другой метод балансировки
3. Увеличьте количество данных

### Проблемы с оптимизацией
1. Уменьшите пространство поиска
2. Увеличьте количество итераций
3. Проверьте метрику оптимизации

## Производительность

### Рекомендации по оптимизации
1. Используйте кэширование данных
2. Настройте количество свечей для каждого таймфрейма
3. Выбирайте подходящие методы балансировки
4. Используйте чекпоинты для оптимизации

### Мониторинг ресурсов
- Память: ~2-5 GB для полного пайплайна
- Время: 5-15 минут для полного цикла
- Дисковое пространство: ~1-2 GB для кэша

## Безопасность

### Рекомендации
1. Не храните API ключи в коде
2. Используйте виртуальные окружения
3. Регулярно обновляйте зависимости
4. Мониторьте логи на предмет ошибок

## Поддержка

### Логи
- Основные логи: `logs/trading_system.log`
- Ротация: ежедневно
- Хранение: 30 дней

### Отладка
```python
# Включение детального логирования
logger.add("logs/debug.log", level="DEBUG")
```

### Мониторинг
- Статистика в реальном времени
- Метрики производительности
- Алерты при ошибках 
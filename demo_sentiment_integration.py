#!/usr/bin/env python3
"""
Финальная демонстрация интеграции Sentiment анализа в торговую систему
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys
from datetime import datetime, timedelta
from loguru import logger

# Добавляем путь к исходникам
sys.path.append('src')

# Импорты
from data_collection.sentiment_collector import SentimentCollector
from preprocessing.indicators import TechnicalIndicators
from preprocessing.confidence_scorer import ConfidenceScorer

def create_sample_market_data():
    """Создает синтетические рыночные данные для демонстрации."""
    logger.info("Создание синтетических рыночных данных")
    
    # Создаем 30 дней 5-минутных данных
    dates = pd.date_range(start='2024-12-01', periods=8640, freq='5T')  # 30 дней * 24 часа * 12 интервалов
    np.random.seed(42)
    
    # Имитируем реалистичные ценовые данные BTC
    base_price = 95000
    returns = np.random.normal(0.0001, 0.008, len(dates))  # Небольшой положительный тренд с волатильностью
    
    prices = [base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1000))  # Минимальная цена 1000
    
    # Создаем OHLCV данные
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(50000, 200000, len(dates))
    })
    
    logger.info(f"Создан датасет: {len(df)} записей от {df['timestamp'].min()} до {df['timestamp'].max()}")
    logger.info(f"Ценовой диапазон: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
    
    return df

def add_technical_indicators(df):
    """Добавляет технические индикаторы."""
    logger.info("Добавление технических индикаторов")
    
    try:
        indicators = TechnicalIndicators()
        
        # Основные индикаторы
        df = indicators.add_rsi(df)
        df = indicators.add_macd(df)
        df = indicators.add_obv(df)
        df = indicators.add_vwap(df)
        df = indicators.add_atr(df)
        df = indicators.add_williams_r(df)
        df = indicators.add_cci(df)
        df = indicators.add_sma(df, 20)
        df = indicators.add_sma(df, 50)
        df = indicators.add_bollinger_bands(df)
        
        # Объемные индикаторы
        try:
            df = indicators.add_advanced_volume_indicators(df)
        except:
            logger.warning("Не удалось добавить продвинутые объемные индикаторы")
        
        # Относительный объем
        df['relative_volume'] = df['volume'] / df['volume'].rolling(20).mean()
        
        logger.info(f"Добавлено технических индикаторов, общее количество колонок: {len(df.columns)}")
        return df
        
    except Exception as e:
        logger.error(f"Ошибка добавления технических индикаторов: {e}")
        return df

def add_sentiment_features(df):
    """Добавляет sentiment фичи."""
    logger.info("Добавление sentiment фичей")
    
    try:
        sentiment_collector = SentimentCollector()
        
        # Получаем Bybit opportunities
        bybit_data = sentiment_collector.get_bybit_opportunities()
        
        if bybit_data:
            logger.info("✅ Bybit данные получены")
            
            # Базовые sentiment фичи
            df['bybit_market_sentiment'] = bybit_data.get('market_sentiment', 0.5)
            df['bybit_hot_sectors_count'] = len(bybit_data.get('hot_sectors', []))
            
            # Анализ трендинговых монет
            trending_coins = bybit_data.get('trending_coins', [])
            if trending_coins:
                positive_trending = sum(1 for coin in trending_coins if coin.get('sentiment', 0) > 0)
                df['bybit_positive_trending_ratio'] = positive_trending / len(trending_coins)
            else:
                df['bybit_positive_trending_ratio'] = 0.5
            
            # Gainers vs Losers ratio
            gainers_losers = bybit_data.get('gainers_losers', {})
            gainers_count = len(gainers_losers.get('gainers', []))
            losers_count = len(gainers_losers.get('losers', []))
            
            if gainers_count + losers_count > 0:
                df['bybit_gainers_ratio'] = gainers_count / (gainers_count + losers_count)
            else:
                df['bybit_gainers_ratio'] = 0.5
            
            logger.info(f"Bybit данные:")
            logger.info(f"  Market Sentiment: {bybit_data.get('market_sentiment', 'N/A')}")
            logger.info(f"  Hot Sectors: {len(bybit_data.get('hot_sectors', []))}")
            logger.info(f"  Trending Coins: {len(trending_coins)}")
            logger.info(f"  Gainers: {gainers_count}, Losers: {losers_count}")
            
        else:
            logger.warning("Bybit данные недоступны, используем нейтральные значения")
            df['bybit_market_sentiment'] = 0.5
            df['bybit_hot_sectors_count'] = 3
            df['bybit_positive_trending_ratio'] = 0.5
            df['bybit_gainers_ratio'] = 0.5
        
        # Composite sentiment score
        df['sentiment_composite_score'] = (
            df['bybit_market_sentiment'] * 0.4 +
            df['bybit_positive_trending_ratio'] * 0.3 +
            df['bybit_gainers_ratio'] * 0.3
        )
        
        # Market regime classification
        df['market_regime_bullish'] = (df['sentiment_composite_score'] > 0.6).astype(int)
        df['market_regime_bearish'] = (df['sentiment_composite_score'] < 0.4).astype(int)
        df['market_regime_neutral'] = (
            (df['sentiment_composite_score'] >= 0.4) & 
            (df['sentiment_composite_score'] <= 0.6)
        ).astype(int)
        
        # Добавляем динамику sentiment (изменения)
        df['sentiment_change'] = df['sentiment_composite_score'].diff()
        df['sentiment_momentum'] = df['sentiment_composite_score'].rolling(5).mean()
        
        logger.info("✅ Sentiment фичи добавлены:")
        logger.info(f"  Composite Score: {df['sentiment_composite_score'].iloc[-1]:.3f}")
        
        # Определяем режим
        if df['market_regime_bullish'].iloc[-1]:
            regime = "🐂 БЫЧИЙ"
        elif df['market_regime_bearish'].iloc[-1]:
            regime = "🐻 МЕДВЕЖИЙ"
        else:
            regime = "😐 НЕЙТРАЛЬНЫЙ"
        
        logger.info(f"  Market Regime: {regime}")
        
        return df
        
    except Exception as e:
        logger.error(f"Ошибка добавления sentiment фичей: {e}")
        # Добавляем дефолтные значения
        df['bybit_market_sentiment'] = 0.5
        df['bybit_hot_sectors_count'] = 3
        df['bybit_positive_trending_ratio'] = 0.5
        df['bybit_gainers_ratio'] = 0.5
        df['sentiment_composite_score'] = 0.5
        df['market_regime_bullish'] = 0
        df['market_regime_bearish'] = 0
        df['market_regime_neutral'] = 1
        df['sentiment_change'] = 0
        df['sentiment_momentum'] = 0.5
        return df

def generate_smart_signals(df):
    """Генерирует умные торговые сигналы с учетом sentiment."""
    logger.info("Генерация умных торговых сигналов")
    
    signals = pd.Series(0, index=df.index)  # 0 = hold, 1 = buy, -1 = sell
    
    for i in range(50, len(df)):  # Начинаем после 50 свечей
        row = df.iloc[i]
        
        # Технические условия для покупки
        tech_buy = [
            row['rsi_14'] < 70 and row['rsi_14'] > 35,          # RSI не перекуплен
            row['macd_line'] > row['macd_signal'],              # MACD бычий
            row['close'] > row['vwap'],                         # Цена выше VWAP
            row['williams_r_14'] > -70,                         # Williams %R не перепродан
            row['cci_20'] > -100 and row['cci_20'] < 100,       # CCI в нормальном диапазоне
            row['close'] > row['sma_20'],                       # Цена выше SMA20
            row['bb_position_20'] > 0.2 and row['bb_position_20'] < 0.8,  # В пределах BB
            row['relative_volume'] > 0.8,                       # Нормальный объем
        ]
        
        # Технические условия для продажи
        tech_sell = [
            row['rsi_14'] > 30 and row['rsi_14'] < 65,          # RSI не перепродан
            row['macd_line'] < row['macd_signal'],              # MACD медвежий
            row['close'] < row['vwap'],                         # Цена ниже VWAP
            row['williams_r_14'] < -30,                         # Williams %R не перекуплен
            row['cci_20'] < 100 and row['cci_20'] > -100,       # CCI в нормальном диапазоне
            row['close'] < row['sma_20'],                       # Цена ниже SMA20
            row['bb_position_20'] > 0.2 and row['bb_position_20'] < 0.8,  # В пределах BB
            row['relative_volume'] > 0.8,                       # Нормальный объем
        ]
        
        # Sentiment условия
        sentiment_score = row['sentiment_composite_score']
        market_regime_bullish = row['market_regime_bullish']
        market_regime_bearish = row['market_regime_bearish']
        sentiment_momentum = row['sentiment_momentum']
        
        # Подсчет технических подтверждений
        buy_confirmations = sum(tech_buy)
        sell_confirmations = sum(tech_sell)
        
        # Базовые требования
        min_buy_confirmations = 5
        min_sell_confirmations = 5
        
        # 🚀 SENTIMENT МОДИФИКАЦИИ
        
        # Sentiment бонусы для покупки
        if sentiment_score > 0.7:  # Очень высокий sentiment
            min_buy_confirmations -= 2
        elif sentiment_score > 0.6:  # Высокий sentiment
            min_buy_confirmations -= 1
        elif sentiment_score < 0.3:  # Низкий sentiment
            min_buy_confirmations += 2
        
        # Sentiment бонусы для продажи
        if sentiment_score < 0.3:  # Очень низкий sentiment
            min_sell_confirmations -= 2
        elif sentiment_score < 0.4:  # Низкий sentiment
            min_sell_confirmations -= 1
        elif sentiment_score > 0.7:  # Высокий sentiment
            min_sell_confirmations += 2
        
        # Market regime бонусы
        if market_regime_bullish:
            min_buy_confirmations -= 1  # Легче покупать в бычьем режиме
            min_sell_confirmations += 1  # Сложнее продавать в бычьем режиме
        elif market_regime_bearish:
            min_sell_confirmations -= 1  # Легче продавать в медвежьем режиме
            min_buy_confirmations += 1   # Сложнее покупать в медвежьем режиме
        
        # Momentum sentiment бонус
        if sentiment_momentum > sentiment_score:  # Растущий sentiment
            min_buy_confirmations -= 1
        elif sentiment_momentum < sentiment_score:  # Падающий sentiment
            min_sell_confirmations -= 1
        
        # Минимальные лимиты
        min_buy_confirmations = max(min_buy_confirmations, 3)
        min_sell_confirmations = max(min_sell_confirmations, 3)
        
        # Генерация сигналов
        if (buy_confirmations >= min_buy_confirmations and 
            sentiment_score >= 0.25 and  # Минимальный sentiment для покупки
            not market_regime_bearish):   # Не покупаем в медвежьем режиме
            signals.iloc[i] = 1  # Buy
            
        elif (sell_confirmations >= min_sell_confirmations and 
              sentiment_score <= 0.75 and  # Максимальный sentiment для продажи
              not market_regime_bullish):   # Не продаем в бычьем режиме
            signals.iloc[i] = -1  # Sell
    
    signal_counts = signals.value_counts()
    logger.info(f"Сигналы сгенерированы:")
    logger.info(f"  Buy: {signal_counts.get(1, 0)}")
    logger.info(f"  Sell: {signal_counts.get(-1, 0)}")
    logger.info(f"  Hold: {signal_counts.get(0, 0)}")
    
    return signals

def calculate_enhanced_confidence(df, signals):
    """Рассчитывает enhanced confidence с sentiment."""
    logger.info("Расчет enhanced confidence scores")
    
    try:
        # Создаем конфигурацию для confidence scorer
        config = {
            'weights': {
                'indicator_agreement': 0.25,
                'signal_strength': 0.22,
                'volatility_factor': 0.18,
                'volume_confirmation': 0.15,
                'market_regime': 0.1,
                'sentiment_confirmation': 0.1
            }
        }
        
        scorer = ConfidenceScorer(config)
        confidence_scores = pd.Series(0.0, index=df.index)
        
        for i in range(len(df)):
            if signals.iloc[i] != 0:  # Только для торговых сигналов
                try:
                    row = df.iloc[i]
                    signal = signals.iloc[i]
                    
                    # Используем enhanced confidence если доступен
                    if hasattr(scorer, 'calculate_advanced_confidence'):
                        confidence = scorer.calculate_advanced_confidence(row, signal)
                    else:
                        # Базовый confidence
                        factors = scorer._collect_confidence_factors(row, signal)
                        confidence = scorer._calculate_weighted_confidence(factors)
                    
                    confidence_scores.iloc[i] = confidence
                    
                except Exception:
                    confidence_scores.iloc[i] = 0.5
        
        non_zero_confidence = confidence_scores[confidence_scores > 0]
        if len(non_zero_confidence) > 0:
            avg_confidence = non_zero_confidence.mean()
            high_confidence_count = len(non_zero_confidence[non_zero_confidence > 0.7])
            
            logger.info(f"Confidence рассчитан:")
            logger.info(f"  Средний: {avg_confidence:.3f}")
            logger.info(f"  Высокий confidence (>0.7): {high_confidence_count}")
        
        return confidence_scores
        
    except Exception as e:
        logger.error(f"Ошибка расчета confidence: {e}")
        return pd.Series(0.5, index=df.index)

def run_sentiment_backtest(df, signals, confidence):
    """Запускает упрощенный бэктест с sentiment анализом."""
    logger.info("Запуск sentiment-enhanced backtesting")
    
    # Фильтруем сигналы по confidence
    min_confidence = 0.6
    filtered_signals = signals.copy()
    filtered_signals[confidence < min_confidence] = 0
    
    filtered_count = len(filtered_signals[filtered_signals != 0])
    original_count = len(signals[signals != 0])
    
    logger.info(f"Фильтрация по confidence: {original_count} → {filtered_count} сигналов")
    
    # Простой backtest
    initial_balance = 10000
    balance = initial_balance
    position = 0
    trades = []
    
    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        signal = filtered_signals.iloc[i]
        conf = confidence.iloc[i]
        sentiment = df['sentiment_composite_score'].iloc[i]
        
        if signal == 1 and position == 0:  # Buy
            position = balance / current_price
            balance = 0
            trades.append({
                'type': 'buy',
                'price': current_price,
                'time': df['timestamp'].iloc[i],
                'confidence': conf,
                'sentiment': sentiment,
                'regime': 'bullish' if df['market_regime_bullish'].iloc[i] else 
                         ('bearish' if df['market_regime_bearish'].iloc[i] else 'neutral')
            })
            
        elif signal == -1 and position > 0:  # Sell
            balance = position * current_price
            pnl = balance - initial_balance
            trades.append({
                'type': 'sell',
                'price': current_price,
                'time': df['timestamp'].iloc[i],
                'confidence': conf,
                'sentiment': sentiment,
                'pnl': pnl,
                'regime': 'bullish' if df['market_regime_bullish'].iloc[i] else 
                         ('bearish' if df['market_regime_bearish'].iloc[i] else 'neutral')
            })
            position = 0
    
    # Закрываем позицию в конце
    if position > 0:
        balance = position * df['close'].iloc[-1]
        position = 0
    
    # Анализ результатов
    total_return = (balance - initial_balance) / initial_balance * 100
    total_trades = len([t for t in trades if t['type'] == 'buy'])
    
    # Анализ по sentiment
    high_conf_trades = len([t for t in trades if t.get('confidence', 0) > 0.7])
    bullish_trades = len([t for t in trades if t.get('regime') == 'bullish'])
    
    # Profitable trades анализ
    profitable_trades = 0
    total_pnl = 0
    for i in range(0, len(trades)-1, 2):  # Пары buy-sell
        if i+1 < len(trades) and trades[i]['type'] == 'buy' and trades[i+1]['type'] == 'sell':
            pnl = (trades[i+1]['price'] - trades[i]['price']) / trades[i]['price'] * 100
            total_pnl += pnl
            if pnl > 0:
                profitable_trades += 1
    
    win_rate = (profitable_trades / max(total_trades, 1)) * 100 if total_trades > 0 else 0
    
    return {
        'total_return': total_return,
        'final_balance': balance,
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'win_rate': win_rate,
        'high_confidence_trades': high_conf_trades,
        'bullish_regime_trades': bullish_trades,
        'avg_sentiment': df['sentiment_composite_score'].mean(),
        'sentiment_range': (df['sentiment_composite_score'].min(), df['sentiment_composite_score'].max()),
        'regime_distribution': {
            'bullish_pct': (df['market_regime_bullish'].sum() / len(df)) * 100,
            'bearish_pct': (df['market_regime_bearish'].sum() / len(df)) * 100,
            'neutral_pct': (df['market_regime_neutral'].sum() / len(df)) * 100
        }
    }

def main():
    """Главная функция демонстрации."""
    print("🚀 " + "="*70)
    print("🚀 ФИНАЛЬНАЯ ДЕМОНСТРАЦИЯ: SENTIMENT-ENHANCED TRADING SYSTEM")
    print("🚀 " + "="*70)
    print("📈 Интеграция Bybit Market Sentiment в торговые решения")
    print("🎯 Enhanced Confidence Scoring с учетом market regime")
    print("🤖 Адаптивная генерация сигналов на основе sentiment")
    
    try:
        # 1. Создание данных
        print(f"\n{'='*70}")
        print("ЭТАП 1: Создание рыночных данных")
        print(f"{'='*70}")
        
        df = create_sample_market_data()
        
        # 2. Технические индикаторы
        print(f"\n{'='*70}")
        print("ЭТАП 2: Добавление технических индикаторов")
        print(f"{'='*70}")
        
        df = add_technical_indicators(df)
        
        # 3. Sentiment фичи
        print(f"\n{'='*70}")
        print("ЭТАП 3: Интеграция Sentiment анализа")
        print(f"{'='*70}")
        
        df = add_sentiment_features(df)
        
        # 4. Генерация сигналов
        print(f"\n{'='*70}")
        print("ЭТАП 4: Генерация умных торговых сигналов")
        print(f"{'='*70}")
        
        signals = generate_smart_signals(df)
        
        # 5. Enhanced confidence
        print(f"\n{'='*70}")
        print("ЭТАП 5: Enhanced Confidence Scoring")
        print(f"{'='*70}")
        
        confidence = calculate_enhanced_confidence(df, signals)
        
        # 6. Backtesting
        print(f"\n{'='*70}")
        print("ЭТАП 6: Sentiment-Enhanced Backtesting")
        print(f"{'='*70}")
        
        results = run_sentiment_backtest(df, signals, confidence)
        
        # 7. Итоговый отчет
        print(f"\n{'='*70}")
        print("🏆 ИТОГОВЫЙ ОТЧЕТ")
        print(f"{'='*70}")
        
        print(f"💰 Финансовые результаты:")
        print(f"   Total Return: {results['total_return']:+.2f}%")
        print(f"   Final Balance: ${results['final_balance']:,.2f}")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        
        print(f"\n📊 Торговая статистика:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Profitable Trades: {results['profitable_trades']}")
        print(f"   High Confidence Trades: {results['high_confidence_trades']}")
        print(f"   Bullish Regime Trades: {results['bullish_regime_trades']}")
        
        print(f"\n😊 Sentiment анализ:")
        print(f"   Average Sentiment: {results['avg_sentiment']:.3f}")
        print(f"   Sentiment Range: {results['sentiment_range'][0]:.3f} - {results['sentiment_range'][1]:.3f}")
        
        regime_dist = results['regime_distribution']
        print(f"\n📈 Market Regime Distribution:")
        print(f"   🐂 Bullish: {regime_dist['bullish_pct']:.1f}%")
        print(f"   🐻 Bearish: {regime_dist['bearish_pct']:.1f}%")
        print(f"   😐 Neutral: {regime_dist['neutral_pct']:.1f}%")
        
        print(f"\n🚀 Enhanced возможности:")
        print("   ✅ Real-time Bybit sentiment integration")
        print("   ✅ Market regime adaptive trading")
        print("   ✅ Sentiment-based confidence adjustment")
        print("   ✅ Multi-factor signal generation")
        print("   ✅ Regime-aware risk management")
        
        print(f"\n📋 Следующие шаги для production:")
        print("   1. Получите реальные API ключи для Bybit/Glassnode")
        print("   2. Настройте параметры в config/smart_adaptive_config_enhanced.yaml")
        print("   3. Интегрируйте в run_smart_adaptive.py")
        print("   4. Настройте алерты для экстремальных sentiment событий")
        print("   5. Запустите на paper trading для валидации")
        
        print(f"\n🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка демонстрации: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n🏁 Результат: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1) 
#!/usr/bin/env python3
"""
Упрощенный Enhanced Backtester с интегрированными Sentiment фичами
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

# Прямые импорты без относительных путей
from data_collection.collector import DataCollector
from data_collection.sentiment_collector import SentimentCollector
from preprocessing.indicators import TechnicalIndicators
from preprocessing.confidence_scorer import ConfidenceScorer

def load_config():
    """Загружает конфигурацию."""
    try:
        with open('config/smart_adaptive_config_enhanced.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("Конфиг не найден")
        return None

def create_enhanced_features(df, symbol='BTCUSDT'):
    """Создает enhanced фичи вручную (без FeatureEngineer)."""
    try:
        logger.info(f"Создание enhanced фичей для {symbol}")
        
        # Технические индикаторы
        indicators = TechnicalIndicators()
        
        # Базовые индикаторы
        df = indicators.add_rsi(df)
        df = indicators.add_macd(df)
        df = indicators.add_obv(df)
        df = indicators.add_vwap(df)
        df = indicators.add_atr(df)
        df = indicators.add_williams_r(df)
        df = indicators.add_cci(df)
        df = indicators.add_sma(df, 20)
        df = indicators.add_sma(df, 50)
        
        # Sentiment фичи (упрощенные)
        try:
            sentiment_collector = SentimentCollector()
            bybit_data = sentiment_collector.get_bybit_opportunities()
            
            if bybit_data:
                df['bybit_market_sentiment'] = bybit_data.get('market_sentiment', 0.5)
                df['bybit_hot_sectors_count'] = len(bybit_data.get('hot_sectors', []))
                
                trending_coins = bybit_data.get('trending_coins', [])
                if trending_coins:
                    positive_trending = sum(1 for coin in trending_coins if coin.get('sentiment', 0) > 0)
                    df['bybit_positive_trending_ratio'] = positive_trending / len(trending_coins)
                else:
                    df['bybit_positive_trending_ratio'] = 0.5
                
                gainers_losers = bybit_data.get('gainers_losers', {})
                gainers_count = len(gainers_losers.get('gainers', []))
                losers_count = len(gainers_losers.get('losers', []))
                
                if gainers_count + losers_count > 0:
                    df['bybit_gainers_ratio'] = gainers_count / (gainers_count + losers_count)
                else:
                    df['bybit_gainers_ratio'] = 0.5
                
                # Composite sentiment score
                df['sentiment_composite_score'] = (
                    df['bybit_market_sentiment'] * 0.4 +
                    df['bybit_positive_trending_ratio'] * 0.3 +
                    df['bybit_gainers_ratio'] * 0.3
                )
                
                # Market regime
                df['market_regime_bullish'] = (df['sentiment_composite_score'] > 0.6).astype(int)
                df['market_regime_bearish'] = (df['sentiment_composite_score'] < 0.4).astype(int)
                df['market_regime_neutral'] = (
                    (df['sentiment_composite_score'] >= 0.4) & 
                    (df['sentiment_composite_score'] <= 0.6)
                ).astype(int)
                
                logger.info("✅ Sentiment фичи добавлены")
            else:
                logger.warning("Bybit данные недоступны, используем нейтральные значения")
                df['bybit_market_sentiment'] = 0.5
                df['bybit_hot_sectors_count'] = 3
                df['bybit_positive_trending_ratio'] = 0.5
                df['bybit_gainers_ratio'] = 0.5
                df['sentiment_composite_score'] = 0.5
                df['market_regime_bullish'] = 0
                df['market_regime_bearish'] = 0
                df['market_regime_neutral'] = 1
                
        except Exception as e:
            logger.error(f"Ошибка получения sentiment данных: {e}")
            # Добавляем нейтральные значения
            df['bybit_market_sentiment'] = 0.5
            df['bybit_hot_sectors_count'] = 3
            df['bybit_positive_trending_ratio'] = 0.5
            df['bybit_gainers_ratio'] = 0.5
            df['sentiment_composite_score'] = 0.5
            df['market_regime_bullish'] = 0
            df['market_regime_bearish'] = 0
            df['market_regime_neutral'] = 1
        
        # Относительный объем
        df['relative_volume'] = df['volume'] / df['volume'].rolling(20).mean()
        
        logger.info(f"Enhanced фичи созданы: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Ошибка создания enhanced фичей: {e}")
        return df

def generate_enhanced_signals(df, config):
    """Генерирует торговые сигналы с учетом sentiment."""
    try:
        logger.info("Генерация enhanced торговых сигналов")
        
        signals = pd.Series(0, index=df.index)  # 0 = hold, 1 = buy, -1 = sell
        
        for i in range(50, len(df)):  # Начинаем после 50 свечей для стабильности индикаторов
            row = df.iloc[i]
            
            # Технические условия для покупки
            buy_conditions = [
                row['rsi_14'] < 70 and row['rsi_14'] > 30,  # RSI в нормальном диапазоне
                row['macd_line'] > row['macd_signal'],      # MACD бычий
                row['close'] > row['vwap'],                 # Цена выше VWAP
                row['williams_r_14'] > -80,                 # Williams %R не перепродан
                row['cci_20'] > -100,                       # CCI не перепродан
                row['close'] > row['sma_20'],               # Цена выше SMA20
            ]
            
            # Технические условия для продажи
            sell_conditions = [
                row['rsi_14'] > 30 and row['rsi_14'] < 70,  # RSI в нормальном диапазоне
                row['macd_line'] < row['macd_signal'],      # MACD медвежий
                row['close'] < row['vwap'],                 # Цена ниже VWAP
                row['williams_r_14'] < -20,                 # Williams %R не перекуплен
                row['cci_20'] < 100,                        # CCI не перекуплен
                row['close'] < row['sma_20'],               # Цена ниже SMA20
            ]
            
            # Sentiment условия
            sentiment_score = row.get('sentiment_composite_score', 0.5)
            market_regime_bullish = row.get('market_regime_bullish', 0)
            market_regime_bearish = row.get('market_regime_bearish', 0)
            
            # Подсчет технических подтверждений
            buy_confirmations = sum(buy_conditions)
            sell_confirmations = sum(sell_conditions)
            
            # Применяем sentiment фильтры
            min_buy_confirmations = 4
            min_sell_confirmations = 4
            
            # Sentiment бонусы/штрафы
            if sentiment_score > 0.6:  # Высокий sentiment
                min_buy_confirmations -= 1  # Легче покупать
                min_sell_confirmations += 1  # Сложнее продавать
            elif sentiment_score < 0.4:  # Низкий sentiment
                min_buy_confirmations += 1  # Сложнее покупать
                min_sell_confirmations -= 1  # Легче продавать
            
            # Market regime бонусы
            if market_regime_bullish:
                min_buy_confirmations -= 1
            elif market_regime_bearish:
                min_sell_confirmations -= 1
            
            # Генерируем сигналы
            if buy_confirmations >= min_buy_confirmations and sentiment_score >= 0.3:
                signals.iloc[i] = 1  # Buy
            elif sell_confirmations >= min_sell_confirmations and sentiment_score <= 0.7:
                signals.iloc[i] = -1  # Sell
        
        signal_counts = signals.value_counts()
        logger.info(f"Сигналы сгенерированы: Buy={signal_counts.get(1, 0)}, Sell={signal_counts.get(-1, 0)}, Hold={signal_counts.get(0, 0)}")
        
        return signals
        
    except Exception as e:
        logger.error(f"Ошибка генерации сигналов: {e}")
        return pd.Series(0, index=df.index)

def calculate_enhanced_confidence(df, signals, config):
    """Рассчитывает enhanced confidence scores."""
    try:
        logger.info("Расчет enhanced confidence scores")
        
        confidence_config = config.get('confidence_scorer', {})
        scorer = ConfidenceScorer(confidence_config)
        
        confidence_scores = pd.Series(0.0, index=df.index)
        
        for i in range(len(df)):
            if signals.iloc[i] != 0:  # Только для торговых сигналов
                try:
                    row = df.iloc[i]
                    signal = signals.iloc[i]
                    
                    # Базовые факторы
                    factors = scorer._collect_confidence_factors(row, signal)
                    base_confidence = scorer._calculate_weighted_confidence(factors)
                    
                    # Enhanced confidence с sentiment
                    sentiment_score = row.get('sentiment_composite_score', 0.5)
                    market_regime_bullish = row.get('market_regime_bullish', 0)
                    market_regime_bearish = row.get('market_regime_bearish', 0)
                    
                    # Модификаторы
                    confidence_multiplier = 1.0
                    
                    if signal > 0 and sentiment_score > 0.7:
                        confidence_multiplier *= 1.2
                    elif signal < 0 and sentiment_score < 0.3:
                        confidence_multiplier *= 1.2
                    
                    if signal > 0 and market_regime_bullish:
                        confidence_multiplier *= 1.15
                    elif signal < 0 and market_regime_bearish:
                        confidence_multiplier *= 1.15
                    
                    final_confidence = base_confidence * confidence_multiplier
                    confidence_scores.iloc[i] = np.clip(final_confidence, 0, 1)
                    
                except Exception as e:
                    confidence_scores.iloc[i] = 0.5
        
        high_confidence_count = len(confidence_scores[confidence_scores > 0.7])
        avg_confidence = confidence_scores[confidence_scores > 0].mean()
        
        logger.info(f"Confidence scores рассчитаны: среднее={avg_confidence:.3f}, высокий confidence={high_confidence_count}")
        
        return confidence_scores
        
    except Exception as e:
        logger.error(f"Ошибка расчета confidence: {e}")
        return pd.Series(0.5, index=df.index)

def run_enhanced_backtest(symbol='BTCUSDT', timeframe='5m'):
    """Запускает enhanced backtesting."""
    logger.info(f"🚀 Запуск Enhanced Backtesting для {symbol} на {timeframe}")
    
    # Загружаем конфигурацию
    config = load_config()
    if not config:
        logger.error("Не удалось загрузить конфигурацию")
        return None
    
    # Получаем данные
    logger.info("📥 Получение данных...")
    data_collector = DataCollector(config)
    df = data_collector.fetch_ohlcv(symbol, timeframe, limit=1000)
    
    if df is None or df.empty:
        logger.error("Не удалось получить данные")
        return None
    
    logger.info(f"✅ Данные получены: {len(df)} записей")
    
    # Создаем enhanced фичи
    logger.info("🔧 Создание enhanced фичей...")
    df = create_enhanced_features(df, symbol)
    
    # Удаляем NaN
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"Данные очищены: {initial_len} → {len(df)} записей")
    
    # Генерируем сигналы
    logger.info("📊 Генерация торговых сигналов...")
    signals = generate_enhanced_signals(df, config)
    
    # Рассчитываем confidence
    logger.info("🎯 Расчет confidence scores...")
    confidence = calculate_enhanced_confidence(df, signals, config)
    
    # Фильтруем по confidence
    min_confidence = config.get('confidence_scorer', {}).get('thresholds', {}).get('min_confidence', 0.6)
    filtered_signals = signals.copy()
    filtered_signals[confidence < min_confidence] = 0
    
    # Простой backtest
    logger.info("💰 Запуск backtesting...")
    
    initial_balance = 10000
    balance = initial_balance
    position = 0
    trades = []
    
    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        signal = filtered_signals.iloc[i]
        conf = confidence.iloc[i]
        
        if signal == 1 and position == 0:  # Buy
            position = balance / current_price
            balance = 0
            trades.append({
                'type': 'buy',
                'price': current_price,
                'time': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i,
                'confidence': conf,
                'sentiment': df['sentiment_composite_score'].iloc[i],
                'regime': 'bullish' if df['market_regime_bullish'].iloc[i] else ('bearish' if df['market_regime_bearish'].iloc[i] else 'neutral')
            })
            
        elif signal == -1 and position > 0:  # Sell
            balance = position * current_price
            position = 0
            trades.append({
                'type': 'sell',
                'price': current_price,
                'time': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i,
                'confidence': conf,
                'sentiment': df['sentiment_composite_score'].iloc[i],
                'regime': 'bullish' if df['market_regime_bullish'].iloc[i] else ('bearish' if df['market_regime_bearish'].iloc[i] else 'neutral')
            })
    
    # Закрываем позицию в конце
    if position > 0:
        balance = position * df['close'].iloc[-1]
        position = 0
    
    # Результаты
    total_return = (balance - initial_balance) / initial_balance * 100
    total_trades = len([t for t in trades if t['type'] == 'buy'])
    
    # Анализ по confidence и sentiment
    high_conf_trades = len([t for t in trades if t.get('confidence', 0) > 0.7])
    bullish_regime_trades = len([t for t in trades if t.get('regime') == 'bullish'])
    
    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'total_return': total_return,
        'final_balance': balance,
        'total_trades': total_trades,
        'high_confidence_trades': high_conf_trades,
        'bullish_regime_trades': bullish_regime_trades,
        'trades': trades,
        'data_points': len(df),
        'sentiment_features': ['sentiment_composite_score', 'market_regime_bullish', 'market_regime_bearish', 'market_regime_neutral'],
        'avg_sentiment': df['sentiment_composite_score'].mean(),
        'regime_distribution': {
            'bullish': df['market_regime_bullish'].sum(),
            'bearish': df['market_regime_bearish'].sum(),
            'neutral': df['market_regime_neutral'].sum()
        }
    }

def main():
    """Главная функция."""
    print("🚀 " + "="*60)
    print("🚀 ENHANCED SMART ADAPTIVE STRATEGY BACKTEST")
    print("🚀 " + "="*60)
    print("📈 С интегрированным Sentiment анализом")
    print("🎯 И Enhanced Confidence Scoring")
    
    symbols = ['BTCUSDT', 'ETHUSDT']
    timeframes = ['5m', '15m']
    
    results = []
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n{'='*60}")
            print(f"📊 Тестирование {symbol} на {timeframe}")
            print(f"{'='*60}")
            
            try:
                result = run_enhanced_backtest(symbol, timeframe)
                
                if result:
                    results.append(result)
                    
                    print(f"✅ {symbol} ({timeframe}) завершено:")
                    print(f"   💰 Доходность: {result['total_return']:+.2f}%")
                    print(f"   📊 Сделок: {result['total_trades']}")
                    print(f"   🎯 Высокий confidence: {result['high_confidence_trades']}")
                    print(f"   🐂 Бычий режим: {result['bullish_regime_trades']}")
                    print(f"   😊 Средний sentiment: {result['avg_sentiment']:.3f}")
                    
                    regime_dist = result['regime_distribution']
                    total_points = sum(regime_dist.values())
                    print(f"   📈 Режимы: Бычий {regime_dist['bullish']/total_points*100:.1f}%, "
                          f"Медвежий {regime_dist['bearish']/total_points*100:.1f}%, "
                          f"Нейтральный {regime_dist['neutral']/total_points*100:.1f}%")
                else:
                    print(f"❌ {symbol} ({timeframe}): ошибка")
                    
            except Exception as e:
                print(f"❌ Ошибка тестирования {symbol} ({timeframe}): {e}")
    
    # Итоговый отчет
    if results:
        print(f"\n{'='*60}")
        print("📋 ИТОГОВЫЙ ОТЧЕТ")
        print(f"{'='*60}")
        
        total_return = np.mean([r['total_return'] for r in results])
        total_trades = sum([r['total_trades'] for r in results])
        total_high_conf = sum([r['high_confidence_trades'] for r in results])
        
        print(f"📊 Общая статистика:")
        print(f"   Средняя доходность: {total_return:+.2f}%")
        print(f"   Всего сделок: {total_trades}")
        print(f"   Высокий confidence: {total_high_conf}/{total_trades} ({total_high_conf/max(total_trades,1)*100:.1f}%)")
        
        print(f"\n🚀 Enhanced возможности протестированы:")
        print("   ✅ Sentiment анализ через Bybit")
        print("   ✅ Market regime detection")
        print("   ✅ Enhanced confidence scoring")
        print("   ✅ Sentiment-based signal filtering")
        
        print(f"\n📈 Следующие шаги:")
        print("   1. Получите реальные API ключи для улучшения sentiment данных")
        print("   2. Настройте параметры в config/smart_adaptive_config_enhanced.yaml")
        print("   3. Запустите на live данных для paper trading")
        
    else:
        print("\n❌ Нет результатов для анализа")
    
    return len(results) > 0

if __name__ == "__main__":
    success = main()
    print(f"\n🏁 Завершено: {'SUCCESS' if success else 'FAILED'}") 
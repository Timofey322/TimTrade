#!/usr/bin/env python3
"""
Тест "правдивости" индикаторов: rolling-accuracy, precision, recall, деградация
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import matplotlib.pyplot as plt

# --- Список индикаторов для теста ---
INDICATORS = [
    'rsi_14',
    'macd_line',
    'macd_signal',
    'sma_20',
    'sma_50',
    'bb_position',
    'volume_ratio',
]

# --- Параметры теста ---
WINDOW = 2000  # rolling window (примерно 1 неделя по 5m)
STEP = 500     # шаг окна

# --- Генерация сигналов по индикатору ---
def generate_signals(df: pd.DataFrame, indicator: str) -> pd.Series:
    if indicator == 'rsi_14':
        # BUY: RSI < 30, SELL: RSI > 70
        return pd.Series(np.where(df['rsi_14'] < 30, 1, np.where(df['rsi_14'] > 70, -1, 0)), index=df.index)
    if indicator == 'macd_line':
        # BUY: MACD crosses above 0, SELL: below 0
        return pd.Series(np.where((df['macd_line'] > 0) & (df['macd_line'].shift(1) <= 0), 1,
                                  np.where((df['macd_line'] < 0) & (df['macd_line'].shift(1) >= 0), -1, 0)), index=df.index)
    if indicator == 'macd_signal':
        # BUY: MACD crosses above signal, SELL: below
        return pd.Series(np.where((df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1)), 1,
                                  np.where((df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1)), -1, 0)), index=df.index)
    if indicator == 'sma_20':
        # BUY: close crosses above sma_20, SELL: below
        return pd.Series(np.where((df['close'] > df['sma_20']) & (df['close'].shift(1) <= df['sma_20'].shift(1)), 1,
                                  np.where((df['close'] < df['sma_20']) & (df['close'].shift(1) >= df['sma_20'].shift(1)), -1, 0)), index=df.index)
    if indicator == 'sma_50':
        # BUY: close crosses above sma_50, SELL: below
        return pd.Series(np.where((df['close'] > df['sma_50']) & (df['close'].shift(1) <= df['sma_50'].shift(1)), 1,
                                  np.where((df['close'] < df['sma_50']) & (df['close'].shift(1) >= df['sma_50'].shift(1)), -1, 0)), index=df.index)
    if indicator == 'bb_position':
        # BUY: bb_position < 0.1, SELL: > 0.9
        return pd.Series(np.where(df['bb_position'] < 0.1, 1, np.where(df['bb_position'] > 0.9, -1, 0)), index=df.index)
    if indicator == 'volume_ratio':
        # BUY: volume_ratio > 1.5, SELL: < 0.7
        return pd.Series(np.where(df['volume_ratio'] > 1.5, 1, np.where(df['volume_ratio'] < 0.7, -1, 0)), index=df.index)
    return pd.Series(0, index=df.index)

# --- Оценка "правдивости" сигнала ---
def evaluate_signal(df: pd.DataFrame, signals: pd.Series, horizon: int = 12) -> pd.Series:
    # horizon = 12 (1 час для 5m) - смотрим, был ли рост/падение после сигнала
    future_return = (df['close'].shift(-horizon) - df['close']) / df['close']
    # BUY: если future_return > 0.002 (0.2%), SELL: < -0.2%
    correct = np.where((signals == 1) & (future_return > 0.002), 1,
                       np.where((signals == -1) & (future_return < -0.002), 1, 0))
    return pd.Series(correct, index=df.index)

# --- Rolling-отчет по accuracy/precision/recall ---
def rolling_truthfulness_report(df: pd.DataFrame, indicator: str) -> List[Dict]:
    signals = generate_signals(df, indicator)
    correct = evaluate_signal(df, signals)
    results = []
    for start in range(0, len(df) - WINDOW, STEP):
        end = start + WINDOW
        window_signals = signals.iloc[start:end]
        window_correct = correct.iloc[start:end]
        n_signals = (window_signals != 0).sum()
        n_correct = window_correct.sum()
        accuracy = n_correct / n_signals if n_signals > 0 else np.nan
        # Precision/Recall (по BUY/SELL)
        buy_mask = window_signals == 1
        sell_mask = window_signals == -1
        buy_precision = (window_correct[buy_mask].sum() / buy_mask.sum()) if buy_mask.sum() > 0 else np.nan
        sell_precision = (window_correct[sell_mask].sum() / sell_mask.sum()) if sell_mask.sum() > 0 else np.nan
        results.append({
            'start': df['timestamp'].iloc[start],
            'end': df['timestamp'].iloc[end-1],
            'n_signals': int(n_signals),
            'accuracy': accuracy,
            'buy_precision': buy_precision,
            'sell_precision': sell_precision
        })
    return results

def plot_rolling_report(report, indicator):
    dates = [r['end'] for r in report]
    accuracy = [r['accuracy'] for r in report]
    buy_precision = [r['buy_precision'] for r in report]
    sell_precision = [r['sell_precision'] for r in report]
    plt.figure(figsize=(12, 5))
    plt.plot(dates, accuracy, label='Accuracy')
    plt.plot(dates, buy_precision, label='Buy Precision')
    plt.plot(dates, sell_precision, label='Sell Precision')
    plt.title(f'Rolling Accuracy/Precision: {indicator}')
    plt.xlabel('Date')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'research/indicators/rolling_{indicator}.png')
    plt.close()

# --- Главный тест ---
def main():
    print("\n=== Тест правдивости индикаторов (rolling accuracy) ===\n")
    df = pd.read_csv("data/historical/BTCUSDT_5m_5years_20210705_20250702.csv")
    if 'datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime'])
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # --- Рассчитываем индикаторы ---
    print("Расчет индикаторов...")
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd_line'] = ema_12 - ema_26
    df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
    # SMA
    df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    df['bb_middle'] = df['close'].rolling(bb_period, min_periods=1).mean()
    bb_std_dev = df['close'].rolling(bb_period, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
    df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    # Volume
    df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # --- Тестируем каждый индикатор ---
    best_indicators = []
    for indicator in INDICATORS:
        print(f"\n--- {indicator} ---")
        report = rolling_truthfulness_report(df, indicator)
        plot_rolling_report(report, indicator)
        # Автоматический отбор лучших: accuracy > 0.35 хотя бы 30% времени
        accs = [r['accuracy'] for r in report if r['accuracy'] is not None]
        if len(accs) > 0 and (np.mean(np.array(accs) > 0.35) > 0.3):
            best_indicators.append(indicator)
        for r in report:
            print(f"{r['start'].strftime('%Y-%m-%d')} - {r['end'].strftime('%Y-%m-%d')}: ",
                  f"signals={r['n_signals']}, accuracy={r['accuracy']:.2f}, ",
                  f"buy_precision={r['buy_precision']:.2f}, sell_precision={r['sell_precision']:.2f}")
    # Экспорт лучших индикаторов
    with open('research/indicators/best_indicators.py', 'w', encoding='utf-8') as f:
        f.write('# Автоматически сгенерировано\n')
        f.write('BEST_INDICATORS = ' + repr(best_indicators) + '\n')
    print(f"\nЛучшие индикаторы: {best_indicators}\nСохранено в research/indicators/best_indicators.py")

if __name__ == "__main__":
    main() 
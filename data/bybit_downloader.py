#!/usr/bin/env python3
"""
📊 Bybit Data Downloader for TimAI
Загрузка реальных исторических данных с Bybit API
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import json

class BybitDataDownloader:
    """Загрузчик данных с Bybit API"""
    
    def __init__(self):
        self.base_url = "https://api.bybit.com/v5/market/kline"
        self.intervals = {
            '5': '5',      # 5 минут
            '15': '15',    # 15 минут  
            '60': '60'     # 60 минут (1 час)
        }
        
    def download_klines(self, symbol: str, interval: str, days_back: int = 1825) -> pd.DataFrame:
        """
        Загружает исторические данные с Bybit
        
        Args:
            symbol: Торговая пара (например BTCUSDT)
            interval: Таймфрейм (5, 15, 60)
            days_back: Количество дней назад (по умолчанию 5 лет = 1825 дней)
        """
        
        print(f"📊 Загрузка {symbol} {interval}m за последние {days_back} дней...")
        
        # Вычисляем временные границы
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        all_data = []
        current_start = start_time
        
        # Bybit ограничивает 1000 свечей за запрос
        max_candles_per_request = 1000
        
        if interval == '5':
            time_per_candle = 5 * 60 * 1000  # 5 минут в миллисекундах
        elif interval == '15': 
            time_per_candle = 15 * 60 * 1000  # 15 минут
        elif interval == '60':
            time_per_candle = 60 * 60 * 1000  # 60 минут
        
        chunk_time = max_candles_per_request * time_per_candle
        
        request_count = 0
        total_requests = int((end_time - start_time) / chunk_time) + 1
        
        print(f"   Ожидается {total_requests} запросов к API...")
        
        while current_start < end_time:
            current_end = min(current_start + chunk_time, end_time)
            
            params = {
                'category': 'spot',
                'symbol': symbol,
                'interval': interval,
                'start': current_start,
                'end': current_end,
                'limit': 1000
            }
            
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if data['retCode'] == 0 and 'result' in data and 'list' in data['result']:
                    klines = data['result']['list']
                    
                    if klines:
                        # Обрабатываем данные
                        for kline in klines:
                            # Формат Bybit: [timestamp, open, high, low, close, volume, turnover]
                            all_data.append({
                                'timestamp': int(kline[0]),
                                'datetime': datetime.fromtimestamp(int(kline[0]) / 1000),
                                'open': float(kline[1]),
                                'high': float(kline[2]), 
                                'low': float(kline[3]),
                                'close': float(kline[4]),
                                'volume': float(kline[5]),
                                'turnover': float(kline[6]) if len(kline) > 6 else 0
                            })
                        
                        request_count += 1
                        if request_count % 10 == 0:
                            progress = (current_start - start_time) / (end_time - start_time) * 100
                            print(f"   Прогресс: {progress:.1f}% ({request_count}/{total_requests} запросов)")
                    
                    # Переходим к следующему временному окну
                    current_start = current_end
                    
                else:
                    print(f"   ⚠️ Ошибка API: {data.get('retMsg', 'Unknown error')}")
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Ошибка запроса: {e}")
                break
            except Exception as e:
                print(f"   ❌ Ошибка обработки: {e}")
                break
            
            # Пауза между запросами чтобы не превысить лимиты API
            time.sleep(0.1)
        
        if all_data:
            df = pd.DataFrame(all_data)
            # Сортируем по времени
            df = df.sort_values('timestamp').reset_index(drop=True)
            # Убираем дубликаты
            df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            
            print(f"   ✅ Загружено {len(df)} свечей")
            print(f"   📅 Период: {df['datetime'].min()} - {df['datetime'].max()}")
            
            return df
        else:
            print("   ❌ Не удалось загрузить данные")
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, symbol: str, interval: str) -> str:
        """Сохраняет данные в CSV файл"""
        
        if df.empty:
            return ""
            
        # Создаем имя файла
        start_date = df['datetime'].min().strftime('%Y%m%d')
        end_date = df['datetime'].max().strftime('%Y%m%d') 
        filename = f"{symbol}_{interval}m_5years_{start_date}_{end_date}.csv"
        filepath = os.path.join("historical", filename)
        
        # Создаем папку если не существует
        os.makedirs("historical", exist_ok=True)
        
        # Сохраняем данные
        df.to_csv(filepath, index=False)
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"   💾 Сохранено: {filepath} ({file_size:.1f} MB)")
        
        return filepath
    
    def download_all_timeframes(self, symbol: str = "BTCUSDT") -> dict:
        """Загружает все таймфреймы для символа"""
        
        print(f"🚀 Загрузка всех таймфреймов для {symbol}")
        print("="*60)
        
        results = {}
        
        for interval_name, interval_code in self.intervals.items():
            print(f"\n📊 Таймфрейм {interval_name} минут:")
            
            # Загружаем данные
            df = self.download_klines(symbol, interval_code)
            
            if not df.empty:
                # Сохраняем в файл
                filepath = self.save_data(df, symbol, interval_name)
                results[f"{interval_name}m"] = {
                    'filepath': filepath,
                    'records': len(df),
                    'start_date': df['datetime'].min(),
                    'end_date': df['datetime'].max(),
                    'file_size_mb': os.path.getsize(filepath) / (1024 * 1024) if filepath else 0
                }
            else:
                results[f"{interval_name}m"] = {'error': 'Failed to download'}
        
        return results

def main():
    """Главная функция загрузки данных"""
    
    print("📊 TimAI Data Downloader - Bybit Historical Data")
    print("="*60)
    
    # Создаем загрузчик
    downloader = BybitDataDownloader()
    
    # Основные торговые пары для загрузки
    symbols = ["BTCUSDT", "ETHUSDT"]
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\n🔸 Загрузка данных для {symbol}")
        results = downloader.download_all_timeframes(symbol)
        all_results[symbol] = results
        
        # Показываем результаты
        print(f"\n📋 Результаты для {symbol}:")
        for timeframe, data in results.items():
            if 'error' not in data:
                print(f"   ✅ {timeframe}: {data['records']} записей, {data['file_size_mb']:.1f} MB")
            else:
                print(f"   ❌ {timeframe}: {data['error']}")
    
    # Сохраняем сводку
    summary_file = "historical/download_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📊 ИТОГОВАЯ СВОДКА:")
    print("="*60)
    
    total_files = 0
    total_size = 0
    total_records = 0
    
    for symbol, timeframes in all_results.items():
        print(f"🔸 {symbol}:")
        for tf, data in timeframes.items():
            if 'error' not in data:
                total_files += 1
                total_size += data['file_size_mb']
                total_records += data['records']
                print(f"   ✅ {tf}: {data['records']:,} записей")
    
    print(f"\n🎉 ЗАГРУЗКА ЗАВЕРШЕНА!")
    print(f"   📁 Файлов: {total_files}")
    print(f"   📊 Записей: {total_records:,}")
    print(f"   💾 Размер: {total_size:.1f} MB")
    print(f"   📋 Сводка: {summary_file}")
    
    # Инструкции для использования
    print(f"\n🚀 ГОТОВО К ИСПОЛЬЗОВАНИЮ:")
    print(f"   1. Данные сохранены в: data/historical/")
    print(f"   2. Запустите: python core/timai_core.py")
    print(f"   3. Для API: python api/trading_api.py")
    
    return all_results

if __name__ == "__main__":
    # Переходим в папку data
    if not os.path.basename(os.getcwd()) == 'data':
        if os.path.exists('data'):
            os.chdir('data')
    
    results = main() 
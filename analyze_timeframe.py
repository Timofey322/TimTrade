#!/usr/bin/env python3
"""
Анализ временного периода и доходности бэктеста
"""

from datetime import datetime, timedelta
import pandas as pd

def analyze_backtest_timeframe():
    """Анализирует временной период и доходность бэктеста."""
    
    print("=" * 80)
    print("📊 АНАЛИЗ ВРЕМЕННОГО ПЕРИОДА И ДОХОДНОСТИ")
    print("=" * 80)
    
    # Данные из бэктеста
    start_date = "2025-04-18 06:25:00"
    end_date = "2025-06-26 16:55:00"
    initial_capital = 10000.0
    final_capital = 10941.695739231216
    
    # Парсинг дат
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    
    # Расчет временного периода
    total_duration = end_dt - start_dt
    total_days = total_duration.days
    total_hours = total_duration.total_seconds() / 3600
    
    # Расчет доходности
    total_return_abs = final_capital - initial_capital
    total_return_pct = (final_capital / initial_capital - 1) * 100
    
    print(f"📅 ВРЕМЕННОЙ ПЕРИОД:")
    print(f"   Начало:     {start_date}")
    print(f"   Окончание:  {end_date}")
    print(f"   Продолжительность:")
    print(f"     • {total_days} дней")
    print(f"     • {total_hours:.1f} часов")
    print(f"     • {total_duration.total_seconds()/60:.0f} минут")
    
    # Разбивка по месяцам
    months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    if end_dt.day < start_dt.day:
        months -= 1
    
    weeks = total_days // 7
    remaining_days = total_days % 7
    
    print(f"\n📊 РАЗБИВКА ПЕРИОДА:")
    print(f"   • ~{months} месяца и {remaining_days} дней")
    print(f"   • {weeks} полных недель и {remaining_days} дней")
    
    print(f"\n💰 ФИНАНСОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Начальный капитал:  ${initial_capital:,.2f}")
    print(f"   Конечный капитал:   ${final_capital:,.2f}")
    print(f"   Абсолютная прибыль: ${total_return_abs:,.2f}")
    print(f"   Доходность:         {total_return_pct:.2f}%")
    
    # Аннуализированная доходность
    years = total_days / 365.25
    annualized_return = ((final_capital / initial_capital) ** (1/years) - 1) * 100
    
    print(f"\n📈 АННУАЛИЗИРОВАННАЯ ДОХОДНОСТЬ:")
    print(f"   За {years:.2f} года: {annualized_return:.2f}% годовых")
    
    # Доходность по периодам
    daily_return = total_return_pct / total_days
    weekly_return = daily_return * 7
    monthly_return = total_return_pct / months if months > 0 else 0
    
    print(f"\n🔄 ДОХОДНОСТЬ ПО ПЕРИОДАМ:")
    print(f"   Дневная (средняя):  {daily_return:.3f}%")
    print(f"   Недельная (средняя): {weekly_return:.2f}%")
    print(f"   Месячная (средняя):  {monthly_return:.2f}%")
    
    # Сравнение с рынком
    print(f"\n🏆 ОЦЕНКА РЕЗУЛЬТАТОВ:")
    
    if total_return_pct > 15:
        rating = "🔥 ОТЛИЧНАЯ"
    elif total_return_pct > 10:
        rating = "✅ ХОРОШАЯ"
    elif total_return_pct > 5:
        rating = "👍 УМЕРЕННАЯ"
    elif total_return_pct > 0:
        rating = "⚠️  НИЗКАЯ"
    else:
        rating = "❌ УБЫТОЧНАЯ"
    
    print(f"   Общая оценка: {rating}")
    print(f"   Доходность {total_return_pct:.2f}% за {total_days} дней")
    
    if annualized_return > 20:
        annual_rating = "🚀 ПРЕВОСХОДНАЯ"
    elif annualized_return > 15:
        annual_rating = "🔥 ОТЛИЧНАЯ"
    elif annualized_return > 10:
        annual_rating = "✅ ХОРОШАЯ"
    else:
        annual_rating = "👍 УМЕРЕННАЯ"
    
    print(f"   Годовая доходность: {annual_rating}")
    print(f"   {annualized_return:.2f}% годовых")
    
    print(f"\n📋 КРАТКАЯ СВОДКА:")
    print(f"   🎯 Доходность +{total_return_pct:.2f}% за {total_days} дней")
    print(f"   🎯 Это примерно {annualized_return:.1f}% годовых")
    print(f"   🎯 Средняя доходность {daily_return:.3f}% в день")
    
    print("=" * 80)

if __name__ == "__main__":
    analyze_backtest_timeframe() 
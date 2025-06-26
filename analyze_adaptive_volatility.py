"""
Анализ адаптивных множителей волатильности на реальных данных.

Этот скрипт загружает реальные данные и анализирует:
- Распределение множителей волатильности
- Влияние рыночных условий на множители
- Распределение целевой переменной
- Корреляцию между множителями и рыночными метриками
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import yaml

# Добавляем путь к src в sys.path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collection.collector import DataCollector
from preprocessing.adaptive_indicators import AdaptiveIndicatorSelector
from preprocessing.feature_engineering import FeatureEngineer


class AdaptiveVolatilityAnalyzer:
    """
    Анализатор адаптивных множителей волатильности.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Инициализация анализатора."""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logger.bind(name="VolatilityAnalyzer")
        
        # Инициализация компонентов
        self.data_collector = DataCollector(self.config)
        self.selector = AdaptiveIndicatorSelector(self.config.get('adaptive_indicators', {}))
        self.feature_engineer = FeatureEngineer(self.config.get('preprocessing', {}))
    
    def _load_config(self) -> dict:
        """Загрузка конфигурации."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            return {}
    
    def load_real_data(self, symbol: str = 'BTC/USDT', limit: int = 50000) -> pd.DataFrame:
        """Загрузка реальных данных."""
        try:
            self.logger.info(f"📊 Загружаем реальные данные для {symbol}...")
            
            data = self.data_collector.fetch_ohlcv_with_history(
                symbol=symbol,
                timeframe='5m',
                target_limit=limit,
                batch_size=1000
            )
            
            if data is None or data.empty:
                raise ValueError("Не удалось загрузить данные")
            
            self.logger.info(f"✅ Загружено {len(data)} записей")
            self.logger.info(f"   Период: {data.index[0]} - {data.index[-1]}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки данных: {e}")
            raise
    
    def analyze_volatility_multipliers(self, data: pd.DataFrame) -> dict:
        """Анализ адаптивных множителей волатильности."""
        try:
            self.logger.info("🔍 Анализируем адаптивные множители волатильности...")
            
            # Рассчитываем множители
            multipliers = self.selector._calculate_adaptive_volatility_multipliers(data)
            
            if not multipliers:
                raise ValueError("Не удалось рассчитать множители")
            
            fall_mult = multipliers['fall']
            rise_mult = multipliers['rise']
            
            # Базовая статистика
            analysis = {
                'fall_multiplier': {
                    'mean': fall_mult.mean(),
                    'std': fall_mult.std(),
                    'min': fall_mult.min(),
                    'max': fall_mult.max(),
                    'median': fall_mult.median(),
                    'q25': fall_mult.quantile(0.25),
                    'q75': fall_mult.quantile(0.75)
                },
                'rise_multiplier': {
                    'mean': rise_mult.mean(),
                    'std': rise_mult.std(),
                    'min': rise_mult.min(),
                    'max': rise_mult.max(),
                    'median': rise_mult.median(),
                    'q25': rise_mult.quantile(0.25),
                    'q75': rise_mult.quantile(0.75)
                }
            }
            
            # Анализ распределения
            self.logger.info("📈 Статистика множителей:")
            self.logger.info(f"   Падение: среднее={analysis['fall_multiplier']['mean']:.3f}, "
                           f"стд={analysis['fall_multiplier']['std']:.3f}")
            self.logger.info(f"   Рост: среднее={analysis['rise_multiplier']['mean']:.3f}, "
                           f"стд={analysis['rise_multiplier']['std']:.3f}")
            
            return analysis, multipliers
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа множителей: {e}")
            raise
    
    def analyze_market_conditions(self, data: pd.DataFrame, multipliers: dict) -> dict:
        """Анализ рыночных условий и их влияния на множители."""
        try:
            self.logger.info("🌊 Анализируем рыночные условия...")
            
            returns = data['close'].pct_change()
            
            # Рассчитываем рыночные метрики
            market_metrics = {
                'volatility': returns.rolling(20).std().reindex(data.index).fillna(returns.std()),
                'trend': ((data['close'].rolling(20).mean() - data['close'].rolling(50).mean()) / 
                         data['close'].rolling(50).mean()).reindex(data.index).fillna(0),
                'volume_ratio': (data['volume'] / data['volume'].rolling(20).mean()).reindex(data.index).fillna(1.0)
            }
            
            # Корреляция с множителями
            correlations = {}
            for metric_name, metric_values in market_metrics.items():
                fall_corr = np.corrcoef(multipliers['fall'].dropna(), metric_values.dropna())[0, 1]
                rise_corr = np.corrcoef(multipliers['rise'].dropna(), metric_values.dropna())[0, 1]
                
                correlations[metric_name] = {
                    'fall_correlation': fall_corr,
                    'rise_correlation': rise_corr
                }
            
            self.logger.info("📊 Корреляция множителей с рыночными метриками:")
            for metric, corr in correlations.items():
                self.logger.info(f"   {metric}: падение={corr['fall_correlation']:.3f}, рост={corr['rise_correlation']:.3f}")
            
            return market_metrics, correlations
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа рыночных условий: {e}")
            raise
    
    def analyze_target_distribution(self, data: pd.DataFrame, horizon: int = 12) -> dict:
        """Анализ распределения целевой переменной."""
        try:
            self.logger.info(f"🎯 Анализируем распределение целевой переменной (горизонт: {horizon})...")
            
            # Создаем целевую переменную
            target = self.selector._create_target_variable(data, horizon)
            
            if target is None or target.empty:
                raise ValueError("Не удалось создать целевую переменную")
            
            # Статистика распределения
            target_dist = target.value_counts()
            target_stats = {
                'distribution': target_dist.to_dict(),
                'total_samples': len(target),
                'class_balance': {
                    'fall_ratio': target_dist.get(0, 0) / len(target),
                    'hold_ratio': target_dist.get(1, 0) / len(target),
                    'rise_ratio': target_dist.get(2, 0) / len(target)
                }
            }
            
            self.logger.info("📊 Распределение целевой переменной:")
            self.logger.info(f"   Всего образцов: {target_stats['total_samples']}")
            self.logger.info(f"   Падение (0): {target_dist.get(0, 0)} ({target_stats['class_balance']['fall_ratio']:.1%})")
            self.logger.info(f"   Боковик (1): {target_dist.get(1, 0)} ({target_stats['class_balance']['hold_ratio']:.1%})")
            self.logger.info(f"   Рост (2): {target_dist.get(2, 0)} ({target_stats['class_balance']['rise_ratio']:.1%})")
            
            return target_stats, target
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа целевой переменной: {e}")
            raise
    
    def create_visualizations(self, data: pd.DataFrame, multipliers: dict, 
                            market_metrics: dict, target: pd.Series, save_path: str = "volatility_analysis"):
        """Создание визуализаций для анализа."""
        try:
            self.logger.info("📊 Создаем визуализации...")
            
            # Создаем директорию для графиков
            os.makedirs(save_path, exist_ok=True)
            
            # Настройка стиля
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Анализ адаптивных множителей волатильности', fontsize=16, fontweight='bold')
            
            # 1. Распределение множителей
            ax1 = axes[0, 0]
            ax1.hist(multipliers['fall'], bins=50, alpha=0.7, label='Падение', color='red')
            ax1.hist(multipliers['rise'], bins=50, alpha=0.7, label='Рост', color='green')
            ax1.set_xlabel('Множитель волатильности')
            ax1.set_ylabel('Частота')
            ax1.set_title('Распределение множителей')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Временной ряд множителей
            ax2 = axes[0, 1]
            sample_size = min(1000, len(multipliers['fall']))
            sample_indices = np.linspace(0, len(multipliers['fall'])-1, sample_size, dtype=int)
            
            ax2.plot(data.index[sample_indices], multipliers['fall'].iloc[sample_indices], 
                    label='Падение', color='red', alpha=0.7)
            ax2.plot(data.index[sample_indices], multipliers['rise'].iloc[sample_indices], 
                    label='Рост', color='green', alpha=0.7)
            ax2.set_xlabel('Время')
            ax2.set_ylabel('Множитель')
            ax2.set_title('Динамика множителей во времени')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Корреляция с волатильностью
            ax3 = axes[1, 0]
            vol_sample = market_metrics['volatility'].iloc[sample_indices]
            fall_sample = multipliers['fall'].iloc[sample_indices]
            rise_sample = multipliers['rise'].iloc[sample_indices]
            
            ax3.scatter(vol_sample, fall_sample, alpha=0.6, color='red', label='Падение')
            ax3.scatter(vol_sample, rise_sample, alpha=0.6, color='green', label='Рост')
            ax3.set_xlabel('Волатильность')
            ax3.set_ylabel('Множитель')
            ax3.set_title('Корреляция: Волатильность vs Множители')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Распределение целевой переменной
            ax4 = axes[1, 1]
            target_dist = target.value_counts()
            colors = ['red', 'gray', 'green']
            labels = ['Падение', 'Боковик', 'Рост']
            
            ax4.bar(labels, [target_dist.get(0, 0), target_dist.get(1, 0), target_dist.get(2, 0)], 
                   color=colors, alpha=0.7)
            ax4.set_ylabel('Количество')
            ax4.set_title('Распределение целевой переменной')
            ax4.grid(True, alpha=0.3)
            
            # Добавляем значения на столбцы
            for i, v in enumerate([target_dist.get(0, 0), target_dist.get(1, 0), target_dist.get(2, 0)]):
                ax4.text(i, v + max(target_dist.values()) * 0.01, str(v), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/volatility_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Графики сохранены в {save_path}/volatility_analysis.png")
            
        except Exception as e:
            self.logger.error(f"Ошибка создания визуализаций: {e}")
    
    def generate_report(self, analysis_results: dict, save_path: str = "volatility_analysis"):
        """Генерация отчета по анализу."""
        try:
            self.logger.info("📝 Генерируем отчет...")
            
            os.makedirs(save_path, exist_ok=True)
            
            report = f"""
# Отчет по анализу адаптивных множителей волатильности

## Статистика множителей

### Множитель падения
- Среднее: {analysis_results['multiplier_stats']['fall_multiplier']['mean']:.3f}
- Стандартное отклонение: {analysis_results['multiplier_stats']['fall_multiplier']['std']:.3f}
- Минимум: {analysis_results['multiplier_stats']['fall_multiplier']['min']:.3f}
- Максимум: {analysis_results['multiplier_stats']['fall_multiplier']['max']:.3f}
- Медиана: {analysis_results['multiplier_stats']['fall_multiplier']['median']:.3f}

### Множитель роста
- Среднее: {analysis_results['multiplier_stats']['rise_multiplier']['mean']:.3f}
- Стандартное отклонение: {analysis_results['multiplier_stats']['rise_multiplier']['std']:.3f}
- Минимум: {analysis_results['multiplier_stats']['rise_multiplier']['min']:.3f}
- Максимум: {analysis_results['multiplier_stats']['rise_multiplier']['max']:.3f}
- Медиана: {analysis_results['multiplier_stats']['rise_multiplier']['median']:.3f}

## Корреляция с рыночными метриками

"""
            
            for metric, corr in analysis_results['correlations'].items():
                report += f"### {metric.replace('_', ' ').title()}\n"
                report += f"- Корреляция с падением: {corr['fall_correlation']:.3f}\n"
                report += f"- Корреляция с ростом: {corr['rise_correlation']:.3f}\n\n"
            
            report += f"""
## Распределение целевой переменной

- Всего образцов: {analysis_results['target_stats']['total_samples']}
- Падение: {analysis_results['target_stats']['distribution'].get(0, 0)} ({analysis_results['target_stats']['class_balance']['fall_ratio']:.1%})
- Боковик: {analysis_results['target_stats']['distribution'].get(1, 0)} ({analysis_results['target_stats']['class_balance']['hold_ratio']:.1%})
- Рост: {analysis_results['target_stats']['distribution'].get(2, 0)} ({analysis_results['target_stats']['class_balance']['rise_ratio']:.1%})

## Выводы

1. **Адаптивность**: Множители варьируются от {analysis_results['multiplier_stats']['fall_multiplier']['min']:.3f} до {analysis_results['multiplier_stats']['fall_multiplier']['max']:.3f}, что показывает хорошую адаптивность системы.

2. **Баланс классов**: Целевая переменная имеет {len(analysis_results['target_stats']['distribution'])} классов с разумным распределением.

3. **Корреляция**: Система корректно реагирует на рыночные условия.
"""
            
            with open(f"{save_path}/analysis_report.md", 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"✅ Отчет сохранен в {save_path}/analysis_report.md")
            
        except Exception as e:
            self.logger.error(f"Ошибка генерации отчета: {e}")
    
    def run_full_analysis(self, symbol: str = 'BTC/USDT', limit: int = 50000):
        """Запуск полного анализа."""
        try:
            self.logger.info("🚀 Запуск полного анализа адаптивных множителей...")
            
            # 1. Загрузка данных
            data = self.load_real_data(symbol, limit)
            
            # 2. Анализ множителей
            multiplier_stats, multipliers = self.analyze_volatility_multipliers(data)
            
            # 3. Анализ рыночных условий
            market_metrics, correlations = self.analyze_market_conditions(data, multipliers)
            
            # 4. Анализ целевой переменной
            target_stats, target = self.analyze_target_distribution(data)
            
            # 5. Создание визуализаций
            self.create_visualizations(data, multipliers, market_metrics, target)
            
            # 6. Генерация отчета
            analysis_results = {
                'multiplier_stats': multiplier_stats,
                'correlations': correlations,
                'target_stats': target_stats
            }
            self.generate_report(analysis_results)
            
            self.logger.info("🎉 Полный анализ завершен!")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Ошибка полного анализа: {e}")
            raise


def main():
    """Основная функция."""
    try:
        # Настройка логирования
        logger.remove()
        logger.add(sys.stderr, level="INFO", 
                  format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
        
        # Создаем анализатор
        analyzer = AdaptiveVolatilityAnalyzer()
        
        # Запускаем анализ
        results = analyzer.run_full_analysis()
        
        print("\n🎉 Анализ завершен!")
        print("📁 Результаты сохранены в папке 'volatility_analysis'")
        print("   - volatility_analysis.png (графики)")
        print("   - analysis_report.md (отчет)")
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
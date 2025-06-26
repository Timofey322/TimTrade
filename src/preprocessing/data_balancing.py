"""
Модуль для балансировки данных.

Этот модуль предоставляет функциональность для:
- Балансировки несбалансированных классов с помощью SMOTE
- Анализа дисбаланса данных
- Визуализации распределения классов
- Выбора оптимальной стратегии балансировки
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from loguru import logger

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline

class DataBalancer:
    """
    Класс для балансировки данных.
    
    Поддерживает различные методы балансировки:
    - SMOTE (Synthetic Minority Over-sampling Technique)
    - ADASYN (Adaptive Synthetic Sampling)
    - BorderlineSMOTE
    - Комбинированные методы
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация балансировщика данных.
        
        Args:
            config: Конфигурация балансировки
        """
        self.config = config or {}
        self.logger = logger.bind(name="DataBalancer")
        
        # Статистика балансировки
        self.balancing_stats = {
            'original_distribution': {},
            'balanced_distribution': {},
            'method_used': None,
            'sampling_strategy': None
        }
    
    def analyze_class_imbalance(self, y: pd.Series) -> Dict:
        """
        Анализ дисбаланса классов.
        
        Args:
            y: Целевая переменная
        
        Returns:
            Словарь с анализом дисбаланса
        """
        try:
            class_counts = Counter(y)
            total_samples = len(y)
            
            analysis = {
                'class_counts': dict(class_counts),
                'total_samples': total_samples,
                'class_ratios': {},
                'imbalance_ratio': 0,
                'minority_class': None,
                'majority_class': None
            }
            
            # Расчет соотношений классов
            for class_label, count in class_counts.items():
                ratio = count / total_samples
                analysis['class_ratios'][class_label] = ratio
            
            # Определение minority и majority классов
            minority_class = min(class_counts, key=class_counts.get)
            majority_class = max(class_counts, key=class_counts.get)
            
            analysis['minority_class'] = minority_class
            analysis['majority_class'] = majority_class
            analysis['imbalance_ratio'] = class_counts[majority_class] / class_counts[minority_class]
            
            # Сохранение статистики
            self.balancing_stats['original_distribution'] = analysis
            
            self.logger.info(f"Анализ дисбаланса классов:")
            self.logger.info(f"  Всего образцов: {total_samples}")
            self.logger.info(f"  Распределение: {dict(class_counts)}")
            self.logger.info(f"  Коэффициент дисбаланса: {analysis['imbalance_ratio']:.2f}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа дисбаланса: {e}")
            return {}
    
    def select_balancing_method(self, imbalance_ratio: float, config: Dict = None) -> str:
        """
        Выбор метода балансировки на основе дисбаланса.
        
        Args:
            imbalance_ratio: Коэффициент дисбаланса
            config: Конфигурация
        
        Returns:
            Название выбранного метода
        """
        config = config or self.config
        
        # Приоритет пользовательского выбора
        if 'method' in config:
            return config['method']
        
        # Автоматический выбор на основе дисбаланса
        if imbalance_ratio < 2:
            method = 'none'  # Небольшой дисбаланс
        elif imbalance_ratio < 5:
            method = 'smote'
        elif imbalance_ratio < 10:
            method = 'borderline_smote'
        else:
            method = 'smote_enn'  # Сильный дисбаланс
        
        self.logger.info(f"Выбран метод балансировки: {method}")
        return method
    
    def create_balancer(self, method: str, config: Dict = None) -> object:
        """
        Создание объекта балансировщика.
        
        Args:
            method: Метод балансировки
            config: Конфигурация
        
        Returns:
            Объект балансировщика
        """
        config = config or self.config
        
        if method == 'none' or method == 'auto':
            return None
        
        # Параметры SMOTE
        smote_params = {
            'random_state': config.get('random_state', 42),
            'k_neighbors': config.get('k_neighbors', 5),
            'sampling_strategy': config.get('sampling_strategy', 'auto')
        }
        
        if method == 'smote':
            balancer = SMOTE(**smote_params)
        elif method == 'adasyn':
            balancer = ADASYN(**smote_params)
        elif method == 'borderline_smote':
            balancer = BorderlineSMOTE(**smote_params)
        elif method == 'smote_enn':
            balancer = SMOTEENN(**smote_params)
        elif method == 'smote_tomek':
            balancer = SMOTETomek(**smote_params)
        elif method == 'random_under':
            balancer = RandomUnderSampler(
                random_state=config.get('random_state', 42),
                sampling_strategy=config.get('sampling_strategy', 'auto')
            )
        else:
            self.logger.warning(f"Неизвестный метод балансировки: {method}")
            return None
        
        return balancer
    
    def balance_data(self, X: pd.DataFrame, y: pd.Series, 
                    method: str = 'auto', config: Dict = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Балансировка данных.
        
        Args:
            X: Признаки
            y: Целевая переменная
            method: Метод балансировки
            config: Конфигурация
        
        Returns:
            Tuple с сбалансированными данными
        """
        try:
            config = config or self.config
            
            # Анализ дисбаланса
            imbalance_analysis = self.analyze_class_imbalance(y)
            
            if not imbalance_analysis:
                self.logger.error("Не удалось проанализировать дисбаланс")
                return X, y
            
            # Выбор метода если 'auto'
            if method == 'auto':
                method = self.select_balancing_method(
                    imbalance_analysis['imbalance_ratio'], config
                )
            
            # Если дисбаланс небольшой или метод 'none'
            if method == 'none' or imbalance_analysis['imbalance_ratio'] < 1.5:
                self.logger.info("Дисбаланс небольшой, балансировка не требуется")
                return X, y
            
            # Создание балансировщика
            balancer = self.create_balancer(method, config)
            
            if balancer is None:
                self.logger.info("Балансировка не требуется, возвращаем исходные данные")
                return X, y
            
            # Применение балансировки
            self.logger.info(f"Применяем {method} для балансировки данных")
            
            X_balanced, y_balanced = balancer.fit_resample(X, y)
            
            # Анализ результатов
            balanced_analysis = self.analyze_class_imbalance(y_balanced)
            
            # Сохранение статистики
            self.balancing_stats.update({
                'balanced_distribution': balanced_analysis,
                'method_used': method,
                'sampling_strategy': config.get('sampling_strategy', 'auto')
            })
            
            self.logger.info(f"Балансировка завершена:")
            self.logger.info(f"  Исходный размер: {len(X)} образцов")
            self.logger.info(f"  Сбалансированный размер: {len(X_balanced)} образцов")
            self.logger.info(f"  Новое распределение: {balanced_analysis['class_counts']}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            self.logger.error(f"Ошибка балансировки данных: {e}")
            return X, y
    
    def get_balancing_statistics(self) -> Dict:
        """
        Получение статистики балансировки.
        
        Returns:
            Словарь со статистикой
        """
        return self.balancing_stats.copy()
    
    def plot_class_distribution(self, y_before: pd.Series, y_after: pd.Series = None, 
                              save_path: str = None):
        """
        Визуализация распределения классов.
        
        Args:
            y_before: Целевая переменная до балансировки
            y_after: Целевая переменная после балансировки
            save_path: Путь для сохранения графика
        """
        try:
            fig, axes = plt.subplots(1, 2 if y_after is not None else 1, figsize=(12, 5))
            
            if y_after is None:
                axes = [axes]
            
            # График до балансировки
            before_counts = Counter(y_before)
            axes[0].bar(before_counts.keys(), before_counts.values(), color='skyblue')
            axes[0].set_title('Распределение классов до балансировки')
            axes[0].set_xlabel('Класс')
            axes[0].set_ylabel('Количество образцов')
            
            # Добавление значений на столбцы
            for i, (class_label, count) in enumerate(before_counts.items()):
                axes[0].text(i, count + max(before_counts.values()) * 0.01, 
                           str(count), ha='center', va='bottom')
            
            # График после балансировки
            if y_after is not None:
                after_counts = Counter(y_after)
                axes[1].bar(after_counts.keys(), after_counts.values(), color='lightgreen')
                axes[1].set_title('Распределение классов после балансировки')
                axes[1].set_xlabel('Класс')
                axes[1].set_ylabel('Количество образцов')
                
                # Добавление значений на столбцы
                for i, (class_label, count) in enumerate(after_counts.items()):
                    axes[1].text(i, count + max(after_counts.values()) * 0.01, 
                               str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"График сохранен в {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Ошибка создания графика: {e}")
    
    def create_balanced_pipeline(self, config: Dict = None) -> Pipeline:
        """
        Создание пайплайна с балансировкой.
        
        Args:
            config: Конфигурация
        
        Returns:
            Pipeline с балансировщиком
        """
        config = config or self.config
        
        method = config.get('method', 'smote')
        balancer = self.create_balancer(method, config)
        
        if balancer is None:
            return None
        
        # Создание пайплайна
        pipeline = Pipeline([
            ('balancer', balancer)
        ])
        
        return pipeline 
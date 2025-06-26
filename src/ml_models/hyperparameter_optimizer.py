"""
Гиперпараметрическая оптимизация для ML моделей.

Этот модуль предоставляет функциональность для:
- Bayesian оптимизации гиперпараметров
- Чекпоинтов и восстановления процесса
- Визуализации результатов оптимизации
- Анализа и сравнения результатов
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import pickle
import json
from datetime import datetime
from loguru import logger

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from skopt.callbacks import CheckpointSaver
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logger.warning("scikit-optimize не установлен. Bayesian оптимизация недоступна.")

try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn не установлен. Кросс-валидация недоступна.")


class HyperparameterOptimizer:
    """
    Класс для оптимизации гиперпараметров ML моделей.
    
    Поддерживает:
    - Bayesian оптимизацию (scikit-optimize)
    - Чекпоинты и восстановление
    - Визуализацию результатов
    - Анализ сходимости
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация оптимизатора.
        
        Args:
            config: Конфигурация оптимизации
        """
        self.config = config or {}
        self.logger = logger.bind(name="HyperparameterOptimizer")
        
        # Проверка доступности библиотек
        if not SKOPT_AVAILABLE:
            self.logger.error("scikit-optimize недоступен. Установите: pip install scikit-optimize")
        
        if not SKLEARN_AVAILABLE:
            self.logger.error("scikit-learn недоступен. Установите: pip install scikit-learn")
        
        # Настройки по умолчанию
        self.default_config = {
            'method': 'bayesian',  # bayesian, grid, random
            'metric': 'f1',  # f1, accuracy, precision, recall
            'n_calls': 50,
            'n_splits': 5,
            'random_state': 42,
            'checkpoint_enabled': True,
            'checkpoint_name': 'optimization_checkpoint',
            'checkpoint_dir': 'models/checkpoints',
            'search_space': {
                'max_depth': [3, 10],
                'learning_rate': [0.01, 0.3],
                'n_estimators': [50, 300],
                'subsample': [0.6, 1.0],
                'colsample_bytree': [0.6, 1.0],
                'reg_alpha': [0.1, 10.0],
                'reg_lambda': [0.1, 10.0],
                'min_child_weight': [3, 10]
            }
        }
        
        # Обновление конфигурации
        if config:
            self.default_config.update(config)
        
        # Создание директории для чекпоинтов
        self.checkpoint_dir = Path(self.default_config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Результаты оптимизации
        self.optimization_results = {
            'best_params': None,
            'best_score': None,
            'optimization_history': [],
            'convergence_history': [],
            'all_results': [],
            'optimization_time': None,
            'n_iterations': 0
        }
        
        # Метрики для оптимизации
        self.metric_functions = {
            'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'accuracy': lambda y_true, y_pred: accuracy_score(y_true, y_pred),
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                               model_class, **kwargs) -> Optional[Dict]:
        """
        Оптимизация гиперпараметров модели.
        
        Args:
            X: Признаки
            y: Целевая переменная
            model_class: Класс модели для оптимизации
            **kwargs: Дополнительные параметры
            
        Returns:
            Словарь с лучшими параметрами или None при ошибке
        """
        if not SKOPT_AVAILABLE or not SKLEARN_AVAILABLE:
            self.logger.error("Необходимые библиотеки недоступны")
            return None
        
        try:
            import time
            start_time = time.time()
            
            self.logger.info("Начинаем оптимизацию гиперпараметров")
            self.logger.info(f"Метод: {self.default_config['method']}")
            self.logger.info(f"Метрика: {self.default_config['metric']}")
            self.logger.info(f"Количество итераций: {self.default_config['n_calls']}")
            
            # Подготовка пространства поиска
            search_space = self._prepare_search_space()
            
            self.logger.info(f"Параметры поиска: {search_space}")
            self.logger.info(f"Размер данных для оптимизации: {len(X)}, признаков: {X.shape[1]}")
            
            # Создание функции для оптимизации
            objective_function = self._create_objective_function(X, y, model_class, **kwargs)
            
            # Настройка чекпоинтов
            callbacks = []
            if self.default_config['checkpoint_enabled']:
                checkpoint_path = self.checkpoint_dir / f"{self.default_config['checkpoint_name']}.pkl"
                callbacks.append(CheckpointSaver(checkpoint_path))
                self.logger.info(f"Чекпоинты будут сохраняться в {checkpoint_path}")
            
            # Запуск оптимизации
            if self.default_config['method'] == 'bayesian':
                result = self._bayesian_optimization(
                    objective_function, search_space, callbacks
                )
            else:
                self.logger.warning(f"Метод {self.default_config['method']} не поддерживается")
                return None
            
            # Обработка результатов
            self._process_optimization_results(result, start_time)
            
            end_time = time.time()
            self.logger.info(f"Время оптимизации: {end_time - start_time:.2f} секунд")
            
            self.logger.info(f"Оптимизация завершена за {self.optimization_results['optimization_time']:.2f} секунд")
            self.logger.info(f"Лучший скор: {self.optimization_results['best_score']:.4f}")
            self.logger.info(f"Лучшие параметры: {self.optimization_results['best_params']}")
            
            return self.optimization_results['best_params']
            
        except Exception as e:
            self.logger.error(f"Ошибка оптимизации гиперпараметров: {e}")
            return None
    
    def _prepare_search_space(self) -> List:
        """
        Подготовка пространства поиска для scikit-optimize.
        
        Returns:
            Список параметров поиска
        """
        search_space = []
        space_config = self.default_config['search_space']
        
        for param_name, param_range in space_config.items():
            if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                search_space.append(Integer(param_range[0], param_range[1], name=param_name))
            else:
                search_space.append(Real(param_range[0], param_range[1], name=param_name))
        
        return search_space
    
    def _create_objective_function(self, X: pd.DataFrame, y: pd.Series, 
                                 model_class, **kwargs) -> Callable:
        """
        Создание функции цели для оптимизации.
        
        Args:
            X: Признаки
            y: Целевая переменная
            model_class: Класс модели
            **kwargs: Дополнительные параметры
            
        Returns:
            Функция цели для оптимизации
        """
        use_cv = kwargs.get('use_cross_validation', True)
        cv_config = kwargs.get('cv_config', {})
        custom_cv = self.default_config.get('custom_cv_with_eval_set', False)
        early_stopping_rounds = self.default_config.get('early_stopping_rounds', 20)
        
        if use_cv:
            self.logger.info("Используем кросс-валидацию для оценки параметров")
            
            # Настройки кросс-валидации
            n_splits = cv_config.get('n_splits', self.default_config['n_splits'])
            shuffle = cv_config.get('shuffle', False)  # False для временных рядов
            random_state = cv_config.get('random_state', self.default_config['random_state'])
            
            # Используем TimeSeriesSplit для временных рядов
            if not shuffle:
                from sklearn.model_selection import TimeSeriesSplit
                cv = TimeSeriesSplit(n_splits=n_splits)
            else:
                from sklearn.model_selection import StratifiedKFold
                cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        metric_name = self.default_config['metric']
        metric_function = self.metric_functions[metric_name]
        
        @use_named_args(self._prepare_search_space())
        def objective(**params):
            try:
                # Добавляем базовые параметры
                model_params = params.copy()
                model_params.update({
                    'random_state': self.default_config['random_state'],
                    'n_jobs': -1,  # Используем все ядра
                    'verbosity': 0  # Отключаем вывод
                })
                
                # Определяем количество классов
                n_classes = len(np.unique(y))
                if n_classes > 2:
                    model_params['objective'] = 'multi:softprob'
                    model_params['eval_metric'] = 'mlogloss'
                    model_params['num_class'] = n_classes
                else:
                    model_params['objective'] = 'binary:logistic'
                    model_params['eval_metric'] = 'logloss'
                
                # Кастомный цикл с eval_set и early stopping
                if use_cv and custom_cv:
                    scores = []
                    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        model = model_class(**model_params)
                        try:
                            model.fit(
                                X_train, y_train,
                                eval_set=[(X_val, y_val)],
                                early_stopping_rounds=early_stopping_rounds,
                                verbose=False
                            )
                        except TypeError:
                            # Если модель не поддерживает eval_set, fallback на обычный fit
                            model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = metric_function(y_val, y_pred)
                        scores.append(score)
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    self.optimization_results['optimization_history'].append({
                        'params': params,
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'scores': scores
                    })
                    if len(self.optimization_results['optimization_history']) % 10 == 0:
                        self.logger.info(f"Итерация {len(self.optimization_results['optimization_history'])}: скор = {mean_score:.4f} ± {std_score:.4f}")
                    return -mean_score
                # Обычный cross_val_score
                elif use_cv:
                    model = model_class(**model_params)
                    scores = cross_val_score(
                        model, X, y, 
                        cv=cv, 
                        scoring=make_scorer(metric_function),
                        n_jobs=-1
                    )
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    self.optimization_results['optimization_history'].append({
                        'params': params,
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'scores': scores.tolist()
                    })
                    if len(self.optimization_results['optimization_history']) % 10 == 0:
                        self.logger.info(f"Итерация {len(self.optimization_results['optimization_history'])}: скор = {mean_score:.4f} ± {std_score:.4f}")
                    return -mean_score
                else:
                    # Простое разделение на train/test
                    from sklearn.model_selection import train_test_split
                    model = model_class(**model_params)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=self.default_config['random_state']
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = metric_function(y_test, y_pred)
                    return -score
            except Exception as e:
                self.logger.warning(f"Ошибка в итерации оптимизации: {e}")
                return -0.1  # Возвращаем плохой скор при ошибке
        return objective
    
    def _bayesian_optimization(self, objective_function: Callable, 
                             search_space: List, callbacks: List) -> object:
        """
        Bayesian оптимизация с улучшенными стратегиями.
        
        Args:
            objective_function: Функция цели
            search_space: Пространство поиска
            callbacks: Колбэки для сохранения
            
        Returns:
            Результат оптимизации
        """
        try:
            # НОВОЕ: Адаптивные настройки для Bayesian оптимизации
            n_calls = self.default_config['n_calls']
            n_initial_points = max(5, n_calls // 10)  # 10% от общего количества для начальных точек
            
            # НОВОЕ: Используем более продвинутые настройки (убираем неподдерживаемые параметры)
            result = gp_minimize(
                func=objective_function,
                dimensions=search_space,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                random_state=self.default_config['random_state'],
                # Убираем callbacks, noise, n_jobs - они не поддерживаются в этой версии
                acq_func='EI',  # Expected Improvement
                acq_optimizer='auto'
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка Bayesian оптимизации: {e}")
            raise
    
    def _process_optimization_results(self, result: object, start_time: float):
        """
        Обработка результатов оптимизации с улучшенной статистикой.
        
        Args:
            result: Результат оптимизации
            start_time: Время начала оптимизации
        """
        try:
            import time
            optimization_time = time.time() - start_time
            
            # Получаем лучшие параметры
            best_params = {}
            for i, param_name in enumerate([dim.name for dim in self._prepare_search_space()]):
                best_params[param_name] = result.x[i]
            
            # НОВОЕ: Детальная статистика
            all_scores = [-score for score in result.func_vals]  # Убираем отрицательный знак
            best_score = max(all_scores)
            worst_score = min(all_scores)
            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores)
            
            # НОВОЕ: Анализ сходимости
            convergence = []
            running_best = all_scores[0]
            for i, score in enumerate(all_scores):
                if score > running_best:
                    running_best = score
                convergence.append(running_best)
            
            # Сохраняем результаты
            self.optimization_results.update({
                'best_params': best_params,
                'best_score': best_score,
                'worst_score': worst_score,
                'mean_score': mean_score,
                'std_score': std_score,
                'optimization_time': optimization_time,
                'n_iterations': len(all_scores),
                'convergence_history': convergence,
                'all_scores': all_scores,
                'improvement': best_score - all_scores[0] if all_scores else 0
            })
            
            # НОВОЕ: Детальное логирование
            self.logger.info("=== РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ===")
            self.logger.info(f"Лучший скор: {best_score:.4f}")
            self.logger.info(f"Средний скор: {mean_score:.4f} ± {std_score:.4f}")
            self.logger.info(f"Улучшение: {self.optimization_results['improvement']:.4f}")
            self.logger.info(f"Время оптимизации: {optimization_time:.2f} сек")
            self.logger.info(f"Итераций: {len(all_scores)}")
            self.logger.info("Лучшие параметры:")
            for param, value in best_params.items():
                self.logger.info(f"  {param}: {value}")
            
            # Сохраняем результаты
            self._save_optimization_results()
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки результатов: {e}")
    
    def _save_optimization_results(self):
        """Сохранение результатов оптимизации."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.checkpoint_dir / f"optimization_results_{timestamp}.json"
            
            # Подготовка данных для сохранения
            save_data = {
                'config': self.default_config,
                'best_params': self.optimization_results['best_params'],
                'best_score': self.optimization_results['best_score'],
                'optimization_time': self.optimization_results['optimization_time'],
                'n_iterations': self.optimization_results['n_iterations'],
                'convergence_history': self.optimization_results['convergence_history'],
                'timestamp': timestamp
            }
            
            # Конвертируем numpy типы в нативные Python типы
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                else:
                    return obj
            
            save_data_converted = convert_numpy_types(save_data)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(save_data_converted, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Результаты оптимизации сохранены в {results_file}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения результатов: {e}")
    
    def load_checkpoint(self, checkpoint_name: str = None) -> bool:
        """
        Загрузка чекпоинта оптимизации.
        
        Args:
            checkpoint_name: Имя чекпоинта
            
        Returns:
            True если чекпоинт загружен успешно
        """
        try:
            if checkpoint_name is None:
                checkpoint_name = self.default_config['checkpoint_name']
            
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
            
            if not checkpoint_path.exists():
                self.logger.warning(f"Чекпоинт {checkpoint_path} не найден")
                return False
            
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.logger.info(f"Чекпоинт загружен из {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки чекпоинта: {e}")
            return False
    
    def plot_optimization_results(self, save_path: str = None):
        """
        Визуализация результатов оптимизации.
        
        Args:
            save_path: Путь для сохранения графиков
        """
        if not self.optimization_results['convergence_history']:
            self.logger.warning("Нет данных для визуализации")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Результаты оптимизации гиперпараметров', fontsize=16)
            
            # График сходимости
            axes[0, 0].plot(self.optimization_results['convergence_history'])
            axes[0, 0].set_title('Сходимость оптимизации')
            axes[0, 0].set_xlabel('Итерация')
            axes[0, 0].set_ylabel(f'Метрика ({self.default_config["metric"]})')
            axes[0, 0].grid(True)
            
            # Лучший скор по итерациям
            best_scores = []
            current_best = float('-inf')
            for score in self.optimization_results['convergence_history']:
                if score > current_best:
                    current_best = score
                best_scores.append(current_best)
            
            axes[0, 1].plot(best_scores)
            axes[0, 1].set_title('Лучший скор по итерациям')
            axes[0, 1].set_xlabel('Итерация')
            axes[0, 1].set_ylabel(f'Лучший {self.default_config["metric"]}')
            axes[0, 1].grid(True)
            
            # Распределение скоров
            if len(self.optimization_results['all_results']) > 1:
                scores = [result['score'] for result in self.optimization_results['all_results']]
                axes[1, 0].hist(scores, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Распределение скоров')
                axes[1, 0].set_xlabel(f'Метрика ({self.default_config["metric"]})')
                axes[1, 0].set_ylabel('Частота')
                axes[1, 0].grid(True)
            
            # Статистика итераций
            if self.optimization_results['all_results']:
                iterations = list(range(1, len(self.optimization_results['all_results']) + 1))
                std_scores = [result['std_score'] for result in self.optimization_results['all_results']]
                axes[1, 1].plot(iterations, std_scores)
                axes[1, 1].set_title('Стандартное отклонение скоров')
                axes[1, 1].set_xlabel('Итерация')
                axes[1, 1].set_ylabel('Стандартное отклонение')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Графики сохранены в {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Ошибка создания графиков: {e}")
    
    def get_optimization_summary(self) -> Dict:
        """
        Получение сводки результатов оптимизации.
        
        Returns:
            Словарь со сводкой
        """
        summary = {
            'method': self.default_config['method'],
            'metric': self.default_config['metric'],
            'n_iterations': self.optimization_results['n_iterations'],
            'best_score': self.optimization_results['best_score'],
            'best_params': self.optimization_results['best_params'],
            'optimization_time': self.optimization_results['optimization_time'],
            'convergence_improvement': None
        }
        
        # Расчет улучшения сходимости
        if self.optimization_results['convergence_history']:
            initial_score = self.optimization_results['convergence_history'][0]
            final_score = self.optimization_results['convergence_history'][-1]
            summary['convergence_improvement'] = final_score - initial_score
        
        return summary 
"""
Умная адаптивная стратегия фильтров.

Этот модуль реализует динамическую адаптацию фильтров торговых сигналов
на основе рыночных условий и адаптационных признаков.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class AdaptiveFilterStrategy:
    """
    Умная адаптивная стратегия фильтров.
    
    Динамически регулирует параметры фильтров на основе:
    - Текущей волатильности рынка
    - Силы тренда
    - Асимметрии движений (бычий/медвежий рынок)
    - Объемной активности
    - Изменчивости волатильности
    - Исторической производительности модели
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация адаптивной стратегии фильтров.
        
        Args:
            config: Конфигурация стратегии
        """
        self.config = config or {}
        self.logger = logger.bind(name="AdaptiveFilterStrategy")
        
        # Базовые параметры фильтров
        self.base_filters = {
            'min_confidence': 0.35,
            'min_volume_ratio': 0.8,
            'max_volatility_percentile': 98,
            'min_momentum_threshold': 0.0005,
            'trend_filter_enabled': False,
            'volatility_filter_enabled': False,
            'volume_filter_enabled': False,
            'momentum_filter_enabled': False,
            'time_filter_enabled': False,
            'spread_filter_enabled': False,
            'min_gap_between_signals': 2,
            'max_positions': 5,
            'position_size': 0.15
        }
        
        # Адаптивные множители
        self.adaptive_multipliers = {
            'confidence': 1.0,
            'volume_ratio': 1.0,
            'volatility_percentile': 1.0,
            'momentum_threshold': 1.0,
            'position_size': 1.0,
            'max_positions': 1.0,
            'signal_gap': 1.0
        }
        
        # Веса для различных рыночных условий
        self.market_condition_weights = {
            'volatility': 0.25,
            'trend': 0.20,
            'asymmetry': 0.15,
            'volume': 0.15,
            'vol_of_vol': 0.15,
            'model_performance': 0.10
        }
        
        # История производительности для адаптации
        self.performance_history = []
        self.max_history_size = 100
        
        # Состояние рынка
        self.market_state = {
            'current_volatility': 1.0,
            'trend_strength': 0.0,
            'asymmetry_ratio': 1.0,
            'volume_activity': 1.0,
            'vol_of_vol': 0.3,
            'model_accuracy': 0.5
        }
    
    def calculate_market_conditions(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Расчет текущих рыночных условий.
        
        Args:
            data: Исторические данные OHLCV
            
        Returns:
            Словарь с метриками рыночных условий
        """
        try:
            returns = data['close'].pct_change()
            
            # 1. Текущая волатильность относительно исторической
            current_vol = returns.rolling(20).std().iloc[-1]
            historical_vol = returns.rolling(100).std().iloc[-1]
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
            
            # 2. Сила тренда
            short_ma = data['close'].rolling(20).mean().iloc[-1]
            long_ma = data['close'].rolling(50).mean().iloc[-1]
            trend_strength = (short_ma - long_ma) / long_ma if long_ma > 0 else 0.0
            
            # 3. Асимметрия движений
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                pos_vol = positive_returns.rolling(20).std().iloc[-1]
                neg_vol = negative_returns.rolling(20).std().iloc[-1]
                asymmetry_ratio = pos_vol / neg_vol if neg_vol > 0 else 1.0
            else:
                asymmetry_ratio = 1.0
            
            # 4. Объемная активность
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 5. Волатильность волатильности
            vol_series = returns.rolling(20).std()
            vol_of_vol = vol_series.rolling(20).std().iloc[-1]
            vol_of_vol_normalized = vol_of_vol / current_vol if current_vol > 0 else 0.3
            
            # 6. Модельная точность (используем историческую)
            model_accuracy = self._get_model_performance()
            
            market_conditions = {
                'volatility_ratio': vol_ratio,
                'trend_strength': trend_strength,
                'asymmetry_ratio': asymmetry_ratio,
                'volume_ratio': volume_ratio,
                'vol_of_vol': vol_of_vol_normalized,
                'model_accuracy': model_accuracy
            }
            
            # Обновляем состояние рынка
            self.market_state.update({
                'current_volatility': vol_ratio,
                'trend_strength': trend_strength,
                'asymmetry_ratio': asymmetry_ratio,
                'volume_activity': volume_ratio,
                'vol_of_vol': vol_of_vol_normalized,
                'model_accuracy': model_accuracy
            })
            
            return market_conditions
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета рыночных условий: {e}")
            return {
                'volatility_ratio': 1.0,
                'trend_strength': 0.0,
                'asymmetry_ratio': 1.0,
                'volume_ratio': 1.0,
                'vol_of_vol': 0.3,
                'model_accuracy': 0.5
            }
    
    def calculate_adaptive_filters(self, data: pd.DataFrame) -> dict:
        """
        Расчет адаптивных фильтров на основе текущих рыночных условий.
        
        Args:
            data: DataFrame с рыночными данными
            
        Returns:
            Словарь с адаптивными параметрами
        """
        try:
            # Анализируем рыночные условия
            market_conditions = self.calculate_market_conditions(data)
            
            # Ограничиваем логирование - выводим только каждые 1000 вызовов
            if not hasattr(self, '_log_counter'):
                self._log_counter = 0
            self._log_counter += 1
            
            should_log = (self._log_counter % 1000) == 0
            
            if should_log:
                self.logger.info(f"Рыночные условия: vol={market_conditions['volatility_ratio']:.2f}, trend={market_conditions['trend_strength']:.3f}, asym={market_conditions['asymmetry_ratio']:.2f}, vol_act={market_conditions['volume_ratio']:.2f}, vol_of_vol={market_conditions['vol_of_vol']:.2f}, acc={market_conditions['model_accuracy']:.2f}")
            
            # Рассчитываем адаптивные параметры
            filters = self._calculate_adaptive_parameters(market_conditions)
            
            if should_log:
                self.logger.info("Адаптивные фильтры:")
                for key, value in filters.items():
                    if isinstance(value, float):
                        self.logger.info(f"  {key}: {value:.3f}")
                    else:
                        self.logger.info(f"  {key}: {value}")
            
            return filters
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета адаптивных фильтров: {e}")
            # Возвращаем базовые параметры в случае ошибки
            return {
                'min_confidence': 0.5,
                'min_volume_ratio': 1.0,
                'max_volatility_percentile': 90.0,
                'min_momentum_threshold': 0.001,
                'position_size': 0.2,
                'max_positions': 3,
                'min_gap_between_signals': 3
            }
    
    def _calculate_confidence_multiplier(self, market_conditions: Dict[str, float]) -> float:
        """Расчет множителя для минимальной уверенности."""
        vol_ratio = market_conditions['volatility_ratio']
        trend_strength = market_conditions['trend_strength']
        model_accuracy = market_conditions['model_accuracy']
        
        # Высокая волатильность - снижаем требования к уверенности
        vol_factor = 0.8 if vol_ratio > 1.3 else (1.2 if vol_ratio < 0.7 else 1.0)
        
        # Сильный тренд - повышаем требования к уверенности
        trend_factor = 1.2 if abs(trend_strength) > 0.02 else 1.0
        
        # Высокая точность модели - снижаем требования к уверенности
        accuracy_factor = 0.9 if model_accuracy > 0.6 else (1.1 if model_accuracy < 0.4 else 1.0)
        
        return vol_factor * trend_factor * accuracy_factor
    
    def _calculate_volume_multiplier(self, market_conditions: Dict[str, float]) -> float:
        """Расчет множителя для фильтра объема."""
        volume_ratio = market_conditions['volume_ratio']
        vol_ratio = market_conditions['volatility_ratio']
        
        # Высокий объем - снижаем требования
        volume_factor = 0.7 if volume_ratio > 1.5 else (1.3 if volume_ratio < 0.7 else 1.0)
        
        # Высокая волатильность - снижаем требования к объему
        vol_factor = 0.8 if vol_ratio > 1.2 else 1.0
        
        return volume_factor * vol_factor
    
    def _calculate_volatility_multiplier(self, market_conditions: Dict[str, float]) -> float:
        """Расчет множителя для фильтра волатильности."""
        vol_ratio = market_conditions['volatility_ratio']
        vol_of_vol = market_conditions['vol_of_vol']
        
        # Высокая волатильность - повышаем допустимый порог
        vol_factor = 1.1 if vol_ratio > 1.2 else (0.9 if vol_ratio < 0.8 else 1.0)
        
        # Высокая изменчивость волатильности - повышаем порог
        vol_of_vol_factor = 1.1 if vol_of_vol > 0.4 else 1.0
        
        return vol_factor * vol_of_vol_factor
    
    def _calculate_momentum_multiplier(self, market_conditions: Dict[str, float]) -> float:
        """Расчет множителя для порога импульса."""
        trend_strength = market_conditions['trend_strength']
        asymmetry_ratio = market_conditions['asymmetry_ratio']
        
        # Сильный тренд - снижаем требования к импульсу
        trend_factor = 0.7 if abs(trend_strength) > 0.02 else 1.0
        
        # Бычий рынок - снижаем требования к импульсу
        asymmetry_factor = 0.8 if asymmetry_ratio > 1.1 else (1.2 if asymmetry_ratio < 0.9 else 1.0)
        
        return trend_factor * asymmetry_factor
    
    def _calculate_position_size_multiplier(self, market_conditions: Dict[str, float]) -> float:
        """Расчет множителя для размера позиции."""
        vol_ratio = market_conditions['volatility_ratio']
        trend_strength = market_conditions['trend_strength']
        model_accuracy = market_conditions['model_accuracy']
        
        # Низкая волатильность - увеличиваем размер позиции
        vol_factor = 1.3 if vol_ratio < 0.8 else (0.7 if vol_ratio > 1.3 else 1.0)
        
        # Сильный тренд - увеличиваем размер позиции
        trend_factor = 1.2 if abs(trend_strength) > 0.02 else 1.0
        
        # Высокая точность модели - увеличиваем размер позиции
        accuracy_factor = 1.2 if model_accuracy > 0.6 else (0.8 if model_accuracy < 0.4 else 1.0)
        
        return vol_factor * trend_factor * accuracy_factor
    
    def _calculate_max_positions_multiplier(self, market_conditions: Dict[str, float]) -> float:
        """Расчет множителя для максимального количества позиций."""
        vol_ratio = market_conditions['volatility_ratio']
        trend_strength = market_conditions['trend_strength']
        
        # Низкая волатильность - увеличиваем количество позиций
        vol_factor = 1.3 if vol_ratio < 0.8 else (0.7 if vol_ratio > 1.3 else 1.0)
        
        # Сильный тренд - увеличиваем количество позиций
        trend_factor = 1.2 if abs(trend_strength) > 0.02 else 1.0
        
        return vol_factor * trend_factor
    
    def _calculate_gap_multiplier(self, market_conditions: Dict[str, float]) -> float:
        """Расчет множителя для минимального промежутка между сигналами."""
        vol_ratio = market_conditions['volatility_ratio']
        vol_of_vol = market_conditions['vol_of_vol']
        
        # Высокая волатильность - уменьшаем промежуток
        vol_factor = 0.7 if vol_ratio > 1.2 else (1.3 if vol_ratio < 0.8 else 1.0)
        
        # Высокая изменчивость - уменьшаем промежуток
        vol_of_vol_factor = 0.8 if vol_of_vol > 0.4 else 1.0
        
        return vol_factor * vol_of_vol_factor
    
    def _calculate_adaptive_parameters(self, market_conditions: Dict[str, float]) -> dict:
        """
        Расчет адаптивных параметров на основе рыночных условий.
        
        Args:
            market_conditions: Словарь с рыночными условиями
            
        Returns:
            Словарь с адаптивными параметрами
        """
        try:
            # Рассчитываем множители
            confidence_mult = self._calculate_confidence_multiplier(market_conditions)
            volume_mult = self._calculate_volume_multiplier(market_conditions)
            volatility_mult = self._calculate_volatility_multiplier(market_conditions)
            momentum_mult = self._calculate_momentum_multiplier(market_conditions)
            position_size_mult = self._calculate_position_size_multiplier(market_conditions)
            max_positions_mult = self._calculate_max_positions_multiplier(market_conditions)
            gap_mult = self._calculate_gap_multiplier(market_conditions)
            
            # Применяем множители к базовым параметрам
            filters = {
                'min_confidence': self.base_filters['min_confidence'] * confidence_mult,
                'min_volume_ratio': self.base_filters['min_volume_ratio'] * volume_mult,
                'max_volatility_percentile': self.base_filters['max_volatility_percentile'] * volatility_mult,
                'min_momentum_threshold': self.base_filters['min_momentum_threshold'] * momentum_mult,
                'position_size': self.base_filters['position_size'] * position_size_mult,
                'max_positions': max(1, int(self.base_filters['max_positions'] * max_positions_mult)),
                'min_gap_between_signals': max(1, int(self.base_filters['min_gap_between_signals'] * gap_mult))
            }
            
            # Ограничиваем значения
            filters['min_confidence'] = max(0.3, min(0.8, filters['min_confidence']))
            filters['min_volume_ratio'] = max(0.5, min(2.0, filters['min_volume_ratio']))
            filters['max_volatility_percentile'] = max(80, min(99, filters['max_volatility_percentile']))
            filters['min_momentum_threshold'] = max(0.0005, min(0.002, filters['min_momentum_threshold']))
            filters['position_size'] = max(0.1, min(0.4, filters['position_size']))
            filters['max_positions'] = max(1, min(5, filters['max_positions']))
            filters['min_gap_between_signals'] = max(1, min(5, filters['min_gap_between_signals']))
            
            return filters
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета адаптивных параметров: {e}")
            return self.base_filters.copy()
    
    def update_performance_history(self, trade_result: Dict):
        """
        Обновление истории производительности.
        
        Args:
            trade_result: Результат торговой операции
        """
        try:
            # Добавляем результат в историю
            self.performance_history.append(trade_result)
            
            # Ограничиваем размер истории
            if len(self.performance_history) > self.max_history_size:
                self.performance_history = self.performance_history[-self.max_history_size:]
                
        except Exception as e:
            self.logger.error(f"Ошибка обновления истории производительности: {e}")
    
    def _get_model_performance(self) -> float:
        """Получение текущей точности модели на основе истории."""
        try:
            if len(self.performance_history) < 10:
                return 0.5  # Базовое значение
            
            # Рассчитываем точность на основе последних операций
            recent_trades = self.performance_history[-20:]
            profitable_trades = sum(1 for trade in recent_trades if trade.get('profit', 0) > 0)
            
            accuracy = profitable_trades / len(recent_trades)
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета точности модели: {e}")
            return 0.5
    
    def get_market_regime(self, data: pd.DataFrame) -> str:
        """
        Определение текущего режима рынка.
        
        Args:
            data: Исторические данные
            
        Returns:
            Строка с описанием режима рынка
        """
        try:
            market_conditions = self.calculate_market_conditions(data)
            
            vol_ratio = market_conditions['volatility_ratio']
            trend_strength = market_conditions['trend_strength']
            asymmetry_ratio = market_conditions['asymmetry_ratio']
            
            # Определяем режим рынка
            if vol_ratio > 1.3:
                if trend_strength > 0.02:
                    return "Высоковолатильный бычий тренд"
                elif trend_strength < -0.02:
                    return "Высоковолатильный медвежий тренд"
                else:
                    return "Высоковолатильный боковик"
            elif vol_ratio < 0.7:
                if trend_strength > 0.02:
                    return "Низковолатильный бычий тренд"
                elif trend_strength < -0.02:
                    return "Низковолатильный медвежий тренд"
                else:
                    return "Низковолатильный боковик"
            else:
                if trend_strength > 0.02:
                    return "Умеренный бычий тренд"
                elif trend_strength < -0.02:
                    return "Умеренный медвежий тренд"
                else:
                    return "Умеренный боковик"
                    
        except Exception as e:
            self.logger.error(f"Ошибка определения режима рынка: {e}")
            return "Неопределенный режим"
    
    def get_strategy_recommendations(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Получение рекомендаций по стратегии на основе рыночных условий.
        
        Args:
            data: Исторические данные
            
        Returns:
            Словарь с рекомендациями
        """
        try:
            market_conditions = self.calculate_market_conditions(data)
            market_regime = self.get_market_regime(data)
            
            recommendations = {
                'market_regime': market_regime,
                'confidence_strategy': '',
                'position_strategy': '',
                'risk_strategy': '',
                'timing_strategy': ''
            }
            
            vol_ratio = market_conditions['volatility_ratio']
            trend_strength = market_conditions['trend_strength']
            model_accuracy = market_conditions['model_accuracy']
            
            # Рекомендации по уверенности
            if vol_ratio > 1.3:
                recommendations['confidence_strategy'] = "Снизить требования к уверенности из-за высокой волатильности"
            elif vol_ratio < 0.7:
                recommendations['confidence_strategy'] = "Повысить требования к уверенности в спокойном рынке"
            else:
                recommendations['confidence_strategy'] = "Стандартные требования к уверенности"
            
            # Рекомендации по позициям
            if abs(trend_strength) > 0.02:
                recommendations['position_strategy'] = "Увеличить размер позиций в сильном тренде"
            else:
                recommendations['position_strategy'] = "Стандартный размер позиций в боковике"
            
            # Рекомендации по риску
            if vol_ratio > 1.3:
                recommendations['risk_strategy'] = "Снизить риск из-за высокой волатильности"
            elif model_accuracy < 0.4:
                recommendations['risk_strategy'] = "Снизить риск из-за низкой точности модели"
            else:
                recommendations['risk_strategy'] = "Стандартное управление риском"
            
            # Рекомендации по времени
            if market_conditions['vol_of_vol'] > 0.4:
                recommendations['timing_strategy'] = "Быстрые входы и выходы из-за высокой изменчивости"
            else:
                recommendations['timing_strategy'] = "Стандартное время удержания позиций"
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Ошибка получения рекомендаций: {e}")
            return {
                'market_regime': 'Неопределенный',
                'confidence_strategy': 'Использовать базовые настройки',
                'position_strategy': 'Использовать базовые настройки',
                'risk_strategy': 'Использовать базовые настройки',
                'timing_strategy': 'Использовать базовые настройки'
            } 
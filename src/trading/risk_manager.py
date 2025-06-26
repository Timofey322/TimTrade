"""
Адаптивное управление рисками для торговой системы.

Этот модуль предоставляет функциональность для:
- Динамического расчета Stop Loss и Take Profit
- Адаптивного управления размером позиции
- Анализа волатильности рынка
- Управления максимальной просадкой
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


class AdaptiveRiskManager:
    """
    Адаптивный риск-менеджер для торговой системы.
    
    Особенности:
    - Динамические уровни Stop Loss и Take Profit на основе ATR
    - Адаптивный размер позиции на основе волатильности
    - Управление максимальной просадкой
    - Анализ корреляции активов
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация риск-менеджера.
        
        Args:
            config: Конфигурация риск-менеджмента
        """
        self.config = config or {}
        self.logger = logger.bind(name="AdaptiveRiskManager")
        
        # Настройки по умолчанию
        self.default_config = {
            'max_position_size': 0.1,  # Максимальный размер позиции (10% от капитала)
            'max_drawdown': 0.15,  # Максимальная просадка (15%)
            'risk_per_trade': 0.02,  # Риск на сделку (2%)
            'atr_period': 14,  # Период для расчета ATR
            'volatility_lookback': 20,  # Период для анализа волатильности
            'correlation_threshold': 0.7,  # Порог корреляции
            'position_sizing_method': 'kelly',  # Метод расчета размера позиции
            'stop_loss_multiplier': 2.0,  # Множитель для Stop Loss
            'take_profit_multiplier': 3.0,  # Множитель для Take Profit
            'trailing_stop_enabled': True,  # Включить trailing stop
            'trailing_stop_multiplier': 1.5,  # Множитель для trailing stop
            'max_open_positions': 3,  # Максимальное количество открытых позиций
            'min_risk_reward_ratio': 1.5,  # Минимальное соотношение риск/прибыль
        }
        
        # Обновление конфигурации
        if config:
            self.default_config.update(config)
        
        # Состояние риск-менеджера
        self.current_drawdown = 0.0
        self.open_positions = 0
        self.total_risk = 0.0
        self.position_history = []
        
    def calculate_adaptive_levels(self, data: pd.DataFrame, entry_price: float, 
                                signal_type: str = 'long') -> Dict:
        """
        Расчет адаптивных уровней Stop Loss и Take Profit.
        
        Args:
            data: Исторические данные
            entry_price: Цена входа
            signal_type: Тип сигнала ('long' или 'short')
            
        Returns:
            Словарь с уровнями Stop Loss и Take Profit
        """
        try:
            # Рассчитываем ATR
            atr = self._calculate_atr(data, self.default_config['atr_period'])
            
            # Рассчитываем волатильность
            volatility = self._calculate_volatility(data)
            
            # Рассчитываем адаптивные множители
            # Используем среднюю волатильность за последние 50 периодов
            if len(data) >= 50:
                volatility_series = data['close'].pct_change().rolling(window=50).std()
                avg_volatility = volatility_series.mean()
                volatility_ratio = volatility / avg_volatility if avg_volatility > 0 else 1.0
            else:
                volatility_ratio = 1.0
            
            # Проверяем, что volatility_ratio не NaN
            if np.isnan(volatility_ratio):
                volatility_ratio = 1.0
            
            # Адаптируем множители на основе волатильности
            sl_multiplier = self.default_config['stop_loss_multiplier'] * volatility_ratio
            tp_multiplier = self.default_config['take_profit_multiplier'] * volatility_ratio
            
            # Ограничиваем множители
            sl_multiplier = np.clip(sl_multiplier, 1.0, 5.0)
            tp_multiplier = np.clip(tp_multiplier, 2.0, 8.0)
            
            # Рассчитываем уровни
            if signal_type == 'long':
                stop_loss = entry_price - (atr * sl_multiplier)
                take_profit = entry_price + (atr * tp_multiplier)
            else:
                stop_loss = entry_price + (atr * sl_multiplier)
                take_profit = entry_price - (atr * tp_multiplier)
            
            # Проверяем соотношение риск/прибыль
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Если соотношение слишком плохое, корректируем Take Profit
            if risk_reward_ratio < self.default_config['min_risk_reward_ratio']:
                if signal_type == 'long':
                    take_profit = entry_price + (risk * self.default_config['min_risk_reward_ratio'])
                else:
                    take_profit = entry_price - (risk * self.default_config['min_risk_reward_ratio'])
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': atr,
                'volatility_ratio': volatility_ratio,
                'risk_reward_ratio': risk_reward_ratio,
                'sl_multiplier': sl_multiplier,
                'tp_multiplier': tp_multiplier
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета адаптивных уровней: {e}")
            # Возвращаем базовые уровни
            if signal_type == 'long':
                return {
                    'stop_loss': entry_price * 0.98,
                    'take_profit': entry_price * 1.04,
                    'atr': 0,
                    'volatility_ratio': 1.0,
                    'risk_reward_ratio': 2.0,
                    'sl_multiplier': 2.0,
                    'tp_multiplier': 3.0
                }
            else:
                return {
                    'stop_loss': entry_price * 1.02,
                    'take_profit': entry_price * 0.96,
                    'atr': 0,
                    'volatility_ratio': 1.0,
                    'risk_reward_ratio': 2.0,
                    'sl_multiplier': 2.0,
                    'tp_multiplier': 3.0
                }
    
    def calculate_position_size(self, capital: float, risk_per_trade: float, 
                              stop_loss_distance: float, volatility: float) -> float:
        """
        Расчет размера позиции на основе риска.
        
        Args:
            capital: Текущий капитал
            risk_per_trade: Риск на сделку (в долях)
            stop_loss_distance: Расстояние до Stop Loss
            volatility: Текущая волатильность
            
        Returns:
            Размер позиции
        """
        try:
            # Базовый размер позиции на основе риска
            base_position_size = (capital * risk_per_trade) / stop_loss_distance
            
            # Адаптируем размер на основе волатильности
            volatility_adjustment = 1.0 / (1.0 + volatility)
            adjusted_position_size = base_position_size * volatility_adjustment
            
            # Ограничиваем максимальным размером позиции
            max_position = capital * self.default_config['max_position_size']
            position_size = min(adjusted_position_size, max_position)
            
            # Проверяем лимит открытых позиций
            if self.open_positions >= self.default_config['max_open_positions']:
                position_size = 0
            
            # Проверяем максимальную просадку
            if self.current_drawdown >= self.default_config['max_drawdown']:
                position_size = 0
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета размера позиции: {e}")
            return 0.0
    
    def update_trailing_stop(self, current_price: float, entry_price: float, 
                           current_stop_loss: float, signal_type: str = 'long') -> float:
        """
        Обновление trailing stop.
        
        Args:
            current_price: Текущая цена
            entry_price: Цена входа
            current_stop_loss: Текущий Stop Loss
            signal_type: Тип сигнала
            
        Returns:
            Новый Stop Loss
        """
        try:
            if not self.default_config['trailing_stop_enabled']:
                return current_stop_loss
            
            # Рассчитываем прибыль
            if signal_type == 'long':
                profit = current_price - entry_price
                # Поднимаем Stop Loss только если в прибыли
                if profit > 0:
                    new_stop_loss = current_price - (profit * 0.5)  # 50% от прибыли
                    return max(new_stop_loss, current_stop_loss)
            else:
                profit = entry_price - current_price
                # Опускаем Stop Loss только если в прибыли
                if profit > 0:
                    new_stop_loss = current_price + (profit * 0.5)  # 50% от прибыли
                    return min(new_stop_loss, current_stop_loss)
            
            return current_stop_loss
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления trailing stop: {e}")
            return current_stop_loss
    
    def check_risk_limits(self, new_position_size: float, total_risk: float) -> bool:
        """
        Проверка лимитов риска.
        
        Args:
            new_position_size: Размер новой позиции
            total_risk: Общий риск
            
        Returns:
            True если лимиты не превышены
        """
        try:
            # Проверяем максимальную просадку
            if self.current_drawdown >= self.default_config['max_drawdown']:
                self.logger.warning("Достигнут лимит максимальной просадки")
                return False
            
            # Проверяем общий риск
            max_total_risk = self.default_config['max_position_size']
            if total_risk + new_position_size > max_total_risk:
                self.logger.warning("Превышен лимит общего риска")
                return False
            
            # Проверяем количество открытых позиций
            if self.open_positions >= self.default_config['max_open_positions']:
                self.logger.warning("Достигнут лимит открытых позиций")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка проверки лимитов риска: {e}")
            return False
    
    def update_drawdown(self, current_capital: float, peak_capital: float):
        """
        Обновление текущей просадки.
        
        Args:
            current_capital: Текущий капитал
            peak_capital: Пиковый капитал
        """
        try:
            if peak_capital > 0:
                self.current_drawdown = (peak_capital - current_capital) / peak_capital
            else:
                self.current_drawdown = 0.0
                
        except Exception as e:
            self.logger.error(f"Ошибка обновления просадки: {e}")
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Рассчет ATR (Average True Range)."""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1]
            
            return atr if not np.isnan(atr) else 0.0
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета ATR: {e}")
            return 0.0
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Рассчет волатильности."""
        try:
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(window=self.default_config['volatility_lookback']).std().iloc[-1]
            return volatility if not np.isnan(volatility) else 0.0
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета волатильности: {e}")
            return 0.0
    
    def get_risk_summary(self) -> Dict:
        """
        Получение сводки по рискам.
        
        Returns:
            Словарь с информацией о рисках
        """
        return {
            'current_drawdown': self.current_drawdown,
            'open_positions': self.open_positions,
            'total_risk': self.total_risk,
            'max_drawdown_limit': self.default_config['max_drawdown'],
            'max_positions_limit': self.default_config['max_open_positions'],
            'max_position_size': self.default_config['max_position_size']
        } 
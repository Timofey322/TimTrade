#!/usr/bin/env python3
"""
Модуль для расчета confidence score торговых сигналов
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger

class ConfidenceScorer:
    """
    Класс для расчета confidence score торговых сигналов на основе:
    - Согласованности индикаторов
    - Силы сигнала
    - Волатильности
    - Объемного подтверждения
    - Рыночного режима
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализация scorer'а.
        
        Args:
            config: Конфигурация для расчета confidence
        """
        self.config = config or {}
        self.logger = logger.bind(name="ConfidenceScorer")
        
        # Веса для различных факторов
        self.weights = self.config.get('weights', {
            'indicator_agreement': 0.25,
            'signal_strength': 0.22,
            'volatility_factor': 0.18,
            'volume_confirmation': 0.15,
            'market_regime': 0.1,
            'sentiment_confirmation': 0.1  # НОВОЕ: Добавлен sentiment фактор
        })
        
        # Пороги для различных индикаторов
        self.thresholds = self.config.get('thresholds', {
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'williams_r_overbought': -20,
            'williams_r_oversold': -80,
            'cci_overbought': 100,
            'cci_oversold': -100,
            'high_volatility_threshold': 2.0,
            'low_volume_threshold': 0.5
        })
    
    def calculate_confidence(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Основной метод для расчета confidence score.
        
        Args:
            df: DataFrame с индикаторами
            signals: Series с торговыми сигналами (-1, 0, 1)
        
        Returns:
            Series с confidence scores (0.0 - 1.0)
        """
        try:
            self.logger.info("Расчет confidence scores для сигналов")
            
            confidence_scores = pd.Series(index=df.index, dtype=float)
            
            for i in range(len(df)):
                # Пропускаем строки без сигнала
                if signals.iloc[i] == 0:
                    confidence_scores.iloc[i] = 0.0
                    continue
                
                # Собираем факторы уверенности
                factors = self._collect_confidence_factors(df.iloc[i], signals.iloc[i])
                
                # Рассчитываем взвешенный score
                confidence = self._calculate_weighted_confidence(factors)
                confidence_scores.iloc[i] = confidence
            
            self.logger.info(f"Рассчитан confidence для {len(confidence_scores[confidence_scores > 0])} сигналов")
            self.logger.info(f"Средний confidence: {confidence_scores[confidence_scores > 0].mean():.3f}")
            
            return confidence_scores
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета confidence: {e}")
            return pd.Series(index=df.index, data=0.5)  # Возвращаем нейтральный confidence
    
    def _collect_confidence_factors(self, row: pd.Series, signal: int) -> Dict[str, float]:
        """
        Собирает все факторы уверенности для одной записи.
        
        Args:
            row: Строка данных с индикаторами
            signal: Торговый сигнал (-1, 0, 1)
        
        Returns:
            Словарь с факторами уверенности
        """
        factors = {}
        
        # 1. Согласованность индикаторов
        factors['indicator_agreement'] = self._calculate_indicator_agreement(row, signal)
        
        # 2. Сила сигнала
        factors['signal_strength'] = self._calculate_signal_strength(row, signal)
        
        # 3. Фактор волатильности
        factors['volatility_factor'] = self._calculate_volatility_factor(row)
        
        # 4. Объемное подтверждение
        factors['volume_confirmation'] = self._calculate_volume_confirmation(row, signal)
        
        # 5. Рыночный режим
        factors['market_regime'] = self._calculate_market_regime_factor(row, signal)
        
        # 6. НОВОЕ: Sentiment подтверждение
        factors['sentiment_confirmation'] = self._calculate_sentiment_confirmation(row, signal)
        
        return factors
    
    def _calculate_indicator_agreement(self, row: pd.Series, signal: int) -> float:
        """
        Рассчитывает согласованность индикаторов.
        
        Args:
            row: Строка данных
            signal: Торговый сигнал
        
        Returns:
            Согласованность индикаторов (0.0 - 1.0)
        """
        try:
            agreements = []
            
            # MACD согласие
            if 'macd_line' in row.index and 'macd_signal' in row.index:
                macd_bullish = row['macd_line'] > row['macd_signal']
                agreements.append(1.0 if (signal > 0 and macd_bullish) or (signal < 0 and not macd_bullish) else 0.0)
            
            # RSI согласие
            if 'rsi_14' in row.index:
                rsi = row['rsi_14']
                if signal > 0:  # Покупка
                    agreements.append(1.0 if rsi < self.thresholds['rsi_overbought'] and rsi > self.thresholds['rsi_oversold'] else 0.0)
                else:  # Продажа
                    agreements.append(1.0 if rsi > self.thresholds['rsi_oversold'] else 0.0)
            
            # Williams %R согласие
            if 'williams_r_14' in row.index:
                williams_r = row['williams_r_14']
                if signal > 0:  # Покупка
                    agreements.append(1.0 if williams_r < self.thresholds['williams_r_overbought'] else 0.0)
                else:  # Продажа
                    agreements.append(1.0 if williams_r > self.thresholds['williams_r_oversold'] else 0.0)
            
            # CCI согласие
            if 'cci_20' in row.index:
                cci = row['cci_20']
                if signal > 0:  # Покупка
                    agreements.append(1.0 if cci > self.thresholds['cci_oversold'] and cci < self.thresholds['cci_overbought'] else 0.0)
                else:  # Продажа
                    agreements.append(1.0 if cci < self.thresholds['cci_overbought'] else 0.0)
            
            # VWAP согласие
            if 'close' in row.index and 'vwap' in row.index:
                price_above_vwap = row['close'] > row['vwap']
                agreements.append(1.0 if (signal > 0 and price_above_vwap) or (signal < 0 and not price_above_vwap) else 0.0)
            
            # OBV согласие (используем momentum)
            if 'obv' in row.index:
                # Предполагаем, что есть OBV momentum из предыдущих расчетов
                obv_momentum = row.get('obv_momentum', 0)
                agreements.append(1.0 if (signal > 0 and obv_momentum > 0) or (signal < 0 and obv_momentum < 0) else 0.0)
            
            return np.mean(agreements) if agreements else 0.5
            
        except Exception as e:
            self.logger.warning(f"Ошибка расчета согласованности индикаторов: {e}")
            return 0.5
    
    def _calculate_signal_strength(self, row: pd.Series, signal: int) -> float:
        """
        Рассчитывает силу сигнала.
        
        Args:
            row: Строка данных
            signal: Торговый сигнал
        
        Returns:
            Сила сигнала (0.0 - 1.0)
        """
        try:
            strengths = []
            
            # Сила MACD
            if 'macd_histogram' in row.index:
                macd_hist = abs(row['macd_histogram'])
                macd_strength = min(macd_hist / 0.01, 1.0)  # Нормализуем к 1.0
                strengths.append(macd_strength)
            
            # Сила RSI (расстояние от нейтральной зоны)
            if 'rsi_14' in row.index:
                rsi = row['rsi_14']
                rsi_strength = abs(rsi - 50) / 50  # Расстояние от 50
                strengths.append(rsi_strength)
            
            # Сила Williams %R
            if 'williams_r_14' in row.index:
                williams_r = row['williams_r_14']
                williams_strength = abs(williams_r + 50) / 50  # Расстояние от -50
                strengths.append(williams_strength)
            
            # Сила CCI
            if 'cci_20' in row.index:
                cci = abs(row['cci_20'])
                cci_strength = min(cci / 200, 1.0)  # Нормализуем к 1.0
                strengths.append(cci_strength)
            
            # Ценовой momentum
            if 'momentum_1' in row.index:
                momentum = abs(row['momentum_1'])
                momentum_strength = min(momentum * 100, 1.0)  # Преобразуем в проценты
                strengths.append(momentum_strength)
            
            return np.mean(strengths) if strengths else 0.5
            
        except Exception as e:
            self.logger.warning(f"Ошибка расчета силы сигнала: {e}")
            return 0.5
    
    def _calculate_volatility_factor(self, row: pd.Series) -> float:
        """
        Рассчитывает фактор волатильности (высокая волатильность = низкая уверенность).
        
        Args:
            row: Строка данных
        
        Returns:
            Фактор волатильности (0.0 - 1.0)
        """
        try:
            # ATR как основной индикатор волатильности
            if 'atr_percentage_14' in row.index:
                atr_pct = row['atr_percentage_14']
                # Высокая волатильность (>3%) снижает уверенность
                if atr_pct > 3.0:
                    return 0.3
                elif atr_pct > 2.0:
                    return 0.6
                elif atr_pct > 1.0:
                    return 0.8
                else:
                    return 1.0
            
            # Альтернативно - используем volatility
            if 'volatility_20' in row.index:
                vol = row['volatility_20']
                # Нормализуем волатильность
                vol_factor = 1.0 / (1.0 + vol * 10)
                return vol_factor
            
            return 0.7  # Средний уровень при отсутствии данных
            
        except Exception as e:
            self.logger.warning(f"Ошибка расчета фактора волатильности: {e}")
            return 0.7
    
    def _calculate_volume_confirmation(self, row: pd.Series, signal: int) -> float:
        """
        Рассчитывает улучшенное объемное подтверждение.
        
        Args:
            row: Строка данных
            signal: Торговый сигнал
        
        Returns:
            Объемное подтверждение (0.0 - 1.0)
        """
        try:
            volume_scores = []
            
            # 1. Относительный объем (улучшенный)
            if 'relative_volume' in row.index:
                rel_vol = row['relative_volume']
                if rel_vol > 2.0:  # Экстремально высокий объем
                    volume_scores.append(1.0)
                elif rel_vol > 1.5:  # Высокий объем
                    volume_scores.append(0.9)
                elif rel_vol > 1.2:  # Умеренно высокий объем
                    volume_scores.append(0.7)
                elif rel_vol > 0.8:  # Нормальный объем
                    volume_scores.append(0.5)
                else:  # Низкий объем
                    volume_scores.append(0.2)
            
            # 2. Volume trend (добавлено)
            if 'volume_sma_20' in row.index and 'volume' in row.index:
                current_vol = row['volume']
                avg_vol = row['volume_sma_20']
                vol_trend_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
                
                if vol_trend_ratio > 1.5:
                    volume_scores.append(1.0)
                elif vol_trend_ratio > 1.2:
                    volume_scores.append(0.8)
                elif vol_trend_ratio > 0.8:
                    volume_scores.append(0.6)
                else:
                    volume_scores.append(0.3)
            
            # 3. OBV momentum confirmation (добавлено)
            if 'obv' in row.index and 'obv_sma_10' in row.index:
                obv_current = row['obv']
                obv_avg = row['obv_sma_10']
                obv_momentum = 1.0 if obv_current > obv_avg else -1.0
                
                # OBV должен поддерживать направление сигнала
                if signal > 0:  # Покупка
                    volume_scores.append(1.0 if obv_momentum > 0 else 0.2)
                else:  # Продажа
                    volume_scores.append(1.0 if obv_momentum < 0 else 0.2)
            
            # 4. MFI подтверждение (улучшенное)
            if 'mfi_14' in row.index:
                mfi = row['mfi_14']
                if signal > 0:  # Покупка
                    if 30 < mfi < 70:  # Оптимальная зона для покупки
                        volume_scores.append(1.0)
                    elif mfi < 30:  # Перепроданность
                        volume_scores.append(0.9)
                    elif mfi < 80:  # Еще не перекупленность
                        volume_scores.append(0.6)
                    else:  # Перекупленность
                        volume_scores.append(0.2)
                else:  # Продажа
                    if mfi > 70:  # Перекупленность
                        volume_scores.append(1.0)
                    elif mfi > 50:  # Выше среднего
                        volume_scores.append(0.7)
                    else:  # Низкий MFI
                        volume_scores.append(0.4)
            
            # 5. CMF подтверждение (улучшенное)
            if 'cmf_20' in row.index:
                cmf = row['cmf_20']
                if signal > 0:  # Покупка
                    if cmf > 0.1:  # Сильный денежный поток
                        volume_scores.append(1.0)
                    elif cmf > 0:  # Положительный денежный поток
                        volume_scores.append(0.8)
                    elif cmf > -0.1:  # Нейтральный
                        volume_scores.append(0.5)
                    else:  # Отрицательный денежный поток
                        volume_scores.append(0.2)
                else:  # Продажа
                    if cmf < -0.1:  # Сильный отток
                        volume_scores.append(1.0)
                    elif cmf < 0:  # Отрицательный денежный поток
                        volume_scores.append(0.8)
                    elif cmf < 0.1:  # Нейтральный
                        volume_scores.append(0.5)
                    else:  # Положительный денежный поток против продажи
                        volume_scores.append(0.2)
            
            # 6. Volume-Price Trend (VPT) если доступен
            if 'vpt' in row.index and 'vpt_sma_10' in row.index:
                vpt_current = row['vpt']
                vpt_avg = row['vpt_sma_10']
                vpt_trend = 1.0 if vpt_current > vpt_avg else -1.0
                
                if signal > 0:  # Покупка
                    volume_scores.append(1.0 if vpt_trend > 0 else 0.3)
                else:  # Продажа
                    volume_scores.append(1.0 if vpt_trend < 0 else 0.3)
            
            # Возвращаем средневзвешенное значение с приоритетом для более надежных индикаторов
            if volume_scores:
                # Если есть много подтверждений, усиливаем сигнал
                if len(volume_scores) >= 3:
                    base_score = np.mean(volume_scores)
                    # Бонус за множественные подтверждения
                    confirmation_bonus = min(0.2, (len(volume_scores) - 2) * 0.05)
                    return min(1.0, base_score + confirmation_bonus)
                else:
                    return np.mean(volume_scores)
            else:
                return 0.5  # Нейтральное значение при отсутствии данных
            
        except Exception as e:
            self.logger.warning(f"Ошибка расчета объемного подтверждения: {e}")
            return 0.5
    
    def _calculate_market_regime_factor(self, row: pd.Series, signal: int) -> float:
        """
        Рассчитывает фактор рыночного режима.
        
        Args:
            row: Строка данных
            signal: Торговый сигнал
        
        Returns:
            Фактор рыночного режима (0.0 - 1.0)
        """
        try:
            # Трендовость (SMA20 vs SMA50)
            if 'sma_20' in row.index and 'sma_50' in row.index:
                sma20 = row['sma_20']
                sma50 = row['sma_50']
                
                trend_up = sma20 > sma50
                
                if signal > 0:  # Покупка
                    return 1.0 if trend_up else 0.4
                else:  # Продажа
                    return 1.0 if not trend_up else 0.4
            
            # Позиция относительно Bollinger Bands
            if 'bb_position_20' in row.index:
                bb_pos = row['bb_position_20']
                
                if signal > 0:  # Покупка
                    return 1.0 if 0.2 < bb_pos < 0.8 else 0.5
                else:  # Продажа
                    return 1.0 if bb_pos > 0.8 or bb_pos < 0.2 else 0.5
            
            return 0.7  # Средний уровень при отсутствии данных
            
        except Exception as e:
            self.logger.warning(f"Ошибка расчета фактора рыночного режима: {e}")
            return 0.7
    
    def _calculate_weighted_confidence(self, factors: Dict[str, float]) -> float:
        """
        Рассчитывает взвешенный confidence score.
        
        Args:
            factors: Словарь с факторами уверенности
        
        Returns:
            Итоговый confidence score (0.0 - 1.0)
        """
        try:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for factor_name, factor_value in factors.items():
                weight = self.weights.get(factor_name, 0.0)
                weighted_sum += factor_value * weight
                total_weight += weight
            
            # Нормализуем к диапазону 0-1
            confidence = weighted_sum / total_weight if total_weight > 0 else 0.5
            
            # Ограничиваем диапазон
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"Ошибка расчета взвешенного confidence: {e}")
            return 0.5
    
    def filter_signals_by_confidence(self, signals: pd.Series, confidence: pd.Series, 
                                   min_confidence: float = 0.6) -> pd.Series:
        """
        Фильтрует сигналы по минимальному уровню уверенности.
        
        Args:
            signals: Series с торговыми сигналами
            confidence: Series с confidence scores
            min_confidence: Минимальный уровень уверенности
        
        Returns:
            Отфильтрованные сигналы
        """
        try:
            filtered_signals = signals.copy()
            
            # Обнуляем сигналы с низкой уверенностью
            low_confidence_mask = confidence < min_confidence
            filtered_signals[low_confidence_mask] = 0
            
            original_count = len(signals[signals != 0])
            filtered_count = len(filtered_signals[filtered_signals != 0])
            
            self.logger.info(f"Фильтрация по confidence {min_confidence}: {original_count} → {filtered_count} сигналов")
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Ошибка фильтрации сигналов: {e}")
            return signals
    
    def get_confidence_statistics(self, confidence: pd.Series) -> Dict[str, float]:
        """
        Получает статистику по confidence scores.
        
        Args:
            confidence: Series с confidence scores
        
        Returns:
            Словарь со статистикой
        """
        try:
            non_zero_confidence = confidence[confidence > 0]
            
            if len(non_zero_confidence) == 0:
                return {'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            
            stats = {
                'count': len(non_zero_confidence),
                'mean': non_zero_confidence.mean(),
                'std': non_zero_confidence.std(),
                'min': non_zero_confidence.min(),
                'max': non_zero_confidence.max(),
                'q25': non_zero_confidence.quantile(0.25),
                'q50': non_zero_confidence.quantile(0.5),
                'q75': non_zero_confidence.quantile(0.75),
                'high_confidence_ratio': len(non_zero_confidence[non_zero_confidence > 0.7]) / len(non_zero_confidence)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета статистики confidence: {e}")
            return {}
    
    def _calculate_sentiment_confirmation(self, row: pd.Series, signal: int) -> float:
        """
        Рассчитывает подтверждение на основе sentiment данных.
        
        Args:
            row: Строка данных
            signal: Торговый сигнал
        
        Returns:
            Sentiment подтверждение (0.0 - 1.0)
        """
        try:
            confirmations = []
            
            # Composite sentiment score
            if 'sentiment_composite_score' in row.index:
                sentiment_score = row['sentiment_composite_score']
                
                if signal > 0:  # Покупка
                    # Высокий sentiment поддерживает покупку
                    if sentiment_score > 0.6:
                        confirmations.append(1.0)
                    elif sentiment_score > 0.4:
                        confirmations.append(0.7)
                    else:
                        confirmations.append(0.3)  # Низкий sentiment против покупки
                        
                else:  # Продажа
                    # Низкий sentiment поддерживает продажу
                    if sentiment_score < 0.4:
                        confirmations.append(1.0)
                    elif sentiment_score < 0.6:
                        confirmations.append(0.7)
                    else:
                        confirmations.append(0.3)  # Высокий sentiment против продажи
            
            # Market regime confirmation
            if signal > 0:  # Покупка
                if 'market_regime_bullish' in row.index and row['market_regime_bullish'] == 1:
                    confirmations.append(1.0)  # Бычий режим поддерживает покупку
                elif 'market_regime_neutral' in row.index and row['market_regime_neutral'] == 1:
                    confirmations.append(0.6)  # Нейтральный режим частично поддерживает
                elif 'market_regime_bearish' in row.index and row['market_regime_bearish'] == 1:
                    confirmations.append(0.2)  # Медвежий режим против покупки
                    
            else:  # Продажа
                if 'market_regime_bearish' in row.index and row['market_regime_bearish'] == 1:
                    confirmations.append(1.0)  # Медвежий режим поддерживает продажу
                elif 'market_regime_neutral' in row.index and row['market_regime_neutral'] == 1:
                    confirmations.append(0.6)  # Нейтральный режим частично поддерживает
                elif 'market_regime_bullish' in row.index and row['market_regime_bullish'] == 1:
                    confirmations.append(0.2)  # Бычий режим против продажи
            
            # Bybit gainers/losers ratio
            if 'bybit_gainers_ratio' in row.index:
                gainers_ratio = row['bybit_gainers_ratio']
                
                if signal > 0:  # Покупка
                    # Высокий ratio gainers поддерживает покупку
                    confirmations.append(gainers_ratio)
                else:  # Продажа
                    # Низкий ratio gainers поддерживает продажу
                    confirmations.append(1.0 - gainers_ratio)
            
            # Bybit market sentiment
            if 'bybit_market_sentiment' in row.index:
                market_sentiment = row['bybit_market_sentiment']
                
                if signal > 0:  # Покупка
                    # Позитивный sentiment поддерживает покупку
                    confirmations.append(market_sentiment)
                else:  # Продажа
                    # Негативный sentiment поддерживает продажу
                    confirmations.append(1.0 - market_sentiment)
            
            # Возвращаем среднее подтверждение или нейтральное значение
            return np.mean(confirmations) if confirmations else 0.5
            
        except Exception as e:
            self.logger.warning(f"Ошибка расчета sentiment подтверждения: {e}")
            return 0.5
    
    def calculate_advanced_confidence(self, features_row: pd.Series, signal: int) -> float:
        """
        Расчет продвинутого confidence с учетом fundamentals.
        
        Args:
            features_row: Строка с фичами
            signal: Торговый сигнал
        
        Returns:
            Продвинутый confidence score (0.0 - 1.0)
        """
        try:
            # Базовый confidence
            base_confidence = self._calculate_weighted_confidence(
                self._collect_confidence_factors(features_row, signal)
            )
            
            # Fundamental factors
            sentiment_score = features_row.get('sentiment_composite_score', 0.5)
            market_regime_bullish = features_row.get('market_regime_bullish', 0)
            market_regime_bearish = features_row.get('market_regime_bearish', 0)
            
            # Модификаторы confidence на основе fundamentals
            confidence_multiplier = 1.0
            
            # Если sentiment очень сильно поддерживает сигнал
            if signal > 0 and sentiment_score > 0.7:
                confidence_multiplier *= 1.2  # Увеличиваем на 20% для покупки при высоком sentiment
            elif signal < 0 and sentiment_score < 0.3:
                confidence_multiplier *= 1.2  # Увеличиваем на 20% для продажи при низком sentiment
            
            # Если market regime поддерживает сигнал
            if signal > 0 and market_regime_bullish:
                confidence_multiplier *= 1.15  # Увеличиваем для покупки в бычьем режиме
            elif signal < 0 and market_regime_bearish:
                confidence_multiplier *= 1.15  # Увеличиваем для продажи в медвежьем режиме
            
            # Если sentiment/режим противоречат сигналу
            if signal > 0 and market_regime_bearish:
                confidence_multiplier *= 0.8  # Уменьшаем для покупки в медвежьем режиме
            elif signal < 0 and market_regime_bullish:
                confidence_multiplier *= 0.8  # Уменьшаем для продажи в бычьем режиме
            
            # Финальный confidence
            final_confidence = base_confidence * confidence_multiplier
            
            return np.clip(final_confidence, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета продвинутого confidence: {e}")
            return 0.5 
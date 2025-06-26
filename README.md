# 🚀 Enhanced Crypto Trading System with Sentiment Analysis (+9.42%  69 days) 

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Performance](https://img.shields.io/badge/Performance-+61%25_APY-success.svg)](#performance)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](#)

An advanced cryptocurrency trading system that combines **Machine Learning**, **Technical Analysis**, and **Real-time Sentiment Analysis** to generate profitable trading signals.

## 🎯 **Key Results**

- **📈 +9.42% Return** in 69 days (2+ months)
- **🚀 61.03% Annualized Return** 
- **📊 Sharpe Ratio: 1.885** (excellent risk-adjusted returns)
- **⚡ 0.136% Average Daily Return**
- **🎪 76.5% Average Confidence** in trading signals

## ✨ **Features**

### 🤖 **Machine Learning Models**
- **XGBoost** - Gradient boosting with optimized hyperparameters
- **Random Forest** - Ensemble learning with 300+ estimators  
- **LightGBM** - Fast gradient boosting (53.4% accuracy on 1h timeframe)
- **Hyperparameter Optimization** - Automated parameter tuning

### 📊 **Technical Analysis**
- **47+ Technical Indicators** (MACD, RSI, Bollinger Bands, etc.)
- **Multi-timeframe Analysis** (5m, 15m, 1h)
- **Adaptive Indicators** - Dynamic parameter adjustment
- **Volume Confirmation** - 28% weight in confidence scoring

### 🌐 **Sentiment Analysis** 
- **Real-time Bybit Market Data** - Hot sectors, trending coins
- **Market Regime Detection** - Bullish/Bearish/Neutral identification
- **Sentiment-based Signal Adjustment** - Adaptive thresholds
- **415+ Advanced Features** from fundamental analysis

### 🛡️ **Risk Management**
- **Adaptive Position Sizing** - Based on market regime (+30% in bullish, -30% in bearish)
- **Dynamic Stop Loss/Take Profit** - Volatility-adjusted levels
- **Confidence-based Filtering** - Only high-quality signals (>65% confidence)
- **Max Drawdown Control** - Limited to -8.09%

## 🏗️ **Architecture**

```
📦 Enhanced Trading System
├── 🧠 ML Models (XGBoost, RF, LightGBM)
├── 📊 Technical Indicators (47+ indicators)
├── 🌐 Sentiment Analysis (Bybit + Market Regime)
├── ⚖️ Confidence Scoring (6-factor system)
├── 🛡️ Risk Management (Adaptive position sizing)
└── 📈 Backtesting (Historical validation)
```

## 🚀 **Quick Start**

### 1. **Installation**
```bash
git clone https://github.com/Timofey322/tradingModels.git
cd tradingModels
pip install -r requirements.txt
```

### 2. **Run the Trading System**
```bash
python run_smart_adaptive.py
```

### 3. **Run Sentiment Monitoring**
```bash
python monitor_sentiment.py
```

### 4. **Run Backtesting**
```bash
python enhanced_backtest.py
```

## 📋 **Configuration**

The system uses YAML configuration files:

- `config/smart_adaptive_config.yaml` - Main configuration
- `config/advanced_features_config.yaml` - Advanced features settings

### Key Settings:
```yaml
# Enhanced indicators for each timeframe
indicators:
  '5m': ['macd', 'obv', 'vwap']
  '15m': ['macd', 'obv', 'vwap']  # Optimized from RSI to MACD
  '1h': ['rsi', 'obv', 'vwap']

# Improved confidence thresholds
thresholds:
  min_confidence: 0.65  # Raised from 0.6
  high_confidence: 0.80  # Raised from 0.75

# Enhanced confidence weights
weights:
  volume_confirmation: 0.28  # Increased from 0.15
  indicator_agreement: 0.22
  signal_strength: 0.20
  volatility_factor: 0.15
  market_regime: 0.10
  sentiment_confirmation: 0.05
```

## 📊 **Performance Analysis**

### 📈 **Returns Breakdown**
- **Total Return:** +9.42% in 69 days
- **Annualized:** ~61% APY
- **Daily Average:** 0.136%
- **Weekly Average:** 0.96%
- **Monthly Average:** 4.71%

### ⚖️ **Risk Metrics**
- **Sharpe Ratio:** 1.885
- **Max Drawdown:** -8.09%
- **Win Rate:** 24.7%
- **Profit Factor:** 3.59
- **Total Trades:** 478

### 🎯 **Signal Quality**
- **Average Confidence:** 76.5%
- **High Confidence Signals:** 5,958 out of 7,078
- **Signal Generation:** Enhanced with sentiment factors
- **Market Regime Accuracy:** 100% bullish detection during high sentiment

## 🌟 **Advanced Features**

### 🧠 **Sentiment Integration**
```python
# Real-time Bybit sentiment data
sentiment_score = collector.get_market_sentiment()
market_regime = detect_market_regime(sentiment_score)

# Adaptive signal thresholds
if sentiment_score > 0.7:
    confirmations_needed -= 2  # Bullish market
elif sentiment_score < 0.3:
    confirmations_needed += 2  # Bearish market
```

### 📊 **Multi-timeframe Ensemble**
```python
# Combine signals from multiple timeframes
ensemble_signal = weighted_average([
    model_5m.predict(features_5m) * 0.3,
    model_15m.predict(features_15m) * 0.4,
    model_1h.predict(features_1h) * 0.3
])
```

### 🛡️ **Dynamic Risk Management**
```python
# Sentiment-based position sizing
if market_regime == "bullish":
    position_size *= 1.3  # +30% in bullish markets
elif market_regime == "bearish":
    position_size *= 0.7  # -30% in bearish markets
```

## 📁 **Project Structure**

```
tradingModels/
├── 📊 src/
│   ├── 🤖 ml_models/          # Machine learning models
│   ├── 🔧 preprocessing/       # Feature engineering & indicators
│   ├── 📈 backtesting/         # Backtesting engine
│   ├── 🌐 data_collection/     # Data & sentiment collection
│   ├── 💼 trading/             # Risk management
│   └── 🛠️ utils/              # Utilities
├── ⚙️ config/                  # Configuration files
├── 📊 backtest_results/        # Historical results
├── 🤖 models/                  # Trained models & cache
├── 📈 data/                    # Market data cache
└── 📋 logs/                    # System logs
```

## 🔧 **Technical Requirements**

- **Python 3.8+**
- **Libraries:** pandas, numpy, scikit-learn, xgboost, lightgbm, ta-lib
- **Memory:** 4GB+ RAM recommended
- **Storage:** 2GB+ for data cache

## 📈 **Future Improvements**

- [ ] **Deep Learning Models** (LSTM, Transformer)
- [ ] **More Data Sources** (Twitter sentiment, news analysis)
- [ ] **Real-time Trading** (Binance/Bybit API integration)
- [ ] **Portfolio Optimization** (Multi-asset trading)
- [ ] **Advanced Risk Models** (VaR, CVaR)

## 🤝 **Contributing**

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ **Disclaimer**

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and never invest more than you can afford to lose.

## 📞 **Contact**

- **GitHub:** [@Timofey322](https://github.com/Timofey322)
- **Project Link:** [https://github.com/Timofey322/tradingModels](https://github.com/Timofey322/tradingModels)

---

⭐ **Star this repository if you found it helpful!** ⭐ 

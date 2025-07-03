# 🤖 TimAI — Advanced AI Trading System (+18.7% 90 days)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

## Overview

TimAI is a state-of-the-art AI-powered cryptocurrency trading system. It leverages machine learning, multi-timeframe analysis, advanced feature engineering, and robust risk management to deliver consistent, high-quality trading signals.

## 📊 Backtest Results (3 Months)

| Metric | Value |
|--------|-------|
| Test Period | 2025-04-03 — 2025-07-02 |
| Initial Balance | $1,400 |
| Final Balance | $1,661.80 |
| Net Profit | $261.80 (+18.7%) |
| Number of Trades | 33 |
| Win Rate | 54.5% |
| Max Drawdown | 1.21% |
| Avg. Profit/Trade | 0.57% |
| Best Trade | 2.75% |
| Worst Trade | -1.10% |
| Risk/Reward Ratio | 1.60 |

> **Note**: All results are on strictly out-of-sample data. The system is trained on historical data and tested on unseen 2025 data.

## ✨ Features

- **XGBoost & LightGBM**: Powerful gradient boosting models
- **1D CNN**: Deep learning for pattern recognition
- **Multi-Timeframe Analysis**: 5m, 15m, 1h for robust signals
- **SMOTE**: Class balancing for better generalization
- **Optuna**: Bayesian hyperparameter optimization
- **Weighted Feature Engineering**: 80+ technical indicators
- **Ensemble Methods**: Model stacking and voting
- **Dynamic Risk Management**: Adaptive stop-loss/take-profit
- **Position Scaling**: Dynamic position sizing by signal strength
- **Trend Filters**: Adaptive trend detection for quality trades

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd WW2
pip install -r requirements.txt
```

### 2. Run Backtest

```bash
python timai_test.py
```

### 3. Use in Your Code

```python
from core.timai_core import TimAI

# Initialize TimAI
timai = TimAI(
    symbols=['BTCUSDT'],
    timeframes=['5m', '15m', '1h'],
    use_smote=True,
    use_optuna=True
)

# Train and backtest
results = timai.run_backtest()
print(f"Profit: {results['total_profit']:.2f}%")
```

## 📁 Project Structure

```
WW2/
├── api/                    # Trading API integration
├── config/                 # Configuration files
├── core/                   # Core TimAI system
│   └── timai_core.py      # Main TimAI class
├── data/                   # Historical data
│   ├── historical/        # CSV data files
│   └── bybit_downloader.py
├── models/                 # Trained models
│   ├── production/        # Production models
│   └── experimental/      # Experimental models
├── research/              # Research modules
│   ├── smote_balancer.py  # SMOTE class balancing
│   └── optuna_optimizer.py # Hyperparameter optimization
├── tests/                 # Test files
├── timai_test.py          # Main test script
└── requirements.txt       # Dependencies
```

## 🔧 Key Technologies

- **Machine Learning**: XGBoost, LightGBM, scikit-learn
- **Deep Learning**: TensorFlow/Keras
- **Optimization**: Optuna, Hyperopt
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, plotly

## 📞 Contact & Support

For questions and support, contact via Telegram: [@tim_tim_tim322](https://t.me/tim_tim_tim322)

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational purposes only. Past performance does not guarantee future results. Trading cryptocurrencies involves substantial risk of loss.

---

**Made with ❤️ by the Tim (Im GAY)** 

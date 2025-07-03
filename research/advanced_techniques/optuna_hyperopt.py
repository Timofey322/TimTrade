import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb

# --- Загрузка данных (пример, путь можно менять) ---
df = pd.read_csv('data/historical/BTCUSDT_15m_5years_20210705_20250702.csv')
# Здесь предполагается, что feature engineering уже проведён и есть X, y
# Для примера:
from core.timai_core import TimAIFeatureEngine
fe = TimAIFeatureEngine()
df_feat = fe.engineer_features(df)
returns = df_feat['close'].pct_change().shift(-1)
vol = returns.rolling(50).std()
target = np.where(returns > vol * 0.8, 2, np.where(returns > vol * 0.3, 1, np.where(returns < -vol * 0.8, 0, 1)))
df_feat = df_feat.iloc[:-1]
target = target[:-1]
valid_mask = ~(np.isnan(target) | np.isinf(target))
df_feat = df_feat[valid_mask]
target = target[valid_mask]
exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'datetime']
X = df_feat.select_dtypes(include=[np.number])
X = X[[col for col in X.columns if col not in exclude_cols]]
y = target

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

# --- Optuna для XGBoost ---
def objective_xgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'verbosity': 0,
        'enable_categorical': False
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train.astype(np.float32), y_train)
    preds = model.predict(X_valid.astype(np.float32))
    return f1_score(y_valid, preds, average='weighted')

# --- Optuna для LightGBM ---
def objective_lgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbose': -1
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return f1_score(y_valid, preds, average='weighted')

if __name__ == '__main__':
    print('🔎 Optuna hyperparameter search for XGBoost...')
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=30)
    print('Best XGBoost params:', study_xgb.best_params)

    print('🔎 Optuna hyperparameter search for LightGBM...')
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=30)
    print('Best LightGBM params:', study_lgb.best_params)

    # Сохраняем лучшие параметры
    import json
    with open('models/production/best_xgboost_params.json', 'w') as f:
        json.dump(study_xgb.best_params, f, indent=2)
    with open('models/production/best_lightgbm_params.json', 'w') as f:
        json.dump(study_lgb.best_params, f, indent=2) 
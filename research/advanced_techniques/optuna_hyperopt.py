"""
Модуль для байесовской оптимизации гиперпараметров с помощью Optuna.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

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

def optimize_model(X, y, model_type='xgboost', n_trials=30, scoring='f1_weighted', random_state=42):
    """
    Байесовская оптимизация гиперпараметров для XGBoost или LightGBM.
    Возвращает лучшие параметры.
    """
    def objective(trial):
        if model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': random_state
            }
            model = xgb.XGBClassifier(**params, verbosity=0, n_jobs=-1)
        elif model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': random_state
            }
            model = lgb.LGBMClassifier(**params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        score = cross_val_score(model, X, y, cv=3, scoring=scoring).mean()
        return score
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# Пример использования:
# from optuna_hyperopt import optimize_model
# best_params = optimize_model(X, y, model_type='xgboost')

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
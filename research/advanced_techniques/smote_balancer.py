"""
Модуль для балансировки классов с помощью SMOTE, RandomOverSampler и RandomUnderSampler.
"""
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def balance_data(X, y, method='smote', random_state=42):
    """
    Балансировка классов в данных.
    method: 'smote', 'over', 'under'
    """
    if method == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif method == 'over':
        sampler = RandomOverSampler(random_state=random_state)
    elif method == 'under':
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res

# Пример использования:
# from smote_balancer import balance_data
# X_bal, y_bal = balance_data(X, y, method='smote') 
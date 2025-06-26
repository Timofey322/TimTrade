import pickle

model_path = "models/xgboost_BTC_USDT.pkl"
 
with open(model_path, "rb") as f:
    obj = pickle.load(f)
    print(f"Тип объекта в {model_path}: {type(obj)}")
    print(f"Содержимое: {obj if isinstance(obj, dict) else str(obj)[:500]}") 
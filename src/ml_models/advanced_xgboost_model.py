"""

Advanced Ensemble Model with XGBoost, Random Forest and LightGBM.



Includes:

- Multi-model ensemble (XGBoost + Random Forest + LightGBM)

- Timeframe-specific parameters

- Top-20 feature selection with importance threshold

- Rolling statistics integration

- Advanced regularization techniques

- Adaptive learning

- Meta-learning

- Advanced ensemble techniques

- Dynamic architecture changes

"""



import xgboost as xgb

import pandas as pd

import numpy as np

from typing import Dict, List, Tuple, Optional, Union

import pickle

from datetime import datetime

from pathlib import Path

from loguru import logger

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel

from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor, RandomForestClassifier

from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.svm import SVC, SVR

import time

import warnings

import json

from scipy import stats



try:

    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True

except ImportError:

    LIGHTGBM_AVAILABLE = False

    logger.warning("LightGBM не установлен. Будет использован только XGBoost и Random Forest.")



warnings.filterwarnings('ignore')





class AdvancedEnsembleModel:

    """

    Advanced Ensemble model with XGBoost, Random Forest and LightGBM.

    """

    

    def __init__(self, config: Dict = None):

        """

        Initialize advanced ensemble model.

        

        Args:

            config: Model configuration

        """

        self.config = config or {}

        self.logger = logger.bind(name="AdvancedEnsembleModel")

        

        # Параметры по таймфреймам согласно рекомендациям

        self.timeframe_params = {

            '5m': {

                'xgboost': {

                    'n_estimators': 1000,

                    'max_depth': 12,

                    'learning_rate': 0.02,

                    'subsample': 0.9,

                    'colsample_bytree': 0.9,

                    'reg_alpha': 0.01,

                    'reg_lambda': 0.3,

                    'min_child_weight': 1,

                    'random_state': 42,

                    'verbose': False,

                    'n_jobs': -1

                },

                'random_forest': {

                    'n_estimators': 500,

                    'max_depth': 15,

                    'min_samples_split': 5,

                    'min_samples_leaf': 2,

                    'random_state': 42,

                    'n_jobs': -1

                },

                'lightgbm': {

                    'n_estimators': 800,

                    'max_depth': 10,

                    'learning_rate': 0.03,

                    'subsample': 0.9,

                    'colsample_bytree': 0.9,

                    'reg_alpha': 0.01,

                    'reg_lambda': 0.3,

                    'random_state': 42,

                    'verbose': -1,

                    'n_jobs': -1

                },

                'classification_threshold': 0.001

            },

            '15m': {

                'xgboost': {

                    'n_estimators': 500,

                    'max_depth': 10,

                    'learning_rate': 0.03,

                    'subsample': 0.9,

                    'colsample_bytree': 0.9,

                    'reg_alpha': 0.01,

                    'reg_lambda': 0.3,

                    'min_child_weight': 1,

                    'random_state': 42,

                    'verbose': False,

                    'n_jobs': -1

                },

                'random_forest': {

                    'n_estimators': 300,

                    'max_depth': 12,

                    'min_samples_split': 5,

                    'min_samples_leaf': 2,

                    'random_state': 42,

                    'n_jobs': -1

                },

                'lightgbm': {

                    'n_estimators': 500,

                    'max_depth': 8,

                    'learning_rate': 0.04,

                    'subsample': 0.9,

                    'colsample_bytree': 0.9,

                    'reg_alpha': 0.01,

                    'reg_lambda': 0.3,

                    'random_state': 42,

                    'verbose': -1,

                    'n_jobs': -1

                },

                'classification_threshold': 0.001

            },

            '1h': {

                'xgboost': {

                    'n_estimators': 500,

                    'max_depth': 10,

                    'learning_rate': 0.03,

                    'subsample': 0.9,

                    'colsample_bytree': 0.9,

                    'reg_alpha': 0.01,

                    'reg_lambda': 0.3,

                    'min_child_weight': 1,

                    'random_state': 42,

                    'verbose': False,

                    'n_jobs': -1

                },

                'random_forest': {

                    'n_estimators': 300,

                    'max_depth': 12,

                    'min_samples_split': 5,

                    'min_samples_leaf': 2,

                    'random_state': 42,

                    'n_jobs': -1

                },

                'lightgbm': {

                    'n_estimators': 500,

                    'max_depth': 8,

                    'learning_rate': 0.04,

                    'subsample': 0.9,

                    'colsample_bytree': 0.9,

                    'reg_alpha': 0.01,

                    'reg_lambda': 0.3,

                    'random_state': 42,

                    'verbose': -1,

                    'n_jobs': -1

                },

                'classification_threshold': 0.002

            }

        }

        

        # Update parameters from config

        if 'timeframe_params' in self.config:

            self.timeframe_params.update(self.config['timeframe_params'])

        

        # Feature selection settings

        self.feature_selection_config = {

            'enabled': True,

            'n_features': 20,          # Только топ-20 признаков

            'importance_threshold': 0.01,  # Исключить с важностью < 0.01

            'method': 'mutual_info'

        }

        

        if 'feature_selection' in self.config:

            self.feature_selection_config.update(self.config['feature_selection'])

        

        # Ensemble settings

        self.ensemble_config = {

            'enabled': True,

            'voting_method': 'soft',

            'weights': {

                'xgboost': 0.5,

                'random_forest': 0.3,

                'lightgbm': 0.2

            }

        }

        

        if 'ensemble' in self.config:

            self.ensemble_config.update(self.config['ensemble'])

        

        # Cross-validation settings

        self.cv_config = {

            'enabled': True,

            'method': 'time_series_split',

            'n_splits': 10,

            'test_size': 0.1,

            'shuffle': False,

            'random_state': 42

        }

        

        if 'cv' in self.config:

            self.cv_config.update(self.config['cv'])

        

        # Initialize models

        self.models = {}

        self.selected_features = None

        self.training_results = None

        self.is_trained = False

        self.current_timeframe = None

        

        self.logger.info("AdvancedEnsembleModel initialized")

    

    def train(self, df: pd.DataFrame, symbol: str = None, timeframe: str = '5m') -> bool:

        """

        Train the ensemble model.

        

        Args:

            df: DataFrame with data

            symbol: Trading pair symbol

            timeframe: Trading timeframe

            

        Returns:

            bool: True if training successful

        """

        try:

            self.logger.info(f"Starting ensemble model training for {symbol} on {timeframe}")

            self.current_timeframe = timeframe

            

            # Get timeframe-specific parameters

            if timeframe not in self.timeframe_params:

                self.logger.warning(f"Timeframe {timeframe} not found, using 5m parameters")

                timeframe = '5m'

            

            params = self.timeframe_params[timeframe]

            

            # Prepare data

            X = df.drop(['target'], axis=1, errors='ignore')

            y = df['target']

            

            # Ensure target is integer

            y = y.astype(int)

            

            # Get number of classes

            n_classes = len(y.unique())

            self.logger.info(f"Number of classes: {n_classes}")

            self.logger.info(f"Class distribution: {y.value_counts().to_dict()}")

            

            # Feature selection

            if self.feature_selection_config['enabled']:

                X = self._select_features(X, y)

            

            # Split data

            split_idx = int(len(df) * 0.9)

            X_train, X_val = X[:split_idx], X[split_idx:]

            y_train, y_val = y[:split_idx], y[split_idx:]

            

            self.logger.info(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

            

            # Train individual models

            self.models = {}

            

            # 1. XGBoost

            self.logger.info("Training XGBoost model...")

            xgb_params = params['xgboost'].copy()

            xgb_params['objective'] = 'multi:softprob' if n_classes > 2 else 'binary:logistic'

            if n_classes > 2:

                xgb_params['num_class'] = n_classes

            

            xgb_model = xgb.XGBClassifier(**xgb_params)

            # Убираем проблемные параметры early stopping
            safe_params = {k: v for k, v in xgb_params.items() 
                         if k not in ['early_stopping_rounds', 'eval_metric']}
            xgb_model = xgb.XGBClassifier(**safe_params)
            xgb_model.fit(X_train, y_train)

            self.models['xgboost'] = xgb_model

            

            # 2. Random Forest

            self.logger.info("Training Random Forest model...")

            rf_params = params['random_forest'].copy()

            rf_model = RandomForestClassifier(**rf_params)

            rf_model.fit(X_train, y_train)

            self.models['random_forest'] = rf_model

            

            # 3. LightGBM (if available)

            if LIGHTGBM_AVAILABLE:

                self.logger.info("Training LightGBM model...")

                lgb_params = params['lightgbm'].copy()

                lgb_params['objective'] = 'multiclass' if n_classes > 2 else 'binary'

                if n_classes > 2:

                    lgb_params['num_class'] = n_classes

                

                lgb_model = lgb.LGBMClassifier(**lgb_params)

                lgb_model.fit(X_train, y_train)

                self.models['lightgbm'] = lgb_model

            

            # Create ensemble

            if self.ensemble_config['enabled']:

                self.logger.info("Creating ensemble...")

                estimators = []

                weights = []

                

                for model_name, model in self.models.items():

                    estimators.append((model_name, model))

                    weights.append(self.ensemble_config['weights'].get(model_name, 1.0))

                

                self.ensemble = VotingClassifier(

                    estimators=estimators,

                    voting=self.ensemble_config['voting_method'],

                    weights=weights

                )

                

                # Fit ensemble

                self.ensemble.fit(X_train, y_train)

            

            # Evaluate models

            self._evaluate_models(X_val, y_val, n_classes)

            

            self.is_trained = True

            self.logger.info(f"Ensemble model trained successfully for {timeframe}")

            

            return True

            

        except Exception as e:

            self.logger.error(f"Error training ensemble model: {e}")

            return False

    

    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:

        """

        Select top features based on importance.

        

        Args:

            X: Feature matrix

            y: Target variable

            

        Returns:

            DataFrame with selected features

        """

        try:

            self.logger.info(f"Selecting top {self.feature_selection_config['n_features']} features")

            

            # Remove NaN columns

            X = X.dropna(axis=1)

            

            # Use mutual information for feature selection

            importance_scores = mutual_info_classif(X, y, random_state=42)

            feature_importance = dict(zip(X.columns, importance_scores))

            

            # Sort by importance

            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

            

            # Filter by importance threshold

            threshold = self.feature_selection_config['importance_threshold']

            filtered_features = [(name, score) for name, score in sorted_features if score >= threshold]

            

            # Take top N features

            n_features = self.feature_selection_config['n_features']

            top_features = filtered_features[:n_features]

            

            # Log results

            self.logger.info(f"Found {len(filtered_features)} features with importance >= {threshold}")

            self.logger.info(f"Selected top {len(top_features)} features:")

            

            for i, (feature, importance) in enumerate(top_features, 1):

                self.logger.info(f"{i:2d}. {feature}: {importance:.6f}")

            

            # Create new DataFrame with selected features

            selected_columns = [feature for feature, _ in top_features]

            self.selected_features = selected_columns

            

            return X[selected_columns]

            

        except Exception as e:

            self.logger.error(f"Error in feature selection: {e}")

            return X

    

    def _evaluate_models(self, X_val: pd.DataFrame, y_val: pd.Series, n_classes: int):

        """

        Evaluate all models and ensemble.

        

        Args:

            X_val: Validation features

            y_val: Validation targets

            n_classes: Number of classes

        """

        try:

            results = {}

            

            # Evaluate individual models

            for model_name, model in self.models.items():

                y_pred = model.predict(X_val)

                y_proba = model.predict_proba(X_val)

                

                results[model_name] = {

                    'accuracy': accuracy_score(y_val, y_pred),

                    'f1': f1_score(y_val, y_pred, average='weighted'),

                    'precision': precision_score(y_val, y_pred, average='weighted'),

                    'recall': recall_score(y_val, y_pred, average='weighted')

                }

                

                self.logger.info(f"{model_name}: Accuracy={results[model_name]['accuracy']:.4f}, "

                               f"F1={results[model_name]['f1']:.4f}")

            

            # Evaluate ensemble

            if hasattr(self, 'ensemble'):

                y_pred_ensemble = self.ensemble.predict(X_val)

                y_proba_ensemble = self.ensemble.predict_proba(X_val)

                

                results['ensemble'] = {

                    'accuracy': accuracy_score(y_val, y_pred_ensemble),

                    'f1': f1_score(y_val, y_pred_ensemble, average='weighted'),

                    'precision': precision_score(y_val, y_pred_ensemble, average='weighted'),

                    'recall': recall_score(y_val, y_pred_ensemble, average='weighted')

                }

                

                self.logger.info(f"Ensemble: Accuracy={results['ensemble']['accuracy']:.4f}, "

                               f"F1={results['ensemble']['f1']:.4f}")

            

            self.training_results = results

            

        except Exception as e:

            self.logger.error(f"Error evaluating models: {e}")

    

    def predict(self, X: pd.DataFrame) -> np.ndarray:

        """

        Predict classes using ensemble.

        

        Args:

            X: Feature matrix

            

        Returns:

            Predicted classes

        """

        if not self.is_trained:

            raise ValueError("Model not trained")

        

        # Apply feature selection if needed

        if self.selected_features is not None:

            X = X[self.selected_features]

        

        if hasattr(self, 'ensemble'):

            return self.ensemble.predict(X)

        else:

            # Fallback to XGBoost

            return self.models['xgboost'].predict(X)

    

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:

        """

        Predict class probabilities using ensemble.

        

        Args:

            X: Feature matrix

            

        Returns:

            Predicted probabilities

        """

        if not self.is_trained:

            raise ValueError("Model not trained")

        

        # Apply feature selection if needed

        if self.selected_features is not None:

            X = X[self.selected_features]

        

        if hasattr(self, 'ensemble'):

            return self.ensemble.predict_proba(X)

        else:

            # Fallback to XGBoost

            return self.models['xgboost'].predict_proba(X)

    

    def get_classification_threshold(self) -> float:

        """

        Get classification threshold for current timeframe.

        

        Returns:

            Classification threshold

        """

        if self.current_timeframe and self.current_timeframe in self.timeframe_params:

            return self.timeframe_params[self.current_timeframe]['classification_threshold']

        return 0.001  # Default threshold

    

    def save_model(self, filename: str = None) -> str:

        """

        Save the ensemble model.

        

        Args:

            filename: Model filename

            

        Returns:

            Saved file path

        """

        try:

            if filename is None:

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                filename = f"advanced_ensemble_model_{timestamp}.pkl"

            

            model_data = {

                'models': self.models,

                'ensemble': getattr(self, 'ensemble', None),

                'selected_features': self.selected_features,

                'training_results': self.training_results,

                'timeframe_params': self.timeframe_params,

                'current_timeframe': self.current_timeframe,

                'config': self.config

            }

            

            with open(filename, 'wb') as f:

                pickle.dump(model_data, f)

            

            self.logger.info(f"Model saved to {filename}")

            return filename

            

        except Exception as e:

            self.logger.error(f"Error saving model: {e}")

            return ""

    

    def load_model(self, filepath: str) -> bool:

        """

        Load the ensemble model.

        

        Args:

            filepath: Model file path

            

        Returns:

            bool: True if loading successful

        """

        try:

            with open(filepath, 'rb') as f:

                model_data = pickle.load(f)

            

            self.models = model_data['models']

            self.ensemble = model_data.get('ensemble')

            self.selected_features = model_data['selected_features']

            self.training_results = model_data['training_results']

            self.timeframe_params = model_data['timeframe_params']

            self.current_timeframe = model_data['current_timeframe']

            self.config = model_data['config']

            self.is_trained = True

            

            self.logger.info(f"Model loaded from {filepath}")

            return True

            

        except Exception as e:

            self.logger.error(f"Error loading model: {e}")

            return False

    

    def get_model_summary(self) -> Dict:

        """

        Get model summary.

        

        Returns:

            Model summary dictionary

        """

        return {

            'model_type': 'AdvancedEnsembleModel',

            'is_trained': self.is_trained,

            'current_timeframe': self.current_timeframe,

            'models': list(self.models.keys()),

            'ensemble_enabled': hasattr(self, 'ensemble'),

            'selected_features_count': len(self.selected_features) if self.selected_features else 0,

            'training_results': self.training_results,

            'classification_threshold': self.get_classification_threshold()

        } 
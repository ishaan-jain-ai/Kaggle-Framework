import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                             GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from core.base_trainer import BaseTrainer

class SklearnTrainer(BaseTrainer):
    """
    Versatile trainer for scikit-learn models supporting multiple algorithms
    for both classification and regression tasks.
    """
    
    # Available algorithms
    CLASSIFICATION_MODELS = {
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'logistic_regression': LogisticRegression,
        'svm': SVC,
        'knn': KNeighborsClassifier,
        'decision_tree': DecisionTreeClassifier
    }
    
    REGRESSION_MODELS = {
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'linear_regression': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'svm': SVR,
        'knn': KNeighborsRegressor,
        'decision_tree': DecisionTreeRegressor
    }
    
    def __init__(self, algorithm: str, task_type: str = 'classification', 
                 model_params: Dict[str, Any] = None, random_state: int = 42):
        """
        Initialize the sklearn trainer
        
        Args:
            algorithm: Name of the algorithm to use
            task_type: 'classification' or 'regression'
            model_params: Parameters to pass to the model
            random_state: Random state for reproducibility
        """
        super().__init__(f"{task_type}_{algorithm}")
        
        self.algorithm = algorithm
        self.task_type = task_type
        self.model_params = model_params or {}
        self.random_state = random_state
        
        # Set random state if model supports it
        if 'random_state' not in self.model_params:
            self.model_params['random_state'] = random_state
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on algorithm and task type"""
        if self.task_type == 'classification':
            if self.algorithm not in self.CLASSIFICATION_MODELS:
                raise ValueError(f"Algorithm '{self.algorithm}' not supported for classification. "
                               f"Available: {list(self.CLASSIFICATION_MODELS.keys())}")
            model_class = self.CLASSIFICATION_MODELS[self.algorithm]
        elif self.task_type == 'regression':
            if self.algorithm not in self.REGRESSION_MODELS:
                raise ValueError(f"Algorithm '{self.algorithm}' not supported for regression. "
                               f"Available: {list(self.REGRESSION_MODELS.keys())}")
            model_class = self.REGRESSION_MODELS[self.algorithm]
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")
        
        # Remove random_state if model doesn't support it
        model_params = self.model_params.copy()
        try:
            self.model = model_class(**model_params)
        except TypeError:
            # Remove random_state and try again
            model_params.pop('random_state', None)
            self.model = model_class(**model_params)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              validation_split: float = None, cv_folds: int = None, **kwargs) -> 'SklearnTrainer':
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            validation_split: Fraction of data to use for validation
            cv_folds: Number of cross-validation folds
            **kwargs: Additional parameters
        """
        self.feature_names = list(X_train.columns)
        self.target_name = y_train.name or 'target'
        
        # Fit the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Log training
        self.training_history.append(f"Model trained with {len(X_train)} samples")
        self.training_history.append(f"Features: {len(self.feature_names)}")
        
        # Perform cross-validation if requested
        if cv_folds and cv_folds > 1:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            self.training_history.append(f"CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        # Training score
        train_score = self.model.score(X_train, y_train)
        self.training_history.append(f"Training Score: {train_score:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure features match training data
        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Algorithm '{self.algorithm}' does not support probability predictions")
        
        # Ensure features match training data
        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]
        
        return self.model.predict_proba(X)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        info = {
            'algorithm': self.algorithm,
            'task_type': self.task_type,
            'model_params': self.model_params,
            'is_trained': self.is_trained
        }
        
        if self.is_trained:
            info['feature_count'] = len(self.feature_names)
            info['feature_names'] = self.feature_names
            info['target_name'] = self.target_name
        
        return info
    
    @classmethod
    def get_available_algorithms(cls, task_type: str = None) -> Dict[str, list]:
        """Get list of available algorithms"""
        if task_type == 'classification':
            return {'classification': list(cls.CLASSIFICATION_MODELS.keys())}
        elif task_type == 'regression':
            return {'regression': list(cls.REGRESSION_MODELS.keys())}
        else:
            return {
                'classification': list(cls.CLASSIFICATION_MODELS.keys()),
                'regression': list(cls.REGRESSION_MODELS.keys())
            }
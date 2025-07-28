from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import os

class BaseTrainer(ABC):
    """
    Abstract base class for all model trainers in the Kaggle framework.
    Provides common interface and functionality for training, evaluation, and prediction.
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.__class__.__name__
        self.model = None
        self.is_trained = False
        self.training_history = []
        self.feature_names = None
        self.target_name = None
        
    @abstractmethod
    def train(self, X_train, y_train, **kwargs) -> 'BaseTrainer':
        """Train the model with given data"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions on new data"""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Get prediction probabilities (for classification models)"""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'training_history': self.training_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            import joblib
            joblib.dump(model_data, filepath)
        except ImportError:
            raise ImportError("joblib is required for model saving. Install with: pip install joblib")
        self.training_history.append(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'BaseTrainer':
        """Load a trained model from disk"""
        try:
            import joblib
            model_data = joblib.load(filepath)
        except ImportError:
            raise ImportError("joblib is required for model loading. Install with: pip install joblib")
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.training_history = model_data['training_history']
        self.is_trained = True
        
        self.training_history.append(f"Model loaded from {filepath}")
        return self
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            raise ImportError("pandas and numpy are required for feature importance. Install with: pip install pandas numpy")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        elif hasattr(self.model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(self.model.coef_.flatten())
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None
    
    def get_training_history(self) -> List[str]:
        """Get the training history log"""
        return self.training_history
    
    def reset(self) -> None:
        """Reset the trainer to initial state"""
        self.model = None
        self.is_trained = False
        self.training_history = []
        self.feature_names = None
        self.target_name = None
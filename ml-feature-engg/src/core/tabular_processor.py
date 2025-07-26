import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from core.base_processor import BaseProcessor

class TabularProcessor(BaseProcessor):
    def __init__(self, data):
        super().__init__(data)
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.data.select_dtypes(include=['object']).columns
        
    def clean(self):
        """Clean tabular data"""
        # Handle missing values
        self._handle_missing_values()
        # Remove duplicates
        self._remove_duplicates()
        # Handle outliers
        self._handle_outliers()
        return self
    
    def engineer_features(self):
        """Engineer features for tabular data"""
        # Numeric features
        self._scale_numeric_features()
        # Categorical features
        self._encode_categorical_features()
        # Feature interactions
        self._create_feature_interactions()
        return self
    
    def _handle_missing_values(self):
        # Numeric: fill with median
        for col in self.numeric_columns:
            self.data[col].fillna(self.data[col].median(), inplace=True)
        
        # Categorical: fill with mode
        for col in self.categorical_columns:
            self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        
        self.processing_steps.append("handled_missing_values")
    
    def _handle_outliers(self, threshold=3):
        for col in self.numeric_columns:
            z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
            self.data.loc[z_scores > threshold, col] = self.data[col].median()
        
        self.processing_steps.append("handled_outliers")
import numpy as np
from typing import Union, List, Dict
import pandas as pd


class PreprocessingUtils:
    @staticmethod
    def detect_outliers(data: pd.Series, threshold: float = 3) -> np.ndarray:
        """
        Detect outliers using z-score method
        
        Args:
            data: Pandas series containing numerical data
            threshold: Z-score threshold (default=3)
            
        Returns:
            Boolean mask indicating outlier positions
        """
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold

    @staticmethod
    def calculate_missing_stats(df: pd.DataFrame) -> Dict:
        """
        Calculate missing value statistics for each column
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with column-wise missing value statistics
        """
        missing_stats = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            missing_stats[col] = {
                'missing_count': missing_count,
                'missing_percent': missing_percent
            }
        return missing_stats

    @staticmethod
    def get_numeric_features(df: pd.DataFrame) -> List[str]:
        """Get list of numeric columns"""
        return df.select_dtypes(include=[np.number]).columns.tolist()

    @staticmethod
    def get_categorical_features(df: pd.DataFrame) -> List[str]:
        """Get list of categorical columns"""
        return df.select_dtypes(include=['object']).columns.tolist()

    @staticmethod
    def memory_usage(df: pd.DataFrame) -> Dict:
        """
        Calculate memory usage for each column
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with column-wise memory usage in MB
        """
        memory_stats = {}
        for col in df.columns:
            memory_stats[col] = df[col].memory_usage(deep=True) / 1024 / 1024  # Convert to MB
        memory_stats['total'] = df.memory_usage(deep=True).sum() / 1024 / 1024
        return memory_stats
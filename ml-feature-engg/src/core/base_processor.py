from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseProcessor(ABC):
    def __init__(self, data):
        self.data = data
        self.original_data = data.copy()
        self.processing_steps = []
    
    @abstractmethod
    def clean(self):
        pass
    
    @abstractmethod
    def engineer_features(self):
        pass
    
    def reset(self):
        """Reset data to original state"""
        self.data = self.original_data.copy()
        self.processing_steps = []
    
    def get_processing_history(self):
        """Return list of applied processing steps"""
        return self.processing_steps
import pandas as pd
import os
import psutil
import numpy as np
from core.base_reader import BaseReader

class TabularReader(BaseReader):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.text_columns = None

    def read_data(self):
        if self.file_path.endswith('.csv'):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.endswith(('.xls', '.xlsx')):
            self.data = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
        self._identify_text_columns()

    def _identify_text_columns(self):
        """Identify columns containing text data"""
        if self.data is not None:
            self.text_columns = self.data.select_dtypes(include=['object']).columns.tolist()

    def get_details(self):
        if self.data is not None:
            # Get basic details
            num_columns = self.data.shape[1]
            size = self.data.size
            memory_usage = self.data.memory_usage(deep=True).sum()
            
            # Get text column details
            text_details = self.get_text_columns_details() if self.text_columns else {}
            
            return {
                "num_columns": num_columns,
                "size": size,
                "memory_usage": memory_usage,
                "text_columns": text_details
            }
        else:
            raise ValueError("Data not read. Please call read_data() first.")

    def get_text_columns_details(self):
        """Get detailed information about text columns"""
        if not self.text_columns:
            return {}
        
        text_details = {}
        for col in self.text_columns:
            text_details[col] = {
                "avg_length": self.data[col].str.len().mean(),
                "max_length": self.data[col].str.len().max(),
                "unique_values": self.data[col].nunique(),
                "null_count": self.data[col].isnull().sum(),
                "sample_values": self.data[col].sample(min(5, len(self.data))).tolist()
            }
        return text_details
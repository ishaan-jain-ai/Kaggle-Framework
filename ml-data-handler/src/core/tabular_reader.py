class BaseReader:
    def read_data(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_details(self):
        raise NotImplementedError("Subclasses should implement this method.")


import pandas as pd
import os
import psutil

class TabularReader(BaseReader):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def read_data(self):
        if self.file_path.endswith('.csv'):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.endswith(('.xls', '.xlsx')):
            self.data = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    def get_details(self):
        if self.data is not None:
            num_columns = self.data.shape[1]
            size = self.data.size
            memory_usage = self.data.memory_usage(deep=True).sum()
            return {
                "num_columns": num_columns,
                "size": size,
                "memory_usage": memory_usage
            }
        else:
            raise ValueError("Data not read. Please call read_data() first.")
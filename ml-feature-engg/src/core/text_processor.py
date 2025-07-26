import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from core.base_processor import BaseProcessor

class TextProcessor(BaseProcessor):
    def __init__(self, data, text_columns):
        super().__init__(data)
        self.text_columns = text_columns
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
    def clean(self):
        """Clean text data"""
        for col in self.text_columns:
            # Lower case
            self.data[col] = self.data[col].str.lower()
            # Remove special characters
            self.data[col] = self.data[col].str.replace(r'[^a-zA-Z\s]', '', regex=True)
            # Remove extra whitespace
            self.data[col] = self.data[col].str.strip()
        
        self.processing_steps.append("cleaned_text")
        return self
    
    def engineer_features(self):
        """Engineer features for text data"""
        for col in self.text_columns:
            # Word count
            self.data[f'{col}_word_count'] = self.data[col].str.split().str.len()
            
            # Character count
            self.data[f'{col}_char_count'] = self.data[col].str.len()
            
            # Average word length
            self.data[f'{col}_avg_word_length'] = (
                self.data[col].str.split()
                .apply(lambda x: np.mean([len(word) for word in x]) if x else 0)
            )
            
            # TF-IDF features can be added here
        
        self.processing_steps.append("engineered_text_features")
        return self
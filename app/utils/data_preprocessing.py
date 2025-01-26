import re
import numpy as np
from typing import List, Dict, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from app.core.logging import logger
from app.core.config import settings
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.encoder = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> str:
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            
            # Join tokens back into a string
            processed_text = ' '.join(tokens)
            
            return processed_text
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            raise

    def normalize_numerical_data(self, data: Union[List[Dict[str, float]], np.ndarray, pd.DataFrame], method: str = 'standard') -> np.ndarray:
        try:
            if method not in ['standard', 'minmax']:
                raise ValueError("Method must be either 'standard' or 'minmax'")

            # Convert to numpy array if it's a list of dicts or pandas DataFrame
            if isinstance(data, list):
                numerical_data = np.array([[v for v in d.values()] for d in data])
            elif isinstance(data, pd.DataFrame):
                numerical_data = data.select_dtypes(include=[np.number]).values
            else:
                numerical_data = data

            if method == 'standard':
                self.scaler = StandardScaler()
            else:  # minmax
                self.scaler = MinMaxScaler()

            normalized_data = self.scaler.fit_transform(numerical_data)

            return normalized_data
        except Exception as e:
            logger.error(f"Error in numerical data normalization: {str(e)}")
            raise

    def handle_missing_values(self, data: Union[List[Dict[str, Any]], pd.DataFrame], strategy: str = 'mean') -> Union[List[Dict[str, Any]], pd.DataFrame]:
        try:
            if strategy not in ['mean', 'median', 'most_frequent', 'constant']:
                raise ValueError("Strategy must be 'mean', 'median', 'most_frequent', or 'constant'")

            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()

            # Separate numerical and categorical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns

            # Handle missing values in numerical columns
            self.imputer = SimpleImputer(strategy=strategy)
            df[numerical_cols] = self.imputer.fit_transform(df[numerical_cols])

            # Handle missing values in categorical columns
            if categorical_cols.any():
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

            if isinstance(data, list):
                return df.to_dict('records')
            else:
                return df
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise

    def encode_categorical_variables(self, data: Union[List[Dict[str, Any]], pd.DataFrame], columns: List[str]) -> np.ndarray:
        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()

            self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded_data = self.encoder.fit_transform(df[columns])
            
            feature_names = self.encoder.get_feature_names(columns)
            encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
            
            return pd.concat([df.drop(columns, axis=1), encoded_df], axis=1)
        except Exception as e:
            logger.error(f"Error encoding categorical variables: {str(e)}")
            raise

    def scale_features(self, data: Union[List[Dict[str, float]], np.ndarray, pd.DataFrame]) -> np.ndarray:
        try:
            if self.scaler is None:
                raise ValueError("Scaler not initialized. Call normalize_numerical_data first.")
            
            if isinstance(data, list):
                numerical_data = np.array([[v for v in d.values()] for d in data])
            elif isinstance(data, pd.DataFrame):
                numerical_data = data.select_dtypes(include=[np.number]).values
            else:
                numerical_data = data

            return self.scaler.transform(numerical_data)
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise

    def preprocess_pipeline(self, data: Union[List[Dict[str, Any]], pd.DataFrame], text_column: str = None, categorical_columns: List[str] = None) -> pd.DataFrame:
        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()

            # Handle missing values
            df = self.handle_missing_values(df, strategy=settings.MISSING_VALUE_STRATEGY)

            # Preprocess text data if text column is specified
            if text_column and text_column in df.columns:
                df[text_column] = df[text_column].apply(self.preprocess_text)

            # Encode categorical variables
            if categorical_columns:
                df = self.encode_categorical_variables(df, categorical_columns)

            # Normalize numerical data
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = self.normalize_numerical_data(df[numerical_cols], method=settings.NORMALIZATION_METHOD)

            return df
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

# Initialize the DataPreprocessor
data_preprocessor = DataPreprocessor()

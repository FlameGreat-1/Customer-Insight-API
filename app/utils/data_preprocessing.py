import re
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from app.core.logging import logger

def preprocess_text(text: str) -> str:
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception as e:
        logger.error(f"Error in text preprocessing: {str(e)}")
        raise

def normalize_numerical_data(data: List[Dict[str, float]], method: str = 'standard') -> np.ndarray:
    try:
        if method not in ['standard', 'minmax']:
            raise ValueError("Method must be either 'standard' or 'minmax'")

        # Extract numerical values
        numerical_data = np.array([[v for v in d.values()] for d in data])

        if method == 'standard':
            scaler = StandardScaler()
        else:  # minmax
            scaler = MinMaxScaler()

        normalized_data = scaler.fit_transform(numerical_data)

        return normalized_data
    except Exception as e:
        logger.error(f"Error in numerical data normalization: {str(e)}")
        raise

def handle_missing_values(data: List[Dict[str, any]], strategy: str = 'mean') -> List[Dict[str, any]]:
    try:
        if strategy not in ['mean', 'median', 'mode', 'drop']:
            raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")

        # Separate numerical and categorical columns
        numerical_cols = [col for col in data[0].keys() if isinstance(data[0][col], (int, float))]
        categorical_cols = [col for col in data[0].keys() if col not in numerical_cols]

        if strategy == 'drop':
            return [d for d in data if all(d[col] is not None for col in d)]

        for col in numerical_cols:
            values = [d[col] for d in data if d[col] is not None]
            if strategy == 'mean':
                fill_value = np.mean(values)
            elif strategy == 'median':
                fill_value = np.median(values)
            else:  # mode
                fill_value = max(set(values), key=values.count)

            for d in data:
                if d[col] is None:
                    d[col] = fill_value

        for col in categorical_cols:
            values = [d[col] for d in data if d[col] is not None]
            fill_value = max(set(values), key=values.count)  # mode for categorical

            for d in data:
                if d[col] is None:
                    d[col] = fill_value

        return data
    except Exception as e:
        logger.error(f"Error handling missing values: {str(e)}")
        raise

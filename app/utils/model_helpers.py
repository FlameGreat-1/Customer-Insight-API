# utils/model_helpers.py

import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from typing import Dict, Any, List, Tuple
from app.core.logging import logger

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, task: str = 'classification') -> Dict[str, float]:
    """
    Evaluate model performance based on true and predicted values.
    
    Args:
    y_true: True labels or values
    y_pred: Predicted labels or values
    task: 'classification' or 'regression'
    
    Returns:
    Dictionary of evaluation metrics
    """
    try:
        if task == 'classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
        elif task == 'regression':
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2_score': r2_score(y_true, y_pred)
            }
        else:
            raise ValueError("Task must be either 'classification' or 'regression'")
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise

def cross_validate(model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = 'accuracy') -> Tuple[float, float]:
    """
    Perform cross-validation on the model.
    
    Args:
    model: The machine learning model
    X: Feature matrix
    y: Target vector
    cv: Number of cross-validation folds
    scoring: Scoring metric to use
    
    Returns:
    Tuple of mean score and standard deviation
    """
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return np.mean(scores), np.std(scores)
    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        raise

def feature_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
    """
    Get feature importance from the model.
    
    Args:
    model: Trained model with feature_importances_ attribute
    feature_names: List of feature names
    
    Returns:
    Dictionary of feature importances
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(feature_names, importances))
        else:
            raise AttributeError("Model does not have feature_importances_ attribute")
    except Exception as e:
        logger.error(f"Error in getting feature importance: {str(e)}")
        raise

def learning_curve(model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5, train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5)) -> Dict[str, np.ndarray]:
    """
    Generate learning curve data for the model.
    
    Args:
    model: The machine learning model
    X: Feature matrix
    y: Target vector
    cv: Number of cross-validation folds
    train_sizes: Array of training set sizes to evaluate
    
    Returns:
    Dictionary containing train sizes, train scores, and test scores
    """
    try:
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes)
        
        return {
            'train_sizes': train_sizes,
            'train_scores': np.mean(train_scores, axis=1),
            'test_scores': np.mean(test_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'test_scores_std': np.std(test_scores, axis=1)
        }
    except Exception as e:
        logger.error(f"Error in generating learning curve: {str(e)}")
        raise

def model_explainer(model: Any, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """
    Generate model explanations using SHAP values.
    
    Args:
    model: Trained model
    X: Feature matrix
    feature_names: List of feature names
    
    Returns:
    Dictionary containing SHAP values and summary plot
    """
    try:
        import shap
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        
        return {
            'shap_values': shap_values,
            'summary_plot': shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        }
    except Exception as e:
        logger.error(f"Error in generating model explanations: {str(e)}")
        raise

def hyperparameter_tuning(model: Any, param_grid: Dict[str, List[Any]], X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
    model: The machine learning model
    param_grid: Dictionary of hyperparameters to tune
    X: Feature matrix
    y: Target vector
    cv: Number of cross-validation folds
    scoring: Scoring metric to use
    
    Returns:
    Dictionary containing best parameters and best score
    """
    try:
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {str(e)}")
        raise

def model_calibration(model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Any:
    """
    Calibrate model probabilities.
    
    Args:
    model: The machine learning model
    X: Feature matrix
    y: Target vector
    cv: Number of cross-validation folds
    
    Returns:
    Calibrated model
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
        calibrated_model = CalibratedClassifierCV(model, cv=cv, method='sigmoid')
        calibrated_model.fit(X, y)
        return calibrated_model
    except Exception as e:
        logger.error(f"Error in model calibration: {str(e)}")
        raise

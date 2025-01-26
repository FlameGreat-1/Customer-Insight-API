import numpy as np
from sklearn.model_selection import cross_val_score, KFold, learning_curve, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, Any, List, Tuple, Union
from app.core.logging import logger
from app.core.config import settings
import shap
import matplotlib.pyplot as plt
import joblib
import os

class ModelHelper:
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, task: str = 'classification') -> Dict[str, float]:
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

    @staticmethod
    def cross_validate(model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = 'accuracy') -> Tuple[float, float]:
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=settings.N_JOBS)
            return np.mean(scores), np.std(scores)
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise

    @staticmethod
    def feature_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
                return dict(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))
            else:
                raise AttributeError("Model does not have feature_importances_ or coef_ attribute")
        except Exception as e:
            logger.error(f"Error in getting feature importance: {str(e)}")
            raise

    @staticmethod
    def learning_curve(model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5, train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5)) -> Dict[str, np.ndarray]:
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=cv, n_jobs=settings.N_JOBS, train_sizes=train_sizes)
            
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

    @staticmethod
    def model_explainer(model: Any, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            
            plt.figure()
            summary_plot = shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(settings.MODEL_OUTPUT_DIR, 'shap_summary_plot.png'))
            plt.close()
            
            return {
                'shap_values': shap_values,
                'summary_plot_path': os.path.join(settings.MODEL_OUTPUT_DIR, 'shap_summary_plot.png')
            }
        except Exception as e:
            logger.error(f"Error in generating model explanations: {str(e)}")
            raise

    @staticmethod
    def hyperparameter_tuning(model: Any, param_grid: Dict[str, List[Any]], X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        try:
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=settings.N_JOBS)
            grid_search.fit(X, y)
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            raise

    @staticmethod
    def model_calibration(model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Any:
        try:
            calibrated_model = CalibratedClassifierCV(model, cv=cv, method='sigmoid', n_jobs=settings.N_JOBS)
            calibrated_model.fit(X, y)
            return calibrated_model
        except Exception as e:
            logger.error(f"Error in model calibration: {str(e)}")
            raise

    @staticmethod
    def save_model(model: Any, model_name: str) -> str:
        try:
            model_path = os.path.join(settings.MODEL_OUTPUT_DIR, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    @staticmethod
    def load_model(model_name: str) -> Any:
        try:
            model_path = os.path.join(settings.MODEL_OUTPUT_DIR, f"{model_name}.joblib")
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

model_helper = ModelHelper()

import os
import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features
                train_array[:, -1],   # Target variable
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "KNeighbors Regressor": KNeighborsRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "AdaBoost Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [0.1, 0.01, 0.5, 0.001]
                }, 
                "KNeighbors Regressor": {
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
            }

            model_report:dict = evaluate_models(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, models=models, param=params)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys()) [
                list(model_report.values()).index(best_model_score)]
            model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            predicted = model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2
        
        except Exception as e:
            raise CustomException(e, sys)

        
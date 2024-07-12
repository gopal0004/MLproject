import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass

# from catboost._catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor , AdaBoostRegressor , GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object , evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts" , "models.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self , train_arr ,test_arr):
        try:
            logging.info("splitting train and test dataset")
            
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-neighbor Regressor" : KNeighborsRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor()
            }

            model_report:dict = evaluate_model(x_train ,y_train ,x_test ,y_test ,models)

            best_r2 = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_r2)
            ]
            best_model = models[best_model_name]
            if best_r2 < 0.60:
                raise CustomException("no best model found")
            logging.info("Best Model found on both train and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_r2
        except Exception as e:
            raise CustomException(e , sys)
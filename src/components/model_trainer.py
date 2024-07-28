#Training and Evaluation of model
import sys
import os 
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    
)
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

#Always crearte a config file
@dataclass
class ModelTrainerConfig:
    train_model_config_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting Training and Test input data")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1], #Take out the last columns and feed everything to X_train 
                train_arr[:,-1], #Last Data as Y train value 
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbours Regression":KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoosting Classifer" : AdaBoostRegressor(),
            }
            #Lets Check which model is performing well 
            model_report :dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            #Get Best Model Score 
            best_model_socre = max(sorted(model_report.values()))

            #Get Best model Name
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_socre)
                ]
            
            best_model = models[best_model_name]

            if best_model_socre < 0.6 :
                raise CustomException("No best model Found")
            logging.info(f"Best Model Found on both training and testing dataset")

            logging.info(f"Dumping the best model.")

            #Saving the preprocessing object to a file path
            save_object(
                file_path=self.model_trainer_config.train_model_config_file_path,
                obj = best_model
            )   

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
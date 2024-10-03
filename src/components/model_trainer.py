import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sys,os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.Utils import save_obj,evaluate_models
from src.exception import CustomException
from src.logger import logging



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("spliting training and test input data")
            X_train,y_train,X_test,y_test=(train_array[:,:-1]
                                           ,train_array[:,-1],
                                           test_array[:,:-1],
                                           test_array[:,-1]
                                           )
            

            models = {
                "Logistic Regression": LogisticRegression(multi_class='ovr'),
                "K-Neighbors Classifier": KNeighborsClassifier(),
            }
            
            model_report:dict=evaluate_models(X_train=X_train,
                                             y_train=y_train,y_test=y_test,
                                             X_test=X_test,
                                             models=models)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No Best Model")
            logging.info("best model found on both training and test data.")
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            accuracy=accuracy_score(y_test,predicted)
            return accuracy
        except Exception as e:
            raise CustomException(e,sys)
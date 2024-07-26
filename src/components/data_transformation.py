 ## Transforming the Data 

import sys 
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer #Used to Create the pipeline
from sklearn.impute import SimpleImputer #Handling missing Values
from sklearn.pipeline import Pipeline #It is used to create pipelines 
from sklearn.preprocessing import OneHotEncoder,StandardScaler #Data Scaling


#Importing the Custom Exception and logging file to handle exception and logging 
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


#To ingest the data we will write a data ingestion class  which gives path for the data components 
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    

    def get_data_transformer_object(self):
        '''
        This Function is responsible for data transformation
        '''

        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender','race_ethnicity',
                                    'parental_level_of_education',
                                    'lunch',
                                    'test_preparation_course']
            
            #Creating a pipeline 
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())]
                    )
            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler",StandardScaler())]
                    )
            logging.info(f"Numerical Columns:{numerical_columns}")
            logging.info(f"Categorical colums :{categorical_columns}")

            #Now we combine both numerical and Categorical pipeline together and for that we use Column Transformer
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline",numerical_pipeline,numerical_columns),#The pipeline takes for pipeline and which column we are implementing for 
                    ("categorical_pipeline",categorical_pipeline,categorical_columns)
                ]
                )
            return preprocessor
        
        except Exception as e :
            raise CustomException(e,sys)
     
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read  train and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on train and test data")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_train_df)]

            logging.info(f"Saving the preprocessing object.")

            #Saving the preprocessing object to a file path
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path)


            
        except Exception as e:
            raise CustomException(e, sys)   



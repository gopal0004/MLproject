import os
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split



@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts' , 'train.csv')
    test_data_path = os.path.join('artifacts' , 'test.csv')
    data_path = os.path.join('artifacts' , 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        # sourcery skip: raise-from-previous-error
        logging.info("entered the data ingestion part or components")

        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("read the dataset as dataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path) , exist_ok=True)

            df.to_csv(self.ingestion_config.data_path , index=False , header=True)
            logging.info("train test split initialized")

            train_set,test_set = train_test_split(df , test_size=0.20 , random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path , index=False ,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path , index=False , header=True)

            logging.info("data ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e , sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
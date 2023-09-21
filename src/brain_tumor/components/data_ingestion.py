import os
import sys
import pandas as pd 
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.brain_tumor.logger import logging
from src.brain_tumor.exception import CustomException
from src.brain_tumor.entity.config_entity import DataIngestionConfig
#from src.brain_tumor.config.configuration import ConfugarationManager

class DataIngestion:
    def __init__(self,config_:DataIngestionConfig):
        self.config_ = config_

    def data_ingestion_initialization(self):
        logging.info('Entered data ingestion method and components')
        try:
            df = pd.read_csv(self.config_.raw_data_path)
            logging.info(f'Read data as dataframe and {self.config_.raw_data_path}')

            logging.info("train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42,shuffle=True)
            test_set,val_set   = train_test_split(test_set,test_size=0.1,random_state=42,shuffle=True)

            train_set.to_csv(self.config_.train_data_path,index=False,header=True)
            test_set.to_csv (self.config_.test_data_path,index=False,header=True)
            val_set.to_csv(self.config_.val_data_path,index=False,header=True)
            logging.info('Ingestion of data is completed')
            return (
                self.config_.train_data_path,
                self.config_.test_data_path,
                self.config_.val_data_path
            )
        except Exception as e:
            raise CustomException (e,sys)
    
    
        

if __name__ == "__main__":
    config                      = ConfugarationManager()
    data_ingestion_config       = config.get_data_ingestion_config()
    split_data = DataIngestion(config_=data_ingestion_config)
    dataframe = split_data.data_ingestion_initialization()

    
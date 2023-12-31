import sys
from src.brain_tumor.exception import CustomException
from src.brain_tumor.logger import logging
from src.brain_tumor.config.configuration import ConfugarationManager
from src.brain_tumor.components.data_ingestion import DataIngestion



STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:

    def __init__(self):
        pass
    def main(self):
        config                      = ConfugarationManager()
        data_ingestion_config       = config.get_data_ingestion_config()
        data_ingestion              = DataIngestion(config_=data_ingestion_config)
        train_arr,test_arr,val_arr  = data_ingestion.data_ingestion_initialization()

if __name__ == "__main__":
    try:

        logging.info(f">>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<")
        obj     = DataIngestionTrainingPipeline()
        obj.main()

        logging.info((f">>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx======x"))
    except Exception as e:
        raise CustomException(e,sys)
    

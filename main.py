import sys
from src.brain_tumor.logger import logging
from src.brain_tumor.exception import CustomException
from src.brain_tumor.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logging.info(f">>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<")
    obj     = DataIngestionTrainingPipeline()
    obj.main()
    logging.info((f">>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx======x"))
except Exception as e:
    raise CustomException(e,sys)
    

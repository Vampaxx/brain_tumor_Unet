import sys
from src.brain_tumor.logger import logging
from src.brain_tumor.exception import CustomException
from src.brain_tumor.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.brain_tumor.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.brain_tumor.pipeline.stage_04_training_model import ModelTrainingPipeline
from src.brain_tumor.pipeline.stage_05_model_evaluation import EvaluationPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logging.info(f">>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<")
    obj     = DataIngestionTrainingPipeline()
    obj.main()
    logging.info((f">>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx======x"))
except Exception as e:
    raise CustomException(e,sys)
    

STAGE_NAME = "Prepare base model"
try:
    logging.info('***************************************')
    logging.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<<")
    obj         = PrepareBaseModelTrainingPipeline()
    obj.main()
    logging.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
    logging.info('***************************************')
except Exception as e:
    raise CustomException(e,sys)


    
STAGE_NAME = "Training"
try:
    logging.info('***************************************')
    logging.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<<")
    obj         = ModelTrainingPipeline()
    obj.main()
    logging.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
    logging.info('***************************************')
except Exception as e:
    raise CustomException(e,sys)



STAGE_NAME = "Evaluation stage"
try:
   logging.info(f"*******************")
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = EvaluationPipeline()
   model_evalution.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    raise CustomException (e,sys)
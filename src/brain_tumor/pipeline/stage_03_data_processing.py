import sys
from src.brain_tumor.logger import logging
from src.brain_tumor.exception import CustomException
from src.brain_tumor.config.configuration import ConfugarationManager
from src.brain_tumor.components.data_procesing import DataProcessing


STAGE_NAME = "Data processing"

class PrepareDataProcessingPipeline:

    def __init__(self) -> None:
        pass

    def main(self,data_split:str):
        self.data_split         = data_split
        config                  = ConfugarationManager()  
        processing_path         = config.get_data_processing_config()
        data_processing         = DataProcessing(config=processing_path)
        image_path,mask_path    = data_processing.get_processing_data_path(self.data_split)        
        data                    = data_processing.get_processing_pipeline()
        return data 

if __name__ == "__main__":
    try:
        logging.info('***************************************')
        logging.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        obj         = PrepareDataProcessingPipeline()
        obj.main(data_split='train')
        logging.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
        logging.info('***************************************')
    except Exception as e:
        raise CustomException(e,sys)
        


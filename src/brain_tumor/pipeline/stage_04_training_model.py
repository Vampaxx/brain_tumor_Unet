import sys
from src.brain_tumor.logger import logging
from src.brain_tumor.exception import CustomException
from src.brain_tumor.components.prepare_callbacks import PrepareCallback
from src.brain_tumor.config.configuration import ConfugarationManager
from src.brain_tumor.components.training import Training



STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_for_training         = ConfugarationManager()
        trainig_config              = config_for_training.get_training_config(dataset_type='train')
        prepare_callbacks_config    = config_for_training.get_prepare_callback_config()
        prepare_callbacks           = PrepareCallback(config=prepare_callbacks_config)
        callback_list               = prepare_callbacks.get_tb_ckpt_callbacks()
        
        training                    = Training(config=trainig_config)
        training.get_base_model()
        training.train(callback_list=callback_list)




if __name__ == '__main__':
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e,sys)
        
        



import sys
from src.brain_tumor.logger import logging
from src.brain_tumor.exception import CustomException
from src.brain_tumor.config.configuration import ConfugarationManager
from src.brain_tumor.components.model_evaluation import Evaluation



STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config      = ConfugarationManager()
        val_config  = config.get_validation_config()
        evaluation  = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()



if __name__ == '__main__':
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e,sys)



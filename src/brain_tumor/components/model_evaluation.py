import tensorflow as tf
from pathlib import Path
from src.brain_tumor.logger import logging 
from src.brain_tumor.exception import CustomException
from src.brain_tumor.utils.common import save_json
from src.brain_tumor.utils.loss_functions import *
from src.brain_tumor.config.configuration import  ConfugarationManager
from src.brain_tumor.entity.config_entity import EvaluationConfig






class Evaluation:

    def __init__(self,config : EvaluationConfig) :
        self.config     = config

    @staticmethod
    def load_model(path:Path) -> tf.keras.models:
        return tf.keras.models.load_model(filepath=path,
                                          custom_objects={'dice_coefficient_loss' : dice_coefficient_loss,
                                                          'iou'                   : iou,
                                                          'dice_coefficient'      : dice_coefficient})    
    
    
    def evaluation(self):
        logging.info(f'Model loaded from {self.config.path_of_model}')
        self.model      = self.load_model(self.config.path_of_model)
        logging.info(f'Model evaluation initialization')
        self.score      = self.model.evaluate(self.config.test_data)
        logging.info('Evaluation completed')
    
    def save_score(self):
        scores  = { 'loss'              : self.score[0],
                   'binary_accuracy'    : self.score[1],
                    'iou'               : self.score[2],
                    'dice_coefficient'  : self.score[3] }
        save_json(path=Path("scores.json"),data=scores)
        logging.info(f"Evaluation matrics saved on {'scores.json'}")



if __name__ == "__main__":
    config = ConfugarationManager()
    validation_config = config.get_validation_config()
    evaluation = Evaluation(validation_config)
    evaluation.evaluation()
    evaluation.save_score()
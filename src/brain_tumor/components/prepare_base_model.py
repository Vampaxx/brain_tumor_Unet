import os
import sys
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from src.brain_tumor.logger import logging
from src.brain_tumor.exception import CustomException
from src.brain_tumor.utils.Unet import unet
from src.brain_tumor.utils.common import save_model
from src.brain_tumor.utils.loss_functions import (dice_coefficient_loss,
                                                  iou,jaccured_distance,
                                                  dice_coefficient)
from src.brain_tumor.entity.config_entity import PrepareBaseModelConfig
from src.brain_tumor.config.configuration import ConfugarationManager

class PrepareBaseModel:

    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model_and_updated_model(self):
        logging.info('Base model building initialized')
        self.model = unet(self.config.params_image_size)
        save_model(path=self.config.base_model_path, model=self.model) 
        logging.info(f'Base model saved on {self.config.base_model_path}')

        #load the model
        self.loaded_model = load_model(self.config.base_model_path)
        logging.info('compiling started')
        self.loaded_model.compile(optimizer=Adam(learning_rate=self.config.params_learning_rate),
                                  metrics=["binary_accuracy", iou, dice_coefficient],
                                  loss=dice_coefficient_loss)
        save_model(path=self.config.updated_base_model_path, model=self.loaded_model) 
        logging.info(f'updated base model saved on {self.config.updated_base_model_path}')

    

if __name__ == "__main__":
    try:
        config = ConfugarationManager()  # Corrected the class name
        path = config.get_prepare_base_model_config()
        obj = PrepareBaseModel(config=path)
        obj.get_base_model_and_updated_model()
    except Exception as e:
        raise CustomException(e,sys)



    
    

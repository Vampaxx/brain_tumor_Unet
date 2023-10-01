import tensorflow as tf
from pathlib import Path
from src.brain_tumor.logger import logging
from tensorflow.keras.callbacks import EarlyStopping
from src.brain_tumor.components.prepare_callbacks import PrepareCallback
from src.brain_tumor.entity.config_entity import TrainigConfig
from src.brain_tumor.utils.loss_functions import *
from src.brain_tumor.config.configuration import ConfugarationManager



class Training(ConfugarationManager):

    def __init__(self,config:TrainigConfig):
        super().__init__()
        self.config     = config
    
    def get_base_model(self):
        logging.info(f'Model is loading from {self.config.updated_base_model_path}')
        self.model = tf.keras.models.load_model(filepath=self.config.updated_base_model_path,
                                                custom_objects={
                                                        'dice_coefficient_loss' : dice_coefficient_loss,
                                                        'iou'                   : iou,
                                                        'dice_coefficient'      : dice_coefficient
                                               })
        logging.info('Completed loading model')
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    

    def train(self, callback_list: list):
        early_stop = EarlyStopping(monitor='val_loss',patience=10,verbose=1)
        logging.info('Initializing training')
        self.model.fit(            
            self.get_training_config('train').data_for_pipeline,
            validation_data = self.get_training_config('test').data_for_pipeline,
            epochs          = self.config.params_epochs,
            callbacks       = [callback_list,early_stop])
        logging.info('Training completed')
        logging.info(f'Trained model saving in {self.config.trained_model_path}')
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model)
        logging.info('saving completed')
        
    



if __name__ == "__main__":
    config_for_training         = ConfugarationManager()
    trainig_config              = config_for_training.get_training_config(dataset_type='train')
    prepare_callbacks_config    = config_for_training.get_prepare_callback_config()
    prepare_callbacks           = PrepareCallback(config=prepare_callbacks_config)
    callback_list               = prepare_callbacks.get_tb_ckpt_callbacks()
    
    training                    = Training(config=trainig_config)
    training.get_base_model()
    training.train(callback_list=callback_list)
            
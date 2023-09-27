import os
import pandas as pd
import tensorflow as tf
from src.brain_tumor.logger import logging
from src.brain_tumor.config.configuration import ConfugarationManager
from src.brain_tumor.entity.config_entity import PreprocessingConfig
from src.brain_tumor.utils.Unet import unet
from src.brain_tumor.utils.common import convert_file_into_path
from src.brain_tumor.utils.data_processing import (mappable_function,mapping_fixup)


class DataProcessing(ConfugarationManager):

    def __init__(self, config: PreprocessingConfig):
        super().__init__()
        self.config_ = config

    def get_processing_data_path(self, dataset_type: str):
        self.dataset_type = dataset_type
        image_path = self.config_.image_path
        mask_path = self.config_.mask_path
        logging.info('preprocessing initialized')

        data_file_path = ""

        if self.dataset_type == 'train':
            data_file_path = self.config_.train_data_path
        elif self.dataset_type == 'test':
            data_file_path = self.config_.test_data_path
        elif self.dataset_type == 'val':
            data_file_path = self.config_.val_data_path
        else:
            raise ValueError("Invalid data_type. Use 'train', 'test', or 'val'.")
        logging.info(f'{self.dataset_type} file is created')

        self.image_file, self.mask_file = convert_file_into_path(data_file_path=data_file_path,
                                                                image_path=image_path,
                                                                mask_path=mask_path)
        return (self.image_file, self.mask_file)
    
    def get_processing_pipeline(self):
        logging.info('preprocessing is started')
        self.data   = tf.data.Dataset.list_files(self.image_file)
        self.data   = self.data.shuffle(buffer_size=self.params_.BUFFER_SIZE,reshuffle_each_iteration=False)
        self.data   = self.data.map(mappable_function)
        self.data   = self.data.map(mapping_fixup)
        self.data   = self.data.batch(self.params_.BATCH_SIZE)
        self.data   = self.data.prefetch(tf.data.AUTOTUNE)
        logging.info('preprocessing is completed')
        return self.data
    
if __name__ == "__main__":
    config                  = ConfugarationManager()  
    path                    = config.get_data_processing_config()
    obj                     = DataProcessing(config=path)
    image_path,mask_path    = obj.get_processing_data_path('test')        
    data                    = obj.get_processing_pipeline()
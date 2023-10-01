import os
import pandas as pd
import tensorflow as tf
from src.brain_tumor.logger import logging
#from src.brain_tumor.config.configuration import ConfugarationManager
from src.brain_tumor.entity.config_entity import PreprocessingConfig
from src.brain_tumor.utils.Unet import unet
from src.brain_tumor.utils.common import convert_file_into_path
from src.brain_tumor.utils.data_processing import (mappable_function,mapping_fixup)


class DataProcessing:

    def __init__(self, config: PreprocessingConfig):
        self.config_ = config

    def get_processing_data_path(self, dataset_type: str):
        self.dataset_type   = dataset_type
        image_path          = self.config_.image_path
        mask_path           = self.config_.mask_path
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
    
    def get_processing_pipeline(self,buffer_size,batch_size):
        self.buffer_size    = buffer_size
        self.batch_size     = batch_size
        logging.info('image file converted into tensor')
        data        = tf.data.Dataset.list_files(self.image_file)
        data        = data.shuffle(buffer_size=self.buffer_size,reshuffle_each_iteration=False)
        data        = data.map(mappable_function)
        data        = data.map(mapping_fixup)
        data        = data.batch(self.batch_size)
        data        = data.prefetch(tf.data.AUTOTUNE)
        logging.info('image file converted into tensor is completed')
        
        return data    
if __name__ == "__main__":
    config               = ConfugarationManager()
    processing_path      = config.get_data_processing_config()
    data_processing_obj  = DataProcessing(config=processing_path)
    data_processing_file = data_processing_obj.get_processing_data_path('train')
    data                 = data_processing_obj.get_processing_pipeline(buffer_size=1245,batch_size=32)
    
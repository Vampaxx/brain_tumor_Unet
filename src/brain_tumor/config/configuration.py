import pandas as pd
import tensorflow as tf
from src.brain_tumor.logger import logging
from src.brain_tumor.exception import CustomException
from src.brain_tumor.utils.common import read_yaml,create_directories,convert_file_into_path
from src.brain_tumor.constants import *
from src.brain_tumor.entity.config_entity import DataIngestionConfig
from src.brain_tumor.components.data_ingestion import DataIngestion

class ConfugarationManager:

    def __init__(self,
                 config_file_path=CONFIG_FILE_PATH,
                 params_file_path=PARAMS_FILE_PATH):
        
        # Assuming read_yaml and create_directories are custom functions in your module
        self.config_ = read_yaml(config_file_path)
        self.params_ = read_yaml(params_file_path)
        create_directories([self.config_.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config_ = self.config_.data_ingestion 
        create_directories([config_.root_dir])

        self.data_ingestion_config = DataIngestionConfig(

            train_data_path = Path(config_.train_path),
            test_data_path  = Path(config_.test_path), 
            val_data_path   = Path(config_.val_path),
            raw_data_path   = Path(config_.csv_file_path),
            image_path      = Path(config_.image_path),
            mask_path       = Path(config_.mask_path))
            
        return self.data_ingestion_config 
    
    def converting_image_and_mask_tensor(self,dataset_type:str):
        logging.info('converting image and mask into tensor')
        self.dataset_type   = dataset_type
        image_path          = self.data_ingestion_config.image_path
        mask_path           = self.data_ingestion_config.mask_path

        #self.image_files    = []
        #self.mask_files     = []

        data_file_path = None

        if dataset_type    == 'train':
            data_file_path = self.data_ingestion_config.train_data_path
        elif dataset_type  == 'test':
            data_file_path = self.data_ingestion_config.test_data_path
        elif dataset_type  == 'val':
            data_file_path = self.data_ingestion_config.val_data_path
        else:
            raise ValueError("Invalid data_type. Use 'train', 'test', or 'val'.")
        
        logging.info(f'{dataset_type} file is created')

        image,mask = convert_file_into_path(data_file_path,image_path,mask_path)
        data = tf.data.Dataset.from_tensor_slices((image, mask))

        return data
        
    
if __name__ == "__main__":
    obj = ConfugarationManager()
    file_paths = obj.get_data_ingestion_config()
    ingegestion = DataIngestion(config_=file_paths)
    train_data = obj.converting_image_and_mask_tensor(dataset_type='train')
    



    



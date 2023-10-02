import os
import pandas as pd
import tensorflow as tf
from src.brain_tumor.logger import logging
from src.brain_tumor.exception import CustomException
from src.brain_tumor.utils.common import (read_yaml,
                                          create_directories,
                                          convert_file_into_path,
                                          save_json)
from src.brain_tumor.constants import *
from src.brain_tumor.components.data_procesing import DataProcessing
from src.brain_tumor.entity.config_entity import (DataIngestionConfig,
                                                  PrepareBaseModelConfig,
                                                  PrepareCallbacksConfig,
                                                  PreprocessingConfig,
                                                  TrainigConfig,
                                                  EvaluationConfig)

class ConfugarationManager:

    def __init__(self,
                 config_file_path=CONFIG_FILE_PATH,
                 params_file_path=PARAMS_FILE_PATH):
        
        self.config_ = read_yaml(config_file_path)
        self.params_ = read_yaml(params_file_path)
        create_directories([self.config_.artifacts_root])
        #load Datapreprocessing 
        self.data_processing = DataProcessing(config=self.get_data_processing_config())

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
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:

        config_ = self.config_.prepare_base_model 
        create_directories([config_.root_dir])
        prepare_base_model_Config   = PrepareBaseModelConfig(
            root_dir                = Path (config_.root_dir),
            base_model_path         = Path (config_.base_model_path), 
            updated_base_model_path = Path (config_.updated_base_model_path),
            params_image_size       = list (self.params_.IMAGE_SIZE),
            params_mask_size        = list (self.params_.MASK_SIZE),
            params_learning_rate    = float(self.params_.LEARNING_RATE))
                
        return prepare_base_model_Config
    

    def get_data_processing_config(self) -> PreprocessingConfig:
        config_                     = self.config_.data_ingestion
        self.data_processing_config = PreprocessingConfig(
            train_data_path = Path(config_.train_path),
            test_data_path  = Path(config_.test_path), 
            val_data_path   = Path(config_.val_path),
            raw_data_path   = Path(config_.csv_file_path),
            image_path      = Path(config_.image_path),
            mask_path       = Path(config_.mask_path))
        return self.data_processing_config
    
    
    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config_         = self.config_.prepare_callbacks
        model_ckpt_dir  = os.path.dirname(config_.checkpoint_model_filepath)
        create_directories([Path(model_ckpt_dir),
                            Path(config_.tensorboard_root_log_dir)])
        prepare_callbacks_config    = PrepareCallbacksConfig(
            root_dir                    = Path(config_.root_dir),
            tensorboard_root_log_dir    = Path(config_.tensorboard_root_log_dir),
            checkpoint_model_filepath   = Path(config_.checkpoint_model_filepath))
        return prepare_callbacks_config
    
    def get_training_config(self,dataset_type: str) -> TrainigConfig:
        self.dataset_type               = dataset_type
        training                        = self.config_.training 
        prepare_base_model              = self.config_.prepare_base_model
        params                          = self.params_
        training_image, training_mask   = self.data_processing.get_processing_data_path(self.dataset_type)
        create_directories([Path(training.root_dir)])
        
        training_config                 = TrainigConfig(
            root_dir                        = Path(training.root_dir),
            trained_model_path              = Path(training.trained_model_path),
            updated_base_model_path         = Path(prepare_base_model.updated_base_model_path),
            data_for_pipeline               = self.data_processing.get_processing_pipeline(buffer_size=params.BUFFER_SIZE,
                                                                                        batch_size=params.BATCH_SIZE),
            params_epochs                   = params.EPOCHS,
            params_batch_size               = params.BATCH_SIZE,
            params_is_augumentation         = params.AUGMENTATION,
            params_image_size               = params.IMAGE_SIZE,
            params_mask_size                = params.MASK_SIZE )
        
        return training_config
    
    def get_validation_config(self) -> EvaluationConfig:
        trainig         = self.config_.training 
        self.data_processing.get_processing_data_path(dataset_type='test')
        
        eval_config     = EvaluationConfig(
            path_of_model       = Path(trainig.trained_model_path),
            test_data           = self.data_processing.get_processing_pipeline(buffer_size=self.params_.BUFFER_SIZE,
                                                                           batch_size=self.params_.BATCH_SIZE),
            all_params          = self.params_,
            params_image_size   = self.params_.IMAGE_SIZE,
            params_mask_size    = self.params_.MASK_SIZE,
            params_batch_size   = self.params_.BATCH_SIZE
        )
        return eval_config


if __name__ == "__main__":
    obj = ConfugarationManager()
    #file_paths = obj.get_data_ingestion_config()
    #ingegestion = DataIngestion(config_=file_paths)
    #train_data = obj.converting_image_and_mask_tensor(dataset_type='train')
    #print(train_data[0][:10])
    #obj.get_prepare_callback_config()
    #obj.get_validation_config()
    




    



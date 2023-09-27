import os
import sys
from pathlib import Path
from dataclasses import dataclass




@dataclass(frozen=True)
class DataIngestionConfig:

    train_data_path  : Path
    test_data_path   : Path
    val_data_path    : Path
    raw_data_path    : Path 
    image_path       : Path
    mask_path        : Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:

    root_dir                : Path
    base_model_path         : Path
    updated_base_model_path : Path  
    params_image_size       : list
    params_mask_size        : list
    params_learning_rate    : float


@dataclass(frozen=True)
class PrepareCallbacksConfig:

    root_dir                    : Path
    tensorboard_root_log_dir    : Path
    checkpoint_model_filepath   : Path



@dataclass(frozen=True)
class PreprocessingConfig:

    train_data_path  : Path
    test_data_path   : Path
    val_data_path    : Path
    raw_data_path    : Path 
    image_path       : Path
    mask_path        : Path
    

    
    
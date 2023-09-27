import os
import time
import tensorflow as tf
from src.brain_tumor.logger import logging
from src.brain_tumor.config.configuration import ConfugarationManager
from src.brain_tumor.entity.config_entity import PrepareCallbacksConfig



class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config
    
    @property
    def _create_tb_callbacks(self):
        time_stamp          = time.strftime("%m-%d-%Y-%H-%M-%S")
        tb_running_log_dir  = self.config.tensorboard_root_log_dir
        tb_running_log_dir  = os.path.join(tb_running_log_dir,
                                           f"tb_logs_at_{time_stamp}")
        logging.info("callback file created")

        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    @property
    def _create_ckpt_callbacks(self):
        logging.info("check point created")
        return tf.keras.callbacks.ModelCheckpoint(filepath=self.config.checkpoint_model_filepath,
                                                  save_best_only=True)
        
    
    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]
    

if __name__ == "__main__":
    config = ConfugarationManager()
    paths = config.get_prepare_callback_config()
    obj = PrepareCallback(config=paths)
    obj.get_tb_ckpt_callbacks()

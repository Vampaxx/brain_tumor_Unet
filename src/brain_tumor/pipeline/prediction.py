import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.brain_tumor.utils.loss_functions import *
from src.brain_tumor.utils.data_processing import load_image




class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename

    
    def predict(self):
        model_path      = os.path.join('artifacts','training','model.h5')
        trained_model   = load_model(filepath=model_path,
                                        custom_objects={'dice_coefficient_loss' : dice_coefficient_loss,
                                                        'iou'                   : iou,
                                                        'dice_coefficient'      : dice_coefficient})    
        img             = load_image(self.filename)
        ex_img          = tf.expand_dims(img,axis=0)
        pred            = trained_model.predict(ex_img)
        pred            = tf.squeeze(pred,axis=0).numpy()
        return [{'image'    : pred  }]
    

if __name__ == "__main__":
    prediction = PredictionPipeline(filename='TCGA_DU_7010_19860307_39.tif')
    pred = prediction.predict()
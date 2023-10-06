import os
import cv2
import numpy as np
from pathlib import Path

from flask import Flask, request, render_template
from src.brain_tumor.pipeline.prediction import PredictionPipeline

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from src.brain_tumor.utils.loss_functions import *
from src.brain_tumor.utils.data_processing import load_image


app             = Flask(__name__)
UPLOAD_FOLDER   = "\\Users\\Asus\\vs_code\\brain_tumor_Unet\\static"
upload_file_path        = os.path.join(UPLOAD_FOLDER,'upload')
prediction_mask_path    = os.path.join(UPLOAD_FOLDER,'pred_mask')

class ClientApp:
    def __init__(self):
        self.filename = ""
    def init_classifier(self):
        self.classifier = PredictionPipeline(self.filename)

@app.route("/", methods=["GET","POST"])
def upload_predict():

    if not os.path.exists(prediction_mask_path):
        os.makedirs(prediction_mask_path)
    if not os.path.exists(upload_file_path):
        os.makedirs(upload_file_path)

    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:

            image_location  = os.path.join(upload_file_path,image_file.filename)
            image_file.save(image_location)
            image_jpeg_file = os.path.splitext(image_location)[0] + ".jpeg"
            img_file        = os.path.basename(image_jpeg_file)
            img             = cv2.imread(image_location)
            cv2.imwrite(image_jpeg_file, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            #prediction
            clApp.filename = image_location         
            clApp.init_classifier()  
            pred    = clApp.classifier.predict()
            #save pred file into another folder
            pred_mask_location      = os.path.join(prediction_mask_path,image_file.filename)
            pred_mask_location_jpeg = os.path.splitext(pred_mask_location)[0] + ".jpeg"
            pred_normalized         = (pred * 255).astype(np.uint8)
            mask_file               = os.path.basename(pred_mask_location_jpeg)
            cv2.imwrite(pred_mask_location_jpeg, pred_normalized, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            return render_template('index.html', prediction = 'completed',mask_loc=mask_file,image_loc= img_file )
        
    return render_template('index.html',prediction='Please Upload files',mask_loc=None, image_loc = None)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(debug=True)

    
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Flatten



def dice_coefficient(y_true, y_pred,smooth = 100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union        = K.sum(y_true_flatten) + K.sum(y_pred_flatten)

    return (2 * intersection + smooth ) / (union + smooth)

def dice_coefficient_loss(y_true,y_pred,smooth = 100):
    return -dice_coefficient(y_true,y_pred,smooth)

# Why we use negative because, when dice_coefficient is high its means, it has high accuracy in the prediction, so loss  to need to be some quantity that will be high when the accuracy is low. Thats why take opposite of the accuracy (negative).

def iou (y_true,y_pred, smooth = 100):
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou

def jaccured_distance(y_true,y_pred,smooth = 100):
    y_true_flatten  = K.flatten(y_true)
    y_pred_flatten  = K.flatten(y_pred)
    return -iou(y_true_flatten,y_pred_flatten,smooth)
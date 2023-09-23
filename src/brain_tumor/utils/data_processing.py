import cv2
import os
import numpy as np
import tensorflow as tf

def load_image(path:str):
    image = cv2.imread(path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = tf.image.resize(image,(128,128))/255
    return image


def load_mask(path:str):
    mask = cv2.imread(path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = tf.expand_dims(mask, axis=-1)
    mask= tf.image.resize(mask,(128,128)) / 255
    #mask = mask/255
    zero = tf.zeros(shape=mask.shape)
    ones = tf.ones(shape=mask.shape)
    greater_mask = tf.math.greater(mask, 0.5)
    lesser_equal_mask = tf.math.less_equal(mask, 0.5)

    # Assign 1 to values greater than 0.5
    mask = tf.where(greater_mask,mask,zero)
    # Assign 0 to values less than or equal to 0.5
    mask = tf.where(lesser_equal_mask, mask, ones)
    return mask


def load_data(path:str):
    path = bytes.decode(path.numpy())

    file_name  = path.split('\\')[-1].split('.')[0]
    image_path = os.path.join('dataset','image',f'{file_name}.tif')
    mask_path  = os.path.join('dataset','mask',f'{file_name}_mask.tif')

    image = load_image(image_path)
    mask  = load_mask(mask_path)

    return image,mask

def mappable_function(path:str):
    result = tf.py_function(load_data,[path],(tf.float32,tf.float32))
    return result

def _fixup_shape(images, mask):
    images.set_shape([128, 128, 3])
    mask.set_shape([128, 128, 1])
    # weights.set_shape([None])
    return images, mask
def mapping_fixup(image,mask):
    result = tf.py_function(_fixup_shape,[image,mask],(tf.float32,tf.float32))
    return result


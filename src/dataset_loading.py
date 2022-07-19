import tensorflow as tf
import numpy as np
import PIL
import os

from tensorflow import keras
import pathlib

import logging

def load_train_data():
    """
    
    """
    # path to preprocessed training 
    data_dir = pathlib.Path("./Preprocessed_Images/train")
    
    
    batch_size = 32
    img_height = 128
    img_width = 128
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.1,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    
    logging.info("TRAINING DATASET CREATED")
    logging.info("TRAINING DATASET CLASSES: ")
    logging.info(train_ds.class_names)
    
    
    return train_ds
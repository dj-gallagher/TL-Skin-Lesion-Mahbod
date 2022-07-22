from src.preprocessing import *
from src.metadata_prep import *
from src.dataset_loading import *
from src.model import *

import numpy as np
import cv2
import pathlib

if __name__ == '__main__':
    
    # DEV
    # -------------
    #change_metadata()
    #create_img_filepaths_array()
    
    #images = load_images()
    #res = preprocess_image(images)
    
    #create_imagedata_dir()
    # -------------
    
    
    # TRAINING LOOP
    # -------------
    # Load datasets
    train_gen, val_gen, test_gen = create_dataset_generators()
    
    # Load model
    model = create_baseline()
    
    # Fit training data
    history = model.fit(x=train_gen,
                        validation_data=val_gen,
                        steps_per_epoch=train_gen.n//train_gen.batch_size,
                        validation_steps=val_gen.n//val_gen.batch_size,
                        epochs=15)
    
    # Evaluate 
    model.evaluate(x=test_gen)
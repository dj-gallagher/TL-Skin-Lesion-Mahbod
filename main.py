from src.preprocessing import *
from src.metadata_prep import *
from src.training import *
from src.dataset_loading import *

import matplotlib.pyplot as plt
import matplotlib
import logging


matplotlib.use('Agg') # https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined/3054314#3054314

if __name__ == '__main__':
    
    # DEV
    # --------------------------
    #change_metadata()
    #create_img_filepaths_array()
    
    #images = load_images()
    #res = preprocess_image(images)
    
    #create_imagedata_dir()
    # --------------------------
    
    
    # TRAINING LOOP
    # --------------------------
    # Name of run
    run_name = "BASELINE_RUN_1"
    run_dir = f"./Output/{run_name}"
    
    # Load datasets
    train_gen, val_gen, test_gen = create_dataset_generators()
    
    # set number of epochs
    num_epochs = 75 # 75 epochs to run all augmented data through
    
    # Train model, store training history and test set results
    history, results = run_training_pipeline(run_name, 
                                             train_gen, val_gen, test_gen,
                                             num_epochs)
    # --------------------------
        
    
    # RESULTS SAVING
    # --------------------------
    save_results(run_dir, history, results)
    
    
    
    # --------------------------

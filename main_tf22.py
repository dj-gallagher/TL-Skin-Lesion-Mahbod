from src.tf22.preprocessing import *
from src.tf22.metadata_prep import *
from src.tf22.training import *
from src.tf22.dataset_loading import *

import matplotlib


# execute below to fix plot saving on the GPU
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
    run_name = "DROPOUT_0.05_RUN_1"
    run_dir = f"./Output/{run_name}"
    run_description = ""
    
    # Load datasets
    train_gen, val_gen, test_gen = create_dataset_generators()
    
    # set number of epochs
    num_epochs = 75 # 75 epochs to run all augmented data through
    
    # set random seed
    seed = 4409
    
    # Train model, store training history and test set results
    history, results = run_training_pipeline(run_name, 
                                             train_gen, val_gen, test_gen,
                                             num_epochs, 
                                             seed)
    # --------------------------
        
    
    # RESULTS SAVING
    # --------------------------
    save_results(run_dir, history, results)
    
    
        
    # --------------------------


def main():
    """
    Steps:
        1. 

    """
    
    
    pass
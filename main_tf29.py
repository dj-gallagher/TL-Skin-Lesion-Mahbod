from src.tf29.preprocessing import *
from src.tf29.metadata_prep import *
from src.tf29.training import *
from src.tf29.dataset_loading import *

import matplotlib

matplotlib.use('Agg') # https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined/3054314#3054314



def random_seed_all_test():
    """
    Multiple runs where random seed is varied. Same random seed is used for all random operations.
    50 Epochs only as little change in accuracy and loss after.  
    """
    
    for item in [355, 980, 6664, 3175]:
        
        # set random seed
        seed = item
        
        
        # TRAINING LOOP
        # --------------------------
        # Name of run
        run_name = f"random_seed_all_{seed}"
        run_dir = f"./Output/{run_name}"
        run_description = "Baseline with SGDM. Testing effect of random seed."
        
        # Load datasets
        train_gen, val_gen, test_gen = create_dataset_generators(seed)
        
        # set number of epochs
        num_epochs = 50 # 120 epochs for equivalent 8 fold increase in training data
        
        # Train model, store training history and test set results
        history, results = run_training_pipeline(run_name, 
                                                train_gen, val_gen, test_gen,
                                                num_epochs, 
                                                seed)
        # --------------------------
            
            
        
        # RESULTS SAVING
        # --------------------------
        save_results(run_dir, history, results)
    
    
    
    

def main():
    
    # set random seed
    seed = 4409
    
    
    # TRAINING LOOP
    # --------------------------
    # Name of run
    run_name = "tf_29_SGDM_AugTestData1"
    run_dir = f"./Output/{run_name}"
    run_description = "Baseline with SGDM and augmented test data (8 fold increase)"
    
    # Load datasets
    train_gen, val_gen, test_gen = create_dataset_generators(seed)
    
    # set number of epochs
    num_epochs = 120 # 120 epochs for equivalent 8 fold increase in training data
    
    # Train model, store training history and test set results
    history, results = run_training_pipeline(run_name, 
                                             train_gen, val_gen, test_gen,
                                             num_epochs, 
                                             seed)
    # --------------------------
        
        
    
    # RESULTS SAVING
    # --------------------------
    save_results(run_dir, history, results)
    
    
        
if __name__ == '__main__':
    random_seed_all_test()
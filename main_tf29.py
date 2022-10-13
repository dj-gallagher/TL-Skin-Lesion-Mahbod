from src.tf29.preprocessing import *
from src.tf29.metadata_prep import *
from src.tf29.training import *
from src.tf29.dataset_loading import *

import matplotlib
import tensorflow as tf

matplotlib.use('Agg') # https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined/3054314#3054314



def random_seed_all_test():
    """
    Multiple runs where random seed is varied. Same random seed is used for all random operations.
    50 Epochs only as little change in accuracy and loss after.  
    """
    
    for item in [355, 980, 6664, 3175]:
        
        # clear session at run start to reset Keras name generation sequence
        tf.keras.backend.clear_session()
        
        tf.config.run_functions_eagerly(True)
        
        # set random seed
        seed = item
        
        
        # TRAINING LOOP
        # --------------------------
        # Name of run
        run_name = f"baseline_seed_{seed}"
        run_dir = f"./Output/{run_name}"
        run_description = "Baseline with SGDM. Testing effect of random seed."
        
        # Load datasets
        #train_gen, val_gen, test_gen = create_dataset_generators(seed)
        train_gen, val_gen, test_gen = create_dataset_generators2(seed)
        
        # set number of epochs
        num_epochs = 15 # 120 epochs for equivalent 8 fold increase in training data
        
        # Train model, store training history and test set results
        history, results = run_training_pipeline(run_name, 
                                                train_gen, val_gen, test_gen,
                                                num_epochs, 
                                                seed)
        # --------------------------
            
            
        
        # RESULTS SAVING
        # --------------------------
        save_results(run_dir, history, results)


def cosine_LR_decay():
    """
    Multiple runs where LR decay minimum is varied. Random seed of 6664 is used.
    15 training epochs.  
    """
    
    run_num = 2
    
    for mult in [0.25, 0.1]: # 0.1, 0.05]:
        
        # to name output files as fullstop in min value will cause error
        run_num += 1
                
        # clear session at run start to reset Keras name generation sequence
        tf.keras.backend.clear_session()
        
        # set random seed
        seed = 355
        
        
        # TRAINING LOOP
        # --------------------------
        # Name of run
        run_name = f"LR_decay_mult_{run_num}"
        run_dir = f"./Output/{run_name}"
        run_description = f"Baseline with SGDM and cosine LR decay. Testing rate of LR decay. Fraction of full epochs to decay to 1/10th initial LR = {mult}"
        
        # Load datasets
        #train_gen, val_gen, test_gen = create_dataset_generators(seed)
        train_gen, val_gen, test_gen = create_dataset_generators2(seed)
        
        # set number of epochs
        num_epochs = 15 
        
        # Train model, store training history and test set results
        history, results = run_training_pipeline(run_name, 
                                                train_gen, val_gen, test_gen,
                                                num_epochs, 
                                                seed,
                                                0.1,
                                                0,
                                                0,
                                                mult)
        # --------------------------
            
            
        
        # RESULTS SAVING
        # --------------------------
        save_results(run_dir, history, results)
    
    
def label_smooth():
    """
    Multiple runs where label smoothing rate is varied.
    Random seed of 355 is used.
    15 training epochs.
    LR is decayed to 1/10th its original value in {insert fraction} the full number of epochs.
    Label smoothing factor is varied.
    """
    
    run_num = 0
    
    for rate in [0.5, 0.1, 0.05, 0.01]:
        
        # to name output files as fullstop in min value will cause error
        run_num += 1
                
        # clear session at run start to reset Keras name generation sequence
        tf.keras.backend.clear_session()
        
        # set random seed
        seed = 355
        
        
        # TRAINING LOOP
        # --------------------------
        # Name of run
        run_name = f"Dropout_{run_num}"
        run_dir = f"./Output/{run_name}"
        run_description = f"Baseline with SGDM, cosine LR decay, label smoothing. Testing dropout rate. Rate for run = {rate}. Using LR decay to 0.25 times start LR and smoothing of 0.05."
        
        # Load datasets
        #train_gen, val_gen, test_gen = create_dataset_generators(seed)
        train_gen, val_gen, test_gen = create_dataset_generators2(seed)
        
        # set number of epochs
        num_epochs = 15 
        
        # Train model, store training history and test set results
        history, results = run_training_pipeline(run_name=run_name,
                                                 train_gen=train_gen,
                                                 val_gen=val_gen,
                                                 test_gen=test_gen,
                                                 num_epochs=num_epochs, 
                                                 random_seed=seed,
                                                 alpha=0.1, 
                                                 smoothFactor=rate,
                                                 dropRate=0,
                                                 steps_multiplier=0.5)
        # --------------------------
            
            
        
        # RESULTS SAVING
        # --------------------------
        save_results(run_dir, history, results)


def dropout():
    """
    Multiple runs where droout rate is varied.
    Random seed of 355 is used.
    15 training epochs.
    LR is decayed to 1/10th its original value in {insert fraction} the full number of epochs.
    Label smoothing factor is varied.
    """
    
    run_num = 0
    
    for rate in [0.5, 0.1, 0.05, 0.01]:
        
        # to name output files as fullstop in min value will cause error
        run_num += 1
                
        # clear session at run start to reset Keras name generation sequence
        tf.keras.backend.clear_session()
        
        # set random seed
        seed = 355
        
        
        # TRAINING LOOP
        # --------------------------
        # Name of run
        run_name = f"Actual_Dropout_{run_num}"
        run_dir = f"./Output/{run_name}"
        run_description = f"Baseline with SGDM, cosine LR decay, label smoothing and dropout. Testing dropout rate. Rate for run = {rate}. Using LR decay to 0.25 times start LR and smoothing of 0.05. Using label smooth factor of 0.1"
        
        # Load datasets
        #train_gen, val_gen, test_gen = create_dataset_generators(seed)
        train_gen, val_gen, test_gen = create_dataset_generators2(seed)
        
        # set number of epochs
        num_epochs = 15 
        
        # Train model, store training history and test set results
        history, results = run_training_pipeline(run_name=run_name,
                                                 train_gen=train_gen,
                                                 val_gen=val_gen,
                                                 test_gen=test_gen,
                                                 num_epochs=num_epochs, 
                                                 random_seed=seed,
                                                 alpha=0.1, 
                                                 smoothFactor=0.1,
                                                 dropRate=rate,
                                                 steps_multiplier=0.5)
        # --------------------------
            
            
        
        # RESULTS SAVING
        # --------------------------
        save_results(run_dir, history, results)


def main():
    
    seed_list = [1, 100, 250, 365, 498, 579, 623]
    
    for seed in seed_list:
    
        # clear session at run start to reset Keras name generation sequence
        tf.keras.backend.clear_session()
        
        tf.config.run_functions_eagerly(True)
        
        # set random seed
        #seed = 355
        
        
        # TRAINING LOOP
        # --------------------------
        # Name of run
        run_name = f"Test_1_Seed_{seed}"
        run_dir = f"./Output/{run_name}"
        run_description = "Baseline with different weight inits."
        
        # Load datasets
        #train_gen, val_gen, test_gen = create_dataset_generators(seed)
        train_gen, val_gen, test_gen = create_dataset_generators2()
        
        # set number of epochs
        num_epochs = 4 # 15 standard
        
        # Train model, store training history and test set results
        history, results = run_training_pipeline(run_name, 
                                                train_gen, val_gen, test_gen,
                                                num_epochs, 
                                                seed)
        # --------------------------
            
            
        
        # RESULTS SAVING
        # --------------------------
        save_results(run_dir, history, results)



def cosine_LR_decay_tests():
    """
    Multiple runs where LR decay minimum is varied. Random seed of 6664 is used.
    15 training epochs.  
    """
    
    run_num = 0
    
    for mult in [1]: # 0.1, 0.05]:
        
        # to name output files as fullstop in min value will cause error
        run_num += 1
                
        # clear session at run start to reset Keras name generation sequence
        tf.keras.backend.clear_session()
        
        # set random seed
        seed = 365 # EDIT in based on Test 1 results
        
        
        # TRAINING LOOP
        # --------------------------
        # Name of run
        run_name = f"Test_2_Run_{run_num}"
        run_dir = f"./Output/{run_name}"
        run_description = f"Baseline with SGDM and cosine LR decay. Fraction of full epochs to decay to 1/10th initial LR = {mult}"
        
        # Load datasets
        #train_gen, val_gen, test_gen = create_dataset_generators(seed)
        train_gen, val_gen, test_gen = create_dataset_generators2(seed)
        
        # set number of epochs
        num_epochs = 4
        
        # Train model, store training history and test set results
        history, results = run_training_pipeline(run_name, 
                                                train_gen, val_gen, test_gen,
                                                num_epochs, 
                                                seed,
                                                alpha=0.01,
                                                smoothFactor=0,
                                                dropRate=0,
                                                steps_multiplier=1)
        # --------------------------
            
            
        
        # RESULTS SAVING
        # --------------------------
        save_results(run_dir, history, results)

    
def label_smoothing_tests():
    """
    Multiple runs where LR decay minimum is varied. Random seed of 6664 is used.
    15 training epochs.  
    """
    
    run_num = 0
    
    for factor in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]: # 0.1, 0.05]:
        
        # to name output files as fullstop in min value will cause error
        run_num += 1
                
        # clear session at run start to reset Keras name generation sequence
        tf.keras.backend.clear_session()
        
        # set random seed
        seed = 365 # EDIT in based on Test 1 results
        
        
        # TRAINING LOOP
        # --------------------------
        # Name of run
        run_name = f"Test_3_Run_{run_num}"
        run_dir = f"./Output/{run_name}"
        run_description = f"Baseline with SGDM, cosine LR decay and label smoothing. Smooth factor = {factor}"
        
        # Load datasets
        #train_gen, val_gen, test_gen = create_dataset_generators(seed)
        train_gen, val_gen, test_gen = create_dataset_generators2(seed)
        
        # set number of epochs
        num_epochs = 4
        
        # Train model, store training history and test set results
        history, results = run_training_pipeline(run_name, 
                                                train_gen, val_gen, test_gen,
                                                num_epochs, 
                                                seed,
                                                alpha=0.01,
                                                smoothFactor=factor,
                                                dropRate=0,
                                                steps_multiplier=1)
        # --------------------------
            
            
        
        # RESULTS SAVING
        # --------------------------
        save_results(run_dir, history, results)


def cosine_LR_extra_epochs_tests():
    pass
    
def label_smmothing_extra_epochs_tests():
    """
    Multiple runs where LR decay minimum is varied. Random seed of 6664 is used.
    15 training epochs.  
    """
    
    run_num = 0
    
    for factor in [0.01, 0.1, 0.2]: # other factors tested 0.3, 0.4, 0.5]: 
        
        # to name output files as fullstop in min value will cause error
        run_num += 1
                
        # clear session at run start to reset Keras name generation sequence
        tf.keras.backend.clear_session()
        
        # set random seed
        seed = 365 # EDIT in based on Test 1 results
        
        
        # TRAINING LOOP
        # --------------------------
        # Name of run
        run_name = f"Test_3_Run_{run_num}_extraEpochs"
        run_dir = f"./Output/{run_name}"
        run_description = f"Baseline with SGDM, cosine LR decay, increased epochs and label smoothing. Smooth factor = {factor}"
        
        # Load datasets
        #train_gen, val_gen, test_gen = create_dataset_generators(seed)
        train_gen, val_gen, test_gen = create_dataset_generators2(seed)
        
        # set number of epochs
        num_epochs = 12
        
        # Train model, store training history and test set results
        history, results = run_training_pipeline(run_name, 
                                                train_gen, val_gen, test_gen,
                                                num_epochs, 
                                                seed,
                                                alpha=0.01,
                                                smoothFactor=factor,
                                                dropRate=0,
                                                steps_multiplier=1)
        # --------------------------
            
            
        
        # RESULTS SAVING
        # --------------------------
        save_results(run_dir, history, results)
    
    
        
if __name__ == '__main__':
    label_smmothing_extra_epochs_tests()
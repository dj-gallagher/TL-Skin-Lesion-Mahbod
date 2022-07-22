from src.model import create_basic_ResNet50

import logging
import os


def run_training_pipeline(run_name, train_gen, val_gen, test_gen):
    """
    
    """
    
    run_dir = f"./Output/{run_name}"
    
    # make folder to store run info
    os.mkdir(run_dir)
    
    logging.basicConfig(filename=run_dir + f'/{run_name}.log', 
                        level=logging.INFO,
                        format='%(asctime)s %(message)s')
    
    logging.info(f"STARTING... RUN NAME: {run_name}")
    logging.info("Creating image data generators")
    
    
    logging.info("Creating and compiling keras model")
    
    # Load model
    model = create_basic_ResNet50()
    
    logging.info("Training model...")
    
    # Fit training data
    history = model.fit(x=train_gen,
                        validation_data=val_gen,
                        steps_per_epoch=train_gen.n//train_gen.batch_size,
                        validation_steps=val_gen.n//val_gen.batch_size,
                        epochs=75)
    
    logging.info("Training finished")
    logging.info("Testing trained model...")
    
    # Evaluate 
    results = model.evaluate(x=test_gen)
    
    logging.info("Testing finished.")
    
    return history, results
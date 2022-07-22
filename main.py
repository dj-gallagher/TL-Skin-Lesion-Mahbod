from src.preprocessing import *
from src.metadata_prep import *
from src.training import *
from src.dataset_loading import *

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import logging
import pandas as pd

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
    run_name = "RUN_1"
    run_dir = f"./Output/{run_name}"
    
    # Load datasets
    train_gen, val_gen, test_gen = create_dataset_generators()
    
    # Train model, store training history and test set results
    history, results = run_training_pipeline(run_dir, train_gen, val_gen, test_gen)
    # --------------------------
        
    
    # RESULTS SAVING
    # --------------------------
    # Training and validation metric plots
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, label='Training acc')
    plt.plot(epochs, val_acc, label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(run_dir + '/accuracy.png')
    
    plt.figure()

    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(run_dir + '/loss.png')
    
    
    # Test set metrics
    metrics_df = pd.DataFrame(results, index=[0])
    metrics_df.to_csv(run_dir + "/results.csv")
    
    # confusion matrix 
    # IMPLEMENT LATER
    
    # --------------------------

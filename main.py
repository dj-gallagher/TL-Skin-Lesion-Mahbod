from src.preprocessing import *
from src.metadata_prep import *
from src.dataset_loading import *
from src.model import *

import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import logging

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
    
    # make folder to store run info
    run_dir = f"./Output/{run_name}"
    os.mkdir(run_dir)
    
    logging.basicConfig(filename=run_dir + f'/{run_name}.log', 
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    
    logging.info(f"STARTING... RUN NAME: {run_name}")
    logging.info("Creating image data generators")
    
    # Load datasets
    train_gen, val_gen, test_gen = create_dataset_generators()
    
    logging.info("Creating and compiling keras model")
    
    # Load model
    model = create_baseline()
    
    logging.info("Training model...")
    
    # Fit training data
    history = model.fit(x=train_gen,
                        validation_data=val_gen,
                        steps_per_epoch=train_gen.n//train_gen.batch_size,
                        validation_steps=val_gen.n//val_gen.batch_size,
                        epochs=15)
    
    logging.info("Training finished")
    logging.info("Testing trained model...")
    
    # Evaluate 
    results = model.evaluate(x=test_gen)
    
    logging.info("Testing finished.")
    # --------------------------
    
    print("TEST SET LABLES:")
    print(test_gen.labels)
    
    
    '''# RUN RESULTS SAVING
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
    
    # confusion matrix 
    y_pred = model.predict(x=test_gen) # get predicted labels
    y_pred = y_pred.argmax(axis=1) # convert to ints
    
    print(test_gen.labels)
    
    plt.figure()
    plt.grid(False)
    matrix = confusion_matrix(y_true, y_pred)
    matrix_plot = ConfusionMatrixDisplay(matrix,
                            display_labels=["mel", "seb", "nev"]).plot()
    plt.savefig(f"./output/results/{model.name}/conf_matrix.png")'''
    
    # --------------------------

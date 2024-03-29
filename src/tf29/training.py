from gc import callbacks
from src.tf29.model import *

import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




def run_training_pipeline(run_name, train_gen, val_gen, test_gen, num_epochs, random_seed, alpha=None, smoothFactor=None, dropRate=None, steps_multiplier=1):
    """
    Compiles the desired model, trains it on training and val input data,
    evaluates the trained model and creates a confusion matrix on the test data.
    """
    
    run_dir = f"./Output/{run_name}"
    
    steps_per_epoch = train_gen.n//train_gen.batch_size
    
    
    # make folder to store run info
    os.mkdir(run_dir)
    
    logging.basicConfig(filename=run_dir + f'/{run_name}.log', 
                        level=logging.INFO,
                        format='%(asctime)s %(message)s')
    
    logging.info(f"STARTING... RUN NAME: {run_name}")
    logging.info("Creating image data generators")
    
    
    logging.info("Creating and compiling keras model")
    
    # Load model
    #model = create_basic_ResNet50()
    #model = create_baseline_ResNet50(random_seed)
    model = compile_improved_ResNet50(random_seed=random_seed, 
                                      steps_per_epoch=steps_per_epoch,
                                      enable_dropout=True,
                                      dropout_rate=dropRate,
                                      label_smoothing_factor=smoothFactor,
                                      enable_cosineLR=True,
                                      alpha=alpha,
                                      steps_multiplier=steps_multiplier)
    
    
    logging.info("Training model...")
    
    # Fit training data
    history = model.fit(x=train_gen,
                        validation_data=val_gen,
                        steps_per_epoch=train_gen.n//train_gen.batch_size,
                        validation_steps=val_gen.n//val_gen.batch_size,
                        epochs=num_epochs)
    
    logging.info("Training finished")
    logging.info("Testing trained model...")
    
    # Evaluate 
    results = model.evaluate(x=test_gen)
    
    logging.info("Testing finished.")
    
    
    # confusion matrix
    # ---------------------
    y_true = test_gen.labels
    
    y_pred = model.predict(test_gen) # get predicted labels
    
    y_pred = y_pred.argmax(axis=1) # convert to ints
    
    plt.figure()
    plt.grid(False)
    matrix = confusion_matrix(y_true, y_pred)
    matrix_plot = ConfusionMatrixDisplay(matrix,
                            display_labels=["mel", "nev", "seb"]).plot() # note: name ordering is the same order as directory tree in test images dir
    plt.savefig(f"./Output/{run_name}/conf_matrix.png")
    
    
    
    return history, results


def save_results(run_dir, history, results):
    """
    Save train val metric plots and results csv file to output dir.
    """
    
    # Training and validation metric plots
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss'][4:]
    val_loss = history.history['val_loss'][4:]
    auc = history.history['auc']
    val_auc = history.history['val_auc']
    
    epochs = range(1, len(acc) + 1)

    # save raw data to a CSV file in case its needed later
    raw_data_dict = {'epochs':epochs,
                     'acc':acc,
                     'val_acc':val_acc,
                     'loss':loss,
                     'val_loss':val_loss,
                     'auc':auc,
                     'val_auc':val_auc}
    
    raw_data_DF = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in raw_data_dict.items() ]))

    
    raw_data_DF.to_csv(run_dir + "/raw_data.csv")
    
    

    fig1, ax1 = plt.subplots()
    
    ax1.plot(epochs, acc, label='Training acc')
    ax1.plot(epochs, val_acc, label='Validation acc')
    ax1.set_title('Training and validation accuracy')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    fig1.savefig(run_dir + '/accuracy.png')
    
    fig2, ax2 = plt.subplots()

    ax2.plot(epochs[4:], loss, label='Training loss')
    ax2.plot(epochs[4:], val_loss, label='Validation loss')
    ax2.set_title('Training and validation loss')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    fig2.savefig(run_dir + '/loss.png')
    
    fig3, ax3 = plt.subplots()

    ax3.plot(epochs, auc, label='Training AUC')
    ax3.plot(epochs, val_auc, label='Validation AUC')
    ax3.set_title('Training and validation AUC')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("AUC")
    ax3.legend()

    fig3.savefig(run_dir + '/auc.png')
    
    
    # Test set metrics
    metrics = ["loss", "accuracy", "AUC"]
    metrics_dict = dict(zip(metrics, results))
    metrics_df = pd.DataFrame(metrics_dict, index=[0])
    metrics_df.to_csv(run_dir + "/results.csv")
    
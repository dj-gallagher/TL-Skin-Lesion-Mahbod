from src.model import create_basic_ResNet50

import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def run_training_pipeline(run_name, train_gen, val_gen, test_gen, num_epochs):
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
    
    print("TEST GEN LABELS:")
    print(y_true)
    print("\n")
    print("PRED LABELS:")
    print(y_pred)
    
    
    plt.figure()
    plt.grid(False)
    matrix = confusion_matrix(y_true, y_pred)
    matrix_plot = ConfusionMatrixDisplay(matrix,
                            display_labels=["mel", "seb", "nev"]).plot()
    plt.savefig(f"./Output/{run_name}/conf_matrix.png")
    
    
    
    return history, results


def save_results(run_dir, history, results):
    """
    
    """
    
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
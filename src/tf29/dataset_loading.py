from tensorflow import keras
import pathlib
import logging


def load_train_data():
    """
    Load training data into a tf dataset object
    
    OBSOLETE: Not available in tf 2.2 
    """
    # path to preprocessed training 
    data_dir = pathlib.Path("./Preprocessed_Images/train")
    
    
    batch_size = 32
    img_height = 128
    img_width = 128
    
    
    train_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.1,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    
    logging.info("TRAINING DATASET CREATED")
    logging.info("TRAINING DATASET CLASSES: ")
    logging.info(train_ds.class_names)
    
    
    return train_ds



def create_dataset_generators(random_seed):
    """
    Create image data generator objects for training, validation and test sets. 
    Augmentation included on training and validation but not test sets.
    Returns datagen objects ready for use in a training loop.
    """
    
    # path to preprocessed image data
    train_data_dir = pathlib.Path("./Preprocessed_Images/train")
    test_data_dir = pathlib.Path("./Preprocessed_Images/test")
    
    # image data generators
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=270,
                                                                 horizontal_flip=True,
                                                                 validation_split=0.1)
    
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=270,
                                                                 horizontal_flip=True)
    
    
    # dataset generators
    train_gen = train_datagen.flow_from_directory(directory=train_data_dir, 
                                      target_size=(128,128),
                                      class_mode="categorical",
                                      batch_size=32,
                                      shuffle=True,
                                      seed=random_seed,
                                      subset="training")
    
    val_gen = train_datagen.flow_from_directory(directory=train_data_dir, 
                                      target_size=(128,128),
                                      class_mode="categorical",
                                      batch_size=32,
                                      shuffle=True,
                                      seed=random_seed,
                                      subset="validation")
    
    test_gen = test_datagen.flow_from_directory(directory=test_data_dir,
                                                target_size=(128,128),
                                                class_mode="categorical",
                                                batch_size=1,
                                                shuffle=False,
                                                seed=random_seed)
    
    return train_gen, val_gen, test_gen
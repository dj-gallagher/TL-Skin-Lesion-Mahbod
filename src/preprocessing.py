import pandas as pd
import numpy as np
import cv2

def load_images():
    """
    Read in images and store in an array
    """
    # Read in metadata
    metadata_filepath = "./Metadata/Dev/Dev.csv"
    metadata_df = pd.read_csv(metadata_filepath)
    
    # Load images into numpy array
    image_filepaths = metadata_df["image_filepath"].to_numpy()
    images_array = cv2.imread(image_filepaths)
    
    
    #return images_array
    

def grayworld(image):
    """
    Apply grayworld white balancing algorithm to an image.
    
    Source: https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    """
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            l, a, b = result[x, y, :]
            # fix for CV correction
            l *= 100 / 255.0
            result[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            result[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    
    return result
    

def grayworld_2(image):
    """
    Apply  grayworld white balancing with different approach
    
    source: https://jephraim-manansala.medium.com/image-processing-with-python-color-correction-using-white-balancing-6c6c749886de
    """

    result = ((image * (image.mean() / image.mean(axis=(0, 1)))).clip(0, 255).astype(int))
    
    return result
    

def imagenet_rgb_subtract(image):    
    """
    Subtract the Imagenet average intensity values from each channel in an RGB image.
    
    ImageNet B,G,R intesity means (Range [0,1]): 0.406, 0.456, 0.485
    """
    
    # Calculate imagenet BGR means in range [0,255]
    B, G, R = (int(0.406*255), int(0.456*255), int(0.485*255))
    
    # Copy image array 
    result = np.copy(image)
    
    # subtract means
    result[:,:,0] =  result[:,:,0] - B
    result[:,:,1] =  result[:,:,1] - G
    result[:,:,2] =  result[:,:,2] - R
    
    
    return result


def resize_image(image):
    """
    Change the aspect ratio of an image.
    """   
    # set new dimensions
    new_W = 128
    new_H = 128
    
    # resize
    result = cv2.resize(image, (new_W, new_H))
     
    return result
    
    

def preprocess_image(image):
    """
    Apply all 3 Mahbod preprocessing techniques to an image.
    """
    result = grayworld_2(image)
    result = imagenet_rgb_subtract(result)
    result = resize_image(result)
    
    
    return result


def create_imagedata_dir():
    """
    Takes the train and test metadata csv's and uses them to create the image data directory
    using the train and test datasets in the images folder
    """
    
    '''
    # DEV
    # =========================== 
    # get dev metadata
    metadata_df = pd.read_csv("./Metadata/Dev/Dev_Metadata_new.csv")
    
    # iterate through metadata
    for i in range(metadata_df.shape[0]):
        
        # get image class and filepath
        image_class = metadata_df.loc[i,:].where(metadata_df.loc[i,:]==1).dropna().index.to_numpy()
        
        if image_class == "seborrheic_keratosis":
            
            isic_id = metadata_df.iloc[i,0]
            image_filepath = "./Images/Dev/" + isic_id + ".jpg"
            
            # read image
            image = cv2.imread(image_filepath)
            
            # preprocess image
            prepd_image = preprocess_image(image)
            
            # write preprocessed image to destination folder
            if image_class == "melanoma":
                destination_dir = "Preprocessed_Images/Dev/mel/"
            if image_class == "nevus":
                destination_dir = "Preprocessed_Images/Dev/nev/"
            if image_class == "seborrheic_keratosis":
                destination_dir = "Preprocessed_Images/Dev/seb/"
            
            cv2.imwrite(destination_dir + isic_id + ".jpg", prepd_image)
            
            break
    ''' 
    
    # TRAIN
    # =========================== 
    # get metadata
    metadata_df = pd.read_csv("./Metadata/train.csv")
    
    # iterate through metadata
    for i in range(metadata_df.shape[0]):
        
        # get image class and filepath
        image_class = metadata_df.loc[i,:].where(metadata_df.loc[i,:]==1).dropna().index.to_numpy()
        
        isic_id = metadata_df.iloc[i,0]
        image_filepath = "./Images/train/" + isic_id + ".jpg"
        
        # read image
        image = cv2.imread(image_filepath)
        
        # preprocess image
        prepd_image = preprocess_image(image)
        
        # write preprocessed image to destination folder
        if image_class == "melanoma":
            destination_dir = "Preprocessed_Images/train/mel/"
        if image_class == "nevus":
            destination_dir = "Preprocessed_Images/train/nev/"
        if image_class == "seborrheic_keratosis":
            destination_dir = "Preprocessed_Images/train/seb/"
        
        cv2.imwrite(destination_dir + isic_id + ".jpg", prepd_image)
        
    # TEST
    # =========================== 
    # get metadata
    metadata_df = pd.read_csv("./Metadata/test.csv")
    
    # iterate through metadata
    for i in range(metadata_df.shape[0]):
        
        # get image class and filepath
        image_class = metadata_df.loc[i,:].where(metadata_df.loc[i,:]==1).dropna().index.to_numpy()
        
        isic_id = metadata_df.iloc[i,0]
        image_filepath = "./Images/test/" + isic_id + ".jpg"
        
        # read image
        image = cv2.imread(image_filepath)
        
        # preprocess image
        prepd_image = preprocess_image(image)
        
        # write preprocessed image to destination folder
        if image_class == "melanoma":
            destination_dir = "Preprocessed_Images/test/mel/"
        if image_class == "nevus":
            destination_dir = "Preprocessed_Images/test/nev/"
        if image_class == "seborrheic_keratosis":
            destination_dir = "Preprocessed_Images/test/seb/"
        
        cv2.imwrite(destination_dir + isic_id + ".jpg", prepd_image)
    
    
    # END
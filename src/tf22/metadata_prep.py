import numpy as np
import pandas as pd



def change_metadata():
    
    """
    Function to alter Mahbod original metatdata files to include the nevus
    class label.
    """
    # read in original metadata file
    df = pd.read_csv("./Metadata/Dev/Dev_metadata.csv")
    
    # if 0, 0 found in mel and ske_ker class add nev column with 1
    df["nevus"] = pd.Series( np.zeros(len(df)) )
    for ind in range(len(df)):
        if df.loc[ind, "melanoma"] == 0 and df.loc[ind, "seborrheic_keratosis"] == 0:
            df.loc[ind, "nevus"] = 1
            
    # write new csv file
    df.to_csv("./Metadata/Dev/Dev_metadata_new.csv", index=False)
    



    
def create_img_filepaths_array():
    """
    Create an array of filepaths to each of the ISIC images.
    """
    
    metadata_path = "./Metadata/Dev/Dev_metadata_new.csv"
    images_path = "./Images/Dev"
    
    # Read in metadata
    metadata_df = pd.read_csv(metadata_path)
    
    # Isolate ISIC id's
    id_array = metadata_df["image_id"].to_numpy()
    
    # Write filepaths and store in array
    filepath_array = images_path + "/" + id_array + ".jpg"
    
    # Append filepaths to and write to csv
    metadata_df["image_filepath"] = filepath_array
    
    metadata_df.to_csv("./Metadata/Dev/Dev.csv", index=False)
    
    
    

def combine_train_and_val():
    """
    Combines the train and validation metadata used in the original thesis code into one file.
    """
    train_df = pd.read_csv("./Metadata/train.csv")
    val_df = pd.read_csv("./Metadata/val.csv")
    
    res_df = pd.concat([train_df, val_df])
    
    res_df.to_csv("./Metadata/new/train.csv", index=False)
    
    

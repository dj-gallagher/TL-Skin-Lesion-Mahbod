from glob import glob
import cv2

        

def augment_images(image_dir):
    
    '''
    Create augmented images in the given directory
    '''
    
    # get list of images in train dir
    imagepath_list = glob(image_dir + "/*/*")
    
    # loop through each image in list
    for path in imagepath_list:
    
        # create each augmented image
        image = cv2.imread(path)
        
        aug_A = cv2.flip(src=image, flipCode=1) # flip horizontal
        
        aug_B = cv2.rotate(src=image, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        
        aug_C = cv2.rotate(src=image, rotateCode=cv2.ROTATE_180)
        
        aug_D = cv2.rotate(src=image, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        aug_E = cv2.flip(src=aug_B, flipCode=1)
        
        aug_F = cv2.flip(src=aug_C, flipCode=1)
        
        aug_G = cv2.flip(src=aug_D, flipCode=1)
        
        
        
        
        # save each augmented image with new name
        cv2.imwrite(filename=path[:-4] + '_A' + '.jpg', img=aug_A )
        cv2.imwrite(filename=path[:-4] + '_B' + '.jpg', img=aug_B )
        cv2.imwrite(filename=path[:-4] + '_C' + '.jpg', img=aug_C )
        cv2.imwrite(filename=path[:-4] + '_D' + '.jpg', img=aug_D )
        cv2.imwrite(filename=path[:-4] + '_E' + '.jpg', img=aug_E )
        cv2.imwrite(filename=path[:-4] + '_F' + '.jpg', img=aug_F )
        cv2.imwrite(filename=path[:-4] + '_G' + '.jpg', img=aug_G )
        
        
        

def submain():
    
    train_dir = "./Images/train"
    
    test_dir = "./Images/test"
    
    augment_images(train_dir)
    augment_images(test_dir)
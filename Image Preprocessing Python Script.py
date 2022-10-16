import pandas as pd
import numpy as np
import os
import cv2

def folder_to_csv(folder, classification, save = False):
    """Folder is a folder path, classification is the string classifiction. Save is a boolean
    folder_to_csv() returns a csv with the path of the images in the first column and the classification in the second column
    setting 'save' to true saves the csv to your drive with the name of the file as the name of the classification variable"""
    file_names = []
    for picture in os.listdir(folder):
        path = os.path.join(folder, picture)
        file_names.append([path, classification])
    df = pd.DataFrame(file_names, columns = ["File Path", "Classification"])
    if save == True:
        name = f'{classification}.csv'
        csv = df.to_csv(name, index = False)
    return df.to_csv(index = False)

def folder_to_array(folder, image_size, vectorlocation, save = False):
    """folder is a folder path, image_size is the image size of the post-processed picture, vectorlocation is the index of the one-hot vector
    folder_to_array() returns all of the images in the folder specified into an 2d array with column1 as the image array and column2 as the
    one-hot vector representing the data classification"""
    training_data = []
    for picture in os.listdir(folder):
        if "jpg" in picture:
            path = os.path.join(folder, picture)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (image_size, image_size))
            training_data.append([np.array(img), np.eye(4)[vectorlocation]]) #<- second index is the one-hot vector
    training_data = np.array(training_data, list)
    if save == True:
        np.save(folder, training_data)
    return

#example function calls:
#folder_to_csv(r"AugmentedAlzheimerDataset\VeryMildDemented", "verymilddemented", True)
#folder_to_array(r"AugmentedAlzheimerDataset\VeryMildDemented", "verymilddemented", 100, 0, True)
        



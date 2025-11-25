# galaxy-morphology
#Deep learning based Galaxy morphology (using CNN's)

#importing necessary libraries
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, GlobalMaxPooling2D
from tensorflow.keras.utils import to_categorical

#path to the dataset 
IMAGE_FOLDER=r"images_training_rev1\images_training_rev1"
CSV_PATH=r"training_solutions_rev1\training_solutions_rev1.csv"

#loading the dataset 
df=pd.read_csv(CSV_PATH)
print("CSV loaded:", df.shape)

# Add full image paths (GalaxyID.jpg)
df["image_path"] = df["GalaxyID"].astype(str) + ".jpg"
df["image_path"] = df["image_path"].apply(lambda x: os.path.join(IMAGE_FOLDER, x))

# Keep only existing images
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
print("Images matched:", df.shape)



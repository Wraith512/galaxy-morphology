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

# mapping the images to the classes
#Spiral → Class4.2
df["spiral_prob"] = df["Class4.2"]

#Elliptical (smooth) → Class1.1
df["elliptical_prob"] = df["Class1.1"]

#Irregular → combine 3 disturbance classes
df["irregular_prob"] = df["Class7.1"] + df["Class7.2"] + df["Class7.3"]
def assign_label(row):
    probs = [row["spiral_prob"], row["elliptical_prob"], row["irregular_prob"]]
    idx = np.argmax(probs)
    return ["spiral", "elliptical", "irregular"][idx]


df["label"] = df.apply(assign_label, axis=1)
label_map = {"spiral": 0, "elliptical": 1, "irregular": 2}
df["label_id"] = df["label"].map(label_map)

print("\nLabel Distribution:")
print(df["label"].value_counts())

#Keep only existing images
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
print("Images matched:", df.shape)

# the function to show 5 random galaxy images
def show_random_images(df, sample=5):
    sample = df.sample(min(sample, len(df)))
    plt.figure(figsize=(16,5))
    for i, row in enumerate(sample.itertuples()):
        img = imread(row.image_path)
        plt.subplot(1, sample.shape[0], i+1)
        plt.imshow(img)
        plt.title(os.path.basename(row.image_path))
        plt.axis("off")
    plt.show()

print("random sample images:")
show_random_images(df, 5)

# splitting the dataset into train_test split 80:20 ratio
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label_id"])


print("Train:", len(train_df), "| Test:", len(test_df))

# now resizing the images from 484 X 484 pixel size to 64 X 64 pixel size for better computation while retaining the data
IMG_SHAPE = (64, 64)


def load_and_resize(path):
    try:
        img = imread(path)
        img = resize(img, IMG_SHAPE, anti_aliasing=True)
        return img
    except:
        return None

def prepare_images(df):
    X, y = [], []
    for row in tqdm(df.itertuples(), total=len(df), desc="Resizing"):
        img = load_and_resize(row.image_path)
        if img is not None:
            X.append(img)
            y.append(row.label_id)
    return np.array(X), np.array(y)

print("\nResizing training images...")
X_train, y_train = prepare_images(train_df)

print("\nResizing test images...")
X_test, y_test = prepare_images(test_df)


print("Shapes:", X_train.shape, X_test.shape)

# one hot encoding
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat  = to_categorical(y_test,  num_classes=3)

print("y_train_cat shape:", y_train_cat.shape)
print("y_test_cat shape :", y_test_cat.shape)

# building the cnn model from scratch
model = Sequential()

#block 1 
model.add(Conv2D(512, (3,3), activation="relu", input_shape=(64,64,3)))
model.add(Conv2D(256, (3,3), activation="relu"))
model.add(MaxPooling2D())

#block 2 
model.add(Conv2D(256, (3,3), activation="relu"))
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(MaxPooling2D())

#block 3
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(GlobalMaxPooling2D())

#dense layers
model.add(Dropout(0.25))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))

#output layer
model.add(Dense(3, activation="softmax"))

#compile
model.compile(
    loss="categorical_crossentropy",
    optimizer="adamax",
    metrics=["accuracy"]
)

model.summary()

# building an optimizer to optimize the the results and training the data






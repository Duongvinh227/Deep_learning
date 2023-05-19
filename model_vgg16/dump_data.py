from os import listdir
import cv2
import numpy as np
import pickle
import os

# Process the raw image file in the train_data folder and save it to the data file
# Purpose for faster data loading and more convenient handling

raw_folder = "train_data"
dest_size = (128, 128)

print("Start image processing...")

pixels = []
labels = []


# Loop through subfolders in raw folder
for folder in listdir(raw_folder):
    if folder[0] != ".":  # Ignore invalid directories
        print("Processing folder ", folder)

        # Loop through files in each containing folder
        for file in listdir(os.path.join(raw_folder,folder)):
            if file[0] != '.':
                print("---- Processing file = ", file)
                pixels.append(cv2.resize(cv2.imread(os.path.join(raw_folder, folder, file)), dsize=dest_size))
                labels.append(folder)

pixels = np.array(pixels)
labels = np.array(labels)

# Processing labels of data
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
labels = encoder.fit_transform(labels)
print(encoder.classes_)

# Save data to file data.pkl
file = open('data.pkl', 'wb')
pickle.dump((pixels, labels), file)
file.close()

# Save the labels to the file labels.pkl
file = open('labels.pkl', 'wb')
pickle.dump(encoder, file)
file.close()
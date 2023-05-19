from sklearn.model_selection import train_test_split
import os
import random

import tensorflow as tf
import cv2
import numpy as np

# Import library segmentation_models
from segmentation_models.metrics import iou_score
from segmentation_models import Unet
import segmentation_models as sm
sm.set_framework("tf.keras")
sm.framework()

# define variable
data_path  = "dataset"
w, h = 640, 640
batch_size = 2

# Dataset va Dataloader

BACKBONE = "resnet34"
preprocess_input = sm.get_preprocessing(BACKBONE)

# Use to create data and load in batches
class Dataset:
    def __init__(self, image_path, mask_path, w, h):
        self.image_path = image_path
        self.mask_path = mask_path
        self.w = w
        self.h = h

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.image_path[i])
        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_AREA)
        image = preprocess_input(image)

        mask = cv2.imread(self.mask_path[i], cv2.IMREAD_UNCHANGED)
        image_mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_AREA)

        image_mask = [(image_mask == v) for v in [1]]
        image_mask = np.stack(image_mask, axis=-1).astype('float')

        return image, image_mask

class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size,shape, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shape = shape
        self.indexes = np.arange(self.shape)

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return tuple(batch)

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

#  Load folder dataset create 2 variable image_path, mask_path
def load_path(data_path):
    # Get normal image and mask image NG
    classes = ['class', 'class_NG']

    # iterate through directories without error
    normal_image_path = []
    normal_mask_path = []
    for class_ in classes:
        current_folder = os.path.join(data_path, class_)
        for file in os.listdir(current_folder):
            if file.endswith("jpg") and (not file.startswith(".")):
                image_path = os.path.join(current_folder, file)
                mask_path = os.path.join(current_folder + "_mask", file)
                normal_mask_path.append(mask_path)
                normal_image_path.append(image_path)

    # Get defect image and mask
    defect_image_path = []
    defect_mask_path = []
    for class_ in classes:
        class_ = class_ + "_NG"
        current_folder = os.path.join(data_path, class_)
        for file in os.listdir(current_folder):
            if file.endswith("jpg") and (not file.startswith(".")):
                image_path = os.path.join(current_folder, file)
                mask_path = os.path.join(current_folder + "_mask", file)
                defect_mask_path.append(mask_path)
                defect_image_path.append(image_path)

    idx = random.sample(range(len(normal_mask_path)), len(defect_mask_path))

    normal_mask_path_new = []
    normal_image_path_new = []

    for id in idx:
        normal_image_path_new.append(normal_image_path[id])
        normal_mask_path_new.append(normal_mask_path[id])

    image_path = normal_image_path_new + defect_image_path
    mask_path = normal_mask_path_new + defect_mask_path

    return image_path, mask_path

# Thu hien load va train model

# Load path into 2 variables
image_path, mask_path = load_path(data_path)

# Divide dataset train 0.8 , test 0.2
image_train, image_test, mask_train, mask_test = train_test_split(image_path, mask_path, test_size=0.2)

# create dataset va dataloader
train_dataset = Dataset(image_train, mask_train, w, h)
test_dataset = Dataset(image_test, mask_test, w, h)

train_loader = Dataloader(train_dataset, batch_size, shape=len(image_train), shuffle=True)
test_loader = Dataloader(test_dataset, batch_size, shape=len(image_test), shuffle=True)

# initialization model
opt=tf.keras.optimizers.Adam(0.001)
model= Unet(BACKBONE,encoder_weights="imagenet",classes=1,activation="sigmoid",input_shape=(640,640,3),encoder_freeze=True)
loss1 = sm.losses.categorical_focal_dice_loss
model.compile(optimizer=opt,loss=loss1,metrics=[iou_score])

# Train model
is_train = True
if is_train:
    from keras.callbacks import ModelCheckpoint
    filepath = "checkpoint.hdf5"
    callback = ModelCheckpoint(filepath, monitor='val_iou_score', verbose=1, save_best_only=True,mode='max')

    model.fit(train_loader, validation_data=test_loader, epochs=50, callbacks=[callback])

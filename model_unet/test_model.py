import os

import cv2
import numpy as np

import tensorflow as tf
from segmentation_models.metrics import iou_score
from segmentation_models import Unet
import segmentation_models as sm
def detection(img, file_name):
    BACKBONE = "resnet34"

    # create model
    opt=tf.keras.optimizers.Adam(0.001)
    model= Unet(BACKBONE,encoder_weights="imagenet",classes=1,activation="sigmoid",input_shape=(640,640,3),encoder_freeze=True)
    loss1 = sm.losses.categorical_focal_dice_loss
    model.compile(optimizer=opt,loss=loss1,metrics=[iou_score])

    # Load model from checkpoint.hdf5
    model.load_weights("checkpoint.hdf5")

    img = cv2.resize(img, (640,640))
    mask_predict = model.predict(img[np.newaxis, :, :, :])
    mask_binary = (mask_predict[0] > 0.5).astype(np.uint8)
    mask_rgb = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2RGB)

    if cv2.countNonZero(mask_binary) < 200:
        name = "OK"
    else:
        name = "NG"

    # img_mask = mask_predict[0]
    img_mask = cv2.addWeighted(img, 1, mask_rgb * 255, 0.6, 0)

    cv2.putText(img_mask, name, (25, 25), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)

    # plt.imshow(z)
    # plt.show()
    cv2.imshow(f'{file_name}', img)
    cv2.imshow('predict', img_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_mask

import random

def load_file():
    image_folder = './dataset/images'
    image_result = './dataset/result'
    for filename_image in os.listdir(image_folder):
        img = cv2.imread(os.path.join(image_folder, filename_image))
        print(filename_image)
        img_mask = detection(img, filename_image)
        cv2.imwrite(f'{image_result}/{filename_image}', img_mask)

load_file()
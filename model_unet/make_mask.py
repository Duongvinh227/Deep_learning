import json
import os
import shutil

import cv2
import numpy as np
from PIL import Image

from skimage.draw import ellipse

data_path = "dataset"

#  made a map for the boys who didn't feel sorry
def make_map_normal(data_path):
    # Loop through normal folder
    for folder in os.listdir(data_path):
        if (not folder.endswith("NG")) and (not folder.startswith(".")) and (not folder.endswith("mask")):
            print("*" * 10, folder)
            # Make mask folder
            mask_folder = folder + "_mask"
            mask_folder = os.path.join(data_path, mask_folder)
            try:
                shutil.rmtree(mask_folder)
            except:
                pass

            os.mkdir(mask_folder)

            # Loop through file in current folder:
            current_folder = os.path.join(data_path, folder)
            for file in os.listdir(current_folder):
                if file.endswith("jpg"):
                    print(file)
                    # Read image file
                    current_file = os.path.join(current_folder, file)
                    image = cv2.imread(current_file)
                    w, h = image.shape[0], image.shape[1]

                    # Make mask file for normal product - it's blank image, no defect
                    mask_image = np.zeros((w, h), dtype=np.uint8)
                    mask_image = Image.fromarray(mask_image)

                    # Save the file
                    mask_image.save(os.path.join(mask_folder, file))

def draw_NG(file, labels, w, h):
    # get file id
    file_id = int(file.replace(".jpg", ""))

    # get label of file
    label = labels[file_id - 1]

    # Separation of ingredients in a label
    label = label.replace("\t", "").replace("  ", " ").replace("  ", " ").replace("\n", "")
    label_array = label.split(" ")

    # draw ellipse
    major, minor, angle, x_pos, y_pos = float(label_array[1]), float(label_array[2]), float(label_array[3]), float(
        label_array[4]), float(label_array[5])
    rr, cc = ellipse(y_pos, x_pos, r_radius=minor, c_radius=major, rotation=-angle)

    # make black photo ( w, h)
    mask_image = np.zeros((w, h), dtype=np.uint8)

    try:
        mask_image[rr, cc] = 1
    except:
        rr_n = [min(511, rr[i]) for i in rr]
        cc_n = [min(511, cc[i]) for i in cc]
        mask_image[rr_n, cc_n] = 1
        # mask_image = Image.fromarray(mask_image)

    # convert image
    mask_image = np.array(mask_image, dtype=np.uint8)
    mask_image = Image.fromarray(mask_image)

    return mask_image

def make_map_NG(data_path):
    # Loop through defect folder
    for folder in os.listdir(data_path):
        if (folder.endswith("NG")) and (not folder.startswith(".")):
            print("*" * 10, folder)

            # Make mask folder
            mask_folder = folder + "_mask"
            mask_folder = os.path.join(data_path, mask_folder)
            try:
                shutil.rmtree(mask_folder)
            except:
                pass

            os.mkdir(mask_folder)

            # Loop through file in current folder:
            current_folder = os.path.join(data_path, folder)

            # Load txt file
            f = open(os.path.join(current_folder, 'labels.txt'))
            labels = f.readlines()
            f.close()

            for file in os.listdir(current_folder):
                if file.find("(") > -1:
                    # Xoá file nếu bị trùng (do đặc thù dữ liệu)
                    os.remove(os.path.join(current_folder, file))
                    continue

                if file.endswith("jpg"):
                    print(file)
                    # Read image file
                    current_file = os.path.join(current_folder, file)
                    image = cv2.imread(current_file)
                    w, h = image.shape[0], image.shape[1]

                    # Make mask file for defect product - it's blank image with defect
                    mask_image = draw_NG(file, labels, w, h)

                    # Save the file
                    mask_image.save(os.path.join(mask_folder, file))

def draw_NG_format_VGG(filename):

    with open("labels_vgg_ng.json") as f:
        data = json.load(f)

    img = cv2.imread(f"./dataset/class_NG/{filename}")

    try:
        # Generate mask from shape polygons defined in JSON file ( vgg )
        mask = np.zeros(img.shape[:2], np.uint8)
        for region in data[f"{filename}"]["regions"].values():
            pts_x = region["shape_attributes"]["all_points_x"]
            pts_y = region["shape_attributes"]["all_points_y"]
            pts = np.array(list(zip(pts_x, pts_y)))
            pts = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts], color=(1, 1, 1))
        return mask
    except:
        pass


def make_map_format_VGG(data_path):
    for folder in os.listdir(data_path):
        if (folder.endswith("NG")) and (not folder.startswith(".")):
            print("*" * 10, folder)

            # Make mask folder
            mask_folder = folder + "_mask"
            mask_folder = os.path.join(data_path, mask_folder)
            try:
                shutil.rmtree(mask_folder)
            except:
                pass

            os.mkdir(mask_folder)
            print(mask_folder)
            for filename_NG in os.listdir(f"./{data_path}/{folder}"):
                print(filename_NG)
                image_mask = draw_NG_format_VGG(filename_NG)
                try:
                    cv2.imwrite(f'./{mask_folder}/{filename_NG}', image_mask)
                except:
                    pass

make_map_format_VGG(data_path)
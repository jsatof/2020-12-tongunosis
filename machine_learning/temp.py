import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import randint
import numpy as np
from  matplotlib import cm
import json
import cv2
import glob
from app import TongueImages, UNet
from images import augment_image
import pdb
import datetime
import pickle

from masks import combine_image_mask

def image_dir_diff():
    image_path = "train_images"
    images = [os.path.splitext(file)[0].replace("_train", "") for file in os.listdir(image_path)]

    mask_path = "image_masks"
    masks = [os.path.splitext(file)[0].replace("_mask", "") for file in os.listdir(mask_path)]

    diff_list = list(set(images) - set(masks))
    
    if len(diff_list) == 0:
        print("0 differences")
    else:
        print("{} differences:".format(len(diff_list)))
        for i in diff_list:
            print(i)

def populate_coords(json_data, object_list):
    for key, value in json_data["_via_img_metadata"].items():
        filename = value["filename"]
        if not len(value["regions"]) == 0:
            x = value["regions"][0]["shape_attributes"]["all_points_x"]
            y = value["regions"][0]["shape_attributes"]["all_points_y"]
            coords = [[x[i], y[i]] for i in range(len(x))]
            
            item = dict(filename = filename, coords = coords)

            if "_flip" in filename:
                objects.append(item)

def merge_image_mask():
    mask = Image.open("image_temp/mask.13396__P_flip.jpg")
    mask.load()
    og_image = Image.open("train_images/13396__P_flip.jpg")
    og_image.load()

    out_image = Image.composite(og_image, mask, mask)
    out_image.save("temp_images/mask_test.jpg")

    # Create Mask images
    """
    mask = np.zeros((256,256))
    points = np.array(item["coords"], dtype = np.int32)
    cv2.fillPoly(mask, [points], color = (255, 255, 255))
    temp_filename = "image_temp/" + "mask." + item["filename"]
    cv2.imwrite(temp_filename, mask)
    """

def get_distance_x(x, axis):
    return axis - x

def horiztonal_flip(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def load_pth_weights(model):
    path = "/home/jsferrarelli/develop/coop/output/"
    list_files = glob.glob(path + "save_point/*.pth")
    latest_pth = max(list_files, key = os.path.getctime)

    model.load_state_dict(torch.load(latest_pth))
    
    dir_path = os.path.splitext(latest_pth)[0]
    file_tag = dir_path[-6:]

    output_file = file_tag + ".pkl"
    
    with open(path + "export/" + output_file, "wb") as f:
        pickle.dump(model.state_dict(), f)

from app import UNet

if __name__ == "__main__":
    model = None
    
    with open("../../coop/output/export/epoch5.pkl", "rb") as f:
        model = pickle.load(f)

    model.predict(np.zeros(1, 1, 1, 1))
    
    
    
    """items = []

    train_path = "train_images/"
    mask_path = "image_masks/"

    pth_file_list = glob.glob("output/save_point/*.pth")
    latest_pth = max(pth_file_list, key = os.path.getctime)
    print(latest_pth)"""

    """for name in os.listdir(train_path):
        if "original" in name:
            new_train_image_name = name.replace(".jpg", "") + "_flip.jpg"
            new_mask_image_name = "mask." + name.replace(".jpg", "") + "_flip.jpg"
            
            image = Image.open(train_path + name)
            mask =  Image.open(mask_path + "mask." + name)

            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            image.save(train_path + new_train_image_name)
            mask.save(mask_path + new_mask_image_name) """
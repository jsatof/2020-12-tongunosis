import torch
import torchvision
import numpy as np
from PIL import Image
import cv2
import os
from matplotlib import cm
from random import randint

def rename_images():
    for filename in os.listdir("train_images"):
        if "target" in filename:
            new_name = filename.replace("target", "train")
            os.rename("train_images/" + filename, "train_images/" + new_name)

def get_max_dims():
    max_height = 0
    max_height_name = ""
    max_width = 0
    max_width_name = ""

    for filename in os.listdir("train_targets"):
        image = Image.open("train_targets/" + filename)
        width, height = image.size
        if max_height < width:
            max_height = height
            max_height_name = filename
        if max_width < width:
            max_width = width
            max_width_name = filename
        image.close()

    print("Max Width: {}\t({})".format(max_width, max_width_name))
    print("Max Height: {}\t({})".format(max_height, max_height_name))

# returns the scaled dims of image, so that one dim is 256
def get_new_dims(image):
    desired_dim = 256
    scale_factor = 0.
    width, height = image.size

    if height > width:
        scale_factor = float(desired_dim / height)
    else:
        scale_factor = float(desired_dim / width)

    new_width = round(width * scale_factor)
    new_height = round(height * scale_factor)

    return new_width, new_height

# resize
def resize_images():
    for name in os.listdir("train_images"):
        path = "train_images/" + name
        image = Image.open(path)
        new_width, new_height = get_dims(image)
        new_image = image.resize((new_width, new_height))
        new_image.save("image_temp/script_test/" + name)

def resize_image(image):
    desired_dim = 256
    width, height = image.size
    scale = 0.
    if height > width:
        scale = float(desired_dim / height)
    else:
        scale = float(desired_dim / width)

    new_height = round(height * scale)
    new_width = round(width * scale)
    new_image = image.resize((new_width, new_height))
    return new_image

def pad_image(image):
    # assuming image has one dim = 256
    zero_matrix = np.zeros((256, 256))
    padded_image = Image.fromarray(np.uint8(cm.gist_earth(zero_matrix) * 255)) # convert np array to PIL image
    width, height = image.size
    x = 0
    y = 0
    if width == 256:
        y = randint(0, 256 - height)
    elif height == 256:
        x = randint(0, 256 - width)
    
    padded_image.paste(image, (x, y))
    return padded_image
    
# uses the resize_image() and pad_image() 
def augment_image(image):
    resized_image = resize_image(image)
    padded_image = pad_image(resized_image)
    if padded_image:
        return padded_image

def convert_to_png(image):
    pass

def horizontal_flip(image):
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    return flipped

def main():
    for filename in os.listdir("image_temp"):
        if ".jpg" in filename:
            image = Image.open("image_temp/" + filename)
            image.load()
            new_image = horizontal_flip(image)
            new_image.save("image_temp/script_test/" + "flip_" + filename, format = "PNG")
        

    

if __name__ == "__main__":
    main()

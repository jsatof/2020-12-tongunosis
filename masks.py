import json
from PIL import Image
import cv2
import numpy as np
import os

def get_mask_json(jsonfile):
    with open(jsonfile) as f:
        data = json.load(f)
        return data

def populate_items(json_data, object_list):
    for key, value in json_data["_via_img_metadata"].items():
        filename = value["filename"]
        if not len(value["regions"]) == 0:
            x = value["regions"][0]["shape_attributes"]["all_points_x"]
            y = value["regions"][0]["shape_attributes"]["all_points_y"]
            coords = [(x[i], y[i]) for i in range(len(x))]
            
            item = dict(filename = filename, coords = coords)
            object_list.append(item)

def create_mask_image(object_list):
    for item in object_list:
        mask = np.zeros((256, 256))
        points = np.array([item["coords"]], dtype = np.int32)
        cv2.fillPoly(mask, points, color = (255, 255, 255))
        new_name = "eval_input/" + "mask." + item["filename"]
        cv2.imwrite(new_name, mask)

def combine_image_mask(image_path, mask_path, out_path):
    for mask_name in os.listdir(mask_path):
        file_id = mask_name.replace("mask.", "")
        og_image = Image.open(image_path + file_id)
        mask = Image.open(mask_path + mask_name)
        output_image = Image.composite(og_image, mask, mask)
        output_image.save(out_path + "out." + file_id)
        
def main():
    json_file = "dr_liu_masks.json"
    json_data = get_mask_json(json_file)

    objects = []

    # retyped the populate_items() function, maybe passing the objects array didnt work for some reason
    for key, value in json_data["_via_img_metadata"].items():
        filename = value["filename"]
        if not len(value["regions"]) == 0:
            x = value["regions"][0]["shape_attributes"]["all_points_x"]
            y = value["regions"][0]["shape_attributes"]["all_points_y"]
            coords = [(x[i], y[i]) for i in range(len(x))]

            item = dict(filename = filename, coords = coords)
            objects.append(item)

    create_mask_image(objects)

if __name__ == "__main__":
    main()

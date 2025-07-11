import os
from pathlib import Path
from PIL import Image

base_dir = Path("dataset")


def yoloprediction_to_pdlpr_input(x_center, y_center, width, height, image_path):
    #This functions takes in input the prediction from yolo and returns the cropped image (so the input for pdlpr)
    img = Image.open(image_path)

    image_width, image_height = img.size

    x_center_pixel = x_center * image_width
    y_center_pixel = y_center * image_height
    width_pixel = width * image_width
    height_pixel = height * image_height

    x_min = int(x_center_pixel - width_pixel / 2)
    x_max = int(x_center_pixel + width_pixel / 2)
    y_min = int(y_center_pixel - height_pixel / 2)
    y_max = int(y_center_pixel + height_pixel / 2)

    #crop the image according to the bounding box coordinates
    cropped_img = img.crop((x_min, y_min, x_max, y_max))

    return cropped_img

import os
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from globals import *


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

def compute_iou(box_1, box_2):
    #it is a metric that involves the intersection of the two areas
    #over the union, and returns a matching percentage
    
    #coputing the coordinate of the intersections
    x1 = max(box_1[0], box_2[0])
    y1 = max(box_1[1], box_2[1])
    x2 = min(box_1[2], box_2[2])
    y2 = min(box_1[3], box_2[3])

    interArea = max(0, x2 - x1) * max(0, y2 - y1)
    boxAArea = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    boxBArea = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)  #+1e-6 is used to avoid the division per zero
    return iou


def target_to_index(target_list):
    #this function converts the id of the target 
    #into the ids that are returned by the model
    #since it uses the ids from the sorted list of 
    #all the possible characters
    output = []
    province = PROVINCES[target_list[0]]
    alphabet = ALPHABETS[target_list[1]]
    output.append(CHAR_IDX[province])
    output.append(CHAR_IDX[alphabet])
    for char_idx in range(2,8):
        char = ADS[char_idx]
        output.append(CHAR_IDX[char])
    return output


def index_to_target(index_list):
    output=[]
    for idx in index_list:
        output.append(index_list[idx])
    return output


def load_gt_box_from_label_validation(image_path):
    """
    Load the ground truth box from a YOLO-format label file.
    Returns [x1, y1, x2, y2] or None if label file is missing/invalid.
    """
    label_path = Path("dataset/labels/val") / (image_path.stem + ".txt")

    if not label_path.exists():
        print(f"[WARN] No label found for {image_path.name}")
        return None

    with open(label_path, "r") as f:
        lines = f.readlines()

    if not lines:
        print(f"[WARN] Empty label file for {image_path.name}")
        return None

    # Assume the first object only
    try:
        parts = list(map(float, lines[0].strip().split()))
        _, x_center, y_center, w, h = parts
    except Exception:
        print(f"[WARN] Label parse failed for {image_path.name}")
        return None

    # Convert from normalized to absolute coordinates
    img = plt.imread(image_path)
    img_h, img_w = img.shape[:2]

    cx, cy = x_center * img_w, y_center * img_h
    bw, bh = w * img_w, h * img_h

    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy + bh / 2

    return [x1, y1, x2, y2]


def load_gt_box_from_label_test(image_path):
    """
    Load the ground truth box from a YOLO-format label file.
    Returns [x1, y1, x2, y2] or None if label file is missing/invalid.
    """
    label_path = Path("dataset/labels/test") / (image_path.stem + ".txt")

    if not label_path.exists():
        print(f"[WARN] No label found for {image_path.name}")
        return None

    with open(label_path, "r") as f:
        lines = f.readlines()

    if not lines:
        print(f"[WARN] Empty label file for {image_path.name}")
        return None

    # Assume the first object only
    try:
        parts = list(map(float, lines[0].strip().split()))
        _, x_center, y_center, w, h = parts
    except Exception:
        print(f"[WARN] Label parse failed for {image_path.name}")
        return None

    # Convert from normalized to absolute coordinates
    img = plt.imread(image_path)
    img_h, img_w = img.shape[:2]

    cx, cy = x_center * img_w, y_center * img_h
    bw, bh = w * img_w, h * img_h

    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy + bh / 2

    return [x1, y1, x2, y2]

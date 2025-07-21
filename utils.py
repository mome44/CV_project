import os
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from globals import *
from tqdm import tqdm

base_dir = Path("dataset")

def initialize_labels():
    #initialize the labels and creating the folders
    splits = ["train", "val", "test"]
    for split in splits:
        # It is just a safe and readable way to say: go to datasets/ccpd/images/train (or val, or test), depending on which split you're processing.
        image_dir = base_dir / "images" / split
        label_dir = base_dir / "labels" / split
        crops_dir = base_dir / "crops" / split
        label_pdlpr_dir = base_dir / "labels_pdlpr" / split
    
        label_dir.mkdir(parents=True, exist_ok=True)    # creates the folder if it does not exist
        crops_dir.mkdir(parents=True, exist_ok=True)
        label_pdlpr_dir.mkdir(parents=True, exist_ok=True)
        
        # Loop through all .jpg images in the current image directory
        #the tqdm library is useful to plot the loading bar
        for image_path in tqdm(list(image_dir.glob("*.jpg")), desc=f"Processing - {split}", unit="img"):
            #print(f"Found image: {image_path}")
            #print(f"Processing: {image_path.name}")
    
            # Parse bounding box from filename: example => "XXXXX&x1_x2_y1_y2&..."
            try:
                fields = image_path.stem.split("-")    # image_path.stem is the filename without .jpg
                
                # Field 2 (index 2) is bbox: format is "x1&y1_x2&y2"
                bbox_part = fields[2]
                corners = bbox_part.split("_")
                x1, y1 = map(int, corners[0].split("&"))
                x2, y2 = map(int, corners[1].split("&"))
    
                # Define min/max values
                x_min = min(x1, x2)
                x_max = max(x1, x2)
                y_min = min(y1, y2)
                y_max = max(y1, y2)
                
                
                #extracting the information about the plate to create the labels for pdlpr
                #the plate is in this format 0_0_22_27_27_33_16
                plate_number = fields[4]
                character_id_list = plate_number.split("_")
                #get the number for the province and for the letter
                province_id = int(character_id_list[0])
                alphabet_id = int(character_id_list[1])
                #get the actual character for both and join them
                province_char = PROVINCES[province_id]
                alphabet_char = ALPHABETS[alphabet_id]
                plate = province_char + alphabet_char
    
                for i in range(2, 8):
                    #for the remaining 5 characters we do the mapping from the ADS
                    ads_index = int(character_id_list[i])
                    plate += ADS[ads_index]
                
            except Exception as e:
                print(f"Skipping {image_path.name}: {e}")
                continue
            # Read the image to get image size (needed to normalize the coordinates)
            img = Image.open(image_path)
    
            img_width, img_height = img.size
    
            #crop the image according to the bounding box coordinates
            cropped_img = img.crop((x_min, y_min, x_max, y_max))
    
            #Adding crops so cut images into a separate folder
            crops_path = crops_dir / (image_path.stem + ".jpg")
    
            #saving the image into the crops folder
            cropped_img.save(crops_path)
    
            img.close()
    
    
            # Normalize the bounding box for YOLO format
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
    
            # Create YOLO label string
            # 0 is the class ID (only one class - license plate)
            # the rest are floats with 6 digits after the decimal point
            label_str = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
            # Save label file with same name
            label_path = label_dir / (image_path.stem + ".txt")
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(label_str + "\n")
    
            #print(f"Wrote label: {label_path.name}")
    
            #Save the label for PDLPR
            label_pdl_pr_path = label_pdlpr_dir / (image_path.stem + ".txt")
            with open(label_pdl_pr_path, "w", encoding="utf-8") as f:
                f.write(plate + "\n")

    
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
        char = ADS[target_list[char_idx]]
        output.append(CHAR_IDX[char])
    return output


def index_to_target(index_list):
    output=[]
    for idx in index_list:
        output.append(IDX_CHAR[idx])
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

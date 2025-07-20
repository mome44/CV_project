import os
from pathlib import Path
from PIL import Image       # used to read image size
from tqdm import tqdm

# Base directory containing images and labels
base_dir = Path("dataset")

# Subsets to be processed
splits = ["train", "val", "test"]

#character mapping used for the chinese plates
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


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

        


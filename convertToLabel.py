import os
from pathlib import Path
from PIL import Image       # used to read image size

# Base directory containing images and labels
base_dir = Path("dataset")

# Subsets to be processed
splits = ["train", "val", "test"]

for split in splits:
    # It is just a safe and readable way to say: go to datasets/ccpd/images/train (or val, or test), depending on which split you're processing.
    image_dir = base_dir / "images" / split
    label_dir = base_dir / "labels" / split

    label_dir.mkdir(parents=True, exist_ok=True)    # creates the folder if it does not exist

    # Loop through all .jpg images in the current image directory
    for image_path in image_dir.glob("*.jpg"):
        print(f"Found image: {image_path}")
        print(f"Processing: {image_path.name}")

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
        
        except Exception as e:
            print(f"Skipping {image_path.name}: {e}")
            continue

        # Read the image to get image size (needed to normalize the coordinates)
        with Image.open(image_path) as img:
            img_width, img_height = img.size

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
        with open(label_path, "w") as f:
            f.write(label_str + "\n")

        print(f"Wrote label: {label_path.name}")


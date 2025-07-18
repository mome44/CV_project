import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path
from torch.utils.data import DataLoader
from globals import BATCH_SIZE_TRAIN_Y, BATCH_SIZE_TEST_Y


class CCPDDataset(Dataset):
    # This class encapsulates the logic for data loading and pre-processing

    def __init__(self, base_dir, transform=None):
        # Sets up the paths for images, YOLO labels, PDLPR labels, and cropped images,
        # organizing access to your pre-processed data

        self.base_dir = Path(base_dir)                                          # string or Path, is the base directory of your processed dataset ("dataset")
        self.transform = transform                                              # optional transform to be applied on an image  --> keep or remove ??


    def get_dataset(self, split):
        self.image_dir = self.base_dir / "images" / split
        self.label_yolo_dir = self.base_dir / "labels" / split
        self.label_pdlpr_dir = self.base_dir / "labels_pdlpr" / split
        self.crops_dir = self.base_dir / "crops" / split                        # directory for cropped plate images
        
        
        # List all image files in the current split's image directory
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")))
        
        # List the cropped image files if PDLPR needs them directly
        self.cropped_image_files = sorted(list(self.crops_dir.glob("*.jpg")))

        # Basic validation to ensure files exist
        if not self.image_files:
            raise FileNotFoundError(f"No .jpg images found in {self.image_dir}")
        if not self.cropped_image_files:
            raise FileNotFoundError(f"No cropped .jpg images found in {self.crops_dir}")

        return self


    def __len__(self):
        # Returns the total number of samples in the dataset
        return len(self.image_files)
    
    
    def __getitem__(self, index):
        # Retrieves a single sample given an index

        # Load full image and its YOLO label
        img_path = self.image_files[index]
        yolo_label_path = self.label_yolo_dir / (img_path.stem + ".txt")

        # Load cropped image and its PDLPR label
        cropped_img_path = self.cropped_image_files[index]
        pdlpr_label_path = self.label_pdlpr_dir / (img_path.stem + ".txt")


        # Open images
        # Ensure 'RGB' conversion if images might be grayscale to be consistent for models
        # This helps with GPU optimization as models typically expect 3 channels
        full_image = Image.open(img_path).convert("RGB")
        cropped_image = Image.open(cropped_img_path).convert("RGB")

        # Read YOLO label (bounding box) from the text file
        with open(yolo_label_path, "r", encoding="utf-8") as f:
            yolo_label_str = f.readline().strip()                           # strip() removes any whitespace char like \n
        
        # Check if the label file is empty or malformed
        if not yolo_label_str:  
            raise ValueError(f"Empty label in {yolo_label_path}")

        # Assuming YOLO format: "class_id x_center y_center width height"
        # We only have one class (0), so we can discard it or keep it

        parts = list(map(float, yolo_label_str.split()))                    # parts is a list of floats like [0.0, 0.5, 0.4, 0.3, 0.1]
        class_id = int(parts[0])
        yolo_bbox = torch.tensor(parts[1:], dtype=torch.float32)            # discard the first element (class) --> [x_center, y_center, width, height]
                                                                            # convert the list of floats into a PyTorch tensor

        # Read PDLPR label (license plate string)
        with open(pdlpr_label_path, "r", encoding="utf-8") as f:
            pdlpr_plate_str = f.readline().strip()

        # Apply transformations if any --> remove or keep ??
        if self.transform:
            full_image = self.transform(full_image)
            cropped_image = self.transform(cropped_image)

        # For the PDLPR model, you'll likely need to convert the plate string into a sequence of numerical IDs.
        # This part depends on how your PDLPR model expects its input labels.
        # Example (you'll need to define mappings if not already done):
        # char_to_id = {char: i for i, char in enumerate(PROVINCES + ALPHABETS + ADS)} # You need to consolidate your character sets
        # pdlpr_label_encoded = torch.tensor([char_to_id[char] for char in pdlpr_plate_str], dtype=torch.long)

        # For now, we'll return the string. You'll need to implement the encoding based on your PDLPR model.
        return {
            'full_image': full_image,
            'cropped_image': cropped_image,
            'yolo_bbox_label': yolo_bbox,
            'pdlpr_plate_string': pdlpr_plate_str, # You'll likely convert this to numerical IDs later
            'image_name': img_path.name
        }


    def get_dataloaders(base_dir, batch_size = 8, transform = None):

        ds = CCPDDataset(base_dir=base_dir, transform=transform)

        train_loader = DataLoader(ds.get_dataset("train"), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ds.get_dataset("val"), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(ds.get_dataset("test"), batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


import os
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision import transforms
from globals import BATCH_SIZE_TRAIN_Y, BATCH_SIZE_TEST_Y, PROVINCES, ALPHABETS, ADS, CHAR_LIST
from utils import target_to_index

class CCPDDataset(Dataset):
    #This class helps to manage the elements from the CCPDD dataset
    #and also initialized the batchloaders used during the training test and validation phases

    def __init__(self, base_dir, transform=None):

        self.base_dir = Path(base_dir)                                         
        self.transform = transform                                            

    def get_dataset(self, split):
        #this is used to set the split from which we are initializing the dataset
        #like train, test, val

        self.image_dir = self.base_dir / "images" / split
        self.label_yolo_dir = self.base_dir / "labels" / split
        #directories for the labels and cropped images that are going to be used 
        #in the character recognition part with pdlpr
        self.crops_dir = self.base_dir / "crops" / split 
        self.label_pdlpr_dir = self.base_dir / "labels_pdlpr" / split
                  
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
    
    def get_image_names(self):
        return self.image_files


    def __len__(self):
        # Returns the total number of samples in the dataset
        return len(self.image_files)
    
    
    def __getitem__(self, index):
        # Retrieves a single data sample given the index
        
        #initializes the paths
        
        img_path = self.image_files[index]
        yolo_label_path = self.label_yolo_dir / (img_path.stem + ".txt")
        cropped_img_path = self.cropped_image_files[index]
        pdlpr_label_path = self.label_pdlpr_dir / (img_path.stem + ".txt")

        img_name = img_path.name
        
        # Open images
        # Ensure 'RGB' conversion if images might be grayscale to be consistent for models
        # This helps with GPU optimization as models typically expect 3 channels
        full_image = Image.open(img_path).convert("RGB")
        cropped_image = Image.open(cropped_img_path).convert("RGB")

        # Read YOLO label (bounding box) from the text file
        with open(yolo_label_path, "r", encoding="utf-8") as f:
            yolo_label_str = f.readline().strip()        
        
        # Check if the label file is empty or malformed
        if not yolo_label_str:  
            raise ValueError(f"Empty label in {yolo_label_path}")

        # Assuming YOLO format: "class_id x_center y_center width height"
        # We only have one class (0), so we can discard it or keep it

        parts = list(map(float, yolo_label_str.split()))
        # parts is a list of floats like [0.0, 0.5, 0.4, 0.3, 0.1]                    
        class_id = int(parts[0])
        # discard the first element (class) --> [x_center, y_center, width, height]
        yolo_bbox = torch.tensor(parts[1:], dtype=torch.float32)            
        # convert the list of floats into a PyTorch tensor

        # Read PDLPR label (license plate string)
        with open(pdlpr_label_path, "r", encoding="utf-8") as f:
            pdlpr_plate_str = f.readline().strip()

        #Extracting the pdlpr index label that is going to be used by 
        #the CNNCTC model
        fields = img_path.name.split("-")
        plate_number = fields[4]
        character_id_list = plate_number.split("_")
        plate_id = []
        for c in character_id_list:
            plate_id.append(int(c))
        
        #converting the index from the name to the index from the 
        #unified vocabulary
        plate_id= target_to_index(plate_id)
        
        pdlpr_label_idx = torch.tensor(plate_id, dtype=torch.long)

        # apply the transformations
        if self.transform:
            full_image = self.transform(full_image)
            cropped_image = self.transform(cropped_image)
        
        return {
            'full_image': full_image,
            'cropped_image': cropped_image,
            'yolo_bbox_label': yolo_bbox,
            'pdlpr_plate_string': pdlpr_plate_str,
            'pdlpr_plate_idx': pdlpr_label_idx,
            'image_name': img_path.name
        }


    def get_dataloaders(base_dir, batch_size = 8, transform = None):
        #this functions initializes the different dataloaders and returns them
        ds = CCPDDataset(base_dir=base_dir, transform=transform)

        train_loader = DataLoader(ds.get_dataset("train"), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ds.get_dataset("val"), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(ds.get_dataset("test"), batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

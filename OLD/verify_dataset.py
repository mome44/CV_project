
import os
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
from data import CCPDDataset
from torch.utils.data import DataLoader

def show_sample(item):
    img = item["full_image"]
    if isinstance(img, torch.Tensor):
        img = TF.to_pil_image(img)

    plt.imshow(img)
    plt.axis("off")

    # Draw bounding box
    if "yolo_bbox_label" in item:
        xc, yc, w, h = item["yolo_bbox_label"]
        img_w, img_h = img.size
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='lime', linewidth=2)
        )

    if "pdlpr_plate_string" in item:
        print("Plate:", item["pdlpr_plate_string"])

    plt.title(item["image_name"])
    plt.show()

def main():
    dataset = CCPDDataset(base_dir="dataset").get_dataset("train")

    for i in range(5):
        print(f"Sample {i}")
        show_sample(dataset[i])

if __name__ == "__main__":
    main()
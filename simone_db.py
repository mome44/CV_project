import os
import torch
import cv2
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

class PlateDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.imgs = sorted(os.listdir(img_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        ann_path = os.path.join(self.ann_dir, self.imgs[idx].replace('.jpg', '.txt'))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []

        with open(ann_path) as f:
            for line in f:
                class_id, x1, y1, x2, y2 = map(int, line.strip().split())
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([idx])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id
        }

        img = F.to_tensor(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

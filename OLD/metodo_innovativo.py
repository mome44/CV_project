import os
import torch
import cv2
from torchvision.transforms import functional as F


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

from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, RetinaNetClassificationHead

#Hyperparameters
BATCH_SIZE = 4
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 10
# Carica modello
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
#model = retinanet_resnet50_fpn(weights="DEFAULT")

# Modifica classificatore
#Since we want that the last classifier layer to have only an output of two
#so we get the input dimension of it
in_features = model.roi_heads.box_predictor.cls_score.in_features
#and reinitialize it with the new class.
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # background + plate

#in_channels = model.head.classification_head.conv[0].in_channels
#num_anchors = model.head.classification_head.num_anchors
#model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes=2)

# Dataset + dataloader
dataset = PlateDataset("dataset/images", "dataset/annotations")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Ottimizzatore
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# Addestra
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


for epoch in range(NUM_EPOCHS):
    model.train()
    for images, targets in dataloader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} loss: {losses.item():.4f}")

torch.save(model.state_dict(), "fasterrcnn_plate_detector.pth")


model.load_state_dict(torch.load("fasterrcnn_plate_detector.pth"))
model.eval()

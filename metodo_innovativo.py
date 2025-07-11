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


IOU_THRESHOLD = 0.5

#Hyperparameters
BATCH_SIZE = 4
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 10

def compute_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# Carica modello
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
#model = retinanet_resnet50_fpn(weights="DEFAULT")

# Modifica classificatore
#Since we want that the last classifier layer to have only an output of two
#so we get the input dimension of it
in_features = model.roi_heads.box_predictor.cls_score.in_features
#and reinitialize it with the new class, we have to put num classes = 2 because otherwise the model
#will always be sure that there will be a plate
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # background + plate

#in_channels = model.head.classification_head.conv[0].in_channels
#num_anchors = model.head.classification_head.num_anchors
#model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes=2)

#cambiare il dataloader in modo che ce ne sia uno per train validation e test
dataset = PlateDataset("dataset/images", "dataset/annotations")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

#this optimizer uses stochastic gradient descent and has in input the parameters (weights) from 
#the pretrained model
params = model.parameters()
optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

#initialize the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#train we are doing fine tuning on the task of recognizing plate
for e in range(NUM_EPOCHS):
    model.train()
    train_loss= 0.0
    #does the for loop for all the items in the same batch
    for images, labels in dataloader:
        #moves the images and labels to the GPU
        images = list(img.to(device) for img in images)
        labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

        #in this model there are different losses
        loss_dict = model(images, labels)
        #we do the sum of all of them to compute the total loss
        total_step_loss = sum(loss for loss in loss_dict.values())
        
        #we remove the gradients from the previous steps
        optimizer.zero_grad()
        #compute the new gradients
        total_step_loss.backward()
        #update the weights that are computed now.
        optimizer.step()

        train_loss += total_step_loss.item()

    avg_train_loss = train_loss / len(dataloader) #train_dataloader

    print(f"Epoch {e+1}/{NUM_EPOCHS} - accuracyloss: {avg_train_loss.item():.4f}")

    model.eval()
    val_loss = 0.0

    TP = 0  # true positives
    FP = 0  # false positives
    FN = 0  # false negatives
    iou_scores = []

    with torch.no_grad():
        for images, targets in val_dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Calcolo della loss (solo per monitoraggio)
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            val_loss += loss.item()

            # Inference
            outputs = model(images)

            for pred, target in zip(outputs, targets):
                pred_boxes = pred['boxes'].cpu()
                #prediction is a list that contains a score from 0 to 1 that 
                #includes the confidence of the answer, so if this score is less than 0.5
                #we will not consider it
                pred_scores = pred['scores'].cpu()
                true_boxes = target['boxes'].cpu()
                 #selezionare la box che ha lo score più alto
                best_index = pred['scores'].argmax()
                best_box = pred['boxes'][best_index]
                best_score = pred['scores'][best_index]
               
                matched = set()
                for i, pbox in enumerate(pred_boxes):
                    
                    matched_flag = False
                    for j, tbox in enumerate(true_boxes):
                        iou = compute_iou(pbox.numpy(), tbox.numpy())
                        if iou > IOU_THRESHOLD and j not in matched:
                            TP += 1
                            matched.add(j)
                            iou_scores.append(iou)
                            matched_flag = True
                            break
                    if not matched_flag:
                        FP += 1
                FN += len(true_boxes) - len(matched)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss/len(train_dataloader):.4f}")
    print(f"Val Loss: {val_loss/len(val_dataloader):.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | mIoU: {mean_iou:.4f}")

torch.save(model.state_dict(), "fasterrcnn_plate_detector.pth")


model.load_state_dict(torch.load("fasterrcnn_plate_detector.pth"))
model.eval()

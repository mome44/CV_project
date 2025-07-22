import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from globals import *
from utils import *

def get_model_name():
    return f"yolov5_epochs{EPOCHS_TRAIN_Y}_bs{BATCH_SIZE_TRAIN_Y}_lr{LR_INIT_Y}_imgs{IMAGE_SIZE_Y}.pt"

def get_run_name():
    # Crea un nome univoco per la run di training che sta facendo usando gli hyperparams usati nel modello
    return get_model_name().replace(".pt", "")


def train_yolo():
    model_name = get_model_name()
    run_name = get_run_name()

    if os.path.exists(model_name):
        print(f"[INFO] Model {model_name} already exists ---> SKIP training!!")
        return YOLO(model_name)
    
    # Create an untrained model based on the configuration params
    model = YOLO("yolov5s.yaml")
    
    model.train(
        data    = "ccpd.yaml",                      # path to .yaml file for the configuration
        epochs  = EPOCHS_TRAIN_Y,                  
        batch   = BATCH_SIZE_TRAIN_Y,
        lr0     = LR_INIT_Y,
        imgsz   = IMAGE_SIZE_Y,
        save    = True,                             # save the training checkpoints and weigths of the final model
        device  = "mps",                            # TO BE CHANGED ACCORDING TO PC --> "cpu"
        project = "runs/train",                     # directory where to save the outputs of training
        name    = model_name.replace(".pt", ""),    # create a subdir in the project folder, where to save training logs and outputs
        val     = True,                             # run validation here to create results.csv and .png
        plots   = True                              
    )

    model.save(model_name)

    best_model_path = f"runs/train/{run_name}/weights/best.pt"
    
    return best_model_path




# TRAIN
train_model_path = train_yolo()

run_name = get_run_name()

# VALIDATION after training
# Load and use the best model best.pt --> create a model instance initializzed with the trained weights
best_model = YOLO(train_model_path, verbose = False)
# best_model = YOLO("/Users/michelafuselli/Desktop/Michi/Università/Magistrale/Computer Vision/Project/CV_project/runs/train/yolov5_epochs20_bs8_lr0.001_imgs6402/weights/best.pt", verbose = False)

# Inside results: mAP@0.5, mAP@0.5:0.95. precision, recall, confusion matrix, curva PR, curva f1, ... --> are saved in runs/detect
results = best_model.val(
    data    = "ccpd.yaml",
    split   = 'val',
    iou     = IOU_THRESHOLD,
    device  = "cpu",
    name    = f"{run_name}_VAL_iou{int(IOU_THRESHOLD*100)}",
)

image_dir = Path("dataset/images/val")
output_dir = Path("runs/detect") / f"{get_run_name()}_VAL_iou{int(IOU_THRESHOLD * 100)}"
output_dir.mkdir(parents=True, exist_ok=True)

iou_list = []

# Loop over images
for image_path in sorted(image_dir.glob("*.jpg")):
    # Predict
    result = best_model(image_path, max_det=5, verbose = False)[0]
    predictions = result.boxes.xyxy.cpu().numpy()  # shape: (N, 4)

    real_box = load_gt_box_from_label_validation(image_path)
    if real_box is None:
        # skip image if no GT or invalid
        continue

    # Calcola IoU tra ogni box predetta e quella reale
    for predicted_box in predictions:
        iou = compute_iou(predicted_box, real_box)
        iou_list.append(iou)

# Compute average among all iou values 
if iou_list:
    mean_iou = sum(iou_list) / len(iou_list)
else:
    mean_iou = 0.0

# Save in .txt
txt_path = output_dir / "mean_iou.txt"
with open(txt_path, "w") as f:
    f.write(f"Mean IoU over validation set: {mean_iou:.4f}\n")

print(f"[INFO] Mean IoU saved to {txt_path}")

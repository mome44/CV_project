import torch
import pandas as pd
import os
from pathlib import Path
from globals import DATASET_PATH_Y, BATCH_SIZE_TRAIN_Y, LR_INIT_Y, EPOCHS_TRAIN_Y, IMAGE_SIZE_Y, IOU_THRESHOLD
from ultralytics import YOLO
from utils import save_metrics_txt, plot_accuracy, plot_recall, plot_precision, plot_f1score, plot_iou, validate_model_on_split



def get_model_name():
    return f"yolov5_epochs{EPOCHS_TRAIN_Y}_bs{BATCH_SIZE_TRAIN_Y}_lr{LR_INIT_Y}_imgs{IMAGE_SIZE_Y}.pt"

def train_yolo():
    model_name = get_model_name()
    
    if os.path.exists(model_name):
        print(f"[INFO] Model {model_name} already exists. Skipping training.")
        return YOLO(model_name)
    
    model = YOLO("yolov5s.yaml")

    # The train cannot be done manually
    # The YOLO API wraps the entire training loop inside .train()
    # YOLOv5 handles dataloading, loss computation, validation, logging, and saving behind the scenes.
    # Il training non può essere fatto a mano
    # Le API di YOLO gestiscono tutto internamente dentro .train()
    # Dataloading, calcolo della loss, validation, log, e salvataggio vengono fatti tutti dietro le quinte
    model.train(
        data    = "ccpd.yaml",                       # it tells YOLO where the images and labels are
        epochs  = EPOCHS_TRAIN_Y,
        batch   = BATCH_SIZE_TRAIN_Y,                # numero di immagini per batch
        lr0     = LR_INIT_Y,
        imgsz   = IMAGE_SIZE_Y,                      # grandezza finale (resizing dell'input)
        device  = 'cpu',
        save    = True,                              # save the model weigths after training
        project = "runs/train",                      # output directory where logs and checkpoints go
        name    = model_name.replace(".pt", "")      # subfolder name for this run
    )

    model.save(model_name)

    return model



if __name__ == "__main__":

    # Construct model name dynamically
    model_name = get_model_name()

    train_yolo()

    # Run validation on val and train
    validate_model_on_split(model_name, split="val", iou_threshold=IOU_THRESHOLD)
    validate_model_on_split(model_name, split="train", iou_threshold=IOU_THRESHOLD)


    """
    YOLOv5 manages everything internally, so we cannot access directly loss and accuracy for each epoch
    YOLOv5 saves everything automatically in runs/train/<model_name>/results.csv, with columns like these:
           train/box_loss  train/obj_loss  metrics/precision  metrics/recall  metrics/mAP_0.5
    epoch
      0     0.52            0.31            0.67                0.64            0.72
      1     0.41            0.29            0.71                0.66            0.75
    ...

    """
    # Read training metrics
    csv_path = Path("runs/train") / model_name.replace(".pt", "") / "results.csv"
    df = pd.read_csv(csv_path)

    for i, row in df.iterrows():
        loss = row["train/box_loss"] + row["train/cls_loss"] + row["train/dfl_loss"]
        acc = row["metrics/mAP50(B)"]
        print(f"Epoch {i+1} — Loss: {loss:.4f} — mAP@0.5: {acc:.4f}")

    # Save and plots
    # save_metrics_txt(df, model_name)
    plot_accuracy(model_name)
    plot_precision(model_name)
    plot_recall(model_name)
    plot_f1score(model_name)
    plot_iou(model_name)

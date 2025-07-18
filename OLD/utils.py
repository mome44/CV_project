import os
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from globals import IOU_THRESHOLD


base_dir = Path("dataset")


def yoloprediction_to_pdlpr_input(x_center, y_center, width, height, image_path):
    #This functions takes in input the prediction from yolo and returns the cropped image (so the input for pdlpr)
    img = Image.open(image_path)

    image_width, image_height = img.size

    x_center_pixel = x_center * image_width
    y_center_pixel = y_center * image_height
    width_pixel = width * image_width
    height_pixel = height * image_height

    x_min = int(x_center_pixel - width_pixel / 2)
    x_max = int(x_center_pixel + width_pixel / 2)
    y_min = int(y_center_pixel - height_pixel / 2)
    y_max = int(y_center_pixel + height_pixel / 2)

    #crop the image according to the bounding box coordinates
    cropped_img = img.crop((x_min, y_min, x_max, y_max))

    return cropped_img


# not used !!!
def save_metrics_txt(metrics, model_name):
    filename = model_name.replace('.pt', '_metrics.txt')
    with open(filename, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")



def plot_accuracy(model_name):
    # it uses the mean Average Precision (mAP@0.5) --> it is the standard accuracy measure in object detection
    
    csv_path = Path("runs/train") / model_name.replace(".pt", "") / "results.csv"
    df = pd.read_csv(csv_path)
    acc = df["metrics/mAP50(B)"]

    plt.plot(acc, label="mAP50(B) (Accuracy)")
    plt.xlabel("Epoch")
    plt.ylabel("mAP50(B)")
    plt.title("Accuracy per Epoch")
    plt.legend()
    plt.savefig(model_name.replace(".pt", "_accuracy.png"))
    plt.close()


def plot_precision(model_name):
    csv_path = Path("runs/train") / model_name.replace(".pt", "") / "results.csv"
    df = pd.read_csv(csv_path)
    precision = df["metrics/precision(B)"]

    plt.plot(precision, label="Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision per Epoch")
    plt.legend()
    plt.savefig(model_name.replace(".pt", "_precision.png"))
    plt.close()


def plot_recall(model_name):
    csv_path = Path("runs/train") / model_name.replace(".pt", "") / "results.csv"
    df = pd.read_csv(csv_path)
    recall = df["metrics/recall(B)"]

    plt.plot(recall, label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall per Epoch")
    plt.legend()
    plt.savefig(model_name.replace(".pt", "_recall.png"))
    plt.close()


def plot_f1score(model_name):
    csv_path = Path("runs/train") / model_name.replace(".pt", "") / "results.csv"
    df = pd.read_csv(csv_path)
    precision = df["metrics/precision(B)"]
    recall = df["metrics/recall(B)"]
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    plt.plot(f1, label="F1-Score", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1-Score per Epoch")
    plt.legend()
    plt.savefig(model_name.replace(".pt", "_f1score.png"))
    plt.close()


# YOLOv5 does not log raw IoU per epoch, it uses mAP@0.5
def plot_iou(model_name):
    csv_path = Path("runs/train") / model_name.replace(".pt", "") / "results.csv"
    df = pd.read_csv(csv_path)
    map95 = df["metrics/mAP50-95(B)"]

    plt.plot(map95, label="mAP50-95(B)")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("mAP50-95(B) per Epoch")
    plt.legend()
    plt.savefig(model_name.replace(".pt", "_mAP50-95(B).png"))
    plt.close()


# After testing, you get one dictionary of aggregated results, no per-epoch results,
# so we cannot create a line plot like during training
# We ccan plot each as a bar
def plot_test_metrics(metrics_dict, model_name):
    labels = []
    values = []

    for k, v in metrics_dict.items():
        labels.append(k)
        values.append(v)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color='skyblue')
    plt.ylabel("Score")
    plt.title("YOLOv5 Test Metrics")

    # Add value on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center')

    filename = model_name.replace(".pt", "_test_metrics_plot.png")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Test metrics plot saved to: {filename}")



def validate_model_on_split(model_name, split, iou_threshold=IOU_THRESHOLD):
    """
    Evaluates a trained YOLOv5 model on the specified dataset split ('train' or 'val').

    Args:
        model_name (str): The name of the trained model .pt file (including hyperparams).
        split (str): 'train' or 'val'.
        iou_threshold (float): IoU threshold for evaluation (default is 0.7).
    """
    model_path = Path("runs/train") / model_name.replace(".pt", "") / "weights" / "best.pt"
    model = YOLO(model_path)

    print(f"\n Evaluating '{split}' set with IoU > {iou_threshold:.7f}")
    # print(f"\n Evaluating '{split}' set with IoU > {iou_threshold:{IOU_THRESHOLD}f}")  --> non so se va bene lo stesso scritto così
    results = model.val(
        data    = "ccpd.yaml",
        split   = split,
        iou     = iou_threshold,
        device  = "cpu",
        name    = f"{model_name.replace('.pt', '')}_{split}_iou{int(iou_threshold*100)}"
    )

    metrics = results.results_dict

    # Print metrics
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Save metrics to file
    out_file = model_name.replace(".pt", f"_{split}_metrics_iou{int(iou_threshold*100)}.txt")
    with open(out_file, "w") as f:
        f.write(f"# Evaluation on '{split}' set (IoU > {iou_threshold})\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"Metrics saved to: {out_file}")


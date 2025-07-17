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

def compute_iou(box_1, box_2):
    #it is a metric that involves the intersection of the two areas
    #over the union, and returns a matching percentage
    
    #coputing the coordinate of the intersections
    x1 = max(box_1[0], box_2[0])
    y1 = max(box_1[1], box_2[1])
    x2 = min(box_1[2], box_2[2])
    y2 = min(box_1[3], box_2[3])

    interArea = max(0, x2 - x1) * max(0, y2 - y1)
    boxAArea = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    boxBArea = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)  #+1e-6 is used to avoid the division per zero
    return iou

"""
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
"""



def load_gt_box_from_label(image_path):
    """
    Load the ground truth box from a YOLO-format label file.
    Returns [x1, y1, x2, y2] or None if label file is missing/invalid.
    """
    label_path = Path("dataset/labels/val") / (image_path.stem + ".txt")

    if not label_path.exists():
        print(f"[WARN] No label found for {image_path.name}")
        return None

    with open(label_path, "r") as f:
        lines = f.readlines()

    if not lines:
        print(f"[WARN] Empty label file for {image_path.name}")
        return None

    # Assume the first object only
    try:
        parts = list(map(float, lines[0].strip().split()))
        _, x_center, y_center, w, h = parts
    except Exception:
        print(f"[WARN] Label parse failed for {image_path.name}")
        return None

    # Convert from normalized to absolute coordinates
    img = plt.imread(image_path)
    img_h, img_w = img.shape[:2]

    cx, cy = x_center * img_w, y_center * img_h
    bw, bh = w * img_w, h * img_h

    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy + bh / 2

    return [x1, y1, x2, y2]

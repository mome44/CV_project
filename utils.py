import os
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from globals import *
from tqdm import tqdm
import torch
import cv2
import numpy as np
import torch
import easyocr
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from globals import IOU_THRESHOLD
import json


# Inizializzare il lettore solo una volta: cinese semplificato e inglese
reader = easyocr.Reader(['ch_sim', 'en'])

base_dir = Path("dataset")

def initialize_labels():
    #initialize the labels and creating the folders
    splits = ["train", "val", "test"]
    for split in splits:
        # It is just a safe and readable way to say: go to datasets/ccpd/images/train (or val, or test), depending on which split you're processing.
        image_dir = base_dir / "images" / split
        label_dir = base_dir / "labels" / split
        crops_dir = base_dir / "crops" / split
        label_pdlpr_dir = base_dir / "labels_pdlpr" / split
    
        label_dir.mkdir(parents=True, exist_ok=True)    # creates the folder if it does not exist
        crops_dir.mkdir(parents=True, exist_ok=True)
        label_pdlpr_dir.mkdir(parents=True, exist_ok=True)
        
        # Loop through all .jpg images in the current image directory
        #the tqdm library is useful to plot the loading bar
        for image_path in tqdm(list(image_dir.glob("*.jpg")), desc=f"Processing - {split}", unit="img"):
            #print(f"Found image: {image_path}")
            #print(f"Processing: {image_path.name}")
    
            # Parse bounding box from filename: example => "XXXXX&x1_x2_y1_y2&..."
            try:
                fields = image_path.stem.split("-")    # image_path.stem is the filename without .jpg
                
                # Field 2 (index 2) is bbox: format is "x1&y1_x2&y2"
                bbox_part = fields[2]
                corners = bbox_part.split("_")
                x1, y1 = map(int, corners[0].split("&"))
                x2, y2 = map(int, corners[1].split("&"))
    
                # Define min/max values
                x_min = min(x1, x2)
                x_max = max(x1, x2)
                y_min = min(y1, y2)
                y_max = max(y1, y2)
                
                
                #extracting the information about the plate to create the labels for pdlpr
                #the plate is in this format 0_0_22_27_27_33_16
                plate_number = fields[4]
                character_id_list = plate_number.split("_")
                #get the number for the province and for the letter
                province_id = int(character_id_list[0])
                alphabet_id = int(character_id_list[1])
                #get the actual character for both and join them
                province_char = PROVINCES[province_id]
                alphabet_char = ALPHABETS[alphabet_id]
                plate = province_char + alphabet_char
    
                for i in range(2, 8):
                    #for the remaining 5 characters we do the mapping from the ADS
                    ads_index = int(character_id_list[i])
                    plate += ADS[ads_index]
                
            except Exception as e:
                print(f"Skipping {image_path.name}: {e}")
                continue
            # Read the image to get image size (needed to normalize the coordinates)
            img = Image.open(image_path)
    
            img_width, img_height = img.size
    
            #crop the image according to the bounding box coordinates
            cropped_img = img.crop((x_min, y_min, x_max, y_max))
    
            #Adding crops so cut images into a separate folder
            crops_path = crops_dir / (image_path.stem + ".jpg")
    
            #saving the image into the crops folder
            cropped_img.save(crops_path)
    
            img.close()
    
    
            # Normalize the bounding box for YOLO format
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
    
            # Create YOLO label string
            # 0 is the class ID (only one class - license plate)
            # the rest are floats with 6 digits after the decimal point
            label_str = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
            # Save label file with same name
            label_path = label_dir / (image_path.stem + ".txt")
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(label_str + "\n")
    
            #print(f"Wrote label: {label_path.name}")
    
            #Save the label for PDLPR
            label_pdl_pr_path = label_pdlpr_dir / (image_path.stem + ".txt")
            with open(label_pdl_pr_path, "w", encoding="utf-8") as f:
                f.write(plate + "\n")

    
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


def target_to_index(target_list):
    #this function converts the id of the target 
    #into the ids that are returned by the model
    #since it uses the ids from the sorted list of 
    #all the possible characters
    output = []
    province = PROVINCES[target_list[0]]
    alphabet = ALPHABETS[target_list[1]]
    output.append(CHAR_IDX[province])
    output.append(CHAR_IDX[alphabet])
    for char_idx in range(2,8):
        char = ADS[target_list[char_idx]]
        output.append(CHAR_IDX[char])
    return output


def index_to_target(index_list):
    output=[]
    for idx in index_list:
        output.append(IDX_CHAR[idx])
    return output


def load_gt_box_from_label_validation(image_path):
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


def load_gt_box_from_label_test(image_path):
    """
    Load the ground truth box from a YOLO-format label file.
    Returns [x1, y1, x2, y2] or None if label file is missing/invalid.
    """
    label_path = Path("dataset/labels/test") / (image_path.stem + ".txt")

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

### baseline functions
def get_label_yolo(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    parts = list(map(float, line.split()))
    return torch.tensor(parts[1:], dtype=torch.float32)     # x, y, width, height


def get_ground_truth_coordinates(yolo_tensor, image_width, image_height):
        cx, cy, w, h = yolo_tensor.tolist()

        x_min = (cx - w / 2) * image_width
        y_min = (cy - h / 2) * image_height
        x_max = (cx + w / 2) * image_width
        y_max = (cy + h / 2) * image_height

        return [x_min, y_min, x_max, y_max]


def plate_detector(image_path, true_coordinates):
    # Carica immagine
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Isolamento delle parti verdi 
    # (le targhe hanno la scritta nera su fondo verde)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)              # Conversione da RGB a HSV (per filtrare i colori sulla base del tono)
    lower_green = np.array([40, 40, 40])                    # Range del tono di verde in HSV
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)       # Crea una binary mask

    # Morphology
    # L'edge detector mi ritorna una serie di tanti contorni frammentati, 
    # devo usare operazioni morfologiche per unire linee che sono vicine tra loro
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Edge detector --> Canny
    edges = cv2.Canny(cleaned_mask, 100, 200)

    # Trova i bordi
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Geometric filter + controllo con OCR per vedere se ci sono numeri/lettere
    # Scartare le regiorni che non sono targe
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h
        if not (2.5 < aspect_ratio < 6 and 1000 < area < 40000):
            continue

        roi = img_rgb[y:y+h, x:x+w]
        result = reader.readtext(roi)

        if result:
            text = result[0][1]
            conf = result[0][2]
            clean_text = text.strip().replace(" ", "").replace("\n", "")
        else:
            clean_text = ""
            conf = 0.0

        iou = compute_iou([x, y, x+w, y+h], true_coordinates)
        ocr_score = len(clean_text) if len(clean_text) >= 4 else 0

        # score = iou + 1 * ocr_score     # OCR pesa di più
        score = 1.5 * (ocr_score / 8.0) + 0.5 * iou     # OCR pesa di più ma non ignoro iou

        candidates.append({
            "bbox": [x, y, x+w, y+h],
            "text": clean_text,
            "score": score
        })

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c["score"])
    return best["bbox"], best["text"]

# functions for pdlpr 
def custom_collate(batch):
    return {
        "cropped_image": torch.stack([item["cropped_image"] for item in batch]),
        "pdlpr_plate_string": [item["pdlpr_plate_string"] for item in batch],
        # add other fields as needed
    }

def custom_collate_2(batch):
    return {
        "cropped_image": torch.stack([item["cropped_image"] for item in batch]),
        "pdlpr_plate_idx": [item["pdlpr_plate_idx"] for item in batch],
    }

def build_vocab(label_folder, file_name, include_blank=True):
    vocab = set()

    for filename in os.listdir(label_folder):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(label_folder, filename), "r", encoding="utf-8") as f:
            label = f.read().strip().upper()
            vocab.update(label)

    vocab = sorted(vocab)

    char_idx = {}
    idx_char = {}
    start_idx = 0

    if include_blank:
        char_idx["-"] = 0  # CTC blank
        idx_char[0] = "-"
        start_idx = 1

    for i, ch in enumerate(vocab, start=start_idx):
        char_idx[ch] = i
        idx_char[i] = ch

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(char_idx, f, ensure_ascii=False, indent=2)
    
    # saving the vocabulary for later
    print(f"[vocab] Built vocabulary with {len(char_idx)} characters.")
    return char_idx, idx_char

# function to load the vocabulary
def load_vocab(path="vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        char_idx = json.load(f)
    idx_char = {int(v): k for k, v in char_idx.items()}
    return char_idx, idx_char

def plot_metrics(train_seq, val_seq, train_char, val_char):
    epochs = range(1, NUM_EPOCHS + 1)

    plt.figure()
    plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in train_seq], label="Train Seq Accuracy")
    plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in val_seq], label="Val Seq Accuracy")
    plt.title("Sequence Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"metrics_images/seq_accs_plot_{NUM_EPOCHS}_{LR}_{BATCH_SIZE}.png", dpi=300)

    plt.figure()
    plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in train_char], label="Train Char Accuracy")
    plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in val_char], label="Val Char Accuracy")
    plt.title("Char Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"metrics_images/char_accs_plot{NUM_EPOCHS}_{LR}_{BATCH_SIZE}.png", dpi=300)


# Evaluator
class Evaluator:
    def __init__(self, idx2char={}, blank_index=0):
        self.idx2char = idx2char
        self.blank_index = blank_index
        self.reset()

    def reset(self):
        self.total_chars = 0
        self.correct_chars = 0
        self.correct_seqs = 0
        self.total_samples = 0

    def greedy_decode(self, logits):
        # logits: [B, T, C]
        predictions = torch.argmax(logits, dim=-1)  # [B, T]
        decoded = []

        for prediction in predictions:
            prev = self.blank_index
            chars = []
            for idx in prediction:
                idx = idx.item()
                if idx != self.blank_index and idx != prev:
                    chars.append(self.idx2char[idx])
                prev = idx
            decoded.append("".join(chars))
        return decoded
    
    def greedy_decode_idx(self, logits):
        predictions = torch.argmax(logits, dim=2)
        predictions= predictions.transpose(0, 1)
        final_predictions = []
        #iterate for each prediction array in the batch
        for prediction in predictions:
            before = 0
            reduced = []
            for t_index in prediction:
                t_index = t_index.item()
                if t_index != 0 and t_index != before:
                    #append the index only if it is not zero and it is different than before
                    reduced.append(t_index)
                before = t_index
            final_predictions.append(reduced)
        return final_predictions

    def update(self, logits, target_strs):
        # logits: [B, T, vocab_size]
        pred_strs = self.greedy_decode(logits)

        for pred, true in zip(pred_strs, target_strs):
            self.total_samples += 1
            self.total_chars += len(true)
            correct = sum(p == t for p, t in zip(pred, true))
            self.correct_chars += correct
            if pred == true:
                self.correct_seqs += 1

    def update_baseline(self, logits, labels):
        
        final_predictions = self.greedy_decode_idx(logits)
        for pred_idx_list, label in zip(final_predictions, labels):
            label_list = label.tolist()
            if pred_idx_list == label_list:
                self.correct_seqs +=1

            self.total_samples += 1
            self.total_chars += len(label)
            correct = 0
            for pred_idx, label_idx in zip(pred_idx_list, label):
                if pred_idx == label_idx:
                    correct += 1
            self.correct_chars += correct

    def compute(self):
        char_acc = self.correct_chars / self.total_chars if self.total_chars > 0 else 0.0
        seq_acc = self.correct_seqs / self.total_samples if self.total_samples > 0 else 0.0
        return {
            "char_accuracy": char_acc,
            "seq_accuracy": seq_acc,
        }

    def print(self):
        metrics = self.compute()
        print(f"Character accuracy:  {metrics['char_accuracy']:.4f}")
        print(f"Sequence accuracy:   {metrics['seq_accuracy']:.4f}")
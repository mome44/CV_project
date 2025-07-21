import cv2
import numpy as np
import torch
import easyocr
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from utils import compute_iou
from globals import IOU_THRESHOLD


# Inizializzare il lettore solo una volta: cinese semplificato e inglese
reader = easyocr.Reader(['ch_sim', 'en'])

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
    lower_green = np.array([45, 80, 60])                    # Range del tono di verde in HSV
    upper_green = np.array([75, 255, 255])
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



images_dir = Path("dataset/images/train/")
labels_dir = Path("dataset/labels/train/")

diff_results_dir = Path("/results")
diff_results_dir.mkdir(parents=True, exist_ok=True)
diff_results_txt = diff_results_dir / f"BL_iou_ocr_5.txt"
open(diff_results_txt, "w").close()

total_iou = 0.0
num_iou = 0
num_passed_iou = 0      # tiene il conto dei valori >= 0.7


for image_path in tqdm(images_dir.glob("*.jpg"), desc="Processing images", unit="img"):
    image_name = image_path.name
    label_path = labels_dir / (image_path.stem + ".txt")

    if not label_path.exists():
        print(f"NO label for {image_name}")
        continue
    
    pil_image = Image.open(image_path).convert("RGB")
    yolo_tensor = get_label_yolo(label_path)
    width, height = pil_image.size

    true_coordinates = get_ground_truth_coordinates(yolo_tensor, width, height)

    candidate_bounding_box = plate_detector(image_path, true_coordinates)

    if not candidate_bounding_box:
        with open(diff_results_txt, "a") as f:
            f.write(f"{image_name}\n")
            f.write("IoU: 0.000\n")
            f.write("OCR: NONE\n")
            f.write(f"Box GT: {true_coordinates}\n")
            f.write("Box Pred: NONE\n")
            f.write("---\n")
        
        # contala come mancante
        total_iou += 0.0
        num_iou += 1
        continue

    predict_bbox, ocr_text  = candidate_bounding_box
    iou_diff = compute_iou(predict_bbox, true_coordinates)

    total_iou += iou_diff
    num_iou += 1
    if iou_diff >= IOU_THRESHOLD:
        num_passed_iou += 1

    with open(diff_results_txt, "a") as f:
        f.write(f"{image_name}\n")
        f.write(f"IoU: {iou_diff:.3f}\n")
        f.write(f"OCR: {ocr_text}\n")
        f.write(f"Box GT: {true_coordinates}\n")
        f.write(f"Box Pred: {predict_bbox}\n")
        f.write("---\n")

if num_iou > 0:
    avg_iou = total_iou / num_iou
    pass_rate = (num_passed_iou / num_iou) * 100
else:
    avg_iou = 0.0
    pass_rate = 0.0
    

with open(diff_results_txt, "a") as f:
    f.write(f"\n AVERAGE IoU over {num_iou} predictions: {avg_iou:0.4f}")
    f.write(f"\n IoU pass rate (>= 0.7) {pass_rate:0.2f}")

print(f"\n AVERAGE IoU over {num_iou} predictions: {avg_iou:0.4f}")
print(f"\n IoU pass rate (>= 0.7) {pass_rate:0.2f}")

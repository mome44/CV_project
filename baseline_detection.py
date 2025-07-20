import cv2
import numpy as np
import torch
import pytesseract
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from utils import compute_iou
from globals import IOU_THRESHOLD


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


def plate_detector(image_path):
    # Carica immagine
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Isolamento delle parti verdi 
    # (le targhe hanno la scritta nera su fondo verde)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)              # Conversione da RGB a HSV (per filtrare i colori sulla base del tono)
    lower_green = np.array([35, 40, 40])                    # Range del tono di verde in HSV
    upper_green = np.array([85, 255, 255])
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
    candidate_bbox = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h
        if not (2.5 < aspect_ratio < 6 and 1000 < area < 40000):
            continue

        roi = img_rgb[y:y+h, x:x+w]
        roi_pil = Image.fromarray(roi)

        text = pytesseract.image_to_string(roi_pil, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学O")
        clean_text = text.strip().replace(" ", "").replace("\n", "")
        # if len(clean_text) == 8:
        candidate_bbox.append(([x, y, x+w, y+h], clean_text))

    return candidate_bbox


if __name__ == "__main__":
    images_dir = Path("dataset/images/train/")
    labels_dir = Path("dataset/labels/train/")

    diff_results_dir = Path("baseline_detection_iou")
    diff_results_dir.mkdir(parents=True, exist_ok=True)
    diff_results_txt = diff_results_dir / "results.txt"
    open(diff_results_txt, "w").close()
    
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

        candidate_bounding_box = plate_detector(image_path)

        if not candidate_bounding_box:
            with open(diff_results_txt, "a") as f:
                f.write(f"{image_name}\n")
                f.write("IoU: 0.000\n")
                f.write("OCR: NONE\n")
                f.write(f"Box GT: {true_coordinates}\n")
                f.write("Box Pred: NONE\n")
                f.write("---\n")
            continue

        total_iou = 0.0
        num_iou = 0
        # correct_ocr = 0
        #  total_ocr = 0
        num_passed_iou = 0      # tiene il conto dei valori >= 0.7

        for predict_bbox, ocr_text in candidate_bounding_box:
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
        f.write(f"\n AVERAGE IoU over {num_iou} predicitons: {avg_iou:0.4f}")
        f.write(f"\n IoU pass rate (>= 0.7) {pass_rate:0.2f}")

    print(f"\n AVERAGE IoU over {num_iou} predicitons: {avg_iou:0.4f}")
    print(f"\n IoU pass rate (>= 0.7) {pass_rate:0.2f}")

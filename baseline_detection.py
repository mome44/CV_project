import easyocr
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from utils import *
from network import*
from globals import *
from data import *

# Initialize the reader just once: simplified chinese and english 
reader = easyocr.Reader(['ch_sim', 'en'])


images_dir = Path("dataset/images/train/")
labels_dir = Path("dataset/labels/train/")

diff_results_dir = Path("results")
diff_results_dir.mkdir(parents=True, exist_ok=True)
diff_results_txt = diff_results_dir / f"BL_iou_ocr6.txt"
open(diff_results_txt, "w").close()

total_iou = 0.0
num_iou = 0
num_passed_iou = 0      # counts values >= 0.7


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
        
        # count it as missing
        total_iou += 0.0
        num_iou += 1
        continue

    predict_bbox, ocr_text  = candidate_bounding_box
    iou_diff = compute_iou(predict_bbox, true_coordinates)

    total_iou += iou_diff
    num_iou += 1
    if iou_diff >= IOU_THRESHOLD:
        num_passed_iou += 1

    with open(diff_results_txt, "a", encoding="utf-8") as f:
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

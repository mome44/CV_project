from pathlib import Path
import torch
import os
from PIL import Image  
from evaluator import Evaluator
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from globals import *
from utils import *
from network import CNN_CTC_model
import matplotlib.pyplot as plt
from data import CCPDDataset
from tqdm import tqdm
#from baseline_detection import plate_detector
import cv2
import numpy as np
import torch
import easyocr
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from utils import compute_iou

reader = easyocr.Reader(['ch_sim', 'en'])

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

CHAR_LIST = sorted(set(PROVINCES+ALPHABETS+ADS))
PLATE_LENGTH = 8

for idx, char in enumerate(CHAR_LIST):
    CHAR_IDX[char] = idx + 1  # start from 1
    IDX_CHAR[idx + 1] = char
IDX_CHAR[0] = '_'  # blank character for CTC

NUM_CHAR = len(CHAR_LIST) + 1 #since we include the blank character

#Hyperparameters
BATCH_SIZE = 32
LR = 0.001
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 60
SAVE_NAME = f"n_epochs_{NUM_EPOCHS}_bs_{BATCH_SIZE}_LR_{LR}_wd_{WEIGHT_DECAY}_a"

cnn_ctc_model = CNN_CTC_model(num_char=NUM_CHAR, hidden_size=256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
preprocess = transforms.Compose([
    transforms.Grayscale(),             
    transforms.Resize((48, 144)),       
    transforms.ToTensor(),             
    transforms.Normalize((0.5,), (0.5,))
])

preprocess_dataset = transforms.Compose([                   
    transforms.ToTensor()
])

#Load the cnnctc model for the second part
print(f"models/CNNCTC-{SAVE_NAME}.pth")

if os.path.exists(f"models/CNNCTC-{SAVE_NAME}.pth"):
    cnn_ctc_model.load_state_dict(torch.load(f"models/CNNCTC-{SAVE_NAME}.pth"))
    cnn_ctc_model.to(device)
else:
    print("model not found. Please train the model first")

#Initialize the path for the languages
image_paths = Path("dataset/images/test")

evaluator = Evaluator(idx2char=IDX_CHAR)
cnn_ctc_model.eval()
plate_accuracies = []
char_accuracies = []
iou_scores = []

i=0
for image_path in image_paths.glob("*.jpg"):
    if i % 10 == 0:
        print(f"processing image {i+1}/{55000}")
    image_name = image_path.name
    plate_label_path = Path("dataset/labels_pdlpr/test/") / (image_path.stem + ".txt")

    with open(plate_label_path, "r", encoding="utf-8") as f:
        pdlpr_plate_str = f.readline().strip()
    #print(pdlpr_plate_str)

    fields = image_path.stem.split("-") 
    bbox_part = fields[2]
    corners = bbox_part.split("_")
    x1, y1 = map(int, corners[0].split("&"))
    x2, y2 = map(int, corners[1].split("&"))
    
    true_box = [x1, y1, x2, y2]
    #print(true_box)
    plate_number = fields[4]
    character_id_list = plate_number.split("_")
    plate_id = []
    for c in character_id_list:
        plate_id.append(int(c))
    
    #converting the index from the name to the index from the 
    #unified vocabulary
    plate_id= target_to_index(plate_id)
    #print(plate_id)

    true_plate_idx = torch.tensor(plate_id, dtype=torch.long).to(device)
    
    image = Image.open(image_path).convert("RGB")
    bbx =[]
    result_detector = plate_detector(image_path, true_box)
    if result_detector is None:
        print(f"image {i}: no plate detected")
        i+=1
        continue

    bbx, text = result_detector
    print(i, bbx, text)
    iou = compute_iou(bbx, true_box)

    print(f"image {i}: iou score {iou}")
    iou_scores.append(iou)
    x1,y1, x2, y2 = bbx
    cropped_image = image.crop((x1, y1, x2, y2))
    #import matplotlib.pyplot as plt
    plt.imshow(cropped_image, cmap="gray")
    plt.title("cropped plate")
    plt.axis("off")
    plt.show()

    processed_image =preprocess(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        #computing the predictions wiht the cnn ctc model
        logits_model_output = cnn_ctc_model(processed_image)
        evaluator.reset()
        evaluator.update_baseline(logits_model_output, [true_plate_idx])
        metrics = evaluator.compute()

        char_acc = metrics["char_accuracy"]
        plate_acc = metrics["seq_accuracy"]
        
        char_accuracies.append(char_acc)
        plate_accuracies.append(plate_acc)
        print(f"character acc: {char_acc:.2f}, plate acc: {plate_acc:.2f}\n")
        plate_prediction = evaluator.greedy_decode_idx(logits_model_output)[0]

        plate_string = index_to_target(plate_prediction)
        print(f"predicted_plate: {''.join(plate_string)}, original plate: {pdlpr_plate_str}")
    

    i+=1

mean_char_acc = sum(char_accuracies) / len(char_accuracies)
mean_plate_acc = sum(plate_accuracies)/len(plate_accuracies)
mean_iou = sum(iou_scores)/len(iou)
print(f"Pipeline test result plate accuracy: {mean_plate_acc:.4f}")
print(f"Pipeline test result char accuracy: {mean_char_acc:.4f}")
print(f"Pipeline test result iou score: {mean_iou:.4f}")
#saving the iou result of the training, validation (last step) and testing
with open(f"results/pipeline-baseline-test-{SAVE_NAME}.txt", "w") as f:
    f.write(f"Final testing plate accuracy: {mean_plate_acc:.4f}\n")
    f.write(f"Final testing character accuracy: {mean_char_acc:.4f}\n")
    f.write(f"Final testing iou score: {mean_iou:.4f}\n")


    


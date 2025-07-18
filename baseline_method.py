import cv2
import numpy as np
from PIL import Image
from data import PlateDataset
import torch
from pathlib import Path
import pytesseract

def score_plate(pil_img):
    img = np.array(pil_img.convert("L"))  # grayscale numpy
    h, w = img.shape
    aspect_ratio = w / h

    # OCR
    text = pytesseract.image_to_string(pil_img, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    ocr_score = len(text.strip())

    # Edge density
    edges = cv2.Canny(img, 100, 200)
    edge_density = np.sum(edges > 0) / (w * h)

    # Colore uniforme
    stddev = np.std(img)

    # Normalizza e combina (puoi pesare ogni termine)
    score = (
        ocr_score * 2.0 +  # testo più lungo ha peso maggiore
        edge_density * 100 +  # scala edge per contare
        max(0, 1 - abs(aspect_ratio - 4.5)) * 5 +  # penalizza aspect strani
        max(0, 50 - stddev) * 0.1  # penalizza rumore
    )
    return score, text.strip()


def plate_detection_traditional(image_input):
    img = cv2.imread(image_input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Equalizzazione dell'istogramma per migliorare contrasto
    gray = cv2.equalizeHist(gray)

    # Edge detection (Canny)
    edges = cv2.Canny(gray, 100, 200)

    # Morphology (dilatazione e chiusura per connettere contorni)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Trova i contorni
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def detect_plate_cv_advanced(image_input, debug=False, use_ocr_check=False):
    """
    Versione avanzata della plate detection con tecniche tradizionali.
    
    Params:
        image_input: path immagine o PIL.Image
        debug: se True, mostra immagini con bounding box
        use_ocr_check: se True, filtra con OCR le regioni non testuali

    Returns:
        - list of cropped plate images (as PIL.Image)
        - (opzionale) immagine con bounding box (in formato RGB)
    """
    img = cv2.imread(str(image_input))
    

    orig_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # 1. Sobel verticale
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)

    # 2. Binarizzazione + morfologia
    _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 3. Trova contorni
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = orig_img[y:y+h, x:x+w]

        # 4. Filtro geometrico
        if not (2 < aspect_ratio < 6 and 1000 < area < 40000):
            continue

        # 5. Edge density interna
        edges_inside = cv2.Canny(roi_gray, 100, 200)
        edge_density = np.sum(edges_inside > 0) / (w * h)
        if edge_density < 0.02:
            continue

        # 6. Uniformità colore
        stddev = np.std(roi_gray)
        if stddev > 55:
            continue

        # 7. (Opzionale) OCR check
        if use_ocr_check:
            text = pytesseract.image_to_string(roi_gray, config="--psm 7")
            if len(text.strip()) < 4:
                continue

        # Se passa tutti i filtri, salva crop e disegna box
        plate_img = Image.fromarray(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
        plates.append(plate_img)

        if debug:
            cv2.rectangle(orig_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if debug:
        debug_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        return plates, Image.fromarray(debug_img)
    return plates

image_paths = Path("dataset/images/train")

train_dataset = PlateDataset("dapaset/images/train", "dapaset/labels/train")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

for image_path in image_paths.glob("*.jpg"):
    print(f"Found image: {image_path}")
    print(f"Processing: {image_path.name}")

    plates, debug_img = detect_plate_cv_advanced(image_path, debug=True)
    
    # Visualizza i risultati
    from matplotlib import pyplot as plt

    for i, plate in enumerate(plates):
        plt.subplot(1, len(plates), i+1)
        plt.imshow(plate)
        plt.axis("off")
    plt.suptitle("Targhe rilevate")
    plt.show()

    # Visualizza l'immagine originale con box
    plt.imshow(debug_img)
    plt.title("Bounding box")
    plt.axis("off")
    plt.show()
        
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
    
    # Crea un modello untrained basato sui params di configurazione 
    model = YOLO("yolov5s.yaml")
    
    model.train(
        data    = "ccpd.yaml",                      # path al file .yaml per la configurazione
        epochs  = EPOCHS_TRAIN_Y,                  
        batch   = BATCH_SIZE_TRAIN_Y,
        lr0     = LR_INIT_Y,
        imgsz   = IMAGE_SIZE_Y,
        save    = True,                             # salva i training checkpoints e i weigths del modello finale
        device  = "mps",                            # DA CAMBIARE A SECONDA DEL PC --> "cpu"
        project = "runs/train",                     # directory in cui salvare gli outputs del training
        name    = model_name.replace(".pt", ""),    # crea una subdir nella cartella del progetto, dove salva i training logs e outputs
        val     = True,                             # run validation qui per creare results.csv e .png
        plots   = True                              
    )

    model.save(model_name)

    best_model_path = f"runs/train/{run_name}/weights/best.pt"
    
    return best_model_path




if __name__ == "__main__":

    # TRAIN
    train_model_path = train_yolo()

    run_name = get_run_name()

    # VALIDATION dopo il training
    # Carica e usa il modello migliore best.pt --> crea una model instance inizializzata con i trained weights
    best_model = YOLO(train_model_path, verbose = False)
    # best_model = YOLO("/Users/michelafuselli/Desktop/Michi/Università/Magistrale/Computer Vision/Project/CV_project/runs/train/yolov5_epochs20_bs8_lr0.001_imgs6402/weights/best.pt", verbose = False)

    # Dentro results: mAP@0.5, mAP@0.5:0.95. precision, recall, confusion matrix, curva PR, curva f1, ... --> vengono salvati in runs/detect/val
    results = best_model.val(
        data    = "ccpd.yaml",
        split   = 'val',
        iou     = IOU_THRESHOLD,
        device  = "cpu",
        name    = f"{run_name}_VAL_iou{int(IOU_THRESHOLD*100)}",
    )

    image_dir = Path("dataset/images/val")
    output_dir = Path("runs/val") / f"{get_run_name()}_VAL_iou{int(IOU_THRESHOLD * 100)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    iou_list = []

    # Loop sulle immagini
    for image_path in sorted(image_dir.glob("*.jpg")):
        # Predict
        result = best_model(image_path, max_det=5, verbose = False)[0]
        predictions = result.boxes.xyxy.cpu().numpy()  # shape: (N, 4)

        # Estrazione delle coordinate reali (Ground Truth) in formato: x1_y1_x2_y2_imageid.jpg
        # name = image_path.stem
        # try:
        #   x1, y1, x2, y2 = map(int, name.split("_")[:4])
        #    real_box = [x1, y1, x2, y2]
        # except:
        #    print(f"[WARN] Skipping {name}, filename does not contain GT info")
        #    continue

        real_box = load_gt_box_from_label_validation(image_path)
        if real_box is None:
            continue  # skip immagine se GT non c'è o è invalid

        # Calcola IoU tra ogni box predetta e quella reale
        for predicted_box in predictions:
            iou = compute_iou(predicted_box, real_box)
            iou_list.append(iou)

    # Calcolare la media tra tutti i valori di iou
    if iou_list:
        mean_iou = sum(iou_list) / len(iou_list)
    else:
        mean_iou = 0.0

    # Salva in .txt
    txt_path = output_dir / "mean_iou.txt"
    with open(txt_path, "w") as f:
        f.write(f"Mean IoU over validation set: {mean_iou:.4f}\n")

    print(f"[INFO] Mean IoU saved to {txt_path}")

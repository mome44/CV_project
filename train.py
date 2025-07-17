import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from globals import *

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
        plots   = False                              
    )

    model.save(model_name)

    best_model_path = f"runs/train/{run_name}/weights/best.pt"
    
    return best_model_path




if __name__ == "__main__":

    # TRAIN
    train_model_path = train_yolo()

    run_name = get_run_name()

   


    # PLOTS dopo il training



    # VALIDATION dopo il training
    # Carica e usa il modello migliore best.pt --> crea una model instance inizializzata con i trained weights
    best_model = YOLO(train_model_path)

    # Dentro results: mAP@0.5, mAP@0.5:0.95. precision, recall, confusion matrix, curva PR, curva f1, ... --> vengono salvati in runs/detect/val
    results = best_model.val(
        data    = "ccpd.yaml",
        split   = 'val',
        iou     = IOU_THRESHOLD,
        device  = "cpu",
        name    = f"{run_name}_VAL_iou{int(IOU_THRESHOLD*100)}"
    )

    # print(results)

    output_dir = Path("runs/detect") / f"{run_name}_VAL_iou{int(IOU_THRESHOLD * 100)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # PLOTS
    # Visto che results.box.map restituisce una lista di valori mAP con iou threshold tra 0.5 e 0.95,
    # è meglio estrarre l'indice esatto a 0.7
    iou_threshold = IOU_THRESHOLD
    iou_index = int((iou_threshold - 0.5) / 0.05)  # 0.7 -> index 4
    iou = results.box.map
    print(iou)

    plt.figure()
    plt.bar([f"IoU@{iou_threshold}"], [iou], color="red")
    plt.title("IoU Curve (mAP at IoU = 0.7)")
    plt.ylabel("mAP")
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()

    # Salva nella stessa dir dove stanno i plots della validation
    plt.savefig(os.path.join(output_dir, "IoU_curve.png"))
    plt.close()
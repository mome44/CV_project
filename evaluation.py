from ultralytics import YOLO
from pathlib import Path
<<<<<<< HEAD
from globals import *
from utils import *


def test_yolo():
    
    model_path = "runs/train/yolov5_epochs30_bs12_lr0.001_imgs640/weights/best.pt"
    
    # prende il nome del training run
    run_name = Path(model_path).parent.parent.name
    
    output_name = f"{Path(model_path).stem}_TEST_iou{int(IOU_THRESHOLD * 100)}"
    output_dir = Path("runs/test") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    results = model.val(
        data    = "ccpd.yaml",
        split   = "test",
        iou     = IOU_THRESHOLD,
        device  = "cpu",
        name    = f"{run_name}_TEST_iou70",
        project = "runs/test"
    )

    return results, model, output_dir


if __name__ == "__main__":
    model_path = "runs/train/yolov5_epochs30_bs12_lr0.001_imgs640"

    run_name = Path(model_path).parent.parent.name
    
    output_name = f"{Path(model_path).stem}_TEST_iou{int(IOU_THRESHOLD * 100)}"
    
    # TESTING
    results, model_test, test_output_dir = test_yolo()

    # Salva le metriche nel file
    metrics_path = test_output_dir / "test_metrics.txt"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"IoU Threshold: {IOU_THRESHOLD}\n\n")
        f.write(f"mAP@0.5:      {results.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\n")
        f.write(f"Precision:    {results.box.mp:.4f}\n")
        f.write(f"Recall:       {results.box.mr:.4f}\n")


    # Calcola IoU
    image_dir = Path("dataset/images/test")

    iou_list = []

    # Loop sulle immagini
    for image_path in sorted(image_dir.glob("*.jpg")):
        # Predict
        result = model_test(str(image_path), max_det=5, verbose = False)[0]
        predictions = result.boxes.xyxy.cpu().numpy()  # shape: (N, 4)

        real_box = load_gt_box_from_label_test(image_path)
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

    # Salva in test_metrics.txt
    with open(metrics_path, "a") as f:
        f.write(f"Mean IoU over test set: {mean_iou:.4f}\n")

    print(f"[INFO] Mean IoU saved to {metrics_path}")


    
    print("\n TESTING complete!")
    print(f"Results saved to: runs/test/{test_output_dir}")
=======
from globals import BATCH_SIZE_TRAIN_Y, LR_INIT_Y, EPOCHS_TRAIN_Y, IMAGE_SIZE_Y, IOU_THRESHOLD
from utils import plot_test_metrics, index_to_target
from PIL import Image
import torch

def get_model_name():
    return f"yolov5_epochs{EPOCHS_TRAIN_Y}_bs{BATCH_SIZE_TRAIN_Y}_lr{LR_INIT_Y}_imgs{IMAGE_SIZE_Y}.pt"


def test_yolo(model_name, iou_threshold=IOU_THRESHOLD):
    # Load the actual trained model weights --> this ensures that testing is run on the best version of the model
    model_path = Path("runs/train") / model_name.replace(".pt", "") / "weights" / "best.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = YOLO(model_path)

    print(f"\nRunning test set evaluation with IoU > {iou_threshold}")
    results = model.val(
        data    = "ccpd.yaml",
        split   = "test",
        iou     = iou_threshold,
        device  = "cpu",
        name    = model_name.replace(".pt", "") + f"_test_iou{int(iou_threshold * 100)}"
    )

    metrics = results.results_dict
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Save to file
    out_file = model_name.replace(".pt", f"_test_metrics_iou{int(iou_threshold*100)}.txt")
    with open(out_file, "w") as f:
        f.write(f"# Test Set Evaluation (IoU > {iou_threshold})\n")
        f.write(f"# Hyperparams: epochs={EPOCHS_TRAIN_Y}, bs={BATCH_SIZE_TRAIN_Y}, lr={LR_INIT_Y}, imgs={IMAGE_SIZE_Y}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"Test metrics saved to: {out_file}")
    
    # Plots
    plot_test_metrics(metrics, model_name)

def pipeline_prediction(yolo_model, pdlpr_model, image_path):
    image = Image.open(image_path).convert("RGB")
    yolo_results = yolo_model(image)
    detection = yolo_results.xyxy[0]

    # Se non ci sono targhe rilevate
    if detection.shape[0] == 0:
        return []

    x1, y1, x2, y2, conf, cls = detection[0].tolist()
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    
    cropped_img = image.crop((x1, y1, x2, y2))
    
    #code to convert to plate tensor
    with torch.no_grad():
            logits = pdlpr_model(plate_tensor)
            output_probabilities = F.log_softmax(logits, dim=2)
            predictions = torch.argmax(output_probabilities, dim=2)
            pred_text = index_to_target(logits)
            plates.append(pred_text)
    
    return predicted_plate, correct_plate

if __name__ == "__main__":
    model_name = get_model_name()

    test_yolo(model_name)


    
>>>>>>> 76a834f10a4e0ed1796e9d8c5fd7754e9dbbdda9


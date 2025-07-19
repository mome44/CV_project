from ultralytics import YOLO
from pathlib import Path
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


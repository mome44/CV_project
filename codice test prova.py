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





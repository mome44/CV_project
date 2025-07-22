from ultralytics import YOLO
from pathlib import Path
from globals import *
from utils import *

def get_model_name():
    return f"yolov5_epochs{EPOCHS_TRAIN_Y}_bs{BATCH_SIZE_TRAIN_Y}_lr{LR_INIT_Y}_imgs{IMAGE_SIZE_Y}.pt"


def get_run_name():
    return get_model_name().replace(".pt", "")


def test_yolo():
    model_name = get_model_name()
    # Load the actual trained model weights --> this ensures that testing is run on the best version of the model
    model_path = Path("runs/train") / model_name.replace(".pt", "") / "weights" / "best.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    
    # Retrieve the name of the training run
    run_name = get_run_name()
    
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



model_path = "runs/train/yolov5_epochs30_bs12_lr0.001_imgs640"

run_name = Path(model_path).parent.parent.name

output_name = f"{Path(model_path).stem}_TEST_iou{int(IOU_THRESHOLD * 100)}"

# TESTING
results, model_test, test_output_dir = test_yolo()

# Save metrics in a file
metrics_path = test_output_dir / "test_metrics.txt"
test_output_dir.mkdir(parents=True, exist_ok=True)
with open(metrics_path, "w") as f:
    f.write(f"Model: {model_path}\n")
    f.write(f"IoU Threshold: {IOU_THRESHOLD}\n\n")
    f.write(f"mAP@0.5:      {results.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\n")
    f.write(f"Precision:    {results.box.mp:.4f}\n")
    f.write(f"Recall:       {results.box.mr:.4f}\n")


# Compute IoU
image_dir = Path("dataset/images/test")

iou_list = []

# Loop sover images
for image_path in sorted(image_dir.glob("*.jpg")):
    # Predict
    result = model_test(str(image_path), max_det=5, verbose = False)[0]
    predictions = result.boxes.xyxy.cpu().numpy()  # shape: (N, 4)

    real_box = load_gt_box_from_label_test(image_path)
    if real_box is None:
        # skip image if no GT or invalid
        continue 

    # Compute IoU between every predicted box and the true one
    for predicted_box in predictions:
        iou = compute_iou(predicted_box, real_box)
        iou_list.append(iou)

# Compute average among all iou values
if iou_list:
    mean_iou = sum(iou_list) / len(iou_list)
else:
    mean_iou = 0.0

# Save in test_metrics.txt
with open(metrics_path, "a") as f:
    f.write(f"Mean IoU over test set: {mean_iou:.4f}\n")

print(f"[INFO] Mean IoU saved to {metrics_path}")



print("\n TESTING complete!")
print(f"Results saved to: runs/test/{test_output_dir}")


from ultralytics import YOLO
from pathlib import Path
from globals import BATCH_SIZE_TRAIN_Y, LR_INIT_Y, EPOCHS_TRAIN_Y, IMAGE_SIZE_Y, IOU_THRESHOLD
from utils import plot_test_metrics, index_to_target
from PIL import Image
import torch


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
    
    return predicted_plate

def baseline_pipeline_prediction(cnnctc_model, image_path):
    return predicted_plate



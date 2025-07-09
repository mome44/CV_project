import torch
from pathlib import Path

"""
Fine-tune YOLOv5 on your custom dataset using Python API.
Make sure 'ccpd.yaml' and image/label folders are correctly prepared.
"""

# === CONFIGURATION ===
data_yaml = 'ccpd.yaml'            # Dataset YAML file
model_cfg = 'yolov5s.yaml'         # Model architecture (optional, auto-handled)
weights = 'yolov5s.pt'             # Pretrained weights to fine-tune from
img_size = 640                     # Input image size
batch_size = 16                    # Batch size
epochs = 80                        # Number of fine-tuning epochs
name = 'ccpd_finetune'             # Run name
device = 'cpu'

# === CUSTOM HYPERPARAMETERS ===
lr0 = 0.001                        # Initial learning rate
lrf = 0.01                         # Final LR multiplier
momentum = 0.9                     # SGD momentum
weight_decay = 0.0002              # L2 regularization
warmup_epochs = 3.0
warmup_bias_lr = 0.1

# === TRAINING ===
print(f"\n Starting YOLOv5 fine-tuning on: {data_yaml}")

torch.hub.load('ultralytics/yolov5', 'train',
    data=data_yaml,
    cfg=model_cfg,
    weights=weights,
    imgsz=img_size,
    batch_size=batch_size,
    epochs=epochs,
    lr0=lr0,
    lrf=lrf,
    momentum=momentum,
    weight_decay=weight_decay,
    warmup_epochs=warmup_epochs,
    warmup_bias_lr=warmup_bias_lr,
    name=name,
    device=device
)